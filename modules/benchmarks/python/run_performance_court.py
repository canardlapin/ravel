#!/usr/bin/env python3
"""Run Ravel's parity-gated JVM, NumPy, and Scala.js performance court."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA = "ravel.performance-court.manifest.v1"
SOURCE_SUFFIXES = {
    ".css",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".properties",
    ".py",
    ".sbt",
    ".scala",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
SOURCE_ROOTS = ("project", "modules", "scripts")
EXCLUDED_PARTS = {"__pycache__", "results", "target"}
ALLOCATION_CASES = (
    "contiguous_subtract_double",
    "inner_stride_multiply_double",
    "broadcast_subtract_double",
    "greater_equal_scalar_double",
    "is_finite_double",
    "cast_double_int",
    "full_mean_double",
    "axis0_mean_double",
    "inplace_multiply_double",
)


@dataclass(frozen=True)
class HostLimits:
    normalized_load_1m: float = 0.15
    minimum_available_memory_ratio: float = 0.10
    minimum_memory_pressure_free_ratio: float = 0.25
    maximum_compressed_memory_ratio: float = 0.35
    maximum_process_cpu_percent: float = 20.0
    maximum_pageins_per_second: float = 100.0
    maximum_decompressions_per_second: float = 100.0


@dataclass(frozen=True)
class HostSnapshot:
    captured_at: str
    logical_cpus: int
    load_1m: float
    normalized_load_1m: float
    available_memory_ratio: float | None
    compressed_memory_ratio: float | None
    busiest_process_pid: int | None
    busiest_process_cpu_percent: float | None
    busiest_process_command: str | None
    memory_pressure_free_ratio: float | None = None
    pageins: int | None = None
    pageouts: int | None = None
    swapins: int | None = None
    swapouts: int | None = None
    compressions: int | None = None
    decompressions: int | None = None


@dataclass(frozen=True)
class Phase:
    name: str
    timing: bool
    outputs: tuple[str, ...]
    command: Callable[["Context"], list[str]]
    before: tuple[Callable[["Context"], list[str]], ...] = ()
    after: tuple[Callable[["Context"], list[str]], ...] = ()


@dataclass(frozen=True)
class Context:
    repo: Path
    out: Path
    python: Path
    java_home: Path | None = None

    def subprocess_environment(self) -> dict[str, str]:
        result = dict(os.environ)
        if self.java_home is not None:
            result["JAVA_HOME"] = str(self.java_home)
            result[
                "PATH"
            ] = f"{self.java_home / 'bin'}{os.pathsep}{result.get('PATH', '')}"
        return result


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_capture(
    command: Sequence[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(repo: Path) -> list[Path]:
    candidates = [repo / "build.sbt"]
    for root_name in SOURCE_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(repo).parts)
        )
    return sorted(set(candidates))


def source_fingerprint(repo: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = source_files(repo)
    for path in files:
        relative = path.relative_to(repo).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    try:
        head = run_capture(["git", "rev-parse", "HEAD"], repo)
        dirty = bool(
            run_capture(["git", "status", "--short", "--untracked-files=no"], repo)
        )
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
        dirty = True
    return {
        "head": head,
        "dirty_tracked_tree": dirty,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {field: source.get(field) for field in ("head", "sha256", "file_count")}


def configuration(
    python: Path,
    java_home: Path,
    limits: HostLimits,
    host_samples: int,
    host_interval_seconds: float,
) -> dict[str, Any]:
    return {
        "python": str(python.resolve()),
        "java_home": str(java_home.resolve()),
        "operation_sides": [64, 256, 1024],
        "cross_runtime_sides": [64, 256, 1024],
        "numpy": {"warmups": 3, "samples": 9, "target_ms": 150},
        "node": {"warmup_ms": 200, "sample_ms": 250, "samples": 7},
        "jmh": {"forks": 2, "warmups": 5, "measurements": 7, "iteration_ms": 500},
        "allocation_jmh": {
            "forks": 1,
            "warmups": 3,
            "measurements": 5,
            "iteration_ms": 300,
            "cases": list(ALLOCATION_CASES),
        },
        "host_qualification": {
            **dataclasses.asdict(limits),
            "samples": host_samples,
            "interval_seconds": host_interval_seconds,
        },
    }


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def empty_memory_metrics() -> dict[str, float | int | None]:
    return {
        "available_memory_ratio": None,
        "compressed_memory_ratio": None,
        "memory_pressure_free_ratio": None,
        "pageins": None,
        "pageouts": None,
        "swapins": None,
        "swapouts": None,
        "compressions": None,
        "decompressions": None,
    }


def memory_metrics() -> dict[str, float | int | None]:
    if sys.platform == "darwin":
        return darwin_memory_metrics()
    if sys.platform.startswith("linux"):
        return linux_memory_metrics()
    return empty_memory_metrics()


def darwin_memory_metrics() -> dict[str, float | int | None]:
    result = empty_memory_metrics()
    try:
        total = int(run_capture(["sysctl", "-n", "hw.memsize"], Path.cwd()))
        output = run_capture(["vm_stat"], Path.cwd())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return result
    page_match = re.search(r"page size of (\d+) bytes", output)
    if not page_match or total <= 0:
        return result
    page_size = int(page_match.group(1))
    values: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\.", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    available_pages = sum(
        values.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative")
    )
    compressed_pages = values.get("Pages occupied by compressor", 0)
    result.update(
        {
            "available_memory_ratio": available_pages * page_size / total,
            "compressed_memory_ratio": compressed_pages * page_size / total,
            "pageins": values.get("Pageins"),
            "pageouts": values.get("Pageouts"),
            "swapins": values.get("Swapins"),
            "swapouts": values.get("Swapouts"),
            "compressions": values.get("Compressions"),
            "decompressions": values.get("Decompressions"),
        }
    )
    try:
        pressure = run_capture(["memory_pressure", "-Q"], Path.cwd())
        match = re.search(r"free percentage:\s*(\d+(?:\.\d+)?)%", pressure)
        if match:
            result["memory_pressure_free_ratio"] = float(match.group(1)) / 100.0
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def linux_memory_metrics() -> dict[str, float | int | None]:
    result = empty_memory_metrics()
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
        total = values["MemTotal"]
        available = values["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return result
    result["available_memory_ratio"] = available / total
    return result


def busiest_process() -> tuple[int | None, float | None, str | None]:
    try:
        output = run_capture(
            ["ps", "-Ao", "pid=,pcpu=,rss=,comm="],
            Path.cwd(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    candidates: list[tuple[float, int, str]] = []
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[0])
            cpu = float(fields[1])
        except ValueError:
            continue
        if pid != os.getpid():
            candidates.append((cpu, pid, fields[3]))
    if not candidates:
        return None, None, None
    cpu, pid, command = max(candidates)
    return pid, cpu, command


def collect_host_snapshot() -> HostSnapshot:
    cpus = os.cpu_count() or 1
    load = os.getloadavg()[0]
    memory = memory_metrics()
    pid, process_cpu, command = busiest_process()
    return HostSnapshot(
        captured_at=utc_now(),
        logical_cpus=cpus,
        load_1m=load,
        normalized_load_1m=load / cpus,
        available_memory_ratio=memory["available_memory_ratio"],  # type: ignore[arg-type]
        compressed_memory_ratio=memory["compressed_memory_ratio"],  # type: ignore[arg-type]
        busiest_process_pid=pid,
        busiest_process_cpu_percent=process_cpu,
        busiest_process_command=command,
        memory_pressure_free_ratio=memory["memory_pressure_free_ratio"],  # type: ignore[arg-type]
        pageins=memory["pageins"],  # type: ignore[arg-type]
        pageouts=memory["pageouts"],  # type: ignore[arg-type]
        swapins=memory["swapins"],  # type: ignore[arg-type]
        swapouts=memory["swapouts"],  # type: ignore[arg-type]
        compressions=memory["compressions"],  # type: ignore[arg-type]
        decompressions=memory["decompressions"],  # type: ignore[arg-type]
    )


def evaluate_host(snapshot: HostSnapshot, limits: HostLimits) -> list[str]:
    reasons: list[str] = []
    if snapshot.normalized_load_1m > limits.normalized_load_1m:
        reasons.append(
            f"normalized load {snapshot.normalized_load_1m:.3f} exceeds "
            f"{limits.normalized_load_1m:.3f}"
        )
    if snapshot.available_memory_ratio is None:
        reasons.append("available-memory ratio is unavailable")
    elif snapshot.available_memory_ratio < limits.minimum_available_memory_ratio:
        reasons.append(
            f"available memory {snapshot.available_memory_ratio:.1%} is below "
            f"{limits.minimum_available_memory_ratio:.1%}"
        )
    if snapshot.memory_pressure_free_ratio is not None and (
        snapshot.memory_pressure_free_ratio < limits.minimum_memory_pressure_free_ratio
    ):
        reasons.append(
            f"memory-pressure free percentage "
            f"{snapshot.memory_pressure_free_ratio:.1%} is below "
            f"{limits.minimum_memory_pressure_free_ratio:.1%}"
        )
    if snapshot.compressed_memory_ratio is not None and (
        snapshot.compressed_memory_ratio > limits.maximum_compressed_memory_ratio
    ):
        reasons.append(
            f"compressed memory {snapshot.compressed_memory_ratio:.1%} exceeds "
            f"{limits.maximum_compressed_memory_ratio:.1%}"
        )
    if snapshot.busiest_process_cpu_percent is None:
        reasons.append("busiest-process CPU is unavailable")
    elif snapshot.busiest_process_cpu_percent > limits.maximum_process_cpu_percent:
        reasons.append(
            f"busiest process CPU {snapshot.busiest_process_cpu_percent:.1f}% exceeds "
            f"{limits.maximum_process_cpu_percent:.1f}%"
        )
    return reasons


def evaluate_memory_activity(
    previous: HostSnapshot,
    current: HostSnapshot,
    elapsed_seconds: float,
    limits: HostLimits,
) -> list[str]:
    if elapsed_seconds <= 0.0:
        return []
    reasons: list[str] = []

    def delta(field: str) -> int | None:
        before = getattr(previous, field)
        after = getattr(current, field)
        if before is None or after is None or after < before:
            return None
        return int(after - before)

    pageins = delta("pageins")
    if pageins is not None:
        rate = pageins / elapsed_seconds
        if rate > limits.maximum_pageins_per_second:
            reasons.append(
                f"pageins {rate:.1f}/s exceed "
                f"{limits.maximum_pageins_per_second:.1f}/s"
            )
    decompressions = delta("decompressions")
    if decompressions is not None:
        rate = decompressions / elapsed_seconds
        if rate > limits.maximum_decompressions_per_second:
            reasons.append(
                f"decompressions {rate:.1f}/s exceed "
                f"{limits.maximum_decompressions_per_second:.1f}/s"
            )
    for field in ("pageouts", "swapins", "swapouts"):
        activity = delta(field)
        if activity is not None and activity > 0:
            reasons.append(f"{field} changed by {activity}")
    return reasons


def qualify_host(
    limits: HostLimits,
    sample_count: int,
    interval_seconds: float,
    collector: Callable[[], HostSnapshot] = collect_host_snapshot,
) -> dict[str, Any]:
    snapshots: list[HostSnapshot] = []
    captured_monotonic: list[float] = []
    failures: list[dict[str, Any]] = []
    for index in range(sample_count):
        snapshot = collector()
        observed_at = time.monotonic()
        snapshots.append(snapshot)
        captured_monotonic.append(observed_at)
        reasons = evaluate_host(snapshot, limits)
        elapsed_seconds = None
        if index > 0:
            elapsed_seconds = observed_at - captured_monotonic[index - 1]
            reasons.extend(
                evaluate_memory_activity(
                    snapshots[index - 1],
                    snapshot,
                    elapsed_seconds,
                    limits,
                )
            )
        if reasons:
            failure: dict[str, Any] = {"sample": index + 1, "reasons": reasons}
            if elapsed_seconds is not None:
                failure["elapsed_seconds"] = elapsed_seconds
            failures.append(failure)
        if index + 1 < sample_count:
            time.sleep(interval_seconds)
    return {
        "qualified": not failures,
        "limits": dataclasses.asdict(limits),
        "snapshots": [dataclasses.asdict(snapshot) for snapshot in snapshots],
        "failures": failures,
    }


def relative(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def js_main(ctx: Context) -> Path:
    signatures = json.loads(
        (ctx.out / "jvm-cross-runtime-signatures.json").read_text(encoding="utf-8")
    )
    scala_version = signatures["metadata"]["scala_version"]
    path = (
        ctx.repo
        / "modules"
        / "benchmarks"
        / "js"
        / "target"
        / f"scala-{scala_version}"
        / "ravel-representation-probe-js-opt"
        / "main.js"
    )
    if not path.is_file():
        raise FileNotFoundError(f"full-linked Scala.js entry point not found: {path}")
    return path


def phases() -> list[Phase]:
    return [
        Phase(
            "representation-proof",
            False,
            (),
            lambda ctx: ["sbt", "representationProof"],
        ),
        Phase(
            "laws-proof",
            False,
            (),
            lambda ctx: ["sbt", ";lawsJVM/test;lawsJS/test"],
        ),
        Phase(
            "browser-correctness",
            False,
            (),
            lambda ctx: ["sbt", "browserTests/test"],
            before=(
                lambda ctx: [
                    "node",
                    str(
                        Path.home()
                        / ".local/share/agent-policy/browser-automation-guard.mjs"
                    ),
                    "--audit",
                ],
            ),
            after=(
                lambda ctx: [
                    "node",
                    str(
                        Path.home()
                        / ".local/share/agent-policy/browser-automation-guard.mjs"
                    ),
                    "--audit",
                ],
            ),
        ),
        Phase(
            "jvm-signatures",
            False,
            ("ravel-signatures.json",),
            lambda ctx: [
                "sbt",
                "representationProbeJVM/runMain ravel.bench.OperationMatrixParity "
                f"--out {ctx.out / 'ravel-signatures.json'} --side 64,256,1024",
            ],
        ),
        Phase(
            "numpy-signatures",
            False,
            ("numpy-signatures.json",),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/numpy_operation_matrix.py",
                "--side",
                "64",
                "--side",
                "256",
                "--side",
                "1024",
                "--signatures-only",
                "--out",
                str(ctx.out / "numpy-signatures.json"),
            ],
        ),
        Phase(
            "numpy-parity",
            False,
            (),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/compare_operation_matrix.py",
                "--numpy",
                str(ctx.out / "numpy-signatures.json"),
                "--signatures",
                str(ctx.out / "ravel-signatures.json"),
                "--parity-only",
            ],
        ),
        Phase(
            "jvm-cross-runtime-signatures",
            False,
            ("jvm-cross-runtime-signatures.json",),
            lambda ctx: [
                "sbt",
                "representationProbeJVM/runMain ravel.bench.CrossRuntimeCourt "
                "--signatures-only --sides 64,256,1024 "
                f"--out {ctx.out / 'jvm-cross-runtime-signatures.json'}",
            ],
        ),
        Phase(
            "scala-js-full-link",
            False,
            (),
            lambda ctx: ["sbt", "representationProbeJS/fullLinkJS"],
        ),
        Phase(
            "scala-js-signatures",
            False,
            ("scala-js-signatures.json",),
            lambda ctx: [
                "node",
                "--expose-gc",
                str(js_main(ctx)),
                "--signatures-only",
                "--sides",
                "64,256,1024",
                "--out",
                str(ctx.out / "scala-js-signatures.json"),
            ],
        ),
        Phase(
            "cross-runtime-parity",
            False,
            (),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/compare_cross_runtime_court.py",
                "--jvm",
                str(ctx.out / "jvm-cross-runtime-signatures.json"),
                "--js",
                str(ctx.out / "scala-js-signatures.json"),
            ],
        ),
        Phase(
            "jvm-timings",
            True,
            ("ravel-jmh.json",),
            lambda ctx: [
                "sbt",
                "representationProbeJVM/Jmh/run -rf json "
                f"-rff {ctx.out / 'ravel-jmh.json'} "
                "ravel.bench.OperationMatrixBenchmarks.operation_matrix",
            ],
        ),
        Phase(
            "numpy-timings",
            True,
            ("numpy.json",),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/numpy_operation_matrix.py",
                "--side",
                "64",
                "--side",
                "256",
                "--side",
                "1024",
                "--out",
                str(ctx.out / "numpy.json"),
            ],
        ),
        Phase(
            "jvm-allocation",
            True,
            ("ravel-jmh-gc.json",),
            lambda ctx: [
                "sbt",
                "representationProbeJVM/Jmh/run -f 1 -wi 3 -i 5 "
                "-w 300ms -r 300ms -prof gc -rf json "
                f"-rff {ctx.out / 'ravel-jmh-gc.json'} "
                "-p side=64,256,1024 "
                f"-p caseName={','.join(ALLOCATION_CASES)} "
                "ravel.bench.OperationMatrixBenchmarks.operation_matrix",
            ],
        ),
        Phase(
            "scala-js-timings",
            True,
            ("node.json",),
            lambda ctx: [
                "node",
                "--expose-gc",
                str(js_main(ctx)),
                "--sides",
                "64,256,1024",
                "--warmup-ms",
                "200",
                "--sample-ms",
                "250",
                "--samples",
                "7",
                "--progress",
                "--out",
                str(ctx.out / "node.json"),
            ],
        ),
        Phase(
            "scala-js-baseline-summary",
            False,
            ("node-summary.md",),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/compare_cross_runtime_court.py",
                "--jvm",
                str(ctx.out / "jvm-cross-runtime-signatures.json"),
                "--js",
                str(ctx.out / "scala-js-signatures.json"),
                "--summarize",
                str(ctx.out / "node.json"),
                "--out",
                str(ctx.out / "node-summary.md"),
            ],
        ),
        Phase(
            "scala-js-gc-diagnostic",
            True,
            ("node-gc.json",),
            lambda ctx: [
                "node",
                "--expose-gc",
                "--trace-gc",
                str(js_main(ctx)),
                "--sides",
                "64,256,1024",
                "--warmup-ms",
                "200",
                "--sample-ms",
                "250",
                "--samples",
                "7",
                "--out",
                str(ctx.out / "node-gc.json"),
            ],
        ),
        Phase(
            "jvm-numpy-report",
            False,
            ("comparison.md",),
            lambda ctx: [
                str(ctx.python),
                "modules/benchmarks/python/compare_operation_matrix.py",
                "--jmh",
                str(ctx.out / "ravel-jmh.json"),
                "--numpy",
                str(ctx.out / "numpy.json"),
                "--signatures",
                str(ctx.out / "ravel-signatures.json"),
                "--out",
                str(ctx.out / "comparison.md"),
            ],
        ),
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_receipt(ctx: Context, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"phase did not produce required artifact: {path}")
    return {
        "path": relative(path, ctx.repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def output_receipts(ctx: Context, output_names: Sequence[str]) -> list[dict[str, Any]]:
    return [artifact_receipt(ctx, ctx.out / name) for name in output_names]


def completed_phase_is_intact(repo: Path, record: dict[str, Any]) -> bool:
    receipts = list(record.get("outputs", []))
    if record.get("log_receipt") is not None:
        receipts.append(record["log_receipt"])
    for receipt in receipts:
        path = repo / receipt["path"]
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            return False
    return True


def execute_logged(
    command: Sequence[str],
    cwd: Path,
    log: Path,
    *,
    append: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="")
        return process.wait()


def execute_phase_logged(phase: Phase, ctx: Context, log: Path) -> int:
    before_commands = [factory(ctx) for factory in phase.before]
    command = phase.command(ctx)
    after_commands = [factory(ctx) for factory in phase.after]
    return_code = 0
    log_started = False
    caught: BaseException | None = None
    process_environment = ctx.subprocess_environment()
    try:
        for before in before_commands:
            return_code = execute_logged(
                before,
                ctx.repo,
                log,
                append=log_started,
                env=process_environment,
            )
            log_started = True
            if return_code != 0:
                break
        if return_code == 0:
            return_code = execute_logged(
                command,
                ctx.repo,
                log,
                append=log_started,
                env=process_environment,
            )
            log_started = True
    except BaseException as error:
        caught = error
    finally:
        for after in after_commands:
            try:
                after_code = execute_logged(
                    after,
                    ctx.repo,
                    log,
                    append=log_started,
                    env=process_environment,
                )
                log_started = True
                if return_code == 0 and after_code != 0:
                    return_code = after_code
            except BaseException as error:
                if caught is None:
                    caught = error
    if caught is not None:
        raise caught
    return return_code


def executable_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def environment(ctx: Context) -> dict[str, Any]:
    process_environment = ctx.subprocess_environment()
    path = process_environment.get("PATH")
    resolved_commands: dict[str, Path] = {}
    for name in ("java", "node", "sbt"):
        resolved = shutil.which(name, path=path)
        if resolved is None:
            raise RuntimeError(f"required runtime executable unavailable: {name}")
        resolved_commands[name] = Path(resolved)
    resolved_commands["python"] = ctx.python

    try:
        java = run_capture(
            ["java", "-version"],
            ctx.repo,
            env=process_environment,
        )
        node = json.loads(
            run_capture(
                [
                    "node",
                    "-p",
                    "JSON.stringify({node:process.version,"
                    "v8:process.versions.v8,arch:process.arch,"
                    "platform:process.platform})",
                ],
                ctx.repo,
                env=process_environment,
            )
        )
        python = json.loads(
            run_capture(
                [
                    str(ctx.python),
                    "-c",
                    "import json,platform,sys;"
                    "print(json.dumps({'version':sys.version,"
                    "'implementation':platform.python_implementation(),"
                    "'executable':sys.executable},sort_keys=True))",
                ],
                ctx.repo,
                env=process_environment,
            )
        )
        numpy = json.loads(
            run_capture(
                [
                    str(ctx.python),
                    "-c",
                    "import json,numpy as np;"
                    "print(json.dumps({'version':np.__version__,"
                    "'file':np.__file__,'configuration':"
                    "np.show_config(mode='dicts')},sort_keys=True))",
                ],
                ctx.repo,
                env=process_environment,
            )
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            f"required runtime environment unavailable: {error}"
        ) from error
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "executables": {
            name: executable_identity(executable)
            for name, executable in resolved_commands.items()
        },
        "java": java,
        "node": node,
        "python": python,
        "numpy": numpy,
    }


def initialize_manifest(
    ctx: Context,
    source: dict[str, Any],
    config: dict[str, Any],
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema": SCHEMA,
        "state": "running",
        "created_at": now,
        "updated_at": now,
        "source": source,
        "configuration": config,
        "configuration_sha256": stable_json_hash(config),
        "environment": runtime_environment,
        "phases": {
            phase.name: {"status": "pending", "timing": phase.timing}
            for phase in phases()
        },
        "host_qualifications": [],
    }


def validate_resume(
    manifest: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    runtime_environment: dict[str, Any] | None = None,
) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if source_identity(manifest.get("source", {})) != source_identity(source):
        raise ValueError("source fingerprint changed; start a new receipt directory")
    if manifest.get("configuration_sha256") != stable_json_hash(config):
        raise ValueError("court configuration changed; start a new receipt directory")
    if (
        runtime_environment is not None
        and manifest.get("environment") != runtime_environment
    ):
        raise ValueError("runtime environment changed; start a new receipt directory")


def validate_execution_context(
    expected_source: dict[str, Any],
    current_source: dict[str, Any],
    expected_environment: dict[str, Any],
    current_environment: dict[str, Any],
) -> None:
    if source_identity(current_source) != source_identity(expected_source):
        raise RuntimeError(
            "source changed during the court; discard this receipt and start "
            "a new directory after concurrent work is complete"
        )
    if current_environment != expected_environment:
        raise RuntimeError(
            "runtime environment changed during the court; discard this receipt "
            "and start a new directory"
        )


def verify_execution_context(
    ctx: Context,
    expected_source: dict[str, Any],
    expected_environment: dict[str, Any],
) -> dict[str, Any]:
    current_source = source_fingerprint(ctx.repo)
    current_environment = environment(ctx)
    validate_execution_context(
        expected_source,
        current_source,
        expected_environment,
        current_environment,
    )
    return {
        "verified_at": utc_now(),
        "source": source_identity(current_source),
        "environment_sha256": stable_json_hash(current_environment),
    }


def render_summary(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    lines = [
        "# Ravel performance-court receipt",
        "",
        f"- State: `{manifest['state']}`",
        f"- Source commit: `{source['head']}`",
        f"- Source fingerprint: `{source['sha256']}`",
        f"- Dirty tracked tree at start: `{source['dirty_tracked_tree']}`",
        "",
        "## Phases",
        "",
        "| phase | kind | status | outputs |",
        "|---|---|---|---|",
    ]
    for name, record in manifest["phases"].items():
        outputs = ", ".join(
            f"`{Path(receipt['path']).name}`" for receipt in record.get("outputs", [])
        )
        lines.append(
            f"| `{name}` | {'timing' if record['timing'] else 'gate'} | "
            f"{record['status']} | {outputs} |"
        )
    lines.extend(
        [
            "",
            "Each timing phase passed a fresh multi-sample host qualification. "
            "See `manifest.json` and the phase logs for exact commands, runtime "
            "versions, thresholds, snapshots, and artifact checksums.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="new or resumable dated receipt directory",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter with the pinned benchmark requirements installed",
    )
    parser.add_argument(
        "--java-home",
        type=Path,
        default=Path(os.environ["JAVA_HOME"]) if "JAVA_HOME" in os.environ else None,
        help="explicit JDK home; defaults to JAVA_HOME and is required otherwise",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-phases", action="store_true")
    parser.add_argument("--host-samples", type=int, default=5)
    parser.add_argument("--host-interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-normalized-load", type=float, default=0.15)
    parser.add_argument("--min-available-memory", type=float, default=0.10)
    parser.add_argument("--min-memory-pressure-free", type=float, default=0.25)
    parser.add_argument("--max-compressed-memory", type=float, default=0.35)
    parser.add_argument("--max-process-cpu", type=float, default=20.0)
    parser.add_argument("--max-pageins-per-second", type=float, default=100.0)
    parser.add_argument("--max-decompressions-per-second", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    out = args.out.resolve()
    python = args.python.resolve()
    if args.java_home is None:
        raise SystemExit("--java-home is required when JAVA_HOME is unset")
    java_home = args.java_home.resolve()
    if not (java_home / "bin/java").is_file():
        raise SystemExit(f"--java-home does not contain bin/java: {java_home}")
    ctx = Context(repo=repo, out=out, python=python, java_home=java_home)
    selected_phases = phases()
    if args.list_phases:
        for phase in selected_phases:
            print(f"{phase.name}\t{'timing' if phase.timing else 'gate'}")
        return
    if args.host_samples < 1 or args.host_interval_seconds < 0:
        raise SystemExit("host samples must be positive and interval non-negative")
    limits = HostLimits(
        normalized_load_1m=args.max_normalized_load,
        minimum_available_memory_ratio=args.min_available_memory,
        minimum_memory_pressure_free_ratio=args.min_memory_pressure_free,
        maximum_compressed_memory_ratio=args.max_compressed_memory,
        maximum_process_cpu_percent=args.max_process_cpu,
        maximum_pageins_per_second=args.max_pageins_per_second,
        maximum_decompressions_per_second=args.max_decompressions_per_second,
    )
    if any(value < 0 for value in dataclasses.asdict(limits).values()):
        raise SystemExit("host qualification thresholds must be non-negative")
    for name, value in (
        ("--min-available-memory", limits.minimum_available_memory_ratio),
        ("--min-memory-pressure-free", limits.minimum_memory_pressure_free_ratio),
        ("--max-compressed-memory", limits.maximum_compressed_memory_ratio),
    ):
        if value > 1.0:
            raise SystemExit(f"{name} must be at most 1.0")

    source = source_fingerprint(repo)
    config = configuration(
        python,
        java_home,
        limits,
        args.host_samples,
        args.host_interval_seconds,
    )
    if args.dry_run:
        print(f"source sha256: {source['sha256']}")
        print(f"configuration sha256: {stable_json_hash(config)}")
        for phase in selected_phases:
            try:
                command = phase.command(ctx)
                rendered = shlex.join(command)
            except (FileNotFoundError, KeyError):
                rendered = "<depends on completed Scala.js link/signature phases>"
            print(f"{phase.name} ({'timing' if phase.timing else 'gate'}): {rendered}")
        return

    manifest_path = out / "manifest.json"
    runtime_environment = environment(ctx)
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f"cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume(manifest, source, config, runtime_environment)
        manifest["state"] = "running"
    else:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(
                f"refusing to mix a new court with non-empty directory {out}"
            )
        out.mkdir(parents=True, exist_ok=True)
        manifest = initialize_manifest(ctx, source, config, runtime_environment)
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)

    try:
        for phase in selected_phases:
            record = manifest["phases"][phase.name]
            if record["status"] == "completed":
                if not completed_phase_is_intact(repo, record):
                    raise RuntimeError(
                        f"completed phase output checksum changed: {phase.name}"
                    )
                print(f"[court] skip completed phase {phase.name}")
                continue

            if phase.timing:
                print(f"[court] qualifying host before {phase.name}")
                qualification = qualify_host(
                    limits,
                    args.host_samples,
                    args.host_interval_seconds,
                )
                qualification["phase"] = phase.name
                manifest["host_qualifications"].append(qualification)
                record["host_qualification"] = qualification
                manifest["updated_at"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                if not qualification["qualified"]:
                    record["status"] = "pending"
                    manifest["state"] = "host_blocked"
                    manifest["updated_at"] = utc_now()
                    atomic_write_json(manifest_path, manifest)
                    raise SystemExit(
                        f"host qualification failed before {phase.name}; "
                        f"resume the same receipt when the host is calm"
                    )

            record["context_before"] = verify_execution_context(
                ctx,
                manifest["source"],
                manifest["environment"],
            )
            command = phase.command(ctx)
            before_commands = [factory(ctx) for factory in phase.before]
            after_commands = [factory(ctx) for factory in phase.after]
            log = out / f"{phase.name}.log"
            record.update(
                {
                    "status": "running",
                    "started_at": utc_now(),
                    "command": command,
                    "before_commands": before_commands,
                    "after_commands": after_commands,
                    "log": relative(log, repo),
                }
            )
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            print(f"[court] start {phase.name}: {shlex.join(command)}")
            return_code = execute_phase_logged(phase, ctx, log)
            record["return_code"] = return_code
            if return_code != 0:
                record["status"] = "failed"
                record["finished_at"] = utc_now()
                manifest["state"] = "failed"
                manifest["updated_at"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                raise SystemExit(f"phase failed ({return_code}): {phase.name}")
            record["context_after"] = verify_execution_context(
                ctx,
                manifest["source"],
                manifest["environment"],
            )
            record["log_receipt"] = artifact_receipt(ctx, log)
            record["outputs"] = output_receipts(ctx, phase.outputs)
            record["status"] = "completed"
            record["finished_at"] = utc_now()
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)

        manifest["completion_context"] = verify_execution_context(
            ctx,
            manifest["source"],
            manifest["environment"],
        )
        manifest["state"] = "completed"
        manifest["updated_at"] = utc_now()
        summary_path = out / "README.md"
        summary_path.write_text(render_summary(manifest), encoding="utf-8")
        manifest["summary"] = output_receipts(ctx, ("README.md",))[0]
        atomic_write_json(manifest_path, manifest)
        print(f"[court] complete: {manifest_path}")
    except KeyboardInterrupt:
        manifest["state"] = "interrupted"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise
    except Exception as error:
        for record in manifest["phases"].values():
            if record["status"] == "running":
                record["status"] = "failed"
                record["finished_at"] = utc_now()
                record["error"] = f"{type(error).__name__}: {error}"
        manifest["state"] = "failed"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
