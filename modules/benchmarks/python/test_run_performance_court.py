from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import run_performance_court as court  # noqa: E402


def snapshot(
    *,
    load: float = 0.10,
    available: float | None = 0.50,
    compressed: float | None = 0.05,
    process_cpu: float | None = 5.0,
    pressure_free: float | None = 0.50,
    pageins: int | None = None,
    pageouts: int | None = None,
    swapins: int | None = None,
    swapouts: int | None = None,
    decompressions: int | None = None,
) -> court.HostSnapshot:
    return court.HostSnapshot(
        captured_at="2026-07-29T00:00:00+00:00",
        logical_cpus=10,
        load_1m=load * 10,
        normalized_load_1m=load,
        available_memory_ratio=available,
        compressed_memory_ratio=compressed,
        busiest_process_pid=42,
        busiest_process_cpu_percent=process_cpu,
        busiest_process_command="test",
        memory_pressure_free_ratio=pressure_free,
        pageins=pageins,
        pageouts=pageouts,
        swapins=swapins,
        swapouts=swapouts,
        decompressions=decompressions,
    )


class PerformanceCourtDriverTest(unittest.TestCase):
    def test_host_gate_accepts_only_consecutive_clean_samples(self) -> None:
        samples = iter([snapshot(), snapshot()])
        result = court.qualify_host(
            court.HostLimits(),
            sample_count=2,
            interval_seconds=0,
            collector=lambda: next(samples),
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["failures"], [])

    def test_host_gate_reports_every_violated_invariant(self) -> None:
        reasons = court.evaluate_host(
            snapshot(load=0.75, available=0.05, compressed=0.45, process_cpu=40.0),
            court.HostLimits(),
        )
        self.assertEqual(len(reasons), 4)
        self.assertTrue(any("normalized load" in reason for reason in reasons))
        self.assertTrue(any("available memory" in reason for reason in reasons))
        self.assertTrue(any("compressed memory" in reason for reason in reasons))
        self.assertTrue(any("busiest process" in reason for reason in reasons))

    def test_host_gate_rejects_active_paging_not_stale_compression_alone(self) -> None:
        limits = court.HostLimits()
        previous = snapshot(
            compressed=0.30,
            pageins=1_000,
            pageouts=10,
            swapins=20,
            swapouts=30,
            decompressions=2_000,
        )
        current = snapshot(
            compressed=0.30,
            pageins=1_500,
            pageouts=10,
            swapins=21,
            swapouts=30,
            decompressions=2_600,
        )
        self.assertEqual(court.evaluate_host(current, limits), [])
        reasons = court.evaluate_memory_activity(
            previous,
            current,
            elapsed_seconds=2.0,
            limits=limits,
        )
        self.assertTrue(any("pageins" in reason for reason in reasons))
        self.assertTrue(any("decompressions" in reason for reason in reasons))
        self.assertTrue(any("swapins" in reason for reason in reasons))

    def test_resume_rejects_source_or_configuration_drift(self) -> None:
        config = {"host": {"samples": 5}}
        manifest = {
            "schema": court.SCHEMA,
            "source": {"head": "head-a", "sha256": "source-a", "file_count": 1},
            "configuration_sha256": court.stable_json_hash(config),
            "environment": {"node": "v1"},
        }
        court.validate_resume(
            manifest,
            {"head": "head-a", "sha256": "source-a", "file_count": 1},
            config,
            {"node": "v1"},
        )
        with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
            court.validate_resume(
                manifest,
                {"head": "head-a", "sha256": "source-b", "file_count": 1},
                config,
            )
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            court.validate_resume(
                manifest,
                {"head": "head-a", "sha256": "source-a", "file_count": 1},
                {"host": {"samples": 3}},
            )
        with self.assertRaisesRegex(ValueError, "runtime environment changed"):
            court.validate_resume(
                manifest,
                {"head": "head-a", "sha256": "source-a", "file_count": 1},
                config,
                {"node": "v2"},
            )

    def test_phase_context_rejects_concurrent_source_or_runtime_drift(self) -> None:
        source = {"head": "head-a", "sha256": "source-a", "file_count": 10}
        runtime = {"node": "v24", "java": "25"}
        court.validate_execution_context(source, dict(source), runtime, dict(runtime))
        with self.assertRaisesRegex(RuntimeError, "source changed during the court"):
            court.validate_execution_context(
                source,
                {**source, "sha256": "source-b"},
                runtime,
                dict(runtime),
            )
        with self.assertRaisesRegex(RuntimeError, "runtime environment changed"):
            court.validate_execution_context(
                source,
                dict(source),
                runtime,
                {**runtime, "node": "v25"},
            )

    def test_completed_phase_requires_unchanged_output_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            log = root / "phase.log"
            artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
            log.write_text("proof passed\n", encoding="utf-8")
            record = {
                "outputs": [
                    {
                        "path": "artifact.json",
                        "sha256": court.sha256_file(artifact),
                    }
                ],
                "log_receipt": {
                    "path": "phase.log",
                    "sha256": court.sha256_file(log),
                },
            }
            self.assertTrue(court.completed_phase_is_intact(root, record))
            artifact.write_text(json.dumps({"ok": False}), encoding="utf-8")
            self.assertFalse(court.completed_phase_is_intact(root, record))
            artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
            log.write_text("proof changed\n", encoding="utf-8")
            self.assertFalse(court.completed_phase_is_intact(root, record))

    def test_source_fingerprint_scope_includes_laws_and_browser_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.sbt").write_text("scalaVersion := x", encoding="utf-8")
            law = root / "modules/laws/shared/src/Law.scala"
            browser = root / "modules/browser-tests/src/guard.js"
            ignored_result = root / "modules/benchmarks/results/receipt.json"
            for path in (law, browser, ignored_result):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")
            files = court.source_files(root)
            self.assertIn(law, files)
            self.assertIn(browser, files)
            self.assertNotIn(ignored_result, files)

    def test_phase_cleanup_runs_even_when_browser_preflight_fails(self) -> None:
        phase = court.Phase(
            "browser",
            False,
            (),
            lambda ctx: ["main"],
            before=(lambda ctx: ["before"],),
            after=(lambda ctx: ["after"],),
        )
        context = court.Context(Path("/repo"), Path("/receipt"), Path("/python"))
        with mock.patch.object(court, "execute_logged", side_effect=[7, 0]) as execute:
            return_code = court.execute_phase_logged(
                phase, context, Path("/receipt/browser.log")
            )
        self.assertEqual(return_code, 7)
        self.assertEqual(
            [call.args[0] for call in execute.call_args_list],
            [["before"], ["after"]],
        )

    def test_plan_requalifies_every_timing_family(self) -> None:
        plan = court.phases()
        names = {phase.name for phase in plan}
        self.assertIn("laws-proof", names)
        self.assertIn("browser-correctness", names)
        browser = next(phase for phase in plan if phase.name == "browser-correctness")
        self.assertEqual(len(browser.before), 1)
        self.assertEqual(len(browser.after), 1)
        timing = {phase.name for phase in plan if phase.timing}
        self.assertEqual(
            timing,
            {
                "jvm-timings",
                "numpy-timings",
                "jvm-allocation",
                "scala-js-timings",
                "scala-js-gc-diagnostic",
            },
        )
        allocation = next(phase for phase in plan if phase.name == "jvm-allocation")
        context = court.Context(Path("/repo"), Path("/receipt"), Path("/python"))
        rendered = " ".join(allocation.command(context))
        self.assertIn("-p side=64,256,1024", rendered)
        for name in court.ALLOCATION_CASES:
            self.assertIn(name, rendered)

        config = court.configuration(
            Path("/python"),
            Path("/jdk-25"),
            court.HostLimits(),
            host_samples=5,
            host_interval_seconds=2.0,
        )
        self.assertEqual(config["operation_sides"], [64, 256, 1024])


if __name__ == "__main__":
    unittest.main()
