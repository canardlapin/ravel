# Performance closeout — 2026-07-28 (OpenJDK 25)

Full parity-gated AccessPattern suite after copy, elementwise, scalar, and
reduction tranches. JDK 21 is not installed on this host; this is a local
development closeout on Homebrew OpenJDK 25.0.1 (same substitution policy as
the 2026-07-27 controls baseline). Not a dedicated quiet-runner release gate.

## Protocol

- Sides: 256, 1024
- JMH: AverageTime, 2 forks, 5×500 ms warmup, 7×500 ms measurement
- Compare cases: 16 public API cases (excludes scalar_write)
- Compute geomean: 12 compute families × 2 sizes = 24 rows (excludes scalar read and view create)
- GC sample: side 256, 1 fork, contiguous_add / full_sum_contiguous / axis1_sum / copy_transpose
- Gates: `coreJVM/test` 112/112; targeted Scala.js 28/28; `numpy-parity-gate.sh` OK

## Acceptance vs `bd-01KYHSM8YGF12ZM435A8D9F45X`

| Criterion | Result |
|---|---|
| Compute geomean ≥ 0.50× NumPy | **Pass — 0.824×** |
| No compute case < 0.25× | **Pass — min 0.293×** (`full_sum_inner_stride` @ 1024) |
| No >10% same-host slowdown vs 2026-07-27 baseline | **Pass** (all compute cases faster) |
| JVM + Scala.js gates | **Pass** |
| Stretch (≥0.70× geomean, no case <0.50×) | **Miss** — reduction family geomean 0.392×; several reduction rows 0.29–0.45× |

## Family geomeans

| Family | Geomean vs NumPy |
|---|---:|
| copy | 1.522× |
| elementwise | 1.145× |
| broadcast | 0.910× |
| reduction | 0.392× |

## GC sample (side 256)

| Case | ns/op | alloc B/op | notes |
|---|---:|---:|---|
| contiguous_add | ~18.2k | 524,992 | output 524,288 + ~704 |
| copy_transpose | ~27.5k | 524,992 | output + ~704 |
| axis1_sum | ~28.8k | 2,688 | output 2,048 + ~640 |
| full_sum_contiguous | ~19.9k | ~24 | thread-local scratch reuse |

## Residual gap

Primary closeout targets pass. The remaining NumPy beat-down is concentrated in
**pairwise floating reductions** (family 0.392×). Exact-schedule controls already
show axis-1 near a schedule floor (~0.38–0.43×). Next step is a measured
residual-gap decision for JDK Vector API and/or an opt-in fast reduction policy,
without changing default bit-stable pairwise semantics.
