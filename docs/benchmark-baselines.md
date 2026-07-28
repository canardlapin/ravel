# Kernel benchmark baselines and regression budget

The timing budget supplements the structural kernel gate. A fast timing result
cannot excuse boxing, callback dispatch, iterator use, or generic numeric
dispatch inside a built-in primitive loop.

## Reference workload

The reference operation adds two `Double` arrays with 65,536 logical elements.
The public benchmark includes planning, output allocation, the kernel, and one
result read. The representative strided case reads alternating elements from
two 131,072-element buffers and reverses one operand.

Evidence collected on 2026-07-27 used Apple silicon, OpenJDK 25.0.1,
Scala 3.7.4, Scala.js 1.22.0, Node 24.1.0, and JMH 1.37.

| Runtime and operation | Baseline |
|---|---:|
| JVM raw contiguous add | 64,201 operations/s |
| JVM raw inner-strided add, 32,768 outputs | 71,560 operations/s |
| JVM public contiguous add | 6,748 operations/s |
| JVM public representative strided add | 6,646 operations/s |
| Node raw contiguous add | 724.2 million elements/s |
| Node public contiguous add | 155.9 million elements/s |
| Node public representative strided add | 155.1 million elements/s |

The JVM public operation allocates a 524,288-byte result buffer. JMH measured
525,465 normalized bytes per operation, including the result wrapper and plan
metadata. The raw JVM probes reuse their output and allocate no buffer inside
the measured operation.

## Release budget

On the reference host and toolchain, a release candidate fails the timing gate
if the median of three warmed runs is below 70 percent of its checked-in
baseline:

| Operation | Minimum |
|---|---:|
| JVM public contiguous | 4,724 operations/s |
| JVM public strided | 4,652 operations/s |
| Node public contiguous | 110 million elements/s |
| Node public strided | 110 million elements/s |

The JVM public operations must also allocate no more than the output buffer
plus 2,048 bytes per operation. Raw `*Into` and representation probes must not
allocate an output buffer.

## NumPy parity budget (compute geomean)

The access-pattern suite in [`numpy-benchmarks.md`](numpy-benchmarks.md)
compares public Ravel JVM operations with matched NumPy work on the same host.
**Semantic parity is the required gate now** (`scripts/numpy-parity-gate.sh`).
The speed targets below are aspirational performance goals for the active
NumPy-gap program while the project is still 0.1-level; they are not a claim
that Ravel is release-ready.

| Gate | Requirement |
|---|---|
| Compute geomean | at least **0.50×** NumPy (`NumPy time / Ravel time`) |
| Per-case floor | no compute case below **0.25×** unless an exact-schedule receipt documents a numerical-contract tax |
| Same-host regression | no case regresses more than **10%** versus the previous checked-in same-host report |
| Stretch | geomean **0.70×**, no case below **0.50×** |

Scalar indexing and zero-copy view creation are reported but excluded from the
compute geomean. Absolute cross-host timings remain diagnostic; CI should
compare a candidate to a baseline collected on the same runner and fail on a
drop greater than 30 percent for the representation-probe budget above. Any
timing pass still requires bytecode and optimized JavaScript inspection.

The first public JVM measurement was about 86 operations/s because all dtype
loops had been inlined into one dispatcher method too large for HotSpot to
compile. Splitting the dispatcher from dtype-specific methods raised the
measured rate to 6,748 operations/s. This incident is retained as evidence for
the structural gate: dispatch must call a small monomorphic method before the
element loop.

## NumPy closeout snapshot (2026-07-28)

Local OpenJDK 25 full-suite compute geomean after monomorphic kernel work:
**0.824×** NumPy (24 compute rows; min 0.293×). Artifacts:
[`modules/benchmarks/results/2026-07-28-local-jdk25-closeout/`](../modules/benchmarks/results/2026-07-28-local-jdk25-closeout/).
Reduction family remains ~0.39×; stretch target (0.70× / floor 0.50×) not met.
JDK 21 quiet-runner ratification is still required for a release gate.
