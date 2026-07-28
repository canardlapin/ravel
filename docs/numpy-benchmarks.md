# Ravel JVM and NumPy access-pattern benchmarks

These suites are the **critical correctness and performance evidence path** for
Ravel while the project is still an early 0.1-level GitHub library:

- the access-pattern suite retains the original closeout cases across
  contiguous, inner-strided, outer-strided, reversed, transposed, broadcast,
  reduction, copy, scalar-access, and zero-copy view paths;
- the public-operation matrix adds 79 cases per size across binary and scalar
  arithmetic, every public unary floating operator, comparisons, predicates,
  casts, every reduction family, widening sums, and in-place mutation. It
  covers `Double`, `Float`, `Int`, and `Long` plus all six input-layout classes.

The matrix is broad reconnaissance. A focused benchmark and structural receipt
must confirm a target before an implementation claim is accepted.

## Correctness gate (required)

A change that alters public numerical results must keep Ravel and NumPy
signatures aligned: logical size, C-layout vs view classification, sum, and
position-weighted sum. The weighted signature distinguishes layouts such as a
transpose that an ordinary sum cannot distinguish.

Run the CI-shaped gate locally (small sides, no JMH):

```sh
bash scripts/numpy-parity-gate.sh
# optional: NUMPY_PARITY_SIDES=32,64,128 bash scripts/numpy-parity-gate.sh
```

That script emits Ravel signatures via `AccessPatternParity` and
`OperationMatrixParity`, emits both matched NumPy suites with
`--signatures-only`, and validates both comparisons before reporting success.
CI runs the same job on sides `32,64`. Helper unit tests live in
`modules/benchmarks/python/test_compare_access_patterns.py` and
`modules/benchmarks/python/test_operation_matrix.py`.

The matrix validates the case family, input and result dtype, input layout,
logical work, result size/layout, and two logical-order result signatures.
Integer and Boolean rows compare exactly. Floating rows use an explicitly
dtype-scaled tolerance because Ravel and NumPy need not use identical
transcendental implementations or reduction association.

NumPy parity does **not** define Ravel's default floating reduction schedule.
Raw-bit assertions against the documented 128-value block-pairwise reference
live in `ReductionLawsSuite`, including non-dyadic `sin` fixtures, sizes that
cross block boundaries, transposed/reversed inputs, and adversarial `Float`
NaN fibers. Those tests run on both JVM and Scala.js and remain the exact
schedule receipt.

## Timed access-pattern comparison (diagnostic)

The comparison is parity-gated. A timed report is produced only after Ravel and
NumPy agree on each result's logical size, sum, and position-weighted sum.
The signature also checks whether an output is a C/row-major owned result, a
strided view, or a scalar.

## What is timed

Elementwise operations, reductions that return arrays, and copies include
planning plus result allocation. Inputs and reusable views are constructed
outside the timed region. View-creation cases time only creation of the view
metadata. Full reductions return a scalar. Scalar-read cases visit every
element through the public two-index API in row-major or column-major order.
NumPy ufuncs are explicitly requested to produce C-order outputs, matching
Ravel's owned-result contract even for transposed inputs.

NumPy and Ravel both use deterministic `float64` inputs. The suite reports the
median NumPy sample and median JMH iteration in nanoseconds per operation, plus
normalized nanoseconds per logical work unit.
For reductions, a work unit is an input element visited; for view creation it
is one view.

The scalar-read comparison is deliberately user-facing. It includes Python's
interpreter/indexing overhead and the JVM public indexing path, so it must not
be described as a native NumPy-kernel comparison.

## Reproducible timed run

Run the semantic signatures first; this also creates the output directory:

```sh
sbt "representationProbeJVM/runMain ravel.bench.AccessPatternParity \
  --out target/access-patterns/ravel-signatures.json --side 256,1024"
```

Run JMH in average-time mode. The annotations use two forks, five warmup
iterations, and seven measurement iterations:

```sh
sbt "representationProbeJVM/Jmh/run -rf json \
  -rff target/access-patterns/ravel-jmh.json \
  ravel.bench.AccessPatternBenchmarks.*"
```

Create an isolated Python environment and run NumPy:

```sh
python3 -m venv target/access-patterns/venv
target/access-patterns/venv/bin/python -m pip install \
  -r modules/benchmarks/python/requirements.txt
target/access-patterns/venv/bin/python \
  modules/benchmarks/python/numpy_access_patterns.py \
  --side 256 --side 1024 \
  --out target/access-patterns/numpy.json
```

Validate parity and render the comparison:

```sh
target/access-patterns/venv/bin/python \
  modules/benchmarks/python/compare_access_patterns.py \
  --jmh target/access-patterns/ravel-jmh.json \
  --numpy target/access-patterns/numpy.json \
  --signatures target/access-patterns/ravel-signatures.json \
  --out target/access-patterns/comparison.md
```

The JSON artifacts retain the runtime versions and timing settings. Preserve
all three inputs with any published report.

## Timed public-operation matrix (diagnostic)

Generate Ravel semantic signatures before timing:

```sh
sbt "representationProbeJVM/runMain ravel.bench.OperationMatrixParity \
  --out target/operation-matrix/ravel-signatures.json --side 256,1024"
```

Run the full parameterized JMH matrix. Its annotations use two forks, three
warmup iterations, and five measurement iterations:

```sh
sbt "representationProbeJVM/Jmh/run -rf json \
  -rff target/operation-matrix/ravel-jmh.json \
  ravel.bench.OperationMatrixBenchmarks.operation_matrix"
```

Run the matched NumPy matrix:

```sh
target/access-patterns/venv/bin/python \
  modules/benchmarks/python/numpy_operation_matrix.py \
  --side 256 --side 1024 \
  --out target/operation-matrix/numpy.json
```

Validate semantic parity again and render family, dtype, layout, and detailed
timing summaries:

```sh
target/access-patterns/venv/bin/python \
  modules/benchmarks/python/compare_operation_matrix.py \
  --jmh target/operation-matrix/ravel-jmh.json \
  --numpy target/operation-matrix/numpy.json \
  --signatures target/operation-matrix/ravel-signatures.json \
  --out target/operation-matrix/comparison.md
```

For a deliberately targeted follow-up receipt, pass `--allow-partial`. The
report still validates the complete saved semantic-signature set before it
renders the intersection of measured JMH rows; the flag does not weaken
semantic parity.

JMH records the selected operation in its `caseName` parameter. In-place rows
reuse a destination and return a live scalar read; allocating rows return the
public result. The parity run separately snapshots mutable results outside the
timed path. This avoids charging Ravel for a diagnostic copy while still
checking the complete logical result against NumPy.

The Stage 2 focused reports live under
[`2026-07-28-local-jdk21-stage2`](../modules/benchmarks/results/2026-07-28-local-jdk21-stage2/)
and
[`2026-07-28-local-jdk25-stage2`](../modules/benchmarks/results/2026-07-28-local-jdk25-stage2/).
They compare the exact serial schedule to exact ILP, the public reduction
paths, the primitive dispatch-once mutable loops, and selected allocation.
They retain the losing reversed-view rows rather than folding them into a
headline claim.

## Targeted Vector API probe

The Stage 3 probe is a separate, nonpublished sbt project with no dependency on
or from `ravel-core`. It pairs scalar and explicit-vector implementations for
exact axis-0 sum, contiguous add, less-than, is-finite, and sine. The exact
axis-0 kernel places independent output columns in lanes and preserves the
128-row block order and merge tree; it does not horizontally reassociate a
sum.

Correctness, timing, and allocation artifacts are checked in for
[JDK 21](../modules/benchmarks/results/2026-07-28-local-jdk21-vector-spike/)
and
[JDK 25](../modules/benchmarks/results/2026-07-28-local-jdk25-vector-spike/).
Both ran on AArch64 with a preferred 128-bit species of two `Double` lanes.
Every exact case matched bit-for-bit, Boolean predicates matched exactly, and
sine remained within its explicit `2e-13` tolerance.

| Side-1024 kernel | JDK 21 scalar / vector | JDK 25 scalar / vector | Decision |
|---|---:|---:|---|
| exact axis-0 sum | 1.40x | 1.71x | candidate |
| contiguous add | 0.84x | 1.00x | reject |
| is-finite | 1.40x | 1.39x | candidate |
| less-than | 1.34x | 1.27x | candidate |
| sine | 0.56x | 1.35x | reject |

JDK 21 vector sine allocated 50,331,875 normalized bytes per operation at side
1024, approximately 48 bytes per element, and triggered 11 collections in the
short GC profile. The other vector controls on both JDKs showed only constant
single- or double-digit normalized bytes per operation.

The current result is a no-go for an optional published artifact. Candidate
kernels must first pass the cross-JDK threshold recorded in
[benchmark baselines](benchmark-baselines.md), gain an x86-64 receipt, and have
an explicit scalar fallback and module-loading design. The Vector API remains
an incubator in
[JDK 21 (JEP 448)](https://openjdk.org/jeps/448) and
[JDK 25 (JEP 508)](https://openjdk.org/jeps/508); the experiment requires
module flags and an sbt-jmh ASM-generator/class-file compatibility workaround.
None of those costs enter the public or core build.

## Interpretation

The report's “Ravel speed vs NumPy” column is `NumPy time / Ravel time`; values
above 1.0 favor Ravel. Ratios identify where to investigate, not why a gap
exists. In particular:

- allocation, garbage collection, JIT compilation, CPU frequency, and other
  processes can move cross-runtime results;
- NumPy may choose different inner loops based on strides and output order;
- view construction and scalar indexing mainly compare host API overhead;
- absolute regression thresholds are valid only against a same-host,
  same-toolchain baseline.

Use JMH's GC profiler when investigating allocations:

```sh
sbt "representationProbeJVM/Jmh/run -prof gc \
  ravel.bench.AccessPatternBenchmarks.*"
```

For the operation matrix, select a small, falsifiable family rather than
profiling all 158 size/case combinations:

```sh
sbt "representationProbeJVM/Jmh/run -prof gc \
  -p side=256 \
  -p caseName=inplace_add_double,contiguous_subtract_double,full_mean_double \
  ravel.bench.OperationMatrixBenchmarks.operation_matrix"
```

Use `-XX:+PrintCompilation -XX:+UnlockDiagnosticVMOptions
-XX:+PrintInlining` for inlining evidence. Use JMH `-prof perfasm` only on a
host with a supported disassembler and performance-counter access; retain the
failure as an environment limitation otherwise. An assembly excerpt or
compiler log is supporting evidence, not a substitute for allocation and
same-host timing controls.

Run the signatures and selected measurements on the supported JDK 21 baseline
and on JDK 25. Each Ravel signature JSON records the Java vendor/version, OS,
architecture, and processor count; NumPy records its runtime and platform
metadata. A published matrix must also state the exact commands and commit.

`AccessPatternControls` supplies lower bounds rather than competing public
implementations. It separates:

- Ravel output allocation with no kernel;
- a raw monomorphic add with reused or newly allocated output;
- the existing storage-dispatched probe beside measured inline-storage and
  erased opaque-buffer prototypes;
- `kernel.addInto` with reusable output but public planning;
- allocating inner-stride and transpose copy loops;
- the exact 128-element pairwise reduction schedule with reused or allocated
  scratch;
- raw axis-fiber reductions; and
- representative raw `Float` and `Int` addition loops.

Run the controls with the same sizes, forks, warmup, and measurement settings:

```sh
sbt "representationProbeJVM/Jmh/run -prof gc \
  ravel.bench.AccessPatternControls.*"
```

The controls answer where time and allocation enter the public path. They are
not API proposals and must not replace the semantic or layout parity gate.

Do not treat cross-runtime ratios as a release regression budget until a quiet,
stable runner has collected repeated baselines. Performance aspiration targets
(not CI blockers while the project is 0.1) are recorded in
[benchmark baselines](benchmark-baselines.md). The representation-probe budgets
remain separate local evidence.
