# Ravel JVM and NumPy access-pattern benchmarks

This suite is the **critical correctness and performance evidence path** for
Ravel while the project is still an early 0.1-level GitHub library. It compares
Ravel's public JVM API with semantically matched NumPy operations across
contiguous, inner-strided, outer-strided, reversed, transposed, broadcast,
reduction, copy, scalar-read, and zero-copy view-creation paths.

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

That script emits Ravel signatures via `AccessPatternParity`, NumPy signatures
with `--signatures-only`, and validates them with
`compare_access_patterns.py --parity-only`. CI runs the same job on sides
`32,64`. Helper unit tests live in
`modules/benchmarks/python/test_compare_access_patterns.py`.

## Timed comparison (diagnostic)

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
