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

## Post-closeout operation matrix

The closeout suite measured `float64` addition, four sums, two copies, scalar
access, and view creation. It cannot support performance claims about the rest
of the public API. `OperationMatrixBenchmarks` and its matched NumPy harness now
add 79 cases per size across:

- binary/scalar/unary/comparison/predicate/cast/in-place kernel families;
- full and axis product, min, max, arg-min, arg-max, mean, fixed-width sums,
  and the two supported widening sums;
- `float64`, `float32`, `int32`, and `int64`; and
- contiguous, inner/outer-strided, reversed, transposed, and broadcast inputs.

The required gate is semantic parity for every row. Same-host diagnostic
receipts now exist for
[JDK 21](../modules/benchmarks/results/2026-07-28-local-jdk21-operation-matrix/)
and
[JDK 25](../modules/benchmarks/results/2026-07-28-local-jdk25-operation-matrix/).
Family/dtype/layout geometric means locate candidate work; they are not a new
aggregate release score and do not replace the original closeout series.

Both JDKs identify in-place mutation as the dominant measured gap: its family
geometric mean is 0.026x on JDK 21 and 0.027x on JDK 25. The JDK 25 allocation
receipt records about 3.15 MiB/op for a 65,536-element in-place Double add,
roughly 48 bytes per element, while a full mean allocates about 25 B/op. This
is structural evidence for per-element boxing or equivalent generic-dispatch
allocation in the mutable loop. The next optimization must dispatch once into
a monomorphic primitive loop and prove zero size-dependent allocation.

Structural acceptance remains dispatch-before-loop, no per-element boxing, and
allocation limited to the documented result policy. Use selected matrix cases
with JMH GC profiling and JIT inlining/assembly evidence before attributing a
ratio to dispatch, auto-vectorization, or memory bandwidth.

## Stage 2 exact-ILP and mutable-kernel snapshot

The dispatch-before-loop rewrite and exact-schedule ILP controls have same-host
receipts for
[JDK 21](../modules/benchmarks/results/2026-07-28-local-jdk21-stage2/)
and
[JDK 25](../modules/benchmarks/results/2026-07-28-local-jdk25-stage2/).
Raw-bit JVM and Scala.js laws retain the fixed 128-value block order and merge
tree.

At side 1024, the exact raw full-sum control improved by 1.52x on JDK 21 and
1.59x on JDK 25; the exact raw axis-1 control improved by 1.79x and 1.83x.
The JDK 25 public full and axis-1 sums improved by 1.71x and 1.88x against the
earlier same-host receipt.

The in-place family moved from 0.026x to 0.967x NumPy on JDK 21 and from
0.027x to 0.990x on JDK 25 across the targeted 18 rows. At side 1024,
contiguous `Double` add improved 32-35x, reversed add improved 41-46x, and
inner-stride multiply improved 40-41x. The post-change GC receipt shows
approximately constant per-operation allocation instead of the former
approximately 48 bytes per element.

These are optimization receipts, not a new release score. At this stage,
reversed mutable traversal remained 0.43-0.52x NumPy. The later
[general packed-layout receipt](../modules/benchmarks/results/2026-07-29-local-jdk25-frontier/)
closes the layout-specific traversal cost without adding a benchmark case
branch: packed permutations and reversals use the same physical-order loop as
canonical storage. The separate Vector API work remains an experiment, not a
dependency change in `ravel-core`.

## Stage 3 Vector API decision

The standalone, nonpublished Vector API probe has same-host AArch64 receipts
for
[JDK 21](../modules/benchmarks/results/2026-07-28-local-jdk21-vector-spike/)
and
[JDK 25](../modules/benchmarks/results/2026-07-28-local-jdk25-vector-spike/).
At side 1024, explicit vectors improved exact axis-0 sum by 1.40x and 1.71x,
is-finite by 1.40x and 1.39x, and less-than by 1.34x and 1.27x on JDK 21 and
25 respectively. The exact sum vectors across independent columns, not across
terms in a fiber, so it retains the fixed reduction schedule.

The controls also show why there is no broad Vector API promotion. Explicit
contiguous addition was 0.84x the scalar control on JDK 21 and flat on JDK 25.
Vector sine was 0.56x on JDK 21 and allocated 48.0 MiB per side-1024 operation,
approximately 48 bytes per element, although it reached 1.35x with constant
allocation on JDK 25. The API remains incubating, the build requires module
flags and a benchmark-specific bytecode-generator workaround, and no x86-64
runner was available for this receipt.

The promotion threshold for an optional JVM artifact is:

- exact output under the operation's existing numerical contract;
- at least 1.20x at side 1024 on both JDK 21 and JDK 25;
- no regression greater than 10 percent for an included kernel;
- no size-proportional allocation; and
- successful AArch64 and x86-64 receipts while the incubator dependency stays
  outside `ravel-core`.

Axis-0 sum, less-than, and is-finite are candidate kernels under the local
speed and allocation parts of that threshold. The artifact decision is
**no-go for now** because cross-architecture proof and an integration-specific
fallback design are missing. Contiguous add and sine are excluded by measured
losses on the supported JDK 21 baseline.

## Stage 4 reassociated-sum demand gate

No `sumFast` or reassociated reduction API is warranted. A repository,
tracker/discussion-board, and GitHub-issue audit found no concrete consumer
request to exchange the fixed schedule for different NaN, signed-zero,
overflow, determinism, or cross-platform behavior.

The remaining measured exact-schedule cost does not justify manufacturing that
demand. In the focused Stage 2 receipts, contiguous side-1024 `Float.sum` is
0.854x NumPy on JDK 21 and 0.819x on JDK 25. Exact ILP already improves the
raw serial control by 1.52-1.59x, and the Stage 3 experiment shows that
independent-lane axis-0 work may have a future exact optimization path.

Decision: do not add `sumFast`, an `Associative` policy, or any other public
reassociation switch. The existing fixed 128-value block-pairwise schedule
remains the only floating-sum contract. Reopen this gate only for a named
consumer and workload that accepts a fully specified alternative contract and
shows a material parity-gated benefit over the exact implementation.

## Residual operation-matrix campaign

The focused JDK 25 same-host receipt at
[`2026-07-28-local-jdk25-residual-war`](../modules/benchmarks/results/2026-07-28-local-jdk25-residual-war/)
compares commit `942ce83` with the parity-gated candidate under the same
two-fork, five-warmup, seven-measurement protocol. Across 15 side-1024 public
rows, the geometric-mean speedup is 3.47x; stable individual gains range from
1.83x for axis-1 Double mean to 6.28x for transposed Double max. No stable row
regressed.

The retained changes break fixed-width sum/product dependency chains with
eight exact modulo-arithmetic lanes, traverse rank-2 extrema and axis-0 product
in cache-friendly physical order, reuse pairwise axis-0 scratch, execute
Float-to-Double pairwise blocks through monomorphic storage, and rely on the
specified primitive floating-to-integral conversion instead of repeating its
range/NaN branches per element.

The numerical contracts remain unchanged: floating sums and means keep the
fixed 128-value block schedule and merge tree; axis products keep fiber order;
extrema preserve NaN propagation and signed zero; fixed-width operations keep
wraparound; and floating casts retain the JVM/Scala.js conversion result.
The NumPy gate matched all 32 access-pattern and 158 operation-matrix
signatures, the full JVM and Scala.js suites passed, and optimized Scala.js
linking passed.

Allocation controls show approximately 8.9 KiB/op for side-1024 axis-0 mean,
including its required 8 KiB output, and no measured GC. Scalar Float mean and
Int product remain effectively allocation-free. This local workstation receipt
is diagnostic and does not replace quiet-runner release evidence.

## General packed-layout campaign

The broad 158-row detector at
[`2026-07-29-local-jdk25-current-matrix`](../modules/benchmarks/results/2026-07-29-local-jdk25-current-matrix/)
passed semantic parity at sides 256 and 1024 before timing. It identified a
remaining reversed scalar-mutation loss, but the production change is based on
a reusable layout invariant rather than that row.

A layout is physically dense when its reachable addresses form exactly one
packed interval after singleton axes are ignored and non-singleton axes are
ordered by absolute stride. The invariant covers C order, Fortran order, axis
permutations, reversals, and their combinations; it rejects gaps, overlap, and
broadcast. Element-independent scalar mutation may therefore visit the packed
interval in physical order. Paired assignment remains in logical order because
alias behavior may depend on visit order.

Exact full reductions reuse the same invariant only when their algebra permits
it. Fixed-width sums and products are associative modulo their dtype width.
Full extrema may regroup while retaining NaN propagation and signed-zero
selection. Floating sums and products keep their documented logical schedules.

On JDK 25 AArch64, reversed side-1024 mutation improved from 337,758 to 142,126
ns/op, or 2.38x. A full-protocol six-layout witness measured 137,695-138,217
ns/op across contiguous, transposed, independently reversed, doubly reversed,
and transpose-plus-reverse views, a spread below 0.4 percent. JVM and Scala.js
generated laws verify the invariant and exact results across ranks and dtypes.

The rank-3 reduction witness measures full `Double.maximum` within 1.4 percent
across contiguous, permuted, reversed, and permute-plus-reverse layouts. The
three transformed `Int.sum` rows are within 0.4 percent. Its contiguous row
retains one outlier and a wider interval, so the receipt establishes that
transformed layouts reach the same fast path; it does not establish that they
are faster than contiguous storage.

The expanded standalone Vector API controls are recorded for
[JDK 25](../modules/benchmarks/results/2026-07-29-local-jdk25-vector-frontier/)
and
[JDK 21](../modules/benchmarks/results/2026-07-29-local-jdk21-vector-frontier/).
JDK 25 shows 2.59-3.04x candidates for exact extrema and `Int` reductions,
1.34x for `Long` sum, and a decisive 0.14x rejection for `Long` product.
The clean JDK 21 rows confirm 2.46-2.67x for the `Int` reductions and reject
`Long` product at 0.13x. Split quiet JDK 21 reruns replace the contaminated
aggregate with 2.61x for maximum, 2.57x for minimum, and 1.33x for `Long` sum.
The loaded attempt remains quarantined. The JDK 21 allocation court reports
constant normalized allocation below 50 B/op and zero collections for every
candidate, matching the JDK 25 conclusion. No artifact can be promoted without
x86-64 timing and an integration-specific scalar fallback and module-loading
design.
