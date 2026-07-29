# JVM and Scala.js performance court

This court complements the JVM-versus-NumPy operation matrix. JMH remains the
authority for JVM timing; the cross-runtime court gives full-linked Scala.js a
same-host Node baseline while requiring the JVM and JavaScript artifacts to
execute one shared public-operation registry.

The registry covers binary, scalar, unary, comparison, predicate, cast,
reduction, and in-place operations. Its generated fixtures span `Float`,
`Double`, `Int`, and `Long`, plus contiguous, offset-strided, reversed,
transposed, and broadcast access. Default sides are 64, 256, and 1024. Scala.js
`Long` rows are semantic witnesses only because Ravel does not claim a
JavaScript typed-array fast path for `Long`; they run at side 64 only. The
shared law suite remains the exhaustive cross-platform `Long` correctness
authority.

## Correctness gate

Run the shared court contract suite as part of the representation gate:

```sh
sbt representationProof
```

Generate JVM signatures:

```sh
sbt "representationProbeJVM/runMain ravel.bench.CrossRuntimeCourt \
  --signatures-only \
  --sides 64,256,1024 \
  --out target/cross-runtime-court/jvm-signatures.json"
```

Build and execute the fully optimized Scala.js artifact:

```sh
sbt representationProbeJS/fullLinkJS
node --expose-gc \
  modules/benchmarks/js/target/scala-3.7.4/ravel-representation-probe-js-opt/main.js \
  --signatures-only \
  --sides 64,256,1024 \
  --out target/cross-runtime-court/js-signatures.json
```

Compare every row:

```sh
python3 modules/benchmarks/python/compare_cross_runtime_court.py \
  --jvm target/cross-runtime-court/jvm-signatures.json \
  --js target/cross-runtime-court/js-signatures.json
```

The comparator requires identical case sets, semantic metadata, result size,
result layout, non-finite counts, and negative-zero counts. Integer and Boolean
signatures are exact. Floating signatures use the same dtype-aware tolerances
as the NumPy court.

## Timed Node receipts

Run timed receipts only on a calm host with the same Node binary, V8, operating
system, architecture, processor availability, and timer:

```sh
node --expose-gc \
  modules/benchmarks/js/target/scala-3.7.4/ravel-representation-probe-js-opt/main.js \
  --sides 64,256,1024 \
  --warmup-ms 200 \
  --sample-ms 250 \
  --samples 7 \
  --out target/cross-runtime-court/node-candidate.json
```

The runner calibrates batches, warms each case independently, records every
sample, and reports median, quartiles, range, and relative standard deviation.
When `--expose-gc` is present it collects unreachable fixture and signature
objects between cases, never inside a timed sample, and records that capability
in the receipt. It never compares Node directly with NumPy. Compare a candidate
only with a receipt from the identical recorded runtime and host:

```sh
python3 modules/benchmarks/python/compare_cross_runtime_court.py \
  --jvm target/cross-runtime-court/jvm-signatures.json \
  --js target/cross-runtime-court/js-signatures.json \
  --baseline modules/benchmarks/results/CALM-HOST-BASELINE/node.json \
  --candidate target/cross-runtime-court/node-candidate.json \
  --out target/cross-runtime-court/node-comparison.md
```

The report includes family geometric means and every measured row. A slowdown
greater than 10 percent is visibly marked rather than averaged away. It
independently recomputes each median and RSD from the stored samples and shows
baseline and candidate instability in both the family and detailed tables.

For the first baseline, render absolute timing and variability without
inventing a before/after comparison:

```sh
python3 modules/benchmarks/python/compare_cross_runtime_court.py \
  --jvm target/cross-runtime-court/jvm-signatures.json \
  --js target/cross-runtime-court/js-signatures.json \
  --summarize target/cross-runtime-court/node-baseline.json \
  --out target/cross-runtime-court/node-baseline.md
```

The baseline report shows normalized time and relative standard deviation for
every row, plus family summaries. Rows above 10 percent relative standard
deviation are visibly unstable and cannot support a performance claim. Timing
reports also reject receipts with fewer than five samples or a row-level sample
count that disagrees with its metadata; a one-sample smoke can never appear
stable merely because its calculated RSD is zero.

## Resumable full-court receipt

The full JVM, NumPy, allocation, and Scala.js court can take long enough that a
single uninterrupted shell session is an unnecessary risk. Use the receipt
driver with a new dated directory:

```sh
python3 modules/benchmarks/python/run_performance_court.py \
  --java-home /opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home \
  --python target/access-patterns/venv/bin/python \
  --out modules/benchmarks/results/YYYY-MM-DD-HOST-court
```

If host qualification or an external interruption stops the court, resume the
same directory:

```sh
python3 modules/benchmarks/python/run_performance_court.py \
  --java-home /opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home \
  --python target/access-patterns/venv/bin/python \
  --out modules/benchmarks/results/YYYY-MM-DD-HOST-court \
  --resume
```

The driver:

- runs representation, reusable-law, guarded browser-correctness, NumPy, and
  JVM/Scala.js semantic gates before timing;
- fingerprints all core and benchmark source plus the complete court
  configuration, and rejects a resume if either changes;
- verifies the source fingerprint and Java, Node, Python, and NumPy environment
  immediately before and after every phase, so concurrent worktree edits cannot
  silently mix artifacts from different implementations;
- records resolved executable identities plus the NumPy build configuration,
  including its BLAS/LAPACK and SIMD features, rather than treating a package
  version alone as a pinned runtime;
- checksums every completed output and refuses to silently reuse a modified
  artifact, including the logs that are the primary evidence for proof-only
  phases;
- requires five consecutive calm-host samples before every timing or
  allocation phase, rather than qualifying once for a multi-hour run;
- records exact commands, runtime versions, thresholds, host snapshots, logs,
  and artifact checksums in `manifest.json`;
- runs the full annotated JVM matrix, the matched NumPy matrix, a focused JMH
  GC allocation court, the full-linked Node matrix, and a separate Node
  `--trace-gc` diagnostic;
- writes `comparison.md`, `node-summary.md`, and a receipt `README.md` only
  after their prerequisite evidence exists.

There is deliberately no “force noisy host” switch. A failed qualification
leaves the next timing phase pending and the manifest resumable.
The default gate requires five consecutive samples with one-minute load no
higher than 0.15 per logical CPU, at least 10 percent reclaimable memory, no
less than 25 percent system memory-pressure freedom on macOS, no more than 35
percent physical compressor occupancy, and no unrelated process above 20
percent CPU. It also rejects active pageout or swap traffic and sustained
page-in or decompression rates above 100 pages per second. This distinguishes
live memory contention from harmless compressed pages retained after an older
workload. Overrides are recorded as part of the configuration fingerprint.

## What Scala.js changes

Scala.js is neither inferred from JVM results nor compared directly with
NumPy. The full-linked Node artifact gets its own same-runtime, same-host
baseline because V8 compilation, JavaScript garbage collection, and typed-array
lowering are different performance systems.

Ravel can still exploit the same general representation facts on both
platforms: specialize by semantic dtype, plan a layout once outside the inner
loop, keep dense typed-array traversal monomorphic, and preserve separate
contiguous, strided, broadcast, and reduction kernels. Those optimizations
reduce dispatch, address calculation, and temporary allocation in both
runtimes.

The expected gains are not identical. JavaScript `Double`, `Float`, and `Int`
storage maps naturally to typed arrays and benefits most from dense,
predictable loops. Scala `Long` does not have an equivalent JavaScript
typed-array fast-path contract in Ravel, so large `Long` rows remain JVM timing
workloads and side-64 Scala.js semantic witnesses. V8 also has different
tiering and escape-analysis behavior, so a JVM allocation win is a hypothesis
for Scala.js until the Node receipt and GC diagnostic confirm it.

Timing receipts do not infer allocation from unstable heap snapshots. Use the
JMH GC profiler for JVM allocation and `node --trace-gc` for JavaScript
diagnostics. Correctness gates belong in ordinary CI; calm-host timing and GC
receipts belong in a periodic or explicitly scheduled performance job.
