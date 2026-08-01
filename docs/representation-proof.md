# Ravel representation proof

This document records the evidence required before Ravel freezes its storage
and kernel ABI. Run:

```sh
sbt representationProof
sbt "representationProbeJVM/Jmh/run -prof gc .*RepresentationProbe.*"
node modules/benchmarks/js/target/scala-3.7.4/ravel-representation-probe-js-opt/main.js
```

## Candidate ABI

- `Storage[A]` is a sealed, platform-specific family.
- Each primitive dtype has a distinct storage case.
- Dispatch occurs once at a kernel boundary.
- A built-in inner loop receives a concrete primitive `Array[T]` on the JVM or
  a concrete typed array on Scala.js.
- `Byte`, `UInt8`, `Short`, and `UInt16` are storage and comparison dtypes, not
  arithmetic dtypes.
- JVM `UInt8`/`UInt16` wrap `Array[Byte]`/`Array[Short]`; Scala.js uses
  `Uint8Array`/`Uint16Array`.
- Scala.js `Long` uses `Array[Long]`. Scala `Long` is opaque at the JavaScript
  boundary, so Ravel 1.0 does not claim typed-array or fast-path performance for
  this dtype.
- Scala.js `Boolean` uses `Uint8Array` and encodes values as exactly `0` and
  `1`, distinct from `UInt8` which uses the full `0`…`255` range.

## Structural inspection

The JVM inspection command is:

```sh
javap -c -p \
  modules/core/jvm/target/scala-3.7.4/classes/ravel/internal/ProbeKernels\\$.class
```

Inspect the private `add*`, `negate*`, and `add*Strided` methods. Their
descriptors must contain primitive array types and their loop bodies must not
invoke `DType`, `Numeric`, iterators, tuple constructors, `Option`, or generic
buffer access.

The Scala.js inspection command is:

```sh
rg -n "Float64Array|Uint8Array|addDouble|addDoubleStrided|DType|Iterator|Tuple|Option" \
  modules/benchmarks/js/target/scala-3.7.4/ravel-representation-probe-js-opt/main.js
```

The optimized JavaScript must allocate the promised typed arrays and express
the hot operation as direct indexed access. A `Long` helper may occur, but it
is outside the JavaScript fast-path contract.

## Results

Evidence was collected on 2026-07-26 on Apple silicon with sbt 1.11.7,
Scala 3.7.4, Scala.js 1.22.0, Node 24.1.0, and OpenJDK 25.0.1 for the forked
JMH process.

- `sbt representationProof`: 4 JVM tests and 6 Scala.js tests passed; full
  optimization completed.
- JVM `javap`: `addDouble` has primitive-array parameters
  `(double[], double[], double[], int)` and consists of `daload`, `dadd`, and
  `dastore` inside the loop. It contains no callback, iterator, tuple,
  `Option`, `DType`, or generic storage call.
- Optimized Scala.js: `addDouble` receives three `Float64Array` values and its
  body is direct `out[i] = x[i] + y[i]`. Allocation sites include
  `new Uint8Array(size)` and `new Float64Array(size)`.
- Node linear `Double` add baseline: approximately 705 million elements/s for
  65,536-element buffers. This is descriptive evidence, not a release
  threshold.
- Short JMH baselines: approximately 67,547 contiguous operations/s and
  70,897 inner-strided operations/s for 65,536-element buffers. The second
  measured iteration reported about 0.1 B/op for both probes and no collection;
  first-iteration profiler noise raised the two-iteration means. There is no
  output-buffer allocation inside the measured operation.

The candidate ABI passed. The accepted ABI is the platform-specific sealed
`Storage[A]` family plus one dtype/layout dispatch into concrete raw-buffer
kernels. Scala.js `Long` remains semantically supported through
`Array[Long]`, but is excluded from the JavaScript fast-path performance
contract.
