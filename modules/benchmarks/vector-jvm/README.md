# JVM Vector API probe

This is a standalone, nonpublished experiment. It is deliberately absent from
the root aggregate and has no dependency edge to `ravel-core`. The probe asks
whether explicit `jdk.incubator.vector` kernels improve work whose lanes are
independent without changing Ravel's numerical contracts.

The paired scalar and vector controls cover:

- exact axis-0 `Double` sum, vectorized across independent columns while
  retaining the 128-row block order and scalar merge tree;
- full `Double` minimum and maximum with NaN propagation and signed-zero
  preservation;
- fixed-width `Int` and `Long` sum and product with exact wraparound;
- contiguous `Double` addition, as an auto-vectorization control;
- `Double` less-than and is-finite predicates; and
- `Double` sine, as a transcendental and allocation control.

Construction validates exact extrema and fixed-width reductions, raw-bit
equality for the exact floating reduction and addition, exact Boolean outputs
for the predicates, and a `2e-13` absolute tolerance for sine. Adversarial
cases cover alternate NaN payloads, both signed-zero orders, infinities,
singleton and non-species-multiple tails, empty identities, and fixed-width
overflow. Output arrays and reduction scratch are allocated outside the timed
methods.

## Why this project is awkward to run

The Vector API remains an incubator module on both supported test JDKs. The
sbt process, the forked application, and each JMH fork therefore need
`--add-modules=jdk.incubator.vector`. The sbt-jmh reflection generator cannot
receive the required module option, so this project uses the ASM generator.
The ASM version bundled by sbt-jmh 0.4.8 cannot inspect class files newer than
Java 16, so this benchmark-only project targets class-file version 60.

These are packaging costs of the experiment, not settings suitable for a
published Ravel artifact.

## Reproduce

Compile and emit the correctness receipt:

```sh
sbt -J--add-modules=jdk.incubator.vector \
  "vectorApiProbeJVM/runMain ravel.bench.VectorApiParity \
  --out $(pwd)/target/vector-api/parity.json --side 256,1024"
```

Run the paired timing controls:

```sh
sbt -J--add-modules=jdk.incubator.vector \
  "vectorApiProbeJVM/Jmh/run -f 2 -wi 5 -i 7 -w 500ms -r 500ms \
  -rf json -rff $(pwd)/target/vector-api/jmh.json \
  ravel.bench.VectorApiBenchmarks.*"
```

Run the allocation control at side 1024:

```sh
sbt -J--add-modules=jdk.incubator.vector \
  "vectorApiProbeJVM/Jmh/run -f 1 -wi 3 -i 5 -w 500ms -r 500ms \
  -p side=1024 -prof gc -rf json -rff $(pwd)/target/vector-api/gc.json \
  ravel.bench.VectorApiBenchmarks.*"
```

Use an absolute output path as shown. Forked sbt projects can otherwise resolve
the relative path below their project working directory.

Checked-in AArch64 receipts and the go/no-go decision are linked from
[`docs/numpy-benchmarks.md`](../../../docs/numpy-benchmarks.md). An optional
Vector API artifact must not be proposed from these receipts alone: x86-64
evidence is still missing.
