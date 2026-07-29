# Expanded Vector API frontier: JDK 25

This is an AArch64 experiment in the standalone, nonpublished
`vectorApiProbeJVM` project. It does not add a Vector API dependency to
`ravel-core`.

The parity receipt ran on Homebrew OpenJDK 25.0.1 with a preferred 128-bit
species: two `Double`, four `Int`, and two `Long` lanes. Scalar and vector
controls matched for exact extrema, fixed-width sums and products, the
existing exact axis-0 sum, contiguous add, and predicates. The adversarial
gate includes alternate NaN payloads, both signed-zero orders, infinities,
singleton and tail lengths, empty reduction identities, and overflow-heavy
fixed-width inputs.

The full JMH run used two forks, five 500 ms warmup iterations, and seven
500 ms measurement iterations. Side-1024 results for the new kernels were:

| Kernel | Scalar ns/op | Vector ns/op | Speedup | Local decision |
|---|---:|---:|---:|---|
| maximum | 261,262 | 101,014 | 2.59x | candidate |
| minimum | 261,836 | 100,736 | 2.60x | candidate |
| `Int` product | 153,122 | 57,942 | 2.64x | candidate |
| `Int` sum | 153,443 | 50,453 | 3.04x | candidate; scalar row is noisy |
| `Long` sum | 135,874 | 101,142 | 1.34x | candidate |
| `Long` product | 167,151 | 1,211,816 | 0.14x | reject |

The allocation control recorded no collections. Normalized allocation was
about 49.5 B/op for each extrema kernel, below 1 B/op for the `Int`
reductions, 1.42 B/op for `Long` sum, and 15.39 B/op for rejected `Long`
product. These are constant per operation rather than size-proportional.

This receipt can nominate kernels for further testing; it cannot promote an
artifact. Promotion still requires clean JDK 21 evidence, x86-64 evidence,
and a scalar fallback and module-loading design.

Files:

- `parity.json`: runtime metadata and exact output comparisons;
- `jmh.json`: complete paired timing controls; and
- `gc.json`: side-1024 allocation controls for the new vector kernels.
