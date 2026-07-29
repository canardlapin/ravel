# Expanded Vector API frontier: JDK 21

This is an AArch64 experiment in the standalone, nonpublished
`vectorApiProbeJVM` project. It does not add a Vector API dependency to
`ravel-core`.

The parity receipt ran on Eclipse Adoptium OpenJDK 21.0.11 with a preferred
128-bit species: two `Double`, four `Int`, and two `Long` lanes. Every exact
and adversarial correctness case passed.

In the initial full two-fork run, the clean side-1024 pairs were:

| Kernel | Scalar ns/op | Vector ns/op | Speedup | Status |
|---|---:|---:|---:|---|
| `Int` product | 155,454 | 58,135 | 2.67x | candidate |
| `Int` sum | 133,588 | 54,362 | 2.46x | candidate |
| `Long` product | 170,329 | 1,321,243 | 0.13x | reject |

The initial `Long` sum and extrema aggregates were contaminated by a heavily
loaded fork: maximum's vector score had a 166 microsecond error, while
minimum's scalar score had a 184 microsecond error. They are not decision
evidence.

`focused-rerun.json` is also quarantined. System load rose from roughly 55 to
83 while it ran, and both scalar and vector controls slowed substantially.
The file is retained to make the loss visible; it must not be combined with
the initial run or cited as confirmation.

The quiet side-1024 extrema and reduction rerun was therefore split into one
scalar/vector pair per invocation. The full-protocol replacements are:

| Kernel | Scalar ns/op | Vector ns/op | Speedup | Status |
|---|---:|---:|---:|---|
| maximum | 257,002 | 98,304 | 2.61x | candidate |
| minimum | 257,446 | 100,322 | 2.57x | candidate |
| `Long` sum | 130,732 | 98,079 | 1.33x | candidate |

These replace the contaminated aggregates and put all three above the 1.20x
cross-JDK timing threshold.

The side-1024 allocation court recorded no collections. Normalized allocation
was 49.35 B/op for each extrema kernel, 0.68 B/op for `Int` sum, 0.74 B/op for
`Int` product, and 1.37 B/op for `Long` sum. These are constant per operation,
not proportional to the 1,048,576 input elements.

The local AArch64 correctness, timing, and allocation gates now pass on both
JDK 21 and JDK 25. X86-64 timing plus a scalar fallback and module-loading
design still remain before an optional artifact can be proposed.

`quiet-maximum.json`, `quiet-minimum.json`, and `quiet-sum-long.json` contain
the clean replacement pairs. `gc.json` is the candidate allocation receipt.
`jmh.json` is the initial complete run, `parity.json` is the correctness
receipt, and `focused-rerun.json` is the deliberately retained contaminated
attempt.
