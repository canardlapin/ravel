# JDK 25 Vector API spike

Local AArch64 receipt from OpenJDK 25.0.1. JMH 1.37 used two forks, three
300 ms warmups, and five 300 ms measurements. Ratios are scalar time divided by
vector time, so values above 1 favor the explicit Vector API.

| Kernel | Side 256 scalar / vector (ns/op) | Ratio | Side 1024 scalar / vector (ns/op) | Ratio |
|---|---:|---:|---:|---:|
| exact axis-0 sum | 15,747 / 10,408 | 1.51x | 283,203 / 165,922 | 1.71x |
| contiguous add | 14,600 / 14,608 | 1.00x | 240,051 / 239,593 | 1.00x |
| is-finite | 17,151 / 16,122 | 1.06x | 306,610 / 220,060 | 1.39x |
| less-than | 14,672 / 11,635 | 1.26x | 257,039 / 201,806 | 1.27x |
| sine | 232,616 / 169,469 | 1.37x | 3,668,056 / 2,726,394 | 1.35x |

`parity.json` proves raw-bit identity for axis-0 sum and addition, exact
Boolean identity for the predicates, and sine error no greater than
`1.1102230246251565e-16`. The preferred species was 128 bits, or two `Double`
lanes.

The side-1024 GC profile reports approximately 7-94 normalized bytes per
operation for vector kernels and zero collections. There is no evidence of
size-proportional allocation on this JDK.

Artifacts:

- `parity.json`: correctness and runtime metadata;
- `jmh.json`: paired timings; and
- `gc.json`: side-1024 allocation controls.
