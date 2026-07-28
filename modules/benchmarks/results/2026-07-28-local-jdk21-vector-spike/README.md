# JDK 21 Vector API spike

Local AArch64 receipt from Eclipse Temurin 21.0.11. JMH 1.37 used two forks,
three 300 ms warmups, and five 300 ms measurements. Ratios are scalar time
divided by vector time, so values above 1 favor the explicit Vector API.

| Kernel | Side 256 scalar / vector (ns/op) | Ratio | Side 1024 scalar / vector (ns/op) | Ratio |
|---|---:|---:|---:|---:|
| exact axis-0 sum | 15,740 / 14,817 | 1.06x | 263,441 / 188,156 | 1.40x |
| contiguous add | 11,109 / 17,294 | 0.64x | 200,341 / 239,896 | 0.84x |
| is-finite | 17,762 / 16,278 | 1.09x | 302,995 / 216,925 | 1.40x |
| less-than | 14,421 / 11,572 | 1.25x | 257,511 / 191,532 | 1.34x |
| sine | 228,280 / 412,958 | 0.55x | 3,640,851 / 6,444,480 | 0.56x |

`parity.json` proves raw-bit identity for axis-0 sum and addition, exact
Boolean identity for the predicates, and zero measured sine error. The
preferred species was 128 bits, or two `Double` lanes.

The side-1024 vector sine control is the decisive loss: it allocated
50,331,875 normalized bytes per operation, approximately 48 bytes per element,
at about 7.14 GB/s and triggered 11 collections during the short profile.
Other vector kernels allocated approximately 6-8 bytes per operation with no
collections. The axis-0 two-fork score also has substantial variance, so its
1.40x mean is promising rather than a promotion-quality result.

Artifacts:

- `parity.json`: correctness and runtime metadata;
- `jmh.json`: paired timings; and
- `gc.json`: side-1024 allocation controls.
