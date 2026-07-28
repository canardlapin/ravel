# JDK 25 public-operation matrix

This directory is a diagnostic same-host Ravel-versus-NumPy receipt collected
from commit `3ce982b8193cd4f152442a9adc7266af28876831` on macOS 14.3 AArch64.

- Ravel: Homebrew OpenJDK 25.0.1, Scala 3.7.4, JMH 1.37
- NumPy: 2.4.3 on Python 3.14.3
- Workload: 79 public operation/dtype/layout cases at sides 256 and 1024
- JMH protocol: two forks, three 300 ms warmups, five 300 ms measurements
- Semantic result: all 158 matched signatures passed

The generated artifacts are:

- `ravel-signatures.json`: complete Ravel result signatures and runtime metadata
- `ravel-jmh.json`: the full 158-row JMH timing court
- `numpy.json`: matched NumPy signatures, timings, and runtime metadata
- `comparison.md`: family, dtype, layout, and per-case ratios
- `ravel-jmh-gc.json`: selected allocation controls

The operation-matrix commands are documented in
`docs/numpy-benchmarks.md`. The full JMH command used an absolute `-rff` path
to this directory. The NumPy command used the checked-in comparison virtual
environment at `target/access-patterns/venv`.

The family geometric means were 0.669x binary, 0.646x scalar, 0.773x unary,
0.690x comparison, 0.278x predicate, 0.425x cast, 0.440x reduction, and
0.027x in-place. Ratios are NumPy time divided by Ravel time.

The allocation receipt distinguishes ordinary result allocation from the
mutable-path defect. A 256 by 256 Double result allocates about 526 KiB, a
Boolean result about 66 KiB, and a Double-to-Int result about 264 KiB. Full
mean allocates about 25 B/op. In contrast, in-place Double add allocates about
3.15 MiB/op, approximately 48 bytes per element, proving that the current
generic mutable loop boxes or otherwise allocates per element.

This was a developer workstation, not a frequency-isolated runner. Several
1024 by 1024 allocating rows have wide confidence intervals, so this broad
matrix locates targets but is not release evidence. Focused same-host
before/after controls and allocation receipts remain required for optimization
claims.
