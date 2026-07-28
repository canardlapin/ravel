# JDK 21 public-operation matrix

This directory is the supported-baseline companion to the JDK 25 diagnostic
receipt. It was collected from commit
`3ce982b8193cd4f152442a9adc7266af28876831` on the same macOS 14.3 AArch64
host.

- Ravel: Eclipse Temurin 21.0.11+10 LTS, Scala 3.7.4, JMH 1.37
- NumPy: 2.4.3 on Python 3.14.3
- Workload: 79 public operation/dtype/layout cases at sides 256 and 1024
- JMH protocol: two forks, three 300 ms warmups, five 300 ms measurements
- Semantic result: all 158 matched signatures passed

The generated artifacts are `ravel-signatures.json`, `ravel-jmh.json`,
`numpy.json`, and `comparison.md`. The JDK was a portable Temurin archive; the
full JMH command used its `Contents/Home` as `JAVA_HOME` and an absolute
`-rff` path to this directory. The operation-matrix commands are documented in
`docs/numpy-benchmarks.md`.

The family geometric means were 0.804x binary, 0.783x scalar, 0.731x unary,
0.683x comparison, 0.312x predicate, 1.034x cast, 0.429x reduction, and
0.026x in-place. Ratios are NumPy time divided by Ravel time.

The severe mutable loss is independent of the newer JDK: 1024 by 1024
in-place Double add/subtract/multiply/divide took about 4.48--4.51 ms, reversed
in-place add about 13.47 ms, and inner-strided multiply about 6.48 ms. The
corresponding JDK 25 values were about 4.66--4.83 ms, 13.58 ms, and 6.34 ms.

This was a developer workstation, not a frequency-isolated runner. Several
1024 by 1024 allocating rows have wide confidence intervals. The complete
court proves coverage and portability; focused same-host controls, not broad
geometric means, are the acceptance evidence for subsequent changes.
