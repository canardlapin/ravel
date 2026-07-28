# Ravel 1.0 release evidence

Evidence was collected on 2026-07-27 with Scala 3.7.4, Scala.js 1.22.0,
sbt 1.11.7, Node 24.1.0, and OpenJDK 25.0.1.

## Semantic and platform gates

`sbt testAllFull` passed:

- 75 core tests on the JVM;
- 77 core tests on Node;
- 2 reusable laws tests on each platform;
- 3 browser tests in headless Chromium;
- full-optimized links for the core and laws Scala.js test bundles.

The suites cover checked layouts, rank evidence, casts, structural views,
broadcasting, specialized/reference differential kernels, reductions,
mutable-layout locality, freeze isolation, borrowed ownership, JVM arrays,
JavaScript typed arrays, equality, and previews.

`sbt representationProof` passed both core suites and the optimized
representation link. JVM bytecode contains primitive array loads, arithmetic,
and stores in the dtype-specific loops. The optimized JavaScript uses concrete
typed arrays.

## Remote CI

Commit `7f7c90fe5cda40ca8aaccd038ea0edd71793622e` is the first commit on
`canardlapin/ravel` `main`. GitHub Actions run
[`30264191210`](https://github.com/canardlapin/ravel/actions/runs/30264191210)
passed on Temurin JDK 21 and Node 22. Its cross-platform job completed both:

- `sbt testAllFull`, covering the JVM, Node, reusable laws, real Chromium, and
  full-optimized Scala.js links; and
- `sbt representationProof`, covering both core suites and the optimized
  representation probe.

## Gale integration

Ravel `ravel-core` snapshots were published locally for integration
verification. In the sibling Gale repository, `sbt interopRavelTest` passed
four JVM and four Node tests. The publishable `gale-interop-ravel` cross-project
depends on `gale-core` and `ravel-core`; neither core depends on the other.
Conversions copy logical values and remove borrowed external aliases.

Re-verify after Ravel changes with:

```sh
# in ravel
sbt publishLocalSnapshot
# in gale
sbt interopRavelTest
```

Gale pins `ravelVersion = "1.0.0-SNAPSHOT"` until the Central `1.0.0` release;
`publishLocalSnapshot` forces that version on the Ravel side.

## Artifact scope

The publishable artifacts are `ravel-core` and `ravel-laws` for JVM and
Scala.js. The root, benchmarks, and browser-test projects have publishing
disabled.

Generated POM inspection found:

- `ravel-core` has only the Scala runtime and, on Scala.js, the Scala.js
  runtime as compile dependencies; test libraries remain test-scoped.
- `ravel-laws` has compile dependencies on `ravel-core`, MUnit,
  MUnit-ScalaCheck, and Discipline, as required for reusable law bundles.
- Dependency eviction reports contain no conflicting Ravel runtime library.
  Scala.js selects the linker-compatible standard library supplied by
  Scala.js 1.22.0. MUnit-ScalaCheck selects ScalaCheck 1.19.0 over Discipline's
  older compatible declaration.

The built JVM jars were about 392 KiB for `ravel-core` and 76 KiB for
`ravel-laws`. The core jar contains no Gale, Breeze, storage-format, I/O,
autodiff, GPU, or sparse-array package.

## Performance gate

The final baselines and the 70-percent regression budget are recorded in
[`benchmark-baselines.md`](benchmark-baselines.md). The JVM public addition
allocates one result buffer plus 1,177 bytes of wrapper and plan overhead,
within the 2,048-byte allowance. The raw probes reuse their output.

## Publication boundary

The repository contains a tag-triggered `sbt-ci-release` workflow for signed
Central Portal publication. No release tag has been pushed and no Maven Central
artifact is claimed in this evidence. Publication remains gated on the
repository's `PGP_PASSPHRASE`, `PGP_SECRET`, `SONATYPE_USERNAME`, and
`SONATYPE_PASSWORD` secrets and confirmation that `io.github.canardlapin` is an
authorized Central Portal namespace.
