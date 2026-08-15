# Ravel 1.0 release evidence

Evidence was collected on 2026-07-27 with Scala 3.7.4, Scala.js 1.22.0,
sbt 1.11.7, Node 24.1.0, and OpenJDK 25.0.1.

## Formatting restoration receipt

On 2026-08-03, `sbt fmt` mechanically reformatted every source reported by the
repository-wide gate across core, packed, stencil, JVM, and Scala.js source
sets. `sbt fmtCheck` is the exact verification command and is required by both
protected-branch CI and the tag workflow. The ownership and API hardening
milestone was committed as `ef04c32`. Each subsequent candidate gate records
its exact commit and runtime environment in
`target/release-gate/<commit>/manifest.txt`; this document is not a substitute
for that receipt.

## Semantic and platform gates

The pre-gate hardening run of `sbt testAllFull` passed:

- 187 core tests on the JVM;
- 182 core tests on Node;
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

## Downstream artifact integration

The authoritative gate pins Gale revision
`98508f8d36ceedfb3a7cb4ea18807116fa6af66b`. Its `interopRavelTest` passes 8
tests on the JVM and 8 on Scala.js against the exact local candidate.
Conversions copy logical values and remove borrowed external aliases. The
gate also pins zarr4s revision
`b7c9840fdd4e5676a58acb4ae516dfdaad9bd177`, applies the checked-in migration
to Ravel's pure checked errors, and compiles its JVM and Scala.js adapter. A
fresh offline consumer then completes a Float32 write, read, Ravel transform,
write, and verification workflow on both platforms.

Re-verify after Ravel changes with:

```sh
bash scripts/verify-sibling-consumers.sh
```

Both consumers use public Ravel APIs, and their resolved classpaths must name
the exact local JVM and Scala.js candidate. The detailed friction and ownership
record is in [`consumer-validation.md`](consumer-validation.md).

## Artifact scope

The current enforced 1.0 matrix publishes only `ravel-core` for JVM and
Scala.js. `ravel-packed` and `ravel-stencil` remain cross-tested source modules
with publishing disabled. `ravel-laws` is also cross-tested, but only from Test:
clean Compile outputs contain no `ravel.laws` classes and its test-framework
dependencies do not enter the Compile classpath. This supersedes the earlier
candidate inspection in this document, which treated laws as publishable.

Generated POM inspection found:

- `ravel-core` has only the Scala runtime and, on Scala.js, the Scala.js
  runtime as compile dependencies; test libraries remain test-scoped.
- Dependency eviction reports contain no conflicting Ravel runtime library.
  Scala.js selects the linker-compatible standard library supplied by
  Scala.js 1.22.0. MUnit-ScalaCheck selects ScalaCheck 1.19.0 over Discipline's
  older compatible declaration.

The inspected core binary jars contain no Gale, Breeze, storage-format, I/O,
autodiff, GPU, or sparse-array package.

The current artifact verifier consumes the build-emitted eight-row publication
manifest. Both published core platforms must provide exact-version POM,
binary, source, and API jars; negative tests remove one platform row and one
artifact target to prove those omissions fail rather than false-green.

## Performance gate

The final baselines and the 70-percent regression budget are recorded in
[`benchmark-baselines.md`](benchmark-baselines.md). The JVM public addition
allocates one result buffer plus 1,177 bytes of wrapper and plan overhead,
within the 2,048-byte allowance. The raw probes reuse their output.

The experimental stencil court separately reports preparation allocation and
per-run allocation for the sequential `PreparedDirectNeighborhoodExecutor`,
including dtype, layout class, and reusable-workspace status. Its shared
conformance matrix covers every border mode, canonical/reversed/permuted/sliced
rank-three and rank-four inputs, immutable and mutable sources, and empty
outputs on JVM and Scala.js. Stencil remains unpublished and is not part of the
core 1.0 performance promise.

## Publication boundary

The repository contains a tag-triggered `sbt-ci-release` workflow for signed
Central Portal publication. No release tag has been pushed and no Maven Central
artifact is claimed in this evidence. Publication remains gated on the
repository's `PGP_PASSPHRASE`, `PGP_SECRET`, `SONATYPE_USERNAME`, and
`SONATYPE_PASSWORD` secrets and confirmation that `io.github.canardlapin` is an
authorized Central Portal namespace.
