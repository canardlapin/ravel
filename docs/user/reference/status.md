# Availability and project scope

Ravel is an early, source-available project. No Maven Central artifacts or
stable release are currently claimed. Local `1.0.0-SNAPSHOT` publication is a
`ravel-core` integration convenience, not a released version.

## Supported development surface

| Module | Current source surface | Relationship |
|---|---|---|
| `ravel-core` | Primitive dense arrays, views, broadcasting, eager computation, mutation, and interop | Ordinary entry point |
| `ravel-packed` | One-, two-, and four-bit codes plus one-bit set algebra | Separate representation; not a `DType` |
| `ravel-stencil` | Border mapping and neighborhood execution | Depends on `ravel-core` |
| `ravel-laws` | Reusable MUnit and ScalaCheck laws | Test support depending on `ravel-core` |

All four modules are cross-built for the JVM and Scala.js in the current
source tree. The executable guide runs on the JVM; repository suites and API
compilation provide the separate Scala.js evidence.

The enforced 1.0 publication matrix contains only `ravel-core`. Laws, packed,
and stencil set `publish / skip := true` and remain experimental source APIs.

## Deliberate exclusions

Ravel does not own matrix semantics, decompositions, sparse storage, BLAS
selection, random-number policy, I/O, chunking, GPU execution, automatic
differentiation, or lazy fusion.

## Compatibility

Until a first public release, source and binary compatibility may change.
MiMa is configured as release scaffolding but has no previous public artifact
to compare. Treat the guide and generated APIs as documentation of the
repository revision that produced them, not as a stable compatibility promise.

For contributors, release gates, benchmark evidence, and implementation
contracts remain in the repository's internal `docs/` files. They are kept out
of this public guide so user-facing navigation is not dominated by project
management material.
