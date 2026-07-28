# Availability and project scope

Ravel is an early, source-available project. No Maven Central artifacts or
stable release are currently claimed. Local `1.0.0-SNAPSHOT` publication is an
integration convenience, not a released version.

## Supported development surface

- Scala 3 shared sources for JVM and Scala.js
- primitive dense rectangular arrays
- runtime shapes with dynamic or statically refined rank
- checked structural views and broadcasting
- eager elementwise operations and deterministic reductions
- explicit mutable copies, consuming immutable builders, and reusable kernel
  destinations
- copy and borrowed interop for JVM arrays and Scala.js typed arrays

## Deliberate exclusions

Ravel does not own matrix semantics, decompositions, sparse storage, BLAS
selection, random-number policy, I/O, chunking, GPU execution, automatic
differentiation, or lazy fusion.

## Compatibility

Until a first public release, source and binary compatibility may change.
MiMa is configured as release scaffolding but has no previous public artifact
to compare. Treat the guide and generated API as documentation of the current
main branch.

For contributors, release gates, benchmark evidence, and implementation
contracts remain in the repository's internal `docs/` files. They are kept out
of this public guide so user-facing navigation is not dominated by project
management material.
