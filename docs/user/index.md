# Ravel

Ravel is a small, cross-platform library for dense, rectangular
multidimensional arrays in Scala 3. Its shared API runs on the JVM and
Scala.js, stores primitive values without boxing in the supported platform
representations, and makes copying, viewing, borrowing, and mutation explicit.

Ravel is pre-release software. There are no published Maven Central artifacts
yet, and the API may change. The guide describes the current `main` branch for
source-based evaluation and local integration; it does not imply a 1.0 release.

## Your first ten minutes

1. Follow [Getting started](getting-started.md) to create, transform, and reduce
   an array.
2. Read [Core concepts](core-concepts.md) for the shape, rank, dtype, layout,
   and ownership model.
3. Choose a [task-oriented guide](guides/index.md) for views, numerical
   computation, controlled mutation, or platform interop.

Every Scala block on the main learning path is compiled and executed by mdoc
when the site is built. Laika renders the site, orders its navigation, and
rejects broken internal links.

## Choose a path

- **Create and compute:** [Numerical computation](guides/computation.md)
  introduces constructors, broadcasting, elementwise operations, and
  reductions.
- **Index without accidental copies:** [Indexing and views](guides/views.md)
  explains fixed-rank reads, slices, transposes, broadcasting, and reshape.
- **Fill a destination deliberately:** [Mutation and builders](guides/mutation-and-builders.md)
  separates explicit mutable work from consuming immutable construction.
- **Cross a platform boundary:** [JVM and Scala.js interop](guides/interop.md)
  distinguishes copying from borrowing.

## Look something up

The [reference section](reference/index.md) contains a copy/view table, dtype
and rank capabilities, a NumPy operation map, and an honest availability
statement. Use the generated
[API reference](https://canardlapin.github.io/ravel/api/) for symbol-level
signatures.

Internal implementation plans, audit reports, benchmark receipts, and release
evidence remain versioned in the repository but are not part of this user
guide.
