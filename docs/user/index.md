# Ravel

Ravel is a Scala 3 library for eager, dense multidimensional arrays. Its shared
API runs on the JVM and Scala.js, stores supported dtypes in platform-specific
representations, and makes copying, viewing, borrowing, and mutation explicit.

Ravel is pre-release software. There are no published Maven Central artifacts
yet, and the API may change. The guide is versioned with the source and describes
the revision that built it; it does not imply a 1.0 release.

## Your first ten minutes

1. Follow [Getting started](getting-started.md) to create, transform, and reduce
   an array.
2. Read [Core concepts](core-concepts.md) for the shape, rank, dtype, layout,
   and ownership model.
3. Choose a [task-oriented guide](guides/index.md) for views, numerical
   computation, controlled mutation, packed codes, neighborhoods, or platform
   interop.

Executable examples in the guide run on the JVM through mdoc when the site is
built. Separate repository suites verify the shared JVM and Scala.js behavior;
a successful mdoc run alone is not cross-platform evidence. Laika renders the
site, orders its navigation, and rejects broken internal links.

## Choose the surface that owns your problem

| Goal | Start with | Why |
|---|---|---|
| Dense primitive arrays, views, broadcasting, and reductions | `ravel-core` | This is Ravel's ordinary array API. |
| One-, two-, or four-bit storage codes | `ravel-packed` | Packed codes are a separate representation, not an `NDArray` dtype. |
| N-dimensional neighborhood traversal | `ravel-stencil` | Stencil execution operates over core arrays but leaves image meaning to higher layers. |
| Repository conformance checks | `ravel-laws` | Internal generated tests; there is no downstream laws artifact. |

Only `ravel-core` is slated for 1.0 publication. Packed and stencil are
experimental source modules. Laws is an internal test harness; the repository
cross-runs it on both platforms without exposing a main-source API.

## Choose a path

- **Create and compute:** [Numerical computation](guides/computation.md)
  introduces constructors, broadcasting, elementwise operations, and
  reductions.
- **Index without accidental copies:** [Indexing and views](guides/views.md)
  explains fixed-rank reads, slices, transposes, broadcasting, and reshape.
- **Fill a destination deliberately:** [Mutation and builders](guides/mutation-and-builders.md)
  separates explicit mutable work from consuming immutable construction.
- **Store compact codes:** [Packed codes](guides/packed-codes.md) covers
  validated construction, views, serialization words, and one-bit set algebra.
- **Run a neighborhood:** [Neighborhood computations](guides/neighborhoods.md)
  explains border modes, reducers, destinations, and direct executors.
- **Cross a platform boundary:** [JVM and Scala.js interop](guides/interop.md)
  distinguishes copying from borrowing.

## Look something up

The [reference section](reference/index.md) contains a copy/view table, dtype
and rank capabilities, failure behavior, a NumPy operation map, module API
links, and an honest availability statement.

Internal implementation plans, audit reports, benchmark receipts, and release
evidence remain versioned in the repository but are not part of this user
guide.
