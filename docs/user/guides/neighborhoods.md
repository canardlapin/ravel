# Run neighborhood computations

> `ravel-stencil` is an experimental source module and is not part of the core
> 1.0 artifact or compatibility promise.

Use `ravel-stencil` when a library needs to visit a fixed set of offsets around
every destination sample. The module handles N-dimensional traversal, border
mapping, strided Ravel views, and explicit output storage. It does not assign
image axes, physical units, colour meaning, or display behavior.

The public surface is intentionally low-level: callers provide a source, a
mutable destination, a neighborhood specification, a reducer, and the constant
used by constant-border reads.

## Run a three-sample sum

This example sums each value with its left and right neighbours. Replicate
border mode repeats the nearest edge value:

```scala mdoc:silent
import ravel.*
import ravel.stencil.*

val signal: Array1[Int] =
  NDArray.fromSeq(Shape(5), Seq(1, 2, 3, 4, 5))
val destination = MutableNDArray.zeros[Int, Rank[1]](signal.shape)

val window = NeighborhoodSpec(
  spatialAxes = 1,
  offsets = Vector(Vector(-1), Vector(0), Vector(1)),
  border = BorderMode.Replicate,
  outputOrigin = Vector(0),
  outputSpatialShape = Vector(signal.shape(0))
)

val sumThree = new NeighborhoodReducer[Int, Int, Int]:
  def zero: Int = 0
  def accumulate(acc: Int, value: Int, offsetIndex: Int): Int =
    acc + value
  def finish(acc: Int): Int = acc

ReferenceNeighborhoodExecutor.run(
  signal,
  destination,
  window,
  sumThree,
  constant = 0
)

val summed: Array1[Int] = destination.freezeCopy()
```

```scala mdoc
summed.iterator.toList
```

The reducer receives `offsetIndex` as well as the sampled value, so weighted
filters can associate coefficients with the declared offset order.

## Choose border behavior explicitly

| Mode | Exterior index consequence |
|---|---|
| `Constant` | supplies the `constant` argument |
| `Replicate` | clamps to the nearest edge sample |
| `Wrap` | uses modular indexing |
| `ReflectWithoutEdge` | reflects without repeating the edge sample; NumPy `reflect` / OpenCV `REFLECT_101` convention |
| `ReflectWithEdge` | reflects while repeating the edge sample; NumPy `symmetric` / OpenCV `REFLECT` convention |

The two reflection names are deliberately explicit because libraries disagree
about what a bare “mirror” mode means.

`outputOrigin` maps destination coordinate zero into source coordinates.
`outputSpatialShape` describes the destination's leading spatial axes. The API
does not currently provide `same`, `valid`, or `full` convenience constructors;
the calling library must calculate both values.

## Keep batch axes fixed

Only the leading `spatialAxes` participate in offset traversal. Any trailing
axes are batch axes: their coordinates are copied from destination to source
without applying neighborhood offsets. This lets a two-dimensional footprint
run independently over channels or batches, provided those dimensions are
trailing axes.

## Move from the reference to the direct executor

`ReferenceNeighborhoodExecutor` is the clarity-first conformance oracle.
`DirectNeighborhoodExecutor` follows the same contract while calculating
physical addresses directly for canonical, sliced, reversed, permuted, and
broadcast source views.

```scala mdoc:silent
val directDestination = MutableNDArray.zeros[Int, Rank[1]](signal.shape)

DirectNeighborhoodExecutor.run(
  signal,
  directDestination,
  window,
  sumThree,
  constant = 0
)
```

```scala mdoc
directDestination.freezeCopy().sameElements(summed)
```

For repeated passes over the same logical source and destination shapes,
`DirectNeighborhoodExecutor.prepare` creates a reusable schedule and workspace.
The prepared executor has primitive `runBoolean`, `runByte`, `runShort`,
`runFloat`, and `runDouble` paths. Mutable-to-mutable prepared passes support
ping-pong workspaces, but source and destination must not share storage.

`StencilExecutionPolicy` is currently a placeholder: its method and parallelism
fields do not change executor selection or scheduling. Choose the reference or
direct executor explicitly, and do not infer parallel execution from the
default `Auto` values.

## Know what changes

| Input or operation | Values | Shape and layout | Ownership |
|---|---|---|---|
| source `NDArray` | read only | arbitrary valid view layout is accepted | remains owned and immutable |
| destination `MutableNDArray` | overwritten once per logical output sample | must match the declared output spatial shape and source batch axes | remains mutable and caller-owned |
| `freezeCopy()` after a pass | copied result values | preserves destination logical shape in canonical owned storage | new immutable owner |
| prepared executor | no array values retained | bound to source and destination logical shapes, not storage identities | owns reusable scheduling workspace |

Invalid specifications, rank mismatches, destination-shape mismatches, and
incompatible prepared runs currently throw `IllegalArgumentException`. Validate
shape and axis choices before entering a long-running pipeline; see
[Failures and recovery](../reference/failures.md).

Use the [Stencil API](https://canardlapin.github.io/ravel/api/stencil/) for the
primitive reducer signatures and prepared-executor overloads.
