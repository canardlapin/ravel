package ravel.stencil

import ravel.AnyRank
import ravel.MutableNDArray
import ravel.NDArray

/** Allocation-conscious direct executor for arbitrary Ravel views.
  *
  * It calculates physical source and destination addresses from the arrays' existing layouts, so
  * positive, negative, broadcast, and permuted source strides remain valid. It reuses
  * logical-coordinate arrays for the entire pass and performs no per-sample index-array allocation.
  *
  * The generic [[run]] is useful for non-numeric neighborhood reductions. Use [[runDouble]] for
  * floating filters: it calls Ravel's primitive Double storage accessors and keeps the accumulator
  * unboxed.
  */
object DirectNeighborhoodExecutor extends NeighborhoodExecutor:
  /** Prepare a reusable direct-execution workspace for a fixed logical shape. */
  def prepare[A, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
    PreparedDirectNeighborhoodExecutor.prepare(source, destination, spec)

  /** Prepare a reusable workspace for mutable-to-mutable ping-pong passes. */
  def prepare[A, B, R <: AnyRank](
      source: MutableNDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
    PreparedDirectNeighborhoodExecutor.prepare(source, destination, spec)

  def run[A, Acc, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      reducer: NeighborhoodReducer[A, Acc, B],
      constant: A
  ): Unit =
    val context = Context.validate(source, destination, spec)
    val destinationIndices = new Array[Int](context.rank)
    var linear = 0
    while linear < destination.size do
      Context.unravel(linear, destination, destinationIndices)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val sourceAddress =
          Context.sourceAddress(
            source,
            destinationIndices,
            spec,
            context,
            spec.offsets(offsetIndex)
          )
        val value =
          if sourceAddress < 0 then constant
          else source.readGeneric(sourceAddress)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeGeneric(
        Context.destinationAddress(destination, destinationIndices),
        reducer.finish(acc)
      )
      linear += 1

  /** Primitive Double path for linear filters and local floating statistics. */
  def runDouble[R <: AnyRank](
      source: NDArray[Double, R],
      destination: MutableNDArray[Double, R],
      spec: NeighborhoodSpec,
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    val context = Context.validate(source, destination, spec)
    val destinationIndices = new Array[Int](context.rank)
    var linear = 0
    while linear < destination.size do
      Context.unravel(linear, destination, destinationIndices)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val sourceAddress =
          Context.sourceAddress(
            source,
            destinationIndices,
            spec,
            context,
            spec.offsets(offsetIndex)
          )
        val value =
          if sourceAddress < 0 then constant
          else source.readDouble(sourceAddress)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeDouble(
        Context.destinationAddress(destination, destinationIndices),
        reducer.finish(acc)
      )
      linear += 1

  /** Primitive Boolean path for binary morphology. */
  def runBoolean[R <: AnyRank](
      source: NDArray[Boolean, R],
      destination: MutableNDArray[Boolean, R],
      spec: NeighborhoodSpec,
      reducer: BooleanNeighborhoodReducer,
      constant: Boolean
  ): Unit =
    val context = Context.validate(source, destination, spec)
    val destinationIndices = new Array[Int](context.rank)
    var linear = 0
    while linear < destination.size do
      Context.unravel(linear, destination, destinationIndices)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val sourceAddress =
          Context.sourceAddress(
            source,
            destinationIndices,
            spec,
            context,
            spec.offsets(offsetIndex)
          )
        val value =
          if sourceAddress < 0 then constant
          else source.readBoolean(sourceAddress)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeBoolean(
        Context.destinationAddress(destination, destinationIndices),
        reducer.finish(acc)
      )
      linear += 1

  private final case class Context(
      rank: Int,
      sourceSpatialExtents: Array[Int]
  )

  private object Context:
    def validate[A, B, R <: AnyRank](
        source: NDArray[A, R],
        destination: MutableNDArray[B, R],
        spec: NeighborhoodSpec
    ): Context =
      spec.validate()
      val rank = source.rank
      if destination.rank != rank then
        throw IllegalArgumentException(
          s"source rank $rank != destination rank ${destination.rank}"
        )
      if spec.spatialAxes > rank then
        throw IllegalArgumentException(
          s"spatialAxes ${spec.spatialAxes} exceeds array rank $rank"
        )
      StencilArithmetic.requirePositiveSpatialExtents(spec.spatialAxes, source.shape(_))
      var axis = 0
      while axis < spec.spatialAxes do
        if destination.shape(axis) != spec.outputSpatialShape(axis) then
          throw IllegalArgumentException(
            s"destination spatial shape mismatch on axis $axis"
          )
        axis += 1
      while axis < rank do
        if destination.shape(axis) != source.shape(axis) then
          throw IllegalArgumentException(
            s"batch axis $axis shape mismatch between source and destination"
          )
        axis += 1
      Context(
        rank,
        Array.tabulate(spec.spatialAxes)(source.shape(_))
      )

    def sourceAddress[A, R <: AnyRank](
        source: NDArray[A, R],
        destinationIndices: Array[Int],
        spec: NeighborhoodSpec,
        context: Context,
        offset: Vector[Int]
    ): Int =
      var address = source.layout.offset.toLong
      var axis = 0
      while axis < spec.spatialAxes do
        val logical =
          StencilArithmetic.logicalCoordinate(
            spec.outputOrigin(axis),
            destinationIndices(axis),
            offset(axis)
          )
        val mapped =
          BorderIndex.direct(
            logical,
            context.sourceSpatialExtents(axis),
            spec.border
          )
        if mapped < 0 then return -1
        address += mapped.toLong * source.layout.strides(axis).toLong
        axis += 1
      while axis < context.rank do
        address +=
          destinationIndices(axis).toLong * source.layout.strides(axis).toLong
        axis += 1
      address.toInt

    def destinationAddress[A, R <: AnyRank](
        destination: MutableNDArray[A, R],
        destinationIndices: Array[Int]
    ): Int =
      var address = destination.layout.offset.toLong
      var axis = 0
      while axis < destination.rank do
        address +=
          destinationIndices(axis).toLong *
            destination.layout.strides(axis).toLong
        axis += 1
      address.toInt

    def unravel[A, R <: AnyRank](
        linear: Int,
        array: MutableNDArray[A, R],
        out: Array[Int]
    ): Unit =
      var rem = linear
      var axis = array.rank - 1
      while axis >= 0 do
        val extent = array.shape(axis)
        out(axis) = rem % extent
        rem /= extent
        axis -= 1
