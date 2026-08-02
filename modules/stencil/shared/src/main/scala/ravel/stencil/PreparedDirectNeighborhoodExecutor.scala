package ravel.stencil

import ravel.AnyRank
import ravel.MutableNDArray
import ravel.NDArray
import ravel.Shape
import ravel.internal.Layout

/** Reusable schedule and workspace for direct primitive neighborhood passes.
  *
  * A plan is bound to the source and destination logical shapes used at preparation time, but not
  * to storage identities or layouts. It can therefore be reused across canonical, sliced, reversed,
  * permuted, and broadcast source views of that shape without exposing or aliasing their buffers.
  *
  * Mutable sources are supported for multi-pass ping-pong schedules. A mutable source and
  * destination must not share storage; callers swap distinct workspaces between axes instead of
  * freezing intermediate buffers.
  */
final class PreparedDirectNeighborhoodExecutor private[stencil] (
    private val spec: NeighborhoodSpec,
    private val rank: Int,
    private val sourceShape: Array[Int],
    private val destinationShape: Array[Int],
    private val destinationIndices: Array[Int]
):
  /** Run a primitive Double neighborhood pass without per-run workspace allocation. */
  def runDouble[R <: AnyRank](
      source: NDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    validateCompatible(source.rank, source.shape, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readDouble(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeDouble(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Double pass reading a mutable workspace source. */
  def runDouble[R <: AnyRank](
      source: MutableNDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    validateCompatibleMutable(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readDouble(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeDouble(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Float neighborhood pass without per-run workspace allocation. */
  def runFloat[R <: AnyRank](
      source: NDArray[Float, R],
      destination: MutableNDArray[Float, R],
      reducer: FloatNeighborhoodReducer,
      constant: Float
  ): Unit =
    validateCompatible(source.rank, source.shape, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readFloat(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeFloat(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Float pass reading a mutable workspace source. */
  def runFloat[R <: AnyRank](
      source: MutableNDArray[Float, R],
      destination: MutableNDArray[Float, R],
      reducer: FloatNeighborhoodReducer,
      constant: Float
  ): Unit =
    validateCompatibleMutable(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readFloat(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeFloat(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Byte neighborhood pass without per-run workspace allocation. */
  def runByte[R <: AnyRank](
      source: NDArray[Byte, R],
      destination: MutableNDArray[Byte, R],
      reducer: ByteNeighborhoodReducer,
      constant: Byte
  ): Unit =
    validateCompatible(source.rank, source.shape, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readByte(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeByte(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Byte pass reading a mutable workspace source. */
  def runByte[R <: AnyRank](
      source: MutableNDArray[Byte, R],
      destination: MutableNDArray[Byte, R],
      reducer: ByteNeighborhoodReducer,
      constant: Byte
  ): Unit =
    validateCompatibleMutable(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readByte(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeByte(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Short neighborhood pass without per-run workspace allocation. */
  def runShort[R <: AnyRank](
      source: NDArray[Short, R],
      destination: MutableNDArray[Short, R],
      reducer: ShortNeighborhoodReducer,
      constant: Short
  ): Unit =
    validateCompatible(source.rank, source.shape, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readShort(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeShort(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Short pass reading a mutable workspace source. */
  def runShort[R <: AnyRank](
      source: MutableNDArray[Short, R],
      destination: MutableNDArray[Short, R],
      reducer: ShortNeighborhoodReducer,
      constant: Short
  ): Unit =
    validateCompatibleMutable(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source.layout, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readShort(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeShort(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  private def validateCompatibleMutable[A, B, R <: AnyRank](
      source: MutableNDArray[A, R],
      destination: MutableNDArray[B, R]
  ): Unit =
    if source.storage eq destination.storage then
      throw IllegalArgumentException(
        "mutable source and destination must not share storage; use distinct ping-pong workspaces"
      )
    validateCompatible(source.rank, source.shape, destination)

  private def validateCompatible[B, R <: AnyRank](
      sourceRank: Int,
      sourceShapeValue: Shape[R],
      destination: MutableNDArray[B, R]
  ): Unit =
    if sourceRank != rank || destination.rank != rank then
      throw IllegalArgumentException(
        s"prepared rank $rank does not match source/destination ranks " +
          s"$sourceRank/${destination.rank}"
      )
    var axis = 0
    while axis < rank do
      if sourceShapeValue(axis) != sourceShape(axis) then
        throw IllegalArgumentException(
          s"prepared source shape mismatch on axis $axis"
        )
      if destination.shape(axis) != destinationShape(axis) then
        throw IllegalArgumentException(
          s"prepared destination shape mismatch on axis $axis"
        )
      axis += 1

  private def sourceAddress(
      sourceLayout: Layout,
      offset: Vector[Int]
  ): Int =
    var address = sourceLayout.offset.toLong
    var axis = 0
    while axis < spec.spatialAxes do
      val logical =
        spec.outputOrigin(axis) + destinationIndices(axis) + offset(axis)
      val mapped =
        BorderIndex.direct(logical, sourceShape(axis), spec.border)
      if mapped < 0 then return -1
      address += mapped.toLong * sourceLayout.strides(axis).toLong
      axis += 1
    while axis < rank do
      address += destinationIndices(axis).toLong * sourceLayout.strides(axis).toLong
      axis += 1
    address.toInt

  private def destinationAddress[A, R <: AnyRank](
      destination: MutableNDArray[A, R]
  ): Int =
    var address = destination.layout.offset.toLong
    var axis = 0
    while axis < rank do
      address +=
        destinationIndices(axis).toLong * destination.layout.strides(axis).toLong
      axis += 1
    address.toInt

  private def unravel[A, R <: AnyRank](
      linear: Int,
      array: MutableNDArray[A, R]
  ): Unit =
    var rem = linear
    var axis = rank - 1
    while axis >= 0 do
      val extent = array.shape(axis)
      destinationIndices(axis) = rem % extent
      rem /= extent
      axis -= 1

object PreparedDirectNeighborhoodExecutor:
  /** Validate a direct pass and allocate private, reusable scheduling workspace. */
  def prepare[A, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      policy: StencilExecutionPolicy = StencilExecutionPolicy()
  ): PreparedDirectNeighborhoodExecutor =
    prepareShapes(source.rank, source.shape, destination, spec, policy)

  /** Validate a mutable-to-mutable direct pass and allocate reusable scheduling workspace. */
  def prepare[A, B, R <: AnyRank](
      source: MutableNDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
    prepare(source, destination, spec, StencilExecutionPolicy())

  def prepare[A, B, R <: AnyRank](
      source: MutableNDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      policy: StencilExecutionPolicy
  ): PreparedDirectNeighborhoodExecutor =
    if source.storage eq destination.storage then
      throw IllegalArgumentException(
        "mutable source and destination must not share storage; use distinct ping-pong workspaces"
      )
    prepareShapes(source.rank, source.shape, destination, spec, policy)

  private def prepareShapes[B, R <: AnyRank](
      sourceRank: Int,
      sourceShapeValue: Shape[R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      policy: StencilExecutionPolicy
  ): PreparedDirectNeighborhoodExecutor =
    val _ = policy
    spec.validate()
    val rank = sourceRank
    if destination.rank != rank then
      throw IllegalArgumentException(
        s"source rank $rank != destination rank ${destination.rank}"
      )
    if spec.spatialAxes > rank then
      throw IllegalArgumentException(
        s"spatialAxes ${spec.spatialAxes} exceeds array rank $rank"
      )
    var axis = 0
    while axis < spec.spatialAxes do
      if destination.shape(axis) != spec.outputSpatialShape(axis) then
        throw IllegalArgumentException(
          s"destination spatial shape mismatch on axis $axis"
        )
      axis += 1
    while axis < rank do
      if destination.shape(axis) != sourceShapeValue(axis) then
        throw IllegalArgumentException(
          s"batch axis $axis shape mismatch between source and destination"
        )
      axis += 1
    PreparedDirectNeighborhoodExecutor(
      spec,
      rank,
      Array.tabulate(rank)(sourceShapeValue(_)),
      Array.tabulate(rank)(destination.shape(_)),
      new Array[Int](rank)
    )
