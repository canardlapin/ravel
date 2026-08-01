package ravel.stencil

import ravel.AnyRank
import ravel.MutableNDArray
import ravel.NDArray

/** Reusable schedule and workspace for direct primitive neighborhood passes.
  *
  * A plan is bound to the source and destination logical shapes used at preparation time, but not
  * to storage identities or layouts. It can therefore be reused across canonical, sliced, reversed,
  * permuted, and broadcast source views of that shape without exposing or aliasing their buffers.
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
    validateCompatible(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source, spec.offsets(offsetIndex))
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
    validateCompatible(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source, spec.offsets(offsetIndex))
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
    validateCompatible(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source, spec.offsets(offsetIndex))
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
    validateCompatible(source, destination)
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val address = sourceAddress(source, spec.offsets(offsetIndex))
        val value = if address < 0 then constant else source.readShort(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeShort(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  private def validateCompatible[A, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R]
  ): Unit =
    if source.rank != rank || destination.rank != rank then
      throw IllegalArgumentException(
        s"prepared rank $rank does not match source/destination ranks " +
          s"${source.rank}/${destination.rank}"
      )
    var axis = 0
    while axis < rank do
      if source.shape(axis) != sourceShape(axis) then
        throw IllegalArgumentException(
          s"prepared source shape mismatch on axis $axis"
        )
      if destination.shape(axis) != destinationShape(axis) then
        throw IllegalArgumentException(
          s"prepared destination shape mismatch on axis $axis"
        )
      axis += 1

  private def sourceAddress[A, R <: AnyRank](
      source: NDArray[A, R],
      offset: Vector[Int]
  ): Int =
    var address = source.layout.offset.toLong
    var axis = 0
    while axis < spec.spatialAxes do
      val logical =
        spec.outputOrigin(axis) + destinationIndices(axis) + offset(axis)
      val mapped =
        BorderIndex.direct(logical, sourceShape(axis), spec.border)
      if mapped < 0 then return -1
      address += mapped.toLong * source.layout.strides(axis).toLong
      axis += 1
    while axis < rank do
      address += destinationIndices(axis).toLong * source.layout.strides(axis).toLong
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
    val _ = policy
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
    PreparedDirectNeighborhoodExecutor(
      spec,
      rank,
      Array.tabulate(rank)(source.shape(_)),
      Array.tabulate(rank)(destination.shape(_)),
      new Array[Int](rank)
    )
