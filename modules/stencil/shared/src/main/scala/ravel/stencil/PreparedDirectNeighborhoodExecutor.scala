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
  *
  * The plan owns its coordinate arrays and offset workspace for its entire lifetime. Public runs on
  * one plan are synchronized, so concurrent callers are serialized and cannot race that workspace.
  * Separate plan instances may execute concurrently. Sources and destinations remain caller-owned:
  * do not mutate either from outside the plan until its run returns. No source, destination, or
  * reducer is retained after a run.
  */
final class PreparedDirectNeighborhoodExecutor private[stencil] (
    private val spec: NeighborhoodSpec,
    private val rank: Int,
    private val sourceShape: Array[Int],
    private val destinationShape: Array[Int],
    private val destinationIndices: Array[Int],
    private val offsets: Array[Array[Int]],
    private val outputOrigin: Array[Int]
):
  private val offsetCount = offsets.length
  private val spatialAxes = spec.spatialAxes
  private val border = spec.border

  /** Run a primitive Double neighborhood pass without per-run workspace allocation. */
  def runDouble[R <: AnyRank](
      source: NDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit = this.synchronized:
    validateCompatible(source.rank, source.shape, destination)
    if destination.size == 0 then return
    if spatialAxes == 1 then
      runDoubleLines(source, destination, reducer, constant)
      return
    if spatialAxes == rank && (rank == 2 || rank == 3) then
      runDoubleSpatial(source, destination, reducer, constant)
      return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatibleMutable(source, destination)
    if destination.size == 0 then return
    if spatialAxes == 1 then
      runDoubleLines(source, destination, reducer, constant)
      return
    if spatialAxes == rank && (rank == 2 || rank == 3) then
      runDoubleSpatial(source, destination, reducer, constant)
      return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatible(source.rank, source.shape, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatibleMutable(source, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
        val value = if address < 0 then constant else source.readFloat(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeFloat(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Boolean neighborhood pass without per-run workspace allocation. */
  def runBoolean[R <: AnyRank](
      source: NDArray[Boolean, R],
      destination: MutableNDArray[Boolean, R],
      reducer: BooleanNeighborhoodReducer,
      constant: Boolean
  ): Unit = this.synchronized:
    validateCompatible(source.rank, source.shape, destination)
    if destination.size == 0 then return
    if spatialAxes == 1 then
      runBooleanLines(source, destination, reducer, constant)
      return
    if spatialAxes == rank && (rank == 2 || rank == 3) then
      runBooleanSpatial(source, destination, reducer, constant)
      return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
        val value = if address < 0 then constant else source.readBoolean(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        if reducer.isTerminal(acc) then offsetIndex = offsetCount
        else offsetIndex += 1
      destination.writeBoolean(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Boolean pass reading a mutable workspace source. */
  def runBoolean[R <: AnyRank](
      source: MutableNDArray[Boolean, R],
      destination: MutableNDArray[Boolean, R],
      reducer: BooleanNeighborhoodReducer,
      constant: Boolean
  ): Unit = this.synchronized:
    validateCompatibleMutable(source, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
        val value = if address < 0 then constant else source.readBoolean(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        if reducer.isTerminal(acc) then offsetIndex = offsetCount
        else offsetIndex += 1
      destination.writeBoolean(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  /** Run a primitive Byte neighborhood pass without per-run workspace allocation. */
  def runByte[R <: AnyRank](
      source: NDArray[Byte, R],
      destination: MutableNDArray[Byte, R],
      reducer: ByteNeighborhoodReducer,
      constant: Byte
  ): Unit = this.synchronized:
    validateCompatible(source.rank, source.shape, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatibleMutable(source, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatible(source.rank, source.shape, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
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
  ): Unit = this.synchronized:
    validateCompatibleMutable(source, destination)
    if destination.size == 0 then return
    var linear = 0
    while linear < destination.size do
      unravel(linear, destination)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < offsetCount do
        val address = sourceAddress(source.layout, offsets(offsetIndex))
        val value = if address < 0 then constant else source.readShort(address)
        acc = reducer.accumulate(acc, value, offsetIndex)
        offsetIndex += 1
      destination.writeShort(destinationAddress(destination), reducer.finish(acc))
      linear += 1

  private def runDoubleLines[R <: AnyRank](
      source: NDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    val sourceLayout = source.layout
    val destinationLayout = destination.layout
    val sourceStride = sourceLayout.strides(0).toLong
    val destinationStride = destinationLayout.strides(0).toLong
    val outputExtent = destinationShape(0)
    val lineCount = destination.size / outputExtent
    var line = 0
    while line < lineCount do
      setBatchIndices(line)
      val sourceBase = batchAddress(sourceLayout)
      val destinationBase = batchAddress(destinationLayout)
      var row = 0
      while row < outputExtent do
        var acc = reducer.zero
        var offsetIndex = 0
        while offsetIndex < offsetCount do
          val offset = offsets(offsetIndex)
          val mapped = BorderIndex.direct(
            StencilArithmetic.logicalCoordinate(outputOrigin(0), row, offset(0)),
            sourceShape(0),
            border
          )
          val value =
            if mapped < 0 then constant
            else source.readDouble((sourceBase.toLong + mapped * sourceStride).toInt)
          acc = reducer.accumulate(acc, value, offsetIndex)
          offsetIndex += 1
        destination.writeDouble(
          (destinationBase.toLong + row * destinationStride).toInt,
          reducer.finish(acc)
        )
        row += 1
      line += 1

  private def runDoubleLines[R <: AnyRank](
      source: MutableNDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    val sourceLayout = source.layout
    val destinationLayout = destination.layout
    val sourceStride = sourceLayout.strides(0).toLong
    val destinationStride = destinationLayout.strides(0).toLong
    val outputExtent = destinationShape(0)
    val lineCount = destination.size / outputExtent
    var line = 0
    while line < lineCount do
      setBatchIndices(line)
      val sourceBase = batchAddress(sourceLayout)
      val destinationBase = batchAddress(destinationLayout)
      var row = 0
      while row < outputExtent do
        var acc = reducer.zero
        var offsetIndex = 0
        while offsetIndex < offsetCount do
          val offset = offsets(offsetIndex)
          val mapped = BorderIndex.direct(
            StencilArithmetic.logicalCoordinate(outputOrigin(0), row, offset(0)),
            sourceShape(0),
            border
          )
          val value =
            if mapped < 0 then constant
            else source.readDouble((sourceBase.toLong + mapped * sourceStride).toInt)
          acc = reducer.accumulate(acc, value, offsetIndex)
          offsetIndex += 1
        destination.writeDouble(
          (destinationBase.toLong + row * destinationStride).toInt,
          reducer.finish(acc)
        )
        row += 1
      line += 1

  private def runDoubleSpatial[R <: AnyRank](
      source: NDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    if rank == 2 then
      val sourceLayout = source.layout
      val destinationLayout = destination.layout
      val sourceOffset = sourceLayout.offset.toLong
      val destinationOffset = destinationLayout.offset.toLong
      val sourceStride0 = sourceLayout.strides(0).toLong
      val sourceStride1 = sourceLayout.strides(1).toLong
      val destinationStride0 = destinationLayout.strides(0).toLong
      val destinationStride1 = destinationLayout.strides(1).toLong
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var acc = reducer.zero
          var offsetIndex = 0
          while offsetIndex < offsetCount do
            val offset = offsets(offsetIndex)
            val mappedFirst = BorderIndex.direct(
              StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
              sourceShape(0),
              border
            )
            val value =
              if mappedFirst < 0 then constant
              else
                val mappedSecond = BorderIndex.direct(
                  StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                  sourceShape(1),
                  border
                )
                if mappedSecond < 0 then constant
                else
                  source.readDouble(
                    (
                      sourceOffset +
                        mappedFirst.toLong * sourceStride0 +
                        mappedSecond.toLong * sourceStride1
                    ).toInt
                  )
            acc = reducer.accumulate(acc, value, offsetIndex)
            offsetIndex += 1
          destination.writeDouble(
            (
              destinationOffset +
                first.toLong * destinationStride0 +
                second.toLong * destinationStride1
            ).toInt,
            reducer.finish(acc)
          )
          second += 1
        first += 1
    else
      val sourceLayout = source.layout
      val destinationLayout = destination.layout
      val sourceOffset = sourceLayout.offset.toLong
      val destinationOffset = destinationLayout.offset.toLong
      val sourceStride0 = sourceLayout.strides(0).toLong
      val sourceStride1 = sourceLayout.strides(1).toLong
      val sourceStride2 = sourceLayout.strides(2).toLong
      val destinationStride0 = destinationLayout.strides(0).toLong
      val destinationStride1 = destinationLayout.strides(1).toLong
      val destinationStride2 = destinationLayout.strides(2).toLong
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var third = 0
          while third < destinationShape(2) do
            var acc = reducer.zero
            var offsetIndex = 0
            while offsetIndex < offsetCount do
              val offset = offsets(offsetIndex)
              val mappedFirst = BorderIndex.direct(
                StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
                sourceShape(0),
                border
              )
              val value =
                if mappedFirst < 0 then constant
                else
                  val mappedSecond = BorderIndex.direct(
                    StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                    sourceShape(1),
                    border
                  )
                  if mappedSecond < 0 then constant
                  else
                    val mappedThird = BorderIndex.direct(
                      StencilArithmetic.logicalCoordinate(outputOrigin(2), third, offset(2)),
                      sourceShape(2),
                      border
                    )
                    if mappedThird < 0 then constant
                    else
                      source.readDouble(
                        (
                          sourceOffset +
                            mappedFirst.toLong * sourceStride0 +
                            mappedSecond.toLong * sourceStride1 +
                            mappedThird.toLong * sourceStride2
                        ).toInt
                      )
              acc = reducer.accumulate(acc, value, offsetIndex)
              offsetIndex += 1
            destination.writeDouble(
              (
                destinationOffset +
                  first.toLong * destinationStride0 +
                  second.toLong * destinationStride1 +
                  third.toLong * destinationStride2
              ).toInt,
              reducer.finish(acc)
            )
            third += 1
          second += 1
        first += 1

  private def runDoubleSpatial[R <: AnyRank](
      source: MutableNDArray[Double, R],
      destination: MutableNDArray[Double, R],
      reducer: DoubleNeighborhoodReducer,
      constant: Double
  ): Unit =
    if rank == 2 then
      val sourceLayout = source.layout
      val destinationLayout = destination.layout
      val sourceOffset = sourceLayout.offset.toLong
      val destinationOffset = destinationLayout.offset.toLong
      val sourceStride0 = sourceLayout.strides(0).toLong
      val sourceStride1 = sourceLayout.strides(1).toLong
      val destinationStride0 = destinationLayout.strides(0).toLong
      val destinationStride1 = destinationLayout.strides(1).toLong
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var acc = reducer.zero
          var offsetIndex = 0
          while offsetIndex < offsetCount do
            val offset = offsets(offsetIndex)
            val mappedFirst = BorderIndex.direct(
              StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
              sourceShape(0),
              border
            )
            val value =
              if mappedFirst < 0 then constant
              else
                val mappedSecond = BorderIndex.direct(
                  StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                  sourceShape(1),
                  border
                )
                if mappedSecond < 0 then constant
                else
                  source.readDouble(
                    (
                      sourceOffset +
                        mappedFirst.toLong * sourceStride0 +
                        mappedSecond.toLong * sourceStride1
                    ).toInt
                  )
            acc = reducer.accumulate(acc, value, offsetIndex)
            offsetIndex += 1
          destination.writeDouble(
            (
              destinationOffset +
                first.toLong * destinationStride0 +
                second.toLong * destinationStride1
            ).toInt,
            reducer.finish(acc)
          )
          second += 1
        first += 1
    else
      val sourceLayout = source.layout
      val destinationLayout = destination.layout
      val sourceOffset = sourceLayout.offset.toLong
      val destinationOffset = destinationLayout.offset.toLong
      val sourceStride0 = sourceLayout.strides(0).toLong
      val sourceStride1 = sourceLayout.strides(1).toLong
      val sourceStride2 = sourceLayout.strides(2).toLong
      val destinationStride0 = destinationLayout.strides(0).toLong
      val destinationStride1 = destinationLayout.strides(1).toLong
      val destinationStride2 = destinationLayout.strides(2).toLong
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var third = 0
          while third < destinationShape(2) do
            var acc = reducer.zero
            var offsetIndex = 0
            while offsetIndex < offsetCount do
              val offset = offsets(offsetIndex)
              val mappedFirst = BorderIndex.direct(
                StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
                sourceShape(0),
                border
              )
              val value =
                if mappedFirst < 0 then constant
                else
                  val mappedSecond = BorderIndex.direct(
                    StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                    sourceShape(1),
                    border
                  )
                  if mappedSecond < 0 then constant
                  else
                    val mappedThird = BorderIndex.direct(
                      StencilArithmetic.logicalCoordinate(outputOrigin(2), third, offset(2)),
                      sourceShape(2),
                      border
                    )
                    if mappedThird < 0 then constant
                    else
                      source.readDouble(
                        (
                          sourceOffset +
                            mappedFirst.toLong * sourceStride0 +
                            mappedSecond.toLong * sourceStride1 +
                            mappedThird.toLong * sourceStride2
                        ).toInt
                      )
              acc = reducer.accumulate(acc, value, offsetIndex)
              offsetIndex += 1
            destination.writeDouble(
              (
                destinationOffset +
                  first.toLong * destinationStride0 +
                  second.toLong * destinationStride1 +
                  third.toLong * destinationStride2
              ).toInt,
              reducer.finish(acc)
            )
            third += 1
          second += 1
        first += 1

  private def runBooleanLines[R <: AnyRank](
      source: NDArray[Boolean, R],
      destination: MutableNDArray[Boolean, R],
      reducer: BooleanNeighborhoodReducer,
      constant: Boolean
  ): Unit =
    val sourceLayout = source.layout
    val destinationLayout = destination.layout
    val sourceStride = sourceLayout.strides(0).toLong
    val destinationStride = destinationLayout.strides(0).toLong
    val outputExtent = destinationShape(0)
    val lineCount = destination.size / outputExtent
    var line = 0
    while line < lineCount do
      setBatchIndices(line)
      val sourceBase = batchAddress(sourceLayout)
      val destinationBase = batchAddress(destinationLayout)
      var row = 0
      while row < outputExtent do
        var acc = reducer.zero
        var offsetIndex = 0
        while offsetIndex < offsetCount do
          val offset = offsets(offsetIndex)
          val mapped = BorderIndex.direct(
            StencilArithmetic.logicalCoordinate(outputOrigin(0), row, offset(0)),
            sourceShape(0),
            border
          )
          val value =
            if mapped < 0 then constant
            else source.readBoolean((sourceBase.toLong + mapped * sourceStride).toInt)
          acc = reducer.accumulate(acc, value, offsetIndex)
          if reducer.isTerminal(acc) then offsetIndex = offsetCount
          else offsetIndex += 1
        destination.writeBoolean(
          (destinationBase.toLong + row * destinationStride).toInt,
          reducer.finish(acc)
        )
        row += 1
      line += 1

  private def runBooleanSpatial[R <: AnyRank](
      source: NDArray[Boolean, R],
      destination: MutableNDArray[Boolean, R],
      reducer: BooleanNeighborhoodReducer,
      constant: Boolean
  ): Unit =
    val sourceLayout = source.layout
    val destinationLayout = destination.layout
    val sourceOffset = sourceLayout.offset.toLong
    val destinationOffset = destinationLayout.offset.toLong
    val sourceStride0 = sourceLayout.strides(0).toLong
    val sourceStride1 = sourceLayout.strides(1).toLong
    val destinationStride0 = destinationLayout.strides(0).toLong
    val destinationStride1 = destinationLayout.strides(1).toLong
    if rank == 2 then
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var acc = reducer.zero
          var offsetIndex = 0
          while offsetIndex < offsetCount do
            val offset = offsets(offsetIndex)
            val mappedFirst = BorderIndex.direct(
              StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
              sourceShape(0),
              border
            )
            val value =
              if mappedFirst < 0 then constant
              else
                val mappedSecond = BorderIndex.direct(
                  StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                  sourceShape(1),
                  border
                )
                if mappedSecond < 0 then constant
                else
                  source.readBoolean(
                    (
                      sourceOffset +
                        mappedFirst.toLong * sourceStride0 +
                        mappedSecond.toLong * sourceStride1
                    ).toInt
                  )
            acc = reducer.accumulate(acc, value, offsetIndex)
            if reducer.isTerminal(acc) then offsetIndex = offsetCount
            else offsetIndex += 1
          destination.writeBoolean(
            (
              destinationOffset +
                first.toLong * destinationStride0 +
                second.toLong * destinationStride1
            ).toInt,
            reducer.finish(acc)
          )
          second += 1
        first += 1
    else
      val sourceStride2 = sourceLayout.strides(2).toLong
      val destinationStride2 = destinationLayout.strides(2).toLong
      var first = 0
      while first < destinationShape(0) do
        var second = 0
        while second < destinationShape(1) do
          var third = 0
          while third < destinationShape(2) do
            var acc = reducer.zero
            var offsetIndex = 0
            while offsetIndex < offsetCount do
              val offset = offsets(offsetIndex)
              val mappedFirst = BorderIndex.direct(
                StencilArithmetic.logicalCoordinate(outputOrigin(0), first, offset(0)),
                sourceShape(0),
                border
              )
              val value =
                if mappedFirst < 0 then constant
                else
                  val mappedSecond = BorderIndex.direct(
                    StencilArithmetic.logicalCoordinate(outputOrigin(1), second, offset(1)),
                    sourceShape(1),
                    border
                  )
                  if mappedSecond < 0 then constant
                  else
                    val mappedThird = BorderIndex.direct(
                      StencilArithmetic.logicalCoordinate(outputOrigin(2), third, offset(2)),
                      sourceShape(2),
                      border
                    )
                    if mappedThird < 0 then constant
                    else
                      source.readBoolean(
                        (
                          sourceOffset +
                            mappedFirst.toLong * sourceStride0 +
                            mappedSecond.toLong * sourceStride1 +
                            mappedThird.toLong * sourceStride2
                        ).toInt
                      )
              acc = reducer.accumulate(acc, value, offsetIndex)
              if reducer.isTerminal(acc) then offsetIndex = offsetCount
              else offsetIndex += 1
            destination.writeBoolean(
              (
                destinationOffset +
                  first.toLong * destinationStride0 +
                  second.toLong * destinationStride1 +
                  third.toLong * destinationStride2
              ).toInt,
              reducer.finish(acc)
            )
            third += 1
          second += 1
        first += 1

  private def setBatchIndices(linear: Int): Unit =
    var rem = linear
    var axis = rank - 1
    while axis >= 1 do
      val extent = destinationShape(axis)
      destinationIndices(axis) = rem % extent
      rem /= extent
      axis -= 1
    destinationIndices(0) = 0

  private def batchAddress(layout: Layout): Int =
    var address = layout.offset.toLong
    var axis = 1
    while axis < rank do
      address += destinationIndices(axis).toLong * layout.strides(axis).toLong
      axis += 1
    address.toInt

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
      offset: Array[Int]
  ): Int =
    var address = sourceLayout.offset.toLong
    var axis = 0
    while axis < spec.spatialAxes do
      val logical =
        StencilArithmetic.logicalCoordinate(
          outputOrigin(axis),
          destinationIndices(axis),
          offset(axis)
        )
      val mapped =
        BorderIndex.direct(logical, sourceShape(axis), border)
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
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
    prepareShapes(source.rank, source.shape, destination, spec)

  /** Validate a mutable-to-mutable direct pass and allocate reusable scheduling workspace. */
  def prepare[A, B, R <: AnyRank](
      source: MutableNDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
    if source.storage eq destination.storage then
      throw IllegalArgumentException(
        "mutable source and destination must not share storage; use distinct ping-pong workspaces"
      )
    prepareShapes(source.rank, source.shape, destination, spec)

  private def prepareShapes[B, R <: AnyRank](
      sourceRank: Int,
      sourceShapeValue: Shape[R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec
  ): PreparedDirectNeighborhoodExecutor =
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
    StencilArithmetic.requirePositiveSpatialExtents(spec.spatialAxes, sourceShapeValue(_))
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
      new Array[Int](rank),
      spec.offsets.map(_.toArray).toArray,
      spec.outputOrigin.toArray
    )
