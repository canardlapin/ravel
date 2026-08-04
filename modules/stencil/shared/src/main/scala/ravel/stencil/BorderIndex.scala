package ravel.stencil

/** Pure border-index mapping for one axis.
  *
  * Knows nothing about image roles, physical units, or dtypes. Callers supply a constant fill when
  * [[MappedIndex.Outside]] is returned.
  */
object BorderIndex:
  def map(
      index: Int,
      extent: Int,
      mode: BorderMode
  ): MappedIndex =
    map(index.toLong, extent, mode)

  def map(
      index: Long,
      extent: Int,
      mode: BorderMode
  ): MappedIndex =
    if extent <= 0 then throw IllegalArgumentException(s"extent must be positive, got $extent")
    if index >= 0L && index < extent.toLong then MappedIndex.Inside(index.toInt)
    else
      mode match
        case BorderMode.Constant =>
          MappedIndex.Outside
        case BorderMode.Replicate =>
          MappedIndex.Inside(clamp(index, extent))
        case BorderMode.Wrap =>
          MappedIndex.Inside(wrap(index, extent))
        case BorderMode.ReflectWithoutEdge =>
          MappedIndex.Inside(reflectWithoutEdge(index, extent))
        case BorderMode.ReflectWithEdge =>
          MappedIndex.Inside(reflectWithEdge(index, extent))

  /** Convenience when the mode cannot produce Outside. */
  def mapInside(
      index: Int,
      extent: Int,
      mode: BorderMode
  ): Int =
    mapInside(index.toLong, extent, mode)

  def mapInside(
      index: Long,
      extent: Int,
      mode: BorderMode
  ): Int =
    map(index, extent, mode) match
      case MappedIndex.Inside(mapped) => mapped
      case MappedIndex.Outside =>
        throw IllegalArgumentException(
          s"border mode $mode produced Outside for index=$index extent=$extent"
        )

  /** Allocation-free mapping for optimized executors.
    *
    * Returns `-1` only when `mode` is Constant and the requested index falls outside the domain.
    * All valid mapped indices are non-negative.
    */
  private[stencil] def direct(
      index: Long,
      extent: Int,
      mode: BorderMode
  ): Int =
    if extent <= 0 then throw IllegalArgumentException(s"extent must be positive, got $extent")
    if index >= 0L && index < extent.toLong then index.toInt
    else
      mode match
        case BorderMode.Constant =>
          -1
        case BorderMode.Replicate =>
          clamp(index, extent)
        case BorderMode.Wrap =>
          wrap(index, extent)
        case BorderMode.ReflectWithoutEdge =>
          reflectWithoutEdge(index, extent)
        case BorderMode.ReflectWithEdge =>
          reflectWithEdge(index, extent)

  private def clamp(index: Long, extent: Int): Int =
    if index < 0L then 0
    else if index >= extent.toLong then extent - 1
    else index.toInt

  private def wrap(index: Long, extent: Int): Int =
    val mod = index % extent.toLong
    val normalized = if mod < 0L then mod + extent.toLong else mod
    normalized.toInt

  /** Reflect without duplicating the edge sample (OpenCV REFLECT_101 / NumPy reflect). */
  private def reflectWithoutEdge(index: Long, extent: Int): Int =
    if extent == 1 then 0
    else
      val period = 2L * (extent.toLong - 1L)
      var x = index % period
      if x < 0L then x += period
      val mapped = if x >= extent.toLong then period - x else x
      mapped.toInt

  /** Reflect while duplicating the edge sample (OpenCV REFLECT / NumPy symmetric). */
  private def reflectWithEdge(index: Long, extent: Int): Int =
    val period = 2L * extent.toLong
    var x = index % period
    if x < 0L then x += period
    val mapped = if x >= extent.toLong then period - 1L - x else x
    mapped.toInt

private[stencil] object StencilArithmetic:
  def logicalCoordinate(origin: Int, destination: Int, offset: Int): Long =
    checkedAdd(checkedAdd(origin.toLong, destination.toLong), offset.toLong)

  def requirePositiveSpatialExtents(
      spatialAxes: Int,
      extentAt: Int => Int
  ): Unit =
    var axis = 0
    while axis < spatialAxes do
      val extent = extentAt(axis)
      if extent <= 0 then
        throw IllegalArgumentException(
          s"source spatial extent on axis $axis must be positive, got $extent"
        )
      axis += 1

  private def checkedAdd(left: Long, right: Long): Long =
    val result = left + right
    if ((left ^ result) & (right ^ result)) < 0L then
      throw ArithmeticException(s"stencil coordinate overflow: $left + $right")
    result
