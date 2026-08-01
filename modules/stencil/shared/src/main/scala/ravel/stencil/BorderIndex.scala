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
    if extent <= 0 then throw IllegalArgumentException(s"extent must be positive, got $extent")
    if index >= 0 && index < extent then MappedIndex.Inside(index)
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
      index: Int,
      extent: Int,
      mode: BorderMode
  ): Int =
    if extent <= 0 then throw IllegalArgumentException(s"extent must be positive, got $extent")
    if index >= 0 && index < extent then index
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

  private def clamp(index: Int, extent: Int): Int =
    if index < 0 then 0
    else if index >= extent then extent - 1
    else index

  private def wrap(index: Int, extent: Int): Int =
    val mod = index % extent
    if mod < 0 then mod + extent else mod

  /** Reflect without duplicating the edge sample (OpenCV REFLECT_101 / NumPy reflect). */
  private def reflectWithoutEdge(index: Int, extent: Int): Int =
    if extent == 1 then 0
    else
      val period = 2 * (extent - 1)
      var x = index % period
      if x < 0 then x += period
      if x >= extent then period - x else x

  /** Reflect while duplicating the edge sample (OpenCV REFLECT / NumPy symmetric). */
  private def reflectWithEdge(index: Int, extent: Int): Int =
    val period = 2 * extent
    var x = index % period
    if x < 0 then x += period
    if x >= extent then period - 1 - x else x
