package ravel

/** Axis slice. Open endpoints ([[Slice.all]], [[Slice.from]], …) are normalized against the axis
  * length when applied. Fully specified slices still accept negative indices; for negative steps,
  * stop `-1` means “before the first element”, matching the historical Ravel/Python exclusive-stop
  * convention. `NDArray.narrow` remains the strict exact-bounds operation.
  */
final class Slice private (
    private[ravel] val startRaw: Int,
    private[ravel] val stopRaw: Int,
    val step: Int,
    private[ravel] val startOpen: Boolean,
    private[ravel] val stopOpen: Boolean
):
  def start: Int = startRaw
  def stopExclusive: Int = stopRaw

  private[ravel] def normalize(dimension: Int): Slice =
    if step == 0 then throw InvalidSlice("step must not be zero")
    if dimension < 0 then throw InvalidSlice(s"negative dimension $dimension")
    if startOpen || stopOpen then normalizeOpen(dimension)
    else normalizeResolved(dimension)

  private def normalizeOpen(dimension: Int): Slice =
    val normalizedStart =
      if startOpen then if step > 0 then 0 else math.max(dimension - 1, -1)
      else resolveStart(startRaw, dimension)
    val normalizedStop =
      if stopOpen then if step > 0 then dimension else -1
      else resolveStop(stopRaw, dimension)
    new Slice(normalizedStart, normalizedStop, step, false, false)

  private def normalizeResolved(dimension: Int): Slice =
    if step < 0 && stopRaw < -1 then
      throw InvalidSlice(
        s"negative-step stop $stopRaw is outside [-1, $dimension)"
      )
    val normalizedStart = resolveStart(startRaw, dimension)
    val normalizedStop = resolveStop(stopRaw, dimension)
    if step > 0 then
      if normalizedStart < 0 || normalizedStart > dimension then
        throw InvalidSlice(
          s"positive-step start $startRaw is outside [0, $dimension]"
        )
      if normalizedStop < 0 || normalizedStop > dimension then
        throw InvalidSlice(
          s"positive-step stop $stopRaw is outside [0, $dimension]"
        )
    else
      if normalizedStart < 0 || normalizedStart >= dimension then
        throw InvalidSlice(
          s"negative-step start $startRaw is outside [0, $dimension)"
        )
      if normalizedStop < -1 || normalizedStop >= dimension then
        throw InvalidSlice(
          s"negative-step stop $stopRaw is outside [-1, $dimension)"
        )
    new Slice(normalizedStart, normalizedStop, step, false, false)

  private def resolveStart(index: Int, dimension: Int): Int =
    if index < 0 then index + dimension else index

  private def resolveStop(index: Int, dimension: Int): Int =
    if step < 0 && index == -1 then -1
    else if index < 0 then index + dimension
    else index

object Slice:
  def apply(start: Int, stopExclusive: Int, step: Int = 1): Slice =
    from(start, stopExclusive, step).fold(throw _, identity)

  def from(
      start: Int,
      stopExclusive: Int,
      step: Int = 1
  ): Either[InvalidSlice, Slice] =
    if step == 0 then Left(InvalidSlice("step must not be zero"))
    else Right(new Slice(start, stopExclusive, step, false, false))

  def from(range: Range): Either[InvalidSlice, Slice] =
    val stop =
      if range.isInclusive then
        val shifted = range.end.toLong + range.step.sign.toLong
        if shifted < Int.MinValue.toLong || shifted > Int.MaxValue.toLong then
          return Left(InvalidSlice("inclusive range endpoint cannot be represented exclusively"))
        shifted.toInt
      else range.end
    from(range.start, stop, range.step)

  /** Whole axis with step 1. */
  def all: Slice = new Slice(0, 0, 1, true, true)

  /** From `start` through the end of the axis. */
  def from(start: Int): Slice = new Slice(start, 0, 1, false, true)

  /** From the beginning until exclusive `stopExclusive`. */
  def until(stopExclusive: Int): Slice = new Slice(0, stopExclusive, 1, true, false)

  /** Half-open interval `[start, stopExclusive)`. */
  def between(start: Int, stopExclusive: Int): Slice = apply(start, stopExclusive, 1)

  /** Whole axis with the given step. */
  def every(step: Int): Slice =
    if step == 0 then throw InvalidSlice("step must not be zero")
    new Slice(0, 0, step, true, true)

  /** Whole axis reversed. */
  def reverse: Slice = new Slice(0, 0, -1, true, true)
