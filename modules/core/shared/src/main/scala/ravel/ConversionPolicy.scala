package ravel

/** Rounding applied before converting floating-point values to integral dtypes. */
enum Rounding derives CanEqual:
  case TowardZero
  case NearestEven
  case Floor
  case Ceiling

/** Handling of values outside the target dtype's finite numeric range. */
enum Overflow derives CanEqual:
  /** Reject the complete conversion before allocating an output array. */
  case Reject

  /** Clamp values to the nearest representable target value. */
  case Clamp

  /** Apply Scala/JVM primitive conversion semantics. Intended for explicit low-level interop, never
    * as an image-processing default.
    */
  case Wrap

/** Explicit policy for numeric dtype conversion.
  *
  * The default is deliberate: nearest-even rounding and checked overflow.
  */
final case class ConversionPolicy(
    rounding: Rounding = Rounding.NearestEven,
    overflow: Overflow = Overflow.Reject
) derives CanEqual

sealed trait ConversionError derives CanEqual:
  def message: String

object ConversionError:
  final case class NonFiniteToIntegral(
      logicalIndex: Int,
      source: String,
      target: String
  ) extends ConversionError:
    def message: String =
      s"cannot convert non-finite $source value at logical index $logicalIndex " +
        s"to integral target $target with Overflow.Reject"

  final case class OutOfRange(
      logicalIndex: Int,
      source: String,
      target: String
  ) extends ConversionError:
    def message: String =
      s"$source value at logical index $logicalIndex is outside target $target " +
        s"with Overflow.Reject"
