package ravel

import scala.annotation.implicitNotFound

/** Closed primitive storage family supported by Ravel 1.0. */
@implicitNotFound(
  "No DType evidence for ${A}. Use Boolean, Byte, Short, Int, Long, Float, or Double."
)
sealed trait DType[A]:
  private[ravel] def tag: Byte
  def name: String
  def zero: A

/** Primitive numeric dtypes that can participate in explicit casts. */
@implicitNotFound("${A} is not a numeric dtype. Use Byte, Short, Int, Long, Float, or Double.")
sealed trait NumericDType[A] extends DType[A]:
  def one: A

/** Dtypes with ordering, min/max, and comparisons. */
@implicitNotFound("${A} arrays do not support ordered comparisons. Use a numeric dtype.")
sealed trait OrderedDType[A] extends DType[A]

/** Dtypes with pointwise arithmetic and sum/product kernels. */
@implicitNotFound(
  "${A} arrays do not support arithmetic. Use Int, Long, Float, or Double, or cast explicitly."
)
sealed trait ArithmeticDType[A] extends NumericDType[A]

/** Integral numeric dtypes. */
@implicitNotFound("${A} is not an integral dtype.")
sealed trait IntegralDType[A] extends NumericDType[A] with OrderedDType[A]

/** Integral arithmetic dtypes. Byte and Short deliberately do not extend it. */
@implicitNotFound(
  "${A} arrays do not support integral arithmetic. Use Int or Long, or cast explicitly."
)
sealed trait IntegralArithmeticDType[A] extends IntegralDType[A] with ArithmeticDType[A]

/** Floating arithmetic dtypes. */
@implicitNotFound("${A} arrays do not support floating arithmetic. Use Float or Double.")
sealed trait FloatingDType[A] extends NumericDType[A] with OrderedDType[A] with ArithmeticDType[A]

object DType:
  private[ravel] val BooleanTag: Byte = 0
  private[ravel] val ByteTag: Byte = 1
  private[ravel] val ShortTag: Byte = 2
  private[ravel] val IntTag: Byte = 3
  private[ravel] val LongTag: Byte = 4
  private[ravel] val FloatTag: Byte = 5
  private[ravel] val DoubleTag: Byte = 6

  given booleanDType: DType[Boolean] with
    private[ravel] val tag = BooleanTag
    val name = "Boolean"
    val zero = false

  given byteDType: IntegralDType[Byte] with
    private[ravel] val tag = ByteTag
    val name = "Byte"
    val zero: Byte = 0
    val one: Byte = 1

  given shortDType: IntegralDType[Short] with
    private[ravel] val tag = ShortTag
    val name = "Short"
    val zero: Short = 0
    val one: Short = 1

  given intDType: IntegralArithmeticDType[Int] with
    private[ravel] val tag = IntTag
    val name = "Int"
    val zero = 0
    val one = 1

  given longDType: IntegralArithmeticDType[Long] with
    private[ravel] val tag = LongTag
    val name = "Long"
    val zero = 0L
    val one = 1L

  given floatDType: FloatingDType[Float] with
    private[ravel] val tag = FloatTag
    val name = "Float"
    val zero = 0.0f
    val one = 1.0f

  given doubleDType: FloatingDType[Double] with
    private[ravel] val tag = DoubleTag
    val name = "Double"
    val zero = 0.0
    val one = 1.0

  private[ravel] def castScalar[A, B](
      value: A,
      source: NumericDType[A],
      target: NumericDType[B]
  ): B =
    val result: Any =
      source.tag match
        case ByteTag => castLong(value.asInstanceOf[Byte].toLong, target)
        case ShortTag => castLong(value.asInstanceOf[Short].toLong, target)
        case IntTag => castLong(value.asInstanceOf[Int].toLong, target)
        case LongTag => castLong(value.asInstanceOf[Long], target)
        case FloatTag => castDouble(value.asInstanceOf[Float].toDouble, target)
        case DoubleTag => castDouble(value.asInstanceOf[Double], target)
        case tag => throw new MatchError(tag)
    result.asInstanceOf[B]

  private def castLong[B](value: Long, target: NumericDType[B]): Any =
    target.tag match
      case ByteTag => value.toByte
      case ShortTag => value.toShort
      case IntTag => value.toInt
      case LongTag => value
      case FloatTag => value.toFloat
      case DoubleTag => value.toDouble
      case tag => throw new MatchError(tag)

  private def castDouble[B](value: Double, target: NumericDType[B]): Any =
    target.tag match
      case ByteTag => value.toInt.toByte
      case ShortTag => value.toInt.toShort
      case IntTag => value.toInt
      case LongTag => value.toLong
      case FloatTag => value.toFloat
      case DoubleTag => value
      case tag => throw new MatchError(tag)
