package ravel

import scala.annotation.implicitNotFound

/** Closed primitive storage family supported by Ravel 1.0. */
@implicitNotFound(
  "No DType evidence for ${A}. Use Boolean, Byte, UInt8, Short, UInt16, Int, Long, Float, or Double."
)
sealed trait DType[A]:
  private[ravel] def tag: Byte
  def name: String
  def zero: A
  def format(value: A): String

/** Primitive numeric dtypes that can participate in explicit casts. */
@implicitNotFound(
  "${A} is not a numeric dtype. Use Byte, UInt8, Short, UInt16, Int, Long, Float, or Double."
)
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
  private[ravel] val UInt8Tag: Byte = 7
  private[ravel] val UInt16Tag: Byte = 8

  given booleanDType: DType[Boolean] with
    private[ravel] val tag = BooleanTag
    val name = "Boolean"
    val zero = false
    def format(value: Boolean): String = value.toString

  given byteDType: IntegralDType[Byte] with
    private[ravel] val tag = ByteTag
    val name = "Byte"
    val zero: Byte = 0
    val one: Byte = 1
    def format(value: Byte): String = value.toString

  given shortDType: IntegralDType[Short] with
    private[ravel] val tag = ShortTag
    val name = "Short"
    val zero: Short = 0
    val one: Short = 1
    def format(value: Short): String = value.toString

  given uint8DType: IntegralDType[UInt8] with
    private[ravel] val tag = UInt8Tag
    val name = "UInt8"
    val zero: UInt8 = UInt8.MinValue
    val one: UInt8 = UInt8.unsafe(1)
    def format(value: UInt8): String = value.toInt.toString

  given uint16DType: IntegralDType[UInt16] with
    private[ravel] val tag = UInt16Tag
    val name = "UInt16"
    val zero: UInt16 = UInt16.MinValue
    val one: UInt16 = UInt16.unsafe(1)
    def format(value: UInt16): String = value.toInt.toString

  given intDType: IntegralArithmeticDType[Int] with
    private[ravel] val tag = IntTag
    val name = "Int"
    val zero = 0
    val one = 1
    def format(value: Int): String = value.toString

  given longDType: IntegralArithmeticDType[Long] with
    private[ravel] val tag = LongTag
    val name = "Long"
    val zero = 0L
    val one = 1L
    def format(value: Long): String = value.toString

  given floatDType: FloatingDType[Float] with
    private[ravel] val tag = FloatTag
    val name = "Float"
    val zero = 0.0f
    val one = 1.0f
    def format(value: Float): String =
      if value == 0.0f then if 1.0f / value == Float.NegativeInfinity then "-0.0" else "0.0"
      else if value.isNaN then "NaN"
      else if value == Float.PositiveInfinity then "Infinity"
      else if value == Float.NegativeInfinity then "-Infinity"
      else java.lang.Float.toString(value)

  given doubleDType: FloatingDType[Double] with
    private[ravel] val tag = DoubleTag
    val name = "Double"
    val zero = 0.0
    val one = 1.0
    def format(value: Double): String =
      if value == 0.0 then if 1.0 / value == Double.NegativeInfinity then "-0.0" else "0.0"
      else if value.isNaN then "NaN"
      else if value == Double.PositiveInfinity then "Infinity"
      else if value == Double.NegativeInfinity then "-Infinity"
      else java.lang.Double.toString(value)

  private[ravel] def castScalar[A, B](
      value: A,
      source: NumericDType[A],
      target: NumericDType[B]
  ): B =
    val result: Any =
      source.tag match
        case ByteTag => castLong(value.asInstanceOf[Byte].toLong, target)
        case UInt8Tag => castLong(value.asInstanceOf[UInt8].toLong, target)
        case ShortTag => castLong(value.asInstanceOf[Short].toLong, target)
        case UInt16Tag => castLong(value.asInstanceOf[UInt16].toLong, target)
        case IntTag => castLong(value.asInstanceOf[Int].toLong, target)
        case LongTag => castLong(value.asInstanceOf[Long], target)
        case FloatTag => castDouble(value.asInstanceOf[Float].toDouble, target)
        case DoubleTag => castDouble(value.asInstanceOf[Double], target)
        case tag => throw new MatchError(tag)
    result.asInstanceOf[B]

  private def castLong[B](value: Long, target: NumericDType[B]): Any =
    target.tag match
      case ByteTag => value.toByte
      case UInt8Tag => UInt8.fromRawBits(value.toByte)
      case ShortTag => value.toShort
      case UInt16Tag => UInt16.fromRawBits(value.toShort)
      case IntTag => value.toInt
      case LongTag => value
      case FloatTag => value.toFloat
      case DoubleTag => value.toDouble
      case tag => throw new MatchError(tag)

  private def castDouble[B](value: Double, target: NumericDType[B]): Any =
    target.tag match
      case ByteTag => value.toInt.toByte
      case UInt8Tag => UInt8.fromRawBits(value.toInt.toByte)
      case ShortTag => value.toInt.toShort
      case UInt16Tag => UInt16.fromRawBits(value.toInt.toShort)
      case IntTag => value.toInt
      case LongTag => value.toLong
      case FloatTag => value.toFloat
      case DoubleTag => value
      case tag => throw new MatchError(tag)
