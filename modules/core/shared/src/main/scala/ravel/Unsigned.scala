package ravel

/** Checked construction failure for an unsigned primitive value. */
final case class UnsignedRangeError(
    dtype: String,
    value: Long,
    maximum: Long
) derives CanEqual:
  def message: String =
    s"$value is outside $dtype range [0, $maximum]"

/** Unsigned 8-bit integer with primitive Byte representation.
  *
  * Arithmetic is intentionally unavailable. Cast explicitly to `Int` or
  * `Long` before arithmetic.
  */
opaque type UInt8 = Byte

object UInt8:
  val MinValue: UInt8 = 0.toByte
  val MaxValue: UInt8 = (-1).toByte

  def fromInt(value: Int): Either[UnsignedRangeError, UInt8] =
    if value >= 0 && value <= 255 then Right(value.toByte)
    else Left(UnsignedRangeError("UInt8", value.toLong, 255L))

  def fromLong(value: Long): Either[UnsignedRangeError, UInt8] =
    if value >= 0L && value <= 255L then Right(value.toByte)
    else Left(UnsignedRangeError("UInt8", value, 255L))

  def unsafe(value: Int): UInt8 =
    fromInt(value).fold(error => throw new IllegalArgumentException(error.message), identity)

  private[ravel] def fromRawBits(value: Byte): UInt8 =
    value

  extension (value: UInt8)
    inline def toInt: Int =
      value & 0xff

    inline def toLong: Long =
      toInt.toLong

    private[ravel] inline def rawBits: Byte =
      value

/** Unsigned 16-bit integer with primitive Short representation.
  *
  * Arithmetic is intentionally unavailable. Cast explicitly to `Int` or
  * `Long` before arithmetic.
  */
opaque type UInt16 = Short

object UInt16:
  val MinValue: UInt16 = 0.toShort
  val MaxValue: UInt16 = (-1).toShort

  def fromInt(value: Int): Either[UnsignedRangeError, UInt16] =
    if value >= 0 && value <= 65535 then Right(value.toShort)
    else Left(UnsignedRangeError("UInt16", value.toLong, 65535L))

  def fromLong(value: Long): Either[UnsignedRangeError, UInt16] =
    if value >= 0L && value <= 65535L then Right(value.toShort)
    else Left(UnsignedRangeError("UInt16", value, 65535L))

  def unsafe(value: Int): UInt16 =
    fromInt(value).fold(error => throw new IllegalArgumentException(error.message), identity)

  private[ravel] def fromRawBits(value: Short): UInt16 =
    value

  extension (value: UInt16)
    inline def toInt: Int =
      value & 0xffff

    inline def toLong: Long =
      toInt.toLong

    private[ravel] inline def rawBits: Short =
      value
