package ravel.internal

import ravel.*

/** Policy-aware numeric casts.
  *
  * The implementation dispatches once on the sealed storage family, validates `Overflow.Reject`
  * before allocating output, and keeps primitive reads and writes inside the conversion loop. It
  * deliberately does not route through `Iterator`, `Vector`, or `Any`-typed per-element values.
  */
private[ravel] object PolicyCastKernels:
  def convert[A, B](
      source: Storage[A],
      layout: Layout,
      sourceType: NumericDType[A],
      targetType: NumericDType[B],
      policy: ConversionPolicy
  ): Either[ConversionError, Storage[B]] =
    validate(source, layout, sourceType, targetType, policy).map { _ =>
      val target = ProbeApi.allocate[B](layout.size)(using targetType)
      convertInto(source, layout, target, sourceType, targetType, policy)
      target
    }

  private def validate[A, B](
      source: Storage[A],
      layout: Layout,
      sourceType: NumericDType[A],
      targetType: NumericDType[B],
      policy: ConversionPolicy
  ): Either[ConversionError, Unit] =
    if policy.overflow != Overflow.Reject then Right(())
    else
      source match
        case values: ByteStorage =>
          validateIntegral(
            layout,
            values.raw.apply,
            sourceType.name,
            targetType
          )
        case values: ShortStorage =>
          validateIntegral(
            layout,
            index => values.raw(index).toLong,
            sourceType.name,
            targetType
          )
        case values: IntStorage =>
          validateIntegral(
            layout,
            index => values.raw(index).toLong,
            sourceType.name,
            targetType
          )
        case values: LongStorage =>
          validateIntegral(
            layout,
            values.raw.apply,
            sourceType.name,
            targetType
          )
        case values: FloatStorage =>
          validateFloating(
            layout,
            index => values.raw(index).toDouble,
            sourceType.name,
            targetType,
            policy.rounding
          )
        case values: DoubleStorage =>
          validateFloating(
            layout,
            values.raw.apply,
            sourceType.name,
            targetType,
            policy.rounding
          )
        case _: BooleanStorage =>
          throw new MatchError(source)

  private def convertInto[A, B](
      source: Storage[A],
      layout: Layout,
      target: Storage[B],
      sourceType: NumericDType[A],
      targetType: NumericDType[B],
      policy: ConversionPolicy
  ): Unit =
    source match
      case values: ByteStorage
          if policy.overflow == Overflow.Clamp &&
            isIntegral(targetType) =>
        convertIntegralClamp(layout, index => values.raw(index).toLong, target)
      case values: ShortStorage
          if policy.overflow == Overflow.Clamp &&
            isIntegral(targetType) =>
        convertIntegralClamp(layout, index => values.raw(index).toLong, target)
      case values: IntStorage
          if policy.overflow == Overflow.Clamp &&
            isIntegral(targetType) =>
        convertIntegralClamp(layout, index => values.raw(index).toLong, target)
      case values: LongStorage
          if policy.overflow == Overflow.Clamp &&
            isIntegral(targetType) =>
        convertIntegralClamp(layout, values.raw.apply, target)
      case values: FloatStorage if isIntegral(targetType) =>
        convertFloatingToIntegral(
          layout,
          index => values.raw(index).toDouble,
          target,
          policy
        )
      case values: DoubleStorage if isIntegral(targetType) =>
        convertFloatingToIntegral(layout, values.raw.apply, target, policy)
      case values: DoubleStorage
          if targetType.tag == DType.FloatTag &&
            policy.overflow == Overflow.Clamp =>
        target match
          case output: FloatStorage =>
            castLogical(layout, values.raw.apply, output.raw.update)(
              clampDoubleToFloat
            )
          case _ =>
            throw new MatchError(target)
      case _ =>
        CastKernels.convert(source, layout, target, sourceType, targetType)

  private inline def validateIntegral[B](
      layout: Layout,
      inline read: Int => Long,
      sourceName: String,
      targetType: NumericDType[B]
  ): Either[ConversionError, Unit] =
    if !isIntegral(targetType) then Right(())
    else
      var logical = 0
      var error: Option[ConversionError] = None
      foreachPhysical(layout) { address =>
        if error.isEmpty && !fitsIntegral(read(address), targetType) then
          error = Some(
            ConversionError.OutOfRange(logical, sourceName, targetType.name)
          )
        logical += 1
      }
      error.toLeft(())

  private inline def validateFloating[B](
      layout: Layout,
      inline read: Int => Double,
      sourceName: String,
      targetType: NumericDType[B],
      rounding: Rounding
  ): Either[ConversionError, Unit] =
    var logical = 0
    var error: Option[ConversionError] = None
    foreachPhysical(layout) { address =>
      if error.isEmpty then
        val value = read(address)
        if isIntegral(targetType) then
          if !value.isFinite then
            error = Some(
              ConversionError.NonFiniteToIntegral(
                logical,
                sourceName,
                targetType.name
              )
            )
          else if !fitsIntegral(rounded(value, rounding), targetType) then
            error = Some(
              ConversionError.OutOfRange(
                logical,
                sourceName,
                targetType.name
              )
            )
        else if targetType.tag == DType.FloatTag &&
          value.isFinite &&
          (value > Float.MaxValue || value < -Float.MaxValue)
        then
          error = Some(
            ConversionError.OutOfRange(logical, sourceName, targetType.name)
          )
      logical += 1
    }
    error.toLeft(())

  private inline def convertIntegralClamp[B](
      layout: Layout,
      inline read: Int => Long,
      target: Storage[B]
  ): Unit =
    target match
      case output: ByteStorage =>
        castLogical(layout, read, output.raw.update)(clampByte)
      case output: ShortStorage =>
        castLogical(layout, read, output.raw.update)(clampShort)
      case output: IntStorage =>
        castLogical(layout, read, output.raw.update)(clampInt)
      case output: LongStorage =>
        castLogical(layout, read, output.raw.update)(identity)
      case _ =>
        throw new MatchError(target)

  private inline def convertFloatingToIntegral[B](
      layout: Layout,
      inline read: Int => Double,
      target: Storage[B],
      policy: ConversionPolicy
  ): Unit =
    target match
      case output: ByteStorage =>
        castLogical(layout, read, output.raw.update)(value => floatingToByte(value, policy))
      case output: ShortStorage =>
        castLogical(layout, read, output.raw.update)(value => floatingToShort(value, policy))
      case output: IntStorage =>
        castLogical(layout, read, output.raw.update)(value => floatingToInt(value, policy))
      case output: LongStorage =>
        castLogical(layout, read, output.raw.update)(value => floatingToLong(value, policy))
      case _ =>
        throw new MatchError(target)

  private def floatingToByte(value: Double, policy: ConversionPolicy): Byte =
    policy.overflow match
      case Overflow.Clamp =>
        clampRounded(value, Byte.MinValue, Byte.MaxValue, policy.rounding).toByte
      case _ =>
        rounded(value, policy.rounding).toInt.toByte

  private def floatingToShort(value: Double, policy: ConversionPolicy): Short =
    policy.overflow match
      case Overflow.Clamp =>
        clampRounded(value, Short.MinValue, Short.MaxValue, policy.rounding).toShort
      case _ =>
        rounded(value, policy.rounding).toInt.toShort

  private def floatingToInt(value: Double, policy: ConversionPolicy): Int =
    policy.overflow match
      case Overflow.Clamp =>
        clampRounded(value, Int.MinValue, Int.MaxValue, policy.rounding).toInt
      case _ =>
        rounded(value, policy.rounding).toInt

  private def floatingToLong(value: Double, policy: ConversionPolicy): Long =
    policy.overflow match
      case Overflow.Clamp =>
        clampRounded(value, Long.MinValue, Long.MaxValue, policy.rounding)
      case _ =>
        rounded(value, policy.rounding).toLong

  private def clampRounded(
      value: Double,
      min: Long,
      max: Long,
      rounding: Rounding
  ): Long =
    if value.isNaN then 0L
    else
      val roundedValue = rounded(value, rounding)
      if roundedValue <= min.toDouble then min
      else if roundedValue >= max.toDouble then max
      else roundedValue.toLong

  private def clampDoubleToFloat(value: Double): Float =
    if value.isNaN then Float.NaN
    else if value >= Float.MaxValue then Float.MaxValue
    else if value <= -Float.MaxValue then -Float.MaxValue
    else value.toFloat

  private def clampByte(value: Long): Byte =
    value.max(Byte.MinValue).min(Byte.MaxValue).toByte

  private def clampShort(value: Long): Short =
    value.max(Short.MinValue).min(Short.MaxValue).toShort

  private def clampInt(value: Long): Int =
    value.max(Int.MinValue).min(Int.MaxValue).toInt

  private def rounded(value: Double, mode: Rounding): Double =
    mode match
      case Rounding.TowardZero =>
        if value >= 0.0 then Math.floor(value) else Math.ceil(value)
      case Rounding.NearestEven =>
        Math.rint(value)
      case Rounding.Floor =>
        Math.floor(value)
      case Rounding.Ceiling =>
        Math.ceil(value)

  private def fitsIntegral[B](
      value: Long,
      targetType: NumericDType[B]
  ): Boolean =
    targetType.tag match
      case DType.ByteTag =>
        value >= Byte.MinValue && value <= Byte.MaxValue
      case DType.ShortTag =>
        value >= Short.MinValue && value <= Short.MaxValue
      case DType.IntTag =>
        value >= Int.MinValue && value <= Int.MaxValue
      case DType.LongTag =>
        true
      case _ =>
        true

  private def fitsIntegral[B](
      value: Double,
      targetType: NumericDType[B]
  ): Boolean =
    targetType.tag match
      case DType.ByteTag =>
        value >= Byte.MinValue && value <= Byte.MaxValue
      case DType.ShortTag =>
        value >= Short.MinValue && value <= Short.MaxValue
      case DType.IntTag =>
        value >= Int.MinValue && value <= Int.MaxValue
      case DType.LongTag =>
        value >= Long.MinValue.toDouble && value <= Long.MaxValue.toDouble
      case _ =>
        true

  private def isIntegral[B](dtype: NumericDType[B]): Boolean =
    dtype.tag match
      case DType.ByteTag | DType.ShortTag | DType.IntTag | DType.LongTag =>
        true
      case _ =>
        false

  private inline def castLogical[S, T](
      layout: Layout,
      inline read: Int => S,
      inline write: (Int, T) => Unit
  )(inline conversion: S => T): Unit =
    if layout.isCContiguous then
      var output = 0
      var address = layout.offset
      while output < layout.size do
        write(output, conversion(read(address)))
        address += 1
        output += 1
    else
      var output = 0
      layout.foreachPhysicalIndex { address =>
        write(output, conversion(read(address)))
        output += 1
      }

  private inline def foreachPhysical(
      layout: Layout
  )(inline f: Int => Unit): Unit =
    if layout.isCContiguous then
      var address = layout.offset
      val end = layout.offset + layout.size
      while address < end do
        f(address)
        address += 1
    else layout.foreachPhysicalIndex(f)
