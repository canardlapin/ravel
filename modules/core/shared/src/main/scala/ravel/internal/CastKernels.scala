package ravel.internal

import ravel.*

/** Numeric casts with one storage-family dispatch and no per-element tag match. Conversion
  * semantics match [[DType.castScalar]].
  */
private[ravel] object CastKernels:
  def convert[A, B](
      source: Storage[A],
      layout: Layout,
      target: Storage[B],
      sourceType: NumericDType[A],
      targetType: NumericDType[B]
  ): Unit =
    if layout.size == 0 then return
    if sourceType.tag == targetType.tag then
      CopyKernels.logical(
        source.asInstanceOf[Storage[B]],
        layout,
        target
      )
      return

    (source, target) match
      case (x: ByteStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toShort)
      case (x: ByteStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: ByteStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toLong)
      case (x: ByteStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)
      case (x: ByteStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: ShortStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toByte)
      case (x: ShortStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: ShortStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toLong)
      case (x: ShortStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)
      case (x: ShortStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: IntStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toByte)
      case (x: IntStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toShort)
      case (x: IntStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toLong)
      case (x: IntStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)
      case (x: IntStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: LongStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toByte)
      case (x: LongStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toShort)
      case (x: LongStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: LongStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)
      case (x: LongStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: FloatStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToInt(v.toDouble).toByte)
      case (x: FloatStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToInt(v.toDouble).toShort)
      case (x: FloatStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToInt(v.toDouble))
      case (x: FloatStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToLong(v.toDouble))
      case (x: FloatStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: DoubleStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToInt(v).toByte)
      case (x: DoubleStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(v => floatingToInt(v).toShort)
      case (x: DoubleStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(floatingToInt)
      case (x: DoubleStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(floatingToLong)
      case (x: DoubleStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)

      case _ =>
        throw new IllegalArgumentException(
          s"unsupported cast ${sourceType.name} -> ${targetType.name}"
        )

  private inline def castLogical[S, T](
      layout: Layout,
      inline read: Int => S,
      inline write: (Int, T) => Unit
  )(inline convert: S => T): Unit =
    if layout.isCContiguous then
      var output = 0
      var address = layout.offset
      while output < layout.size do
        write(output, convert(read(address)))
        address += 1
        output += 1
    else
      var output = 0
      layout.foreachPhysicalIndex { address =>
        write(output, convert(read(address)))
        output += 1
      }

  private def floatingToInt(value: Double): Int =
    if value.isNaN then 0
    else if value >= Int.MaxValue.toDouble then Int.MaxValue
    else if value <= Int.MinValue.toDouble then Int.MinValue
    else value.toInt

  private def floatingToLong(value: Double): Long =
    if value.isNaN then 0L
    else if value >= Long.MaxValue.toDouble then Long.MaxValue
    else if value <= Long.MinValue.toDouble then Long.MinValue
    else value.toLong
