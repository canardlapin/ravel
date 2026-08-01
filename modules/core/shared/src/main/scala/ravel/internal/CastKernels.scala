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
      case (x: ByteStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(identity)
      case (x: ByteStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toShort)
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
      case (x: ShortStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toByte)
      case (x: ShortStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(identity)
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
      case (x: IntStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toByte)
      case (x: IntStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toShort)
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
      case (x: LongStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toByte)
      case (x: LongStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toShort)
      case (x: LongStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toShort)
      case (x: LongStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: LongStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)
      case (x: LongStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: FloatStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt.toByte)
      case (x: FloatStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toInt.toByte)
      case (x: FloatStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toInt.toShort)
      case (x: FloatStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt.toShort)
      case (x: FloatStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: FloatStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toLong)
      case (x: FloatStorage, z: DoubleStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toDouble)

      case (x: DoubleStorage, z: ByteStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt.toByte)
      case (x: DoubleStorage, z: UInt8Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toInt.toByte)
      case (x: DoubleStorage, z: UInt16Storage) =>
        castLogical(layout, x.raw.apply, z.setRaw)(_.toInt.toShort)
      case (x: DoubleStorage, z: ShortStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt.toShort)
      case (x: DoubleStorage, z: IntStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toInt)
      case (x: DoubleStorage, z: LongStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toLong)
      case (x: DoubleStorage, z: FloatStorage) =>
        castLogical(layout, x.raw.apply, z.raw.update)(_.toFloat)

      case (x: UInt8Storage, z: ByteStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(identity)
      case (x: UInt8Storage, z: ShortStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xff).toShort)
      case (x: UInt8Storage, z: UInt16Storage) =>
        castLogical(layout, x.getRaw, z.setRaw)(value => (value & 0xff).toShort)
      case (x: UInt8Storage, z: IntStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(_ & 0xff)
      case (x: UInt8Storage, z: LongStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xff).toLong)
      case (x: UInt8Storage, z: FloatStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xff).toFloat)
      case (x: UInt8Storage, z: DoubleStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xff).toDouble)

      case (x: UInt16Storage, z: ByteStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(_.toByte)
      case (x: UInt16Storage, z: UInt8Storage) =>
        castLogical(layout, x.getRaw, z.setRaw)(_.toByte)
      case (x: UInt16Storage, z: ShortStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(identity)
      case (x: UInt16Storage, z: IntStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(_ & 0xffff)
      case (x: UInt16Storage, z: LongStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xffff).toLong)
      case (x: UInt16Storage, z: FloatStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xffff).toFloat)
      case (x: UInt16Storage, z: DoubleStorage) =>
        castLogical(layout, x.getRaw, z.raw.update)(value => (value & 0xffff).toDouble)

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
