package ravel.internal

import ravel.DType

private[ravel] sealed trait Storage[A]:
  def length: Int

private[ravel] final class BooleanStorage(val raw: Array[Boolean]) extends Storage[Boolean]:
  def length: Int = raw.length

private[ravel] final class ByteStorage(val raw: Array[Byte]) extends Storage[Byte]:
  def length: Int = raw.length

private[ravel] final class ShortStorage(val raw: Array[Short]) extends Storage[Short]:
  def length: Int = raw.length

private[ravel] final class IntStorage(val raw: Array[Int]) extends Storage[Int]:
  def length: Int = raw.length

private[ravel] final class LongStorage(val raw: Array[Long]) extends Storage[Long]:
  def length: Int = raw.length

private[ravel] final class FloatStorage(val raw: Array[Float]) extends Storage[Float]:
  def length: Int = raw.length

private[ravel] final class DoubleStorage(val raw: Array[Double]) extends Storage[Double]:
  def length: Int = raw.length

private[ravel] object PlatformStorage:
  def allocate[A](size: Int)(using dtype: DType[A]): Storage[A] =
    if size < 0 then throw new IllegalArgumentException(s"negative buffer size: $size")
    (dtype.tag match
      case DType.BooleanTag => new BooleanStorage(new Array[Boolean](size))
      case DType.ByteTag    => new ByteStorage(new Array[Byte](size))
      case DType.ShortTag   => new ShortStorage(new Array[Short](size))
      case DType.IntTag     => new IntStorage(new Array[Int](size))
      case DType.LongTag    => new LongStorage(new Array[Long](size))
      case DType.FloatTag   => new FloatStorage(new Array[Float](size))
      case DType.DoubleTag  => new DoubleStorage(new Array[Double](size))
      case tag              => throw new MatchError(tag)
    ).asInstanceOf[Storage[A]]
