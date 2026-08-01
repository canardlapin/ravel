package ravel.internal

import ravel.DType
import ravel.UInt16
import ravel.UInt8
import scala.scalajs.js.typedarray.*

private[ravel] sealed trait Storage[A]:
  def length: Int

/** Booleans are encoded as exactly 0 or 1. */
private[ravel] final class BooleanStorage(val raw: Uint8Array) extends Storage[Boolean]:
  def length: Int = raw.length

private[ravel] final class ByteStorage(val raw: Int8Array) extends Storage[Byte]:
  def length: Int = raw.length

private[ravel] final class ShortStorage(val raw: Int16Array) extends Storage[Short]:
  def length: Int = raw.length

private[ravel] final class UInt8Storage(val raw: Uint8Array) extends Storage[UInt8]:
  def length: Int = raw.length

private[ravel] final class UInt16Storage(val raw: Uint16Array) extends Storage[UInt16]:
  def length: Int = raw.length

private[ravel] final class IntStorage(val raw: Int32Array) extends Storage[Int]:
  def length: Int = raw.length

/** Scala.js Long is opaque to JavaScript and has no Ravel 1.0 typed-array contract. This fallback
  * preserves Scala Long semantics but is not a JS fast-path representation.
  */
private[ravel] final class LongStorage(val raw: Array[Long]) extends Storage[Long]:
  def length: Int = raw.length

private[ravel] final class FloatStorage(val raw: Float32Array) extends Storage[Float]:
  def length: Int = raw.length

private[ravel] final class DoubleStorage(val raw: Float64Array) extends Storage[Double]:
  def length: Int = raw.length

private[ravel] object PlatformStorage:
  def allocate[A](size: Int)(using dtype: DType[A]): Storage[A] =
    if size < 0 then throw new IllegalArgumentException(s"negative buffer size: $size")
    (dtype.tag match
      case DType.BooleanTag => new BooleanStorage(new Uint8Array(size))
      case DType.ByteTag => new ByteStorage(new Int8Array(size))
      case DType.UInt8Tag => new UInt8Storage(new Uint8Array(size))
      case DType.ShortTag => new ShortStorage(new Int16Array(size))
      case DType.UInt16Tag => new UInt16Storage(new Uint16Array(size))
      case DType.IntTag => new IntStorage(new Int32Array(size))
      case DType.LongTag => new LongStorage(new Array[Long](size))
      case DType.FloatTag => new FloatStorage(new Float32Array(size))
      case DType.DoubleTag => new DoubleStorage(new Float64Array(size))
      case tag => throw new MatchError(tag)
    ).asInstanceOf[Storage[A]]
