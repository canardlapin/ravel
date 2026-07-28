package ravel

import ravel.internal.*
import scala.annotation.publicInBinary
import scala.compiletime.erasedValue

/** Consuming linear-index writer used by [[NDArray.build]].
  *
  * Writes may arrive in any order. Repeated writes use the last value and unwritten elements keep
  * the dtype's zero value. A builder is valid only during its synchronous `NDArray.build` callback:
  * every write after that callback returns or throws fails with `IllegalStateException`.
  *
  * Builders are mutable and are not thread-safe.
  */
final class ArrayBuilder[A] private[ravel] (
    private val storage: Storage[A],
    val size: Int
):
  private var open = true

  /** Write at a contiguous linear index in `[0, size)`. */
  inline def update(index: Int, value: A): Unit =
    // This match is reduced at the Scala call site. Each cast is guarded by its exact primitive
    // branch and routes to a primitive JVM/Scala.js method; abstract A uses the boxed fallback.
    inline erasedValue[A] match
      case _: Boolean => writeBoolean(index, value.asInstanceOf[Boolean])
      case _: Byte => writeByte(index, value.asInstanceOf[Byte])
      case _: Short => writeShort(index, value.asInstanceOf[Short])
      case _: Int => writeInt(index, value.asInstanceOf[Int])
      case _: Long => writeLong(index, value.asInstanceOf[Long])
      case _: Float => writeFloat(index, value.asInstanceOf[Float])
      case _: Double => writeDouble(index, value.asInstanceOf[Double])
      case _ => writeGeneric(index, value)

  /** Alias for [[update]]. */
  inline def writeLinear(index: Int, value: A): Unit =
    update(index, value)

  private[ravel] def seal(): Unit =
    ensureOpen()
    open = false

  private[ravel] def abandon(): Unit =
    open = false

  private def ensureOpen(): Unit =
    if !open then throw new BuilderClosed()

  private def checkedIndex(index: Int): Int =
    ensureOpen()
    if index < 0 || index >= size then throw InvalidIndex.LinearOutOfBounds(index, size)
    index

  @publicInBinary private[ravel] def writeGeneric(index: Int, value: A): Unit =
    ProbeApi.set(storage, checkedIndex(index), value)

  @publicInBinary private[ravel] def writeBoolean(index: Int, value: Boolean): Unit =
    ProbeApi.setBoolean(storage.asInstanceOf[Storage[Boolean]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeByte(index: Int, value: Byte): Unit =
    ProbeApi.setByte(storage.asInstanceOf[Storage[Byte]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeShort(index: Int, value: Short): Unit =
    ProbeApi.setShort(storage.asInstanceOf[Storage[Short]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeInt(index: Int, value: Int): Unit =
    ProbeApi.setInt(storage.asInstanceOf[Storage[Int]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeLong(index: Int, value: Long): Unit =
    ProbeApi.setLong(storage.asInstanceOf[Storage[Long]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeFloat(index: Int, value: Float): Unit =
    ProbeApi.setFloat(storage.asInstanceOf[Storage[Float]], checkedIndex(index), value)

  @publicInBinary private[ravel] def writeDouble(index: Int, value: Double): Unit =
    ProbeApi.setDouble(storage.asInstanceOf[Storage[Double]], checkedIndex(index), value)
