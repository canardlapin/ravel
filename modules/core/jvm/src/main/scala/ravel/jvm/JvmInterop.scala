package ravel.jvm

import ravel.*
import ravel.internal.*
import scala.reflect.ClassTag

object JvmInterop:
  /** Copies logical row-major values into a new JVM primitive array. */
  def copyToArray[A: ClassTag](array: NDArray[A, ?]): Array[A] =
    copyOwned(array)

  /** Copies logical row-major values without erasing borrowed provenance. */
  def copyToArray[A: ClassTag](array: BorrowedNDArray[A, ?]): Array[A] =
    copyOwned(array.underlying)

  /** Borrows a caller-owned JVM array without copying.
    *
    * Later caller mutation is intentionally observable through the result.
    */
  def unsafeBorrow[A, R <: AnyRank](
      values: Array[A],
      shape: Shape[R]
  )(using dtype: DType[A]): BorrowedNDArray[A, R] =
    if values.length != shape.size then
      throw ShapeMismatch(shape.toString, s"Array(length = ${values.length})")
    val storage =
      (dtype.tag match
        case DType.BooleanTag =>
          new BooleanStorage(values.asInstanceOf[Array[Boolean]])
        case DType.ByteTag =>
          new ByteStorage(values.asInstanceOf[Array[Byte]])
        case DType.ShortTag =>
          new ShortStorage(values.asInstanceOf[Array[Short]])
        case DType.IntTag =>
          new IntStorage(values.asInstanceOf[Array[Int]])
        case DType.LongTag =>
          new LongStorage(values.asInstanceOf[Array[Long]])
        case DType.FloatTag =>
          new FloatStorage(values.asInstanceOf[Array[Float]])
        case DType.DoubleTag =>
          new DoubleStorage(values.asInstanceOf[Array[Double]])
        case tag => throw new MatchError(tag)
      ).asInstanceOf[Storage[A]]
    val view = new NDArray(storage, Layout.contiguous(shape, values.length), dtype)
    new BorrowedNDArray(view)

  private def copyOwned[A: ClassTag](array: NDArray[A, ?]): Array[A] =
    val output = new Array[A](array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output
