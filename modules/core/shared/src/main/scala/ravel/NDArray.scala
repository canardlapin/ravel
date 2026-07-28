package ravel

import ravel.internal.*
import scala.annotation.publicInBinary
import scala.compiletime.erasedValue

final class NDArray[A, +R <: AnyRank] private[ravel] (
    private[ravel] val storage: Storage[A],
    private[ravel] val layout: Layout,
    val dtype: DType[A]
):
  val shape: Shape[R] =
    Shape.unsafeRanked[R](layout.shape)

  def rank: Int = layout.rank
  def size: Int = layout.size
  def isContiguous: Boolean = layout.isCContiguous

  inline def apply(i: Int): A =
    readPrimitive(physicalIndex1(i))

  inline def apply(i: Int, j: Int): A =
    readPrimitive(physicalIndex2(i, j))

  inline def apply(i: Int, j: Int, k: Int): A =
    readPrimitive(physicalIndex3(i, j, k))

  inline def apply(i: Int, j: Int, k: Int, l: Int): A =
    readPrimitive(physicalIndex4(i, j, k, l))

  def at(indices: IArray[Int]): A =
    ProbeApi.get(storage, layout.physicalIndex(indices))

  private inline def readPrimitive(index: Int): A =
    // This match is reduced at the Scala call site. Each cast is guarded by its exact primitive
    // branch and routes to a primitive JVM/Scala.js method; abstract A uses the boxed fallback.
    inline erasedValue[A] match
      case _: Boolean => readBoolean(index).asInstanceOf[A]
      case _: Byte => readByte(index).asInstanceOf[A]
      case _: Short => readShort(index).asInstanceOf[A]
      case _: Int => readInt(index).asInstanceOf[A]
      case _: Long => readLong(index).asInstanceOf[A]
      case _: Float => readFloat(index).asInstanceOf[A]
      case _: Double => readDouble(index).asInstanceOf[A]
      case _ => readGeneric(index)

  @publicInBinary private[ravel] def physicalIndex1(i: Int): Int =
    layout.physicalIndex1(i)

  @publicInBinary private[ravel] def physicalIndex2(i: Int, j: Int): Int =
    layout.physicalIndex2(i, j)

  @publicInBinary private[ravel] def physicalIndex3(i: Int, j: Int, k: Int): Int =
    layout.physicalIndex3(i, j, k)

  @publicInBinary private[ravel] def physicalIndex4(i: Int, j: Int, k: Int, l: Int): Int =
    layout.physicalIndex4(i, j, k, l)

  @publicInBinary private[ravel] def readGeneric(index: Int): A =
    ProbeApi.get(storage, index)

  @publicInBinary private[ravel] def readBoolean(index: Int): Boolean =
    ProbeApi.getBoolean(storage.asInstanceOf[Storage[Boolean]], index)

  @publicInBinary private[ravel] def readByte(index: Int): Byte =
    ProbeApi.getByte(storage.asInstanceOf[Storage[Byte]], index)

  @publicInBinary private[ravel] def readShort(index: Int): Short =
    ProbeApi.getShort(storage.asInstanceOf[Storage[Short]], index)

  @publicInBinary private[ravel] def readInt(index: Int): Int =
    ProbeApi.getInt(storage.asInstanceOf[Storage[Int]], index)

  @publicInBinary private[ravel] def readLong(index: Int): Long =
    ProbeApi.getLong(storage.asInstanceOf[Storage[Long]], index)

  @publicInBinary private[ravel] def readFloat(index: Int): Float =
    ProbeApi.getFloat(storage.asInstanceOf[Storage[Float]], index)

  @publicInBinary private[ravel] def readDouble(index: Int): Double =
    ProbeApi.getDouble(storage.asInstanceOf[Storage[Double]], index)

  def requireRank[N <: Int](using expected: ValueOf[N]): Either[RankMismatch, NDArray[A, Rank[N]]] =
    if rank == expected.value then Right(this.asInstanceOf[NDArray[A, Rank[N]]])
    else Left(RankMismatch(expected.value, rank))

  def slice(axis: Int, slice: Slice): NDArray[A, R] =
    new NDArray[A, R](
      storage,
      ViewLayout.slice(layout, axis, slice, storage.length),
      dtype
    )

  def slice(axis: Int, range: Range): NDArray[A, R] =
    val slice = Slice.from(range).fold(throw _, identity)
    this.slice(axis, slice)

  def narrow(axis: Int, from: Int, length: Int): NDArray[A, R] =
    if length < 0 then throw InvalidSlice(s"negative narrow length $length")
    val stop = Layout.checkedInt(
      Layout.checkedAdd(from.toLong, length.toLong, "narrow endpoint"),
      "narrow endpoint"
    )
    slice(axis, Slice(from, stop, 1))

  def reverse(axis: Int): NDArray[A, R] =
    new NDArray[A, R](
      storage,
      ViewLayout.reverse(layout, axis, storage.length),
      dtype
    )

  def swapAxes(first: Int, second: Int): NDArray[A, R] =
    val left = layout.normalizedAxis(first)
    val right = layout.normalizedAxis(second)
    val order = Array.tabulate(rank)(identity)
    val temporary = order(left)
    order(left) = order(right)
    order(right) = temporary
    permuteAxes(order.toSeq*)

  def permuteAxes(order: Int*): NDArray[A, R] =
    new NDArray[A, R](
      storage,
      ViewLayout.permute(layout, order, storage.length),
      dtype
    )

  def transpose: NDArray[A, R] =
    if rank != 2 then throw InvalidAxis(2, rank)
    swapAxes(0, 1)

  def broadcastTo[S <: AnyRank](target: Shape[S]): NDArray[A, S] =
    new NDArray[A, S](
      storage,
      ViewLayout.broadcastTo(layout, target, storage.length),
      dtype
    )

  def reshapeView[S <: AnyRank](target: Shape[S]): NDArray[A, S] =
    new NDArray[A, S](
      storage,
      ViewLayout.reshape(layout, target, storage.length),
      dtype
    )

  def contiguous: NDArray[A, R] =
    if isContiguous then this else copy

  def flattenCopy: Array1[A] =
    val flat = copy
    new NDArray[A, Rank[1]](
      flat.storage,
      Layout.contiguous(Shape(size), size),
      dtype
    )

  def foreachElement(f: A => Unit): Unit =
    layout.foreachPhysicalIndex(index => f(ProbeApi.get(storage, index)))

  def foreachIndex(f: IArray[Int] => Unit): Unit =
    if size == 0 then return
    if rank == 0 then
      f(IArray.empty)
      return
    val counters = new Array[Int](rank)
    var visited = 0
    while visited < size do
      f(IArray.unsafeFromArray(counters.clone()))
      visited += 1
      if visited < size then
        var axis = rank - 1
        var advanced = false
        while axis >= 0 && !advanced do
          counters(axis) += 1
          if counters(axis) < layout.shape(axis) then advanced = true
          else
            counters(axis) = 0
            axis -= 1

  def elementsIterator: Iterator[A] =
    new Iterator[A]:
      private val values = new Array[Any](NDArray.this.size)
      private var write = 0
      layout.foreachPhysicalIndex { index =>
        values(write) = ProbeApi.get(storage, index)
        write += 1
      }
      private var read = 0
      def hasNext: Boolean = read < values.length
      def next(): A =
        if !hasNext then throw new NoSuchElementException("next on empty iterator")
        val value = values(read).asInstanceOf[A]
        read += 1
        value

  /** Always copies in logical row-major order. */
  def copy: NDArray[A, R] =
    val output = ProbeApi.allocate[A](size)(using dtype)
    var write = 0
    layout.foreachPhysicalIndex { index =>
      ProbeApi.set(output, write, ProbeApi.get(storage, index))
      write += 1
    }
    new NDArray(output, Layout.contiguous(shape, size), dtype)

  def cast[B](using source: NumericDType[A], target: NumericDType[B]): NDArray[B, R] =
    val output = ProbeApi.allocate[B](size)(using target)
    var write = 0
    layout.foreachPhysicalIndex { index =>
      val converted = DType.castScalar(ProbeApi.get(storage, index), source, target)
      ProbeApi.set(output, write, converted)
      write += 1
    }
    new NDArray(output, Layout.contiguous(shape, size), target)

  def sameElements(other: NDArray[A, ?]): Boolean =
    EqualityApi.sameElements(this, other)

  def sameElements(other: BorrowedNDArray[A, ?]): Boolean =
    EqualityApi.sameElements(this, other.underlying)

  def sameElementsBits(other: NDArray[A, ?]): Boolean =
    EqualityApi.sameElementsBits(this, other)

  def sameElementsBits(other: BorrowedNDArray[A, ?]): Boolean =
    EqualityApi.sameElementsBits(this, other.underlying)

  override def toString: String =
    val builder = new StringBuilder
    builder.append("NDArray[")
    builder.append(dtype.name)
    builder.append("](shape = ")
    builder.append(shape)
    builder.append(", contiguous = ")
    builder.append(isContiguous)
    builder.append(", values = ")
    builder.append(Preview.render(this))
    builder.append(")")
    builder.result()

object NDArray:
  def scalar[A](value: A)(using dtype: DType[A]): Array0[A] =
    val storage = ProbeApi.allocate[A](1)
    ProbeApi.set(storage, 0, value)
    new NDArray(storage, Layout.contiguous(Shape.scalar, 1), dtype)

  def zeros[A](d0: Int)(using dtype: DType[A]): Array1[A] =
    zeros(Shape(d0))

  def zeros[A](d0: Int, d1: Int)(using dtype: DType[A]): Array2[A] =
    zeros(Shape(d0, d1))

  def zeros[A](d0: Int, d1: Int, d2: Int)(using dtype: DType[A]): Array3[A] =
    zeros(Shape(d0, d1, d2))

  def zeros[A](d0: Int, d1: Int, d2: Int, d3: Int)(using dtype: DType[A]): Array4[A] =
    zeros(Shape(d0, d1, d2, d3))

  def zeros[A, R <: AnyRank](shape: Shape[R])(using dtype: DType[A]): NDArray[A, R] =
    val storage = ProbeApi.allocate[A](shape.size)
    new NDArray(storage, Layout.contiguous(shape, shape.size), dtype)

  def fill[A, R <: AnyRank](shape: Shape[R], value: A)(using dtype: DType[A]): NDArray[A, R] =
    val result = zeros[A, R](shape)
    ProbeApi.fill(result.storage, value)
    result

  def fromSeq[A, R <: AnyRank](shape: Shape[R], values: IterableOnce[A])(using
      dtype: DType[A]
  ): NDArray[A, R] =
    val storage = ProbeApi.allocate[A](shape.size)
    val iterator = values.iterator
    var index = 0
    while iterator.hasNext && index < shape.size do
      ProbeApi.set(storage, index, iterator.next())
      index += 1
    if iterator.hasNext || index != shape.size then
      throw ShapeMismatch(shape.toString, s"values(count differs from ${shape.size})")
    new NDArray(storage, Layout.contiguous(shape, shape.size), dtype)

  /** Allocate one contiguous destination and fill it through a consuming [[ArrayBuilder]].
    *
    * The returned immutable array owns the builder's destination directly; construction performs no
    * output-sized copy. The builder is sealed after `body` returns or throws and rejects every
    * later write. If `body` throws, no array is returned.
    */
  def build[A, R <: AnyRank](shape: Shape[R])(body: ArrayBuilder[A] => Unit)(using
      dtype: DType[A]
  ): NDArray[A, R] =
    val storage = ProbeApi.allocate[A](shape.size)
    val builder = new ArrayBuilder[A](storage, shape.size)
    try
      body(builder)
      builder.seal()
      new NDArray(storage, Layout.contiguous(shape, shape.size), dtype)
    catch
      case error: Throwable =>
        builder.abandon()
        throw error

  def tabulate[A](d0: Int)(f: Int => A)(using dtype: DType[A]): Array1[A] =
    val result = zeros[A](d0)
    var i = 0
    while i < d0 do
      ProbeApi.set(result.storage, i, f(i))
      i += 1
    result

  def tabulate[A](d0: Int, d1: Int)(f: (Int, Int) => A)(using
      dtype: DType[A]
  ): Array2[A] =
    val result = zeros[A](d0, d1)
    var i = 0
    var out = 0
    while i < d0 do
      var j = 0
      while j < d1 do
        ProbeApi.set(result.storage, out, f(i, j))
        out += 1
        j += 1
      i += 1
    result

  def tabulate[A](d0: Int, d1: Int, d2: Int)(f: (Int, Int, Int) => A)(using
      dtype: DType[A]
  ): Array3[A] =
    val result = zeros[A](d0, d1, d2)
    var i = 0
    var out = 0
    while i < d0 do
      var j = 0
      while j < d1 do
        var k = 0
        while k < d2 do
          ProbeApi.set(result.storage, out, f(i, j, k))
          out += 1
          k += 1
        j += 1
      i += 1
    result

  def tabulate[A](d0: Int, d1: Int, d2: Int, d3: Int)(
      f: (Int, Int, Int, Int) => A
  )(using dtype: DType[A]): Array4[A] =
    val result = zeros[A](d0, d1, d2, d3)
    var i = 0
    var out = 0
    while i < d0 do
      var j = 0
      while j < d1 do
        var k = 0
        while k < d2 do
          var l = 0
          while l < d3 do
            ProbeApi.set(result.storage, out, f(i, j, k, l))
            out += 1
            l += 1
          k += 1
        j += 1
      i += 1
    result
