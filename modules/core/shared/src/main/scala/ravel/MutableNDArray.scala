package ravel

import ravel.internal.*
import scala.annotation.unused

final class MutableNDArray[A, R <: AnyRank] private[ravel] (
    private[ravel] val storage: Storage[A],
    private[ravel] val mutableLayout: MutableLayout,
    val dtype: DType[A]
):
  private[ravel] def layout: Layout = mutableLayout.underlying

  val shape: Shape[R] =
    Shape.unsafeRanked[R](layout.shape)

  def rank: Int = layout.rank
  def size: Int = layout.size

  def apply(i: Int): A =
    ProbeApi.get(storage, layout.physicalIndex1(i))

  def apply(i: Int, j: Int): A =
    ProbeApi.get(storage, layout.physicalIndex2(i, j))

  def apply(i: Int, j: Int, k: Int): A =
    ProbeApi.get(storage, layout.physicalIndex3(i, j, k))

  def apply(i: Int, j: Int, k: Int, l: Int): A =
    ProbeApi.get(storage, layout.physicalIndex4(i, j, k, l))

  def at(indices: IArray[Int]): A =
    ProbeApi.get(storage, layout.physicalIndex(indices))

  def update(i: Int, value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex1(i), value)

  def update(i: Int, j: Int, value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex2(i, j), value)

  def update(i: Int, j: Int, k: Int, value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex3(i, j, k), value)

  def update(i: Int, j: Int, k: Int, l: Int, value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex4(i, j, k, l), value)

  def updateAt(indices: IArray[Int], value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex(indices), value)

  def fill(value: A): Unit =
    if layout.isCContiguous && layout.offset == 0 && layout.size == storage.length then
      ProbeApi.fill(storage, value)
    else
      layout.foreachPhysicalIndex(index => ProbeApi.set(storage, index, value))

  def assign(source: NDArray[A, ?]): Unit =
    MutableNDArray.requireSameShape(layout.shape, source.layout.shape)
    val iterator = source.elementsIterator
    layout.foreachPhysicalIndex { index =>
      ProbeApi.set(storage, index, iterator.next())
    }

  def addInPlace(value: A)(using
      @unused arithmetic: ArithmeticDType[A]
  ): Unit =
    val transformed = KernelApi.scalar(KernelOp.Add, freezeCopy(), value)
    assign(transformed)

  def select(axis: Int, index: Int)(using
      CanDropAxis[R]
  ): MutableNDArray[A, DropAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.select(mutableLayout, axis, index, storage.length),
      dtype
    )

  def slice(axis: Int, slice: Slice): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.slice(mutableLayout, axis, slice, storage.length),
      dtype
    )

  def slice(axis: Int, range: Range): MutableNDArray[A, R] =
    val canonical = Slice.from(range).fold(throw _, identity)
    slice(axis, canonical)

  def narrow(axis: Int, from: Int, length: Int): MutableNDArray[A, R] =
    if length < 0 then throw InvalidSlice(s"negative narrow length $length")
    val stop = Layout.checkedInt(
      Layout.checkedAdd(from.toLong, length.toLong, "mutable narrow endpoint"),
      "mutable narrow endpoint"
    )
    slice(axis, Slice(from, stop))

  def reverse(axis: Int): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.reverse(mutableLayout, axis, storage.length),
      dtype
    )

  def swapAxes(first: Int, second: Int): MutableNDArray[A, R] =
    val left = layout.normalizedAxis(first)
    val right = layout.normalizedAxis(second)
    val order = Array.tabulate(rank)(identity)
    val temporary = order(left)
    order(left) = order(right)
    order(right) = temporary
    permuteAxes(order.toSeq*)

  def permuteAxes(order: Int*): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.permute(mutableLayout, order, storage.length),
      dtype
    )

  def transpose: MutableNDArray[A, R] =
    if rank != 2 then throw InvalidAxis(2, rank)
    swapAxes(0, 1)

  def newAxis(axis: Int): MutableNDArray[A, AddAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.newAxis(mutableLayout, axis, storage.length),
      dtype
    )

  def squeeze(axis: Int)(using
      CanDropAxis[R]
  ): MutableNDArray[A, DropAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.squeeze(mutableLayout, axis, storage.length),
      dtype
    )

  def reshapeView[S <: AnyRank](
      target: Shape[S]
  ): MutableNDArray[A, S] =
    new MutableNDArray(
      storage,
      MutableLayout.reshape(mutableLayout, target, storage.length),
      dtype
    )

  def freezeCopy(): NDArray[A, R] =
    val output = ProbeApi.allocate[A](size)(using dtype)
    CopyKernels.logical(storage, layout, output)
    new NDArray(output, Layout.contiguous(shape, size), dtype)

object MutableNDArray:
  def zeros[A, R <: AnyRank](shape: Shape[R])(using
      dtype: DType[A]
  ): MutableNDArray[A, R] =
    val storage = ProbeApi.allocate[A](shape.size)
    new MutableNDArray(storage, MutableLayout.owned(shape, shape.size), dtype)

  private[ravel] def requireSameShape(
      left: IArray[Int],
      right: IArray[Int]
  ): Unit =
    var same = left.length == right.length
    var i = 0
    while i < left.length && same do
      same = left(i) == right(i)
      i += 1
    if !same then
      throw ShapeMismatch(
        left.mkString("(", ", ", ")"),
        right.mkString("(", ", ", ")")
      )

extension [A, R <: AnyRank](array: NDArray[A, R])
  def mutableCopy: MutableNDArray[A, R] =
    val copied = array.copy
    new MutableNDArray(
      copied.storage,
      MutableLayout.owned(copied.shape, copied.size),
      copied.dtype
    )
