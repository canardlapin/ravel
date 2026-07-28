package ravel

import munit.FunSuite
import ravel.DType.given

final class ViewLawsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("select, slice, narrow, reverse, and negative axes are storage-sharing views") {
    val source = NDArray.tabulate[Int](3, 5)((i, j) => i * 10 + j)
    val selected: Array1[Int] = source.select(0, 1)
    val sliced = source.slice(-1, Slice(0, 5, 2))
    val narrowed = source.narrow(1, 1, 3)
    val reversed = source.reverse(-1)
    assert(selected.storage eq source.storage)
    assert(sliced.storage eq source.storage)
    assert(narrowed.storage eq source.storage)
    assert(reversed.storage eq source.storage)
    assertEquals(values(selected), List(10, 11, 12, 13, 14))
    assertEquals(values(sliced), List(0, 2, 4, 10, 12, 14, 20, 22, 24))
    assertEquals(values(narrowed), List(1, 2, 3, 11, 12, 13, 21, 22, 23))
    assertEquals(values(reversed), List(4, 3, 2, 1, 0, 14, 13, 12, 11, 10, 24, 23, 22, 21, 20))
  }

  test("canonical Slice handles full negative reversal, empties, and Range sugar") {
    val source = NDArray.tabulate[Int](5)(identity)
    assertEquals(values(source.slice(0, Slice(4, -1, -1))), List(4, 3, 2, 1, 0))
    assertEquals(values(source.slice(0, Slice(3, 3, 1))), Nil)
    assertEquals(values(source.slice(0, 0 to 4 by 2)), List(0, 2, 4))
    assertEquals(values(source.slice(0, Slice.from(-1))), List(4))
    assertEquals(values(source.slice(0, Slice.every(2))), List(0, 2, 4))
    assertEquals(values(source.slice(0, Slice.reverse)), List(4, 3, 2, 1, 0))
    assertEquals(values(source.slice(0, Slice.all)), values(source))
    intercept[InvalidSlice](source.slice(0, Slice(4, -2, -1)))
    assert(Slice.from(0, 1, 0).isLeft)
    assert(Slice.from(Range.inclusive(Int.MaxValue, Int.MaxValue)).isLeft)
  }

  test("axis permutation, insertion, and squeeze preserve typed ranks") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val transposed = source.transpose
    val swappedTwice = source.swapAxes(0, 1).swapAxes(-1, -2)
    val inserted: Array3[Int] = source.newAxis(-1)
    val squeezed: Array2[Int] = inserted.squeeze(-1)
    assertEquals(transposed.shape.toString, "(3, 2)")
    assertEquals(values(transposed), List(0, 10, 1, 11, 2, 12))
    assertEquals(values(swappedTwice), values(source))
    assertEquals(inserted.shape.toString, "(2, 3, 1)")
    assertEquals(values(squeezed), values(source))
    intercept[InvalidAxis](source.permuteAxes(0, 0))
    intercept[InvalidAxis](source.newAxis(3))
    intercept[InvalidShape](source.squeeze(0))
  }

  test("view involutions and permutation composition hold") {
    val source = NDArray.tabulate[Int](2, 3, 4)((i, j, k) => i * 100 + j * 10 + k)
    assertEquals(values(source.reverse(1).reverse(1)), values(source))
    assertEquals(values(source.swapAxes(0, 2).swapAxes(0, 2)), values(source))
    val sequential = source.permuteAxes(1, 2, 0).permuteAxes(2, 0, 1)
    assertEquals(values(sequential), values(source))
    assertEquals(sequential.shape.toString, source.shape.toString)
  }

  test("broadcastTo uses zero strides and 1 broadcasts to 0") {
    val source = NDArray.fromSeq(Shape(1, 3), Seq(1, 2, 3))
    val broadcast = source.broadcastTo(Shape(2, 3))
    assert(broadcast.storage eq source.storage)
    assertEquals(values(broadcast), List(1, 2, 3, 1, 2, 3))
    assert(broadcast.layout.hasBroadcastStride)

    val empty = NDArray.fromSeq(Shape(1), Seq(7)).broadcastTo(Shape(0))
    assertEquals(empty.size, 0)
    assertEquals(values(empty), Nil)

    val zero = NDArray.zeros[Int](0)
    intercept[BroadcastMismatch](zero.broadcastTo(Shape(1)))
  }

  test("reshapeView accepts exactly repartitionable physical address sequences") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val flat = source.reshapeView(Shape(6))
    val reshaped = flat.reshapeView(Shape(3, 2))
    assert(flat.storage eq source.storage)
    assertEquals(values(flat), values(source))
    assertEquals(values(reshaped), values(source))

    val reversedFlat = source.reverse(0).reverse(1).reshapeView(Shape(6))
    assertEquals(values(reversedFlat), List(12, 11, 10, 2, 1, 0))

    val transposed = source.transpose
    intercept[NonContiguousLayout](transposed.reshapeView(Shape(6)))
    val same = transposed.reshapeView(Shape(3, 2))
    assertEquals(values(same), values(transposed))

    val broadcast = NDArray.fromSeq(Shape(1), Seq(1)).broadcastTo(Shape(3))
    intercept[NonContiguousLayout](broadcast.reshapeView(Shape(3, 1)))
  }

  test("zero-element reshape is canonical when target strides are representable") {
    val source = NDArray.zeros[Int](0, 3).transpose
    val target = source.reshapeView(Shape(2, 0, 4))
    assertEquals(target.size, 0)
    assert(target.isContiguous)
    assert(target.storage eq source.storage)
  }

  test("contiguous and copying contracts are explicit") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    assert(source.contiguous eq source)
    val transposed = source.transpose
    val materialized = transposed.contiguous
    assert(!(materialized.storage eq source.storage))
    assert(materialized.isContiguous)
    assertEquals(values(materialized), values(transposed))
    val copied = source.copy
    assert(!(copied.storage eq source.storage))
    assertEquals(values(copied), values(source))
    val flat = transposed.flattenCopy
    assertEquals(flat.shape.toString, "(6)")
    assertEquals(values(flat), values(transposed))
  }

  test("deterministic composed views agree with a coordinate reference model") {
    var rows = 1
    while rows <= 5 do
      var columns = 1
      while columns <= 6 do
        val source = NDArray.tabulate[Int](rows, columns)((i, j) => i * 100 + j)
        val actual =
          source.transpose
            .reverse(0)
            .slice(1, Slice(0, rows, 2))
            .elementsIterator
            .toList
        val expected =
          (columns - 1 to 0 by -1).flatMap { j =>
            (0 until rows by 2).map(i => i * 100 + j)
          }.toList
        assertEquals(actual, expected)
        columns += 1
      rows += 1
  }
