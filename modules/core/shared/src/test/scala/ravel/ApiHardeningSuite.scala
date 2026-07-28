package ravel

import munit.FunSuite
import ravel.DType.given

final class ApiHardeningSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("integer / is unavailable; quot truncates; floating / remains") {
    val ints = NDArray.fromSeq(Shape(3), Seq(7, -7, 8))
    assertEquals(values(ints.quot(2)), List(3, -3, 4))
    assertEquals(values(ints.truncDiv(2)), List(3, -3, 4))
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Int](2) / 2
    """).nonEmpty)
    val doubles = NDArray.fromSeq(Shape(2), Seq(7.0, 8.0))
    assertEquals(values(doubles / 2.0), List(3.5, 4.0))
  }

  test("comparisons support scalars and both directions") {
    val left = NDArray.fromSeq(Shape(3), Seq(1, 2, 3))
    val right = NDArray.fromSeq(Shape(3), Seq(2, 2, 2))
    assertEquals(values(left < right), List(true, false, false))
    assertEquals(values(left <= right), List(true, true, false))
    assertEquals(values(left > right), List(false, false, true))
    assertEquals(values(left >= 2), List(false, true, true))
    assertEquals(values(left < 2), List(true, false, false))
  }

  test("reshape triad distinguishes view copy and always-copy") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val viewed = source.reshapeView(Shape(6))
    assert(viewed.storage eq source.storage)
    val transposed = source.transpose
    intercept[NonContiguousLayout](transposed.reshapeView(Shape(6)))
    val reshaped = transposed.reshape(Shape(6))
    assertEquals(values(reshaped), values(transposed))
    assert(!(reshaped.storage eq source.storage))
    val copied = source.reshapeCopy(Shape(3, 2))
    assert(!(copied.storage eq source.storage))
    assertEquals(values(copied), values(source))
  }

  test("item extracts size-one values and rejects larger arrays") {
    assertEquals(NDArray.scalar(3.5).item, 3.5)
    assertEquals(NDArray.fromSeq(Shape(1, 1), Seq(9)).item, 9)
    intercept[InvalidShape](NDArray.zeros[Int](2).item)
  }

  test("sumKeepDims aliases keep-axis reductions") {
    val matrix = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    assertEquals(matrix.sumKeep(0).shape, matrix.sumKeepDims(0).shape)
    assert(matrix.sumKeep(0).sameElements(matrix.sumKeepDims(0)))
  }

  test("one import finds dtype givens from companions") {
    assertEquals(
      compileErrors("""
      import ravel.*
      val x = NDArray.zeros[Double](2, 3)
      val y = x + 1.0
    """),
      ""
    )
  }

  test("rank-two transpose preserves Array2 type") {
    val matrix: Array2[Int] = NDArray.zeros[Int](2, 3)
    val transposed: Array2[Int] = matrix.transpose
    assertEquals(transposed.shape, Shape(3, 2))
  }
