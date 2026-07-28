package ravel

import munit.FunSuite
import ravel.DType.given

final class CopyKernelsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  private def assertMaterializes[A](source: NDArray[A, ?]): Unit =
    val copied = source.copy
    assert(copied.isContiguous)
    assert(!(copied.storage eq source.storage))
    assertEquals(values(copied), values(source))

  private def assertDType[A](input: List[A])(using DType[A]): Unit =
    val source = NDArray.fromSeq(Shape(2, 3), input)
    assertMaterializes(source)
    assertMaterializes(source.transpose)
    assertMaterializes(source.reverse(0))
    assertMaterializes(source.reverse(1))
    assertMaterializes(source.slice(1, Slice(0, 3, 2)))

  test("logical copy dispatches once for every storage dtype") {
    assertDType(List(false, true, true, false, true, false))
    assertDType(List[Byte](0, 1, -2, 3, -4, 5))
    assertDType(List[Short](0, 1, -2, 3, -4, 5))
    assertDType(List(0, 1, -2, 3, -4, 5))
    assertDType(List[Long](0L, 1L, -2L, 3L, -4L, 5L))
    assertDType(List[Float](0.0f, 1.5f, -2.0f, 3.25f, -4.5f, 5.0f))
    assertDType(List(0.0, 1.5, -2.0, 3.25, -4.5, 5.0))
  }

  test("copy covers scalar, empty, broadcast, and general-rank layouts") {
    assertMaterializes(NDArray.scalar(42))
    assertMaterializes(NDArray.zeros[Int](0, 3).transpose)

    val row = NDArray.fromSeq(Shape(1, 3), Seq(2, 4, 6))
    assertMaterializes(row.broadcastTo(Shape(4, 3)))

    val cube = NDArray.tabulate[Int](2, 3, 4)((i, j, k) => i * 100 + j * 10 + k)
    assertMaterializes(cube.permuteAxes(2, 0, 1).reverse(2))
  }

  test("freezeCopy uses logical order and isolates later mutation") {
    val mutable = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j).mutableCopy
    val view = mutable.transpose.reverse(0).slice(1, Slice(0, 3, 2))
    val expected = values(view.freezeCopy())
    val frozen = view.freezeCopy()
    mutable.fill(-1)
    assertEquals(values(frozen), expected)
    assert(frozen.isContiguous)
  }
