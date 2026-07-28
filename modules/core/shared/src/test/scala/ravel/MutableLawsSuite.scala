package ravel

import munit.FunSuite
import ravel.DType.given
import scala.collection.mutable.ArrayBuffer

final class MutableLawsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("mutableCopy owns independent storage and updates only one logical position") {
    val source = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j)
    val mutable = source.mutableCopy
    mutable(1, 2) = 99
    assertEquals(source(1, 2), 12)
    val changed = values(mutable.freezeCopy())
    assertEquals(changed, List(0, 1, 2, 3, 10, 11, 99, 13, 20, 21, 22, 23))
  }

  test("injectivity-preserving mutable views address distinct cells") {
    val mutable = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j).mutableCopy
    val view =
      mutable.transpose
        .reverse(0)
        .slice(1, Slice(0, 3, 2))
        .newAxis(-1)
        .squeeze(-1)
    var next = 100
    var i = 0
    while i < view.shape(0) do
      var j = 0
      while j < view.shape(1) do
        view(i, j) = next
        next += 1
        j += 1
      i += 1
    val result = mutable.freezeCopy()
    assertEquals(values(view.freezeCopy()).distinct.size, view.size)
    assertEquals(result.size, 12)
  }

  test("mutable reshape preserves injectivity and mutable broadcasting is absent") {
    val mutable = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j).mutableCopy
    val flat = mutable.reshapeView(Shape(6))
    flat(4) = 77
    assertEquals(mutable(1, 1), 77)
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Int](1).mutableCopy.broadcastTo(Shape(3))
    """).nonEmpty)
  }

  test("freezeCopy isolates all later mutation") {
    val mutable = NDArray.fromSeq(Shape(3), Seq(1, 2, 3)).mutableCopy
    val frozen = mutable.freezeCopy()
    mutable(0) = 100
    mutable.fill(7)
    assertEquals(values(frozen), List(1, 2, 3))
  }

  test("elementsIterator yields values lazily in logical order") {
    val array = NDArray.fromSeq(Shape(2, 2), Seq(1, 2, 3, 4)).transpose
    val iterator = array.elementsIterator
    assertEquals(iterator.next(), 1)
    assertEquals(iterator.next(), 3)
    assertEquals(iterator.next(), 2)
    assertEquals(iterator.next(), 4)
    assert(!iterator.hasNext)
  }

  test("sameElements fails fast without materializing both arrays") {
    val left = NDArray.fromSeq(Shape(4), Seq(1, 2, 3, 4))
    val right = NDArray.fromSeq(Shape(4), Seq(1, 9, 3, 4))
    assert(!left.sameElements(right))
    assert(left.sameElements(left.copy))
  }

  test("assign and in-place scalar updates mutate without intermediate owned copies") {
    val mutable = NDArray.zeros[Int](2, 3).mutableCopy
    val transposed = mutable.transpose
    transposed.assign(NDArray.fromSeq(Shape(3, 2), Seq(1, 2, 3, 4, 5, 6)))
    assertEquals(values(mutable.freezeCopy()), List(1, 3, 5, 2, 4, 6))
    mutable.addInPlace(10)
    assertEquals(values(mutable.freezeCopy()), List(11, 13, 15, 12, 14, 16))
    mutable.multiplyInPlace(2)
    assertEquals(values(mutable.freezeCopy()), List(22, 26, 30, 24, 28, 32))
    mutable.subtractInPlace(2)
    assertEquals(values(mutable.freezeCopy()), List(20, 24, 28, 22, 26, 30))
  }

  test("ordinary immutable arrays expose no update method") {
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      val x = NDArray.zeros[Int](2)
      x(0) = 1
    """).nonEmpty)
  }

  test("specialized reusable-output kernels agree with eager results") {
    val left = NDArray.tabulate[Double](2, 3)((i, j) => i + j.toDouble)
    val right = NDArray.fromSeq(Shape(3), Seq(2.0, 3.0, 4.0))
    val output = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 3))
    kernel.addInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left + right))
    kernel.multiplyInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left * right))
  }

  test("callback Into kernels preserve logical order and stop on exceptions") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j).transpose
    val output = MutableNDArray.zeros[Int, Rank[2]](Shape(3, 2))
    val observed = ArrayBuffer.empty[Int]
    kernel.mapInto(source, output) { value =>
      observed += value
      value * 2
    }
    assertEquals(observed.toList, values(source))
    assertEquals(values(output.freezeCopy()), values(source).map(_ * 2))

    var calls = 0
    intercept[IllegalStateException] {
      kernel.mapInto(source, output) { value =>
        calls += 1
        if calls == 3 then throw new IllegalStateException("stop")
        value
      }
    }
    assertEquals(calls, 3)
  }

  test("Into kernels validate destination shape and whole-layout requirement") {
    val source = NDArray.zeros[Int](2, 3)
    val wrong = MutableNDArray.zeros[Int, Rank[1]](Shape(6))
    intercept[ShapeMismatch](kernel.addInto(source, source, wrong))

    val parent = MutableNDArray.zeros[Int, Rank[2]](Shape(2, 3))
    val row = parent.select(0, 0)
    intercept[NonContiguousLayout] {
      kernel.addInto(
        NDArray.zeros[Int](3),
        NDArray.zeros[Int](3),
        row
      )
    }
  }
