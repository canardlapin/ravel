package ravel

import munit.FunSuite
import ravel.DType.given
import scala.collection.mutable.ArrayBuffer

final class MutableLawsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  private def assertStridedAssign[A](data: Seq[A])(using DType[A]): Unit =
    val source = NDArray.fromSeq(Shape(2, 3), data).transpose.reverse(0)
    val destination = NDArray.zeros[A](2, 3).mutableCopy.transpose.reverse(0)
    destination.assign(source)
    assertEquals(values(destination.freezeCopy()), values(source))

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

  test("strided assign dispatches every primitive storage without changing logical order") {
    assertStridedAssign[Boolean](Seq(true, false, true, false, false, true))
    assertStridedAssign[Byte](Seq[Byte](1, -2, 3, 4, 5, 6))
    assertStridedAssign[Short](Seq[Short](1, -2, 3, 4, 5, 6))
    assertStridedAssign[Int](Seq(1, -2, 3, 4, 5, 6))
    assertStridedAssign[Long](Seq(1L, -2L, 3L, 4L, 5L, 6L))
    assertStridedAssign[Float](Seq(0.1f, -2.25f, 3.5f, 4.0f, 5.0f, 6.0f))
    assertStridedAssign[Double](Seq(0.1, -2.25, 3.5, 4.0, 5.0, 6.0))
  }

  test("primitive in-place loops preserve negative-stride locality, rounding, and overflow") {
    val mutable =
      NDArray.tabulate[Double](3, 6)((row, column) => row * 10.0 + column).mutableCopy
    val reversedStride =
      mutable.reverse(1).slice(1, Slice(0, 6, 2))
    reversedStride.addInPlace(0.5)
    val expected =
      NDArray.tabulate[Double](3, 6)((row, column) =>
        val initial = row * 10.0 + column
        if column % 2 == 1 then initial + 0.5 else initial
      )
    assertEquals(values(mutable.freezeCopy()), values(expected))

    val floats = NDArray.fromSeq(Shape(3), Seq(0.1f, -2.25f, Float.MaxValue)).mutableCopy
    floats.multiplyInPlace(3.0f)
    assertEquals(
      values(floats.freezeCopy()).map(java.lang.Float.floatToRawIntBits),
      Seq(0.3f, -6.75f, Float.PositiveInfinity).map(java.lang.Float.floatToRawIntBits).toList
    )

    val ints = NDArray.fromSeq(Shape(3), Seq(Int.MaxValue, -2, 3)).mutableCopy
    ints.addInPlace(1)
    assertEquals(values(ints.freezeCopy()), List(Int.MinValue, -1, 4))

    val longs = NDArray.fromSeq(Shape(2), Seq(Long.MaxValue, Long.MinValue)).mutableCopy
    longs.addInPlace(1L)
    assertEquals(values(longs.freezeCopy()), List(Long.MinValue, Long.MinValue + 1L))
  }

  test("dense mutable permutations and reversals use every scalar operation exactly once") {
    val ints =
      NDArray
        .tabulate[Int](2, 3, 4)((plane, row, column) => plane * 100 + row * 10 + column + 8)
        .mutableCopy
    val intView =
      ints
        .permuteAxes(1, 2, 0)
        .reverse(0)
        .reverse(2)
        .newAxis(1)
        .squeeze(1)
    intView.addInPlace(4)
    intView.subtractInPlace(2)
    intView.multiplyInPlace(6)
    intView.quotInPlace(3)
    val expectedInts =
      NDArray.tabulate[Int](2, 3, 4)((plane, row, column) =>
        ((plane * 100 + row * 10 + column + 8 + 4 - 2) * 6) / 3
      )
    assertEquals(values(ints.freezeCopy()), values(expectedInts))

    val longs =
      NDArray.fromSeq(Shape(2, 3), Seq(8L, 10L, 12L, 14L, 16L, 18L)).mutableCopy
    val longView = longs.transpose.reverse(0).reverse(1)
    longView.addInPlace(4L)
    longView.subtractInPlace(2L)
    longView.multiplyInPlace(6L)
    longView.quotInPlace(3L)
    assertEquals(values(longs.freezeCopy()), List(20L, 24L, 28L, 32L, 36L, 40L))

    val floats =
      NDArray.fromSeq(Shape(2, 3), Seq(1.0f, -2.0f, 3.5f, -4.5f, 8.0f, 12.0f)).mutableCopy
    val floatView = floats.transpose.reverse(0).reverse(1)
    floatView.addInPlace(0.5f)
    floatView.subtractInPlace(0.25f)
    floatView.multiplyInPlace(2.0f)
    floatView.divideInPlace(4.0f)
    val expectedFloats =
      Seq(1.0f, -2.0f, 3.5f, -4.5f, 8.0f, 12.0f)
        .map(value => (((value + 0.5f).toFloat - 0.25f).toFloat * 2.0f).toFloat / 4.0f)
        .map(_.toFloat)
    assertEquals(
      values(floats.freezeCopy()).map(java.lang.Float.floatToRawIntBits),
      expectedFloats.map(java.lang.Float.floatToRawIntBits).toList
    )

    val doubles =
      NDArray.fromSeq(Shape(2, 3), Seq(1.0, -2.0, 3.5, -4.5, 8.0, 12.0)).mutableCopy
    val doubleView = doubles.transpose.reverse(0).reverse(1)
    doubleView.addInPlace(0.5)
    doubleView.subtractInPlace(0.25)
    doubleView.multiplyInPlace(2.0)
    doubleView.divideInPlace(4.0)
    val expectedDoubles =
      Seq(1.0, -2.0, 3.5, -4.5, 8.0, 12.0)
        .map(value => ((value + 0.5 - 0.25) * 2.0) / 4.0)
    assertEquals(
      values(doubles.freezeCopy()).map(java.lang.Double.doubleToRawLongBits),
      expectedDoubles.map(java.lang.Double.doubleToRawLongBits).toList
    )
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

  test("mutable arrays are readable operands and support nonaliasing ping-pong kernels") {
    val left = NDArray.tabulate[Int](2, 3)((row, column) => row * 10 + column).mutableCopy
    val right = NDArray.fill(Shape(2, 3), 2).mutableCopy

    val eager: Array2[Int] = left + right
    assertEquals(values(eager), List(2, 3, 4, 12, 13, 14))
    assertEquals(values(left.map(_ * 2)), List(0, 2, 4, 20, 22, 24))
    assertEquals(values(left > right), List(false, false, false, true, true, true))
    assertEquals(left.sum, 36)
    assertEquals(values(left.cast[Double]), List(0.0, 1.0, 2.0, 10.0, 11.0, 12.0))
    assertEquals(
      left.convert[Double]().map(values),
      Right(List(0.0, 1.0, 2.0, 10.0, 11.0, 12.0))
    )

    val firstOutput = MutableNDArray.zeros[Int, Rank[2]](Shape(2, 3))
    kernel.addInto(left, right, firstOutput)
    kernel.multiplyInto(firstOutput, left, right)
    assertEquals(values(right.freezeCopy()), List(0, 3, 8, 120, 143, 168))
  }

  test("mutable read sources reject destination aliases before mutation") {
    val left = NDArray.fromSeq(Shape(3), Seq(1, 2, 3)).mutableCopy
    val right = NDArray.fromSeq(Shape(3), Seq(4, 5, 6)).mutableCopy
    val before = values(left.freezeCopy())

    intercept[IllegalArgumentException](kernel.addInto(left, right, left))
    assertEquals(values(left.freezeCopy()), before)
    intercept[IllegalArgumentException](kernel.mapInto(left, left)(identity))
    assertEquals(values(left.freezeCopy()), before)
    intercept[IllegalArgumentException](left.assign(left))
    assertEquals(values(left.freezeCopy()), before)
  }

  test("mutable reshapeCopy and noncontiguous reshape materialize once and preserve order") {
    val canonical = NDArray.tabulate[Int](2, 3)((row, column) => row * 10 + column).mutableCopy
    val canonicalCopy = canonical.reshapeCopy(Shape(3, 2))
    assertEquals(values(canonicalCopy.freezeCopy()), List(0, 1, 2, 10, 11, 12))
    canonicalCopy(0, 0) = 99
    assertEquals(canonical(0, 0), 0)

    val transposed = canonical.transpose
    val reshaped = transposed.reshape(Shape(6))
    assertEquals(values(reshaped.freezeCopy()), List(0, 10, 1, 11, 2, 12))
    reshaped(0) = 77
    assertEquals(canonical(0, 0), 0)
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
