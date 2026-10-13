package ravel

import ravel.DType.given
import ravel.internal.ProbeApi

class BulkPrimitiveSuite extends munit.FunSuite:
  test("Double copy owns canonical storage and preserves values and signed zero"):
    val input = Array(1.25, -0.0, Double.NaN, Double.PositiveInfinity)
    val result = NDArray.fromArrayCopy(Shape(2, 2), input)
    input(0) = 99.0
    assertEquals(result(0, 0), 1.25)
    assertEquals(java.lang.Double.doubleToRawLongBits(result(0, 1)), Long.MinValue)
    assert(result(1, 0).isNaN)
    assertEquals(result(1, 1), Double.PositiveInfinity)
    assert(result.isCanonicalLayout && result.isWholeBuffer)

  test("Float copy owns storage without widening and rejects wrong lengths"):
    val input = Array(1.25f, -0.0f, Float.NaN, Float.NegativeInfinity)
    val result = NDArray.fromArrayCopy(Shape(4), input)
    input(0) = 99.0f
    assertEquals(result(0), 1.25f)
    assertEquals(java.lang.Float.floatToRawIntBits(result(1)), Int.MinValue)
    assert(result(2).isNaN)
    assertEquals(result(3), Float.NegativeInfinity)
    intercept[ShapeMismatch](NDArray.fromArrayCopy(Shape(3), input))
    intercept[ShapeMismatch](NDArray.fromArrayCopy(Shape(5), input))
    intercept[ShapeMismatch](NDArray.fromArrayCopy(Shape.scalar, Array.emptyDoubleArray))

  test("empty and scalar copies and reductions retain their identities"):
    assert(NDArray.fromArrayCopy(Shape(0, 3), Array.emptyDoubleArray).allFinite)
    assert(NDArray.fromArrayCopy(Shape(0), Array.emptyFloatArray).allFinite)
    assert(NDArray.fromArrayCopy(Shape.scalar, Array(-0.0)).allFinite)
    assert(!NDArray.fromArrayCopy(Shape.scalar, Array(Float.PositiveInfinity)).allFinite)

  test("finite scans agree with logical iteration across packed and sparse layouts"):
    for bad <- Vector(Double.NaN, Double.NegativeInfinity, Double.PositiveInfinity) do
      for position <- 0 until 24 do
        val input = Array.tabulate(24)(_.toDouble)
        input(position) = bad
        val array = NDArray.fromArrayCopy(Shape(4, 6), input)
        val views = Vector(
          array,
          array.transpose,
          array.reverse(0),
          array.reverse(1).transpose,
          array.narrow(0, 1, 2),
          array.narrow(1, 1, 3),
          array.slice(1, 0 until 6 by 2),
          array.slice(0, 0 until 4 by 2).reverse(1),
          array.narrow(0, 2, 1).broadcastTo(Shape(5, 6)),
          array.narrow(1, 3, 1).broadcastTo(Shape(4, 7))
        )
        for view <- views do
          assertEquals(view.allFinite, view.elementsIterator.forall(_.isFinite))
          val floats = view.cast[Float]
          assertEquals(floats.allFinite, floats.elementsIterator.forall(_.isFinite))
          val floatView = floats.reverse(1).transpose
          assertEquals(floatView.allFinite, floatView.elementsIterator.forall(_.isFinite))

  test("borrowed and mutable scans observe current storage and retain no result buffer"):
    val owned = NDArray.fromArrayCopy(Shape(2, 2), Array(1.0, 2.0, 3.0, 4.0))
    val borrowed = new BorrowedNDArray(owned).transpose
    val mutable = owned.mutableCopy
    assert(borrowed.allFinite && mutable.allFinite)
    ProbeApi.setDouble(owned.storage, 1, Double.NaN)
    assert(!borrowed.allFinite)
    assert(mutable.allFinite)
    mutable(0, 0) = Double.PositiveInfinity
    assert(!mutable.allFinite)
    mutable(0, 0) = 2.0
    assert(mutable.allFinite)
