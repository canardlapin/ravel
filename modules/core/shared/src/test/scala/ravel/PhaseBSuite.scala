package ravel

import munit.FunSuite
import ravel.DType.given

final class PhaseBSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("same-shape contiguous multiply and subtract match eager semantics") {
    val left = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val right = NDArray.tabulate[Int](2, 3)((_, j) => j + 1)
    assertEquals(values(left * right), values(left).zip(values(right)).map(_ * _))
    assertEquals(values(left - right), values(left).zip(values(right)).map(_ - _))
  }

  test("scalar arithmetic on contiguous arrays preserves values") {
    val source = NDArray.fromSeq(Shape(4), Seq(1.0, 2.0, 3.0, 4.0))
    assertEquals(values(source + 2.0), List(3.0, 4.0, 5.0, 6.0))
    assertEquals(values(source * 0.5), List(0.5, 1.0, 1.5, 2.0))
  }

  test("cast specializes without changing conversion semantics") {
    val ints = NDArray.fromSeq(Shape(3), Seq(1, -2, 300))
    assertEquals(values(ints.cast[Double]), List(1.0, -2.0, 300.0))
    assertEquals(values(ints.cast[Byte]), List(1.toByte, (-2).toByte, 300.toByte))
    val floats = NDArray.fromSeq(Shape(3), Seq(1.9f, Float.NaN, Float.PositiveInfinity))
    assertEquals(values(floats.cast[Int]), List(1, 0, Int.MaxValue))
    assert(floats.cast[Float].sameElementsBits(floats))
  }

  test("expanded Into kernels agree with eager results") {
    val left = NDArray.tabulate[Double](2, 3)((i, j) => i + j.toDouble)
    val right = NDArray.fromSeq(Shape(3), Seq(2.0, 3.0, 4.0))
    val output = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 3))

    kernel.subtractInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left - right))
    kernel.divideInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left / right))
    kernel.minimumInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left.minimum(right)))
    kernel.maximumInto(left, right, output)
    assertEquals(values(output.freezeCopy()), values(left.maximum(right)))

    kernel.negateInto(left, output)
    assertEquals(values(output.freezeCopy()), values(-left))
    kernel.absInto(left - 10.0, output)
    assertEquals(values(output.freezeCopy()), values((left - 10.0).abs))

    kernel.addScalarInto(left, 1.0, output)
    assertEquals(values(output.freezeCopy()), values(left + 1.0))
    kernel.multiplyScalarInto(left, 2.0, output)
    assertEquals(values(output.freezeCopy()), values(left * 2.0))
    kernel.divideScalarInto(left, 2.0, output)
    assertEquals(values(output.freezeCopy()), values(left / 2.0))

    val ints = NDArray.fromSeq(Shape(3), Seq(7, -7, 8))
    val intOut = MutableNDArray.zeros[Int, Rank[1]](Shape(3))
    kernel.quotInto(ints, NDArray.fill(Shape(3), 2), intOut)
    assertEquals(values(intOut.freezeCopy()), values(ints.quot(2)))
    kernel.quotScalarInto(ints, 2, intOut)
    assertEquals(values(intOut.freezeCopy()), values(ints.quot(2)))
  }

  test("NDArray.build fills by linear index and seals the builder") {
    val built = NDArray.build[Int, Rank[2]](Shape(2, 3)) { builder =>
      var i = 0
      while i < builder.size do
        builder.writeLinear(i, i * 3)
        i += 1
    }
    assertEquals(values(built), List(0, 3, 6, 9, 12, 15))
    intercept[IllegalStateException] {
      NDArray.build[Int, Rank[1]](Shape(2)) { builder =>
        builder.writeLinear(0, 1)
        builder.seal()
        builder.writeLinear(1, 2)
      }
    }
  }

  test("layout-carried shape identity is reused on owned arrays") {
    val array = NDArray.zeros[Int](2, 3)
    assert(array.shape eq array.layout.shapeValue)
    val copy = array.copy
    assertEquals(copy.shape, array.shape)
  }
