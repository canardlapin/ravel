package ravel

import munit.FunSuite
import ravel.DType.given

final class DTypeCastSuite extends FunSuite:
  test("the primitive dtype family and arithmetic capabilities are closed as specified") {
    summon[DType[Boolean]]
    summon[IntegralDType[Byte]]
    summon[IntegralDType[Short]]
    summon[ArithmeticDType[Int]]
    summon[ArithmeticDType[Long]]
    summon[ArithmeticDType[Float]]
    summon[ArithmeticDType[Double]]
    assert(
      compileErrors(
        "import ravel.*; import ravel.DType.given; summon[ArithmeticDType[Byte]]"
      ).nonEmpty
    )
    assert(
      compileErrors(
        "import ravel.*; import ravel.DType.given; summon[NumericDType[Boolean]]"
      ).nonEmpty
    )
  }

  test("integral widening and narrowing follow fixed-width semantics") {
    val bytes = NDArray.fromSeq(Shape(4), Seq[Byte](-1, 0, 1, 127))
    assertEquals(bytes.cast[Int].elementsIterator.toList, List(-1, 0, 1, 127))
    val ints = NDArray.fromSeq(Shape(4), Seq(255, 256, -1, Int.MinValue))
    assertEquals(
      ints.cast[Byte].elementsIterator.toList,
      List((-1).toByte, 0.toByte, (-1).toByte, 0.toByte)
    )
  }

  test("floating to integral casts truncate, clamp, and map NaN to zero") {
    val values = NDArray.fromSeq(
      Shape(7),
      Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity, 3.9, -3.9, 1e100, -1e100)
    )
    assertEquals(
      values.cast[Int].elementsIterator.toList,
      List(0, Int.MaxValue, Int.MinValue, 3, -3, Int.MaxValue, Int.MinValue)
    )
    assertEquals(
      values.cast[Long].elementsIterator.toList,
      List(0L, Long.MaxValue, Long.MinValue, 3L, -3L, Long.MaxValue, Long.MinValue)
    )
  }

  test("Double to Float preserves signed zero and NaN classification") {
    val values = NDArray
      .fromSeq(
        Shape(4),
        Seq(-0.0, 0.0, Double.NaN, Double.MaxValue)
      )
      .cast[Float]
    val converted = values.elementsIterator.toList
    assertEquals(
      java.lang.Float.floatToRawIntBits(converted(0)),
      java.lang.Float.floatToRawIntBits(-0.0f)
    )
    assertEquals(
      java.lang.Float.floatToRawIntBits(converted(1)),
      java.lang.Float.floatToRawIntBits(0.0f)
    )
    assert(converted(2).isNaN)
    assertEquals(converted(3), Float.PositiveInfinity)
  }
