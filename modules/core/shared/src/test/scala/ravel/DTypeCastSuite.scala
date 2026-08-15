package ravel

import munit.FunSuite
import ravel.DType.given

final class DTypeCastSuite extends FunSuite:
  test("the primitive dtype family and arithmetic capabilities are closed as specified") {
    summon[DType[Boolean]]
    summon[IntegralDType[Byte]]
    summon[IntegralDType[Short]]
    summon[IntegralDType[UInt8]]
    summon[IntegralDType[UInt16]]
    summon[OrderedDType[UInt8]]
    summon[OrderedDType[UInt16]]
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
        "import ravel.*; import ravel.DType.given; summon[ArithmeticDType[UInt8]]"
      ).nonEmpty
    )
    assert(
      compileErrors(
        "import ravel.*; import ravel.DType.given; summon[ArithmeticDType[UInt16]]"
      ).nonEmpty
    )
    assert(
      compileErrors(
        "import ravel.*; import ravel.DType.given; summon[NumericDType[Boolean]]"
      ).nonEmpty
    )
  }

  test("unsigned construction rejects out-of-range values") {
    assertEquals(UInt8.fromInt(255), Right(UInt8.unsafe(255)))
    assertEquals(UInt8.fromInt(-1), Left(UnsignedRangeError("UInt8", -1L, 255L)))
    assertEquals(UInt8.fromInt(256), Left(UnsignedRangeError("UInt8", 256L, 255L)))
    assertEquals(UInt16.fromLong(65535L), Right(UInt16.unsafe(65535)))
    assertEquals(UInt16.fromLong(65536L), Left(UnsignedRangeError("UInt16", 65536L, 65535L)))
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

  test("checked conversion defaults to nearest-even rounding and Reject") {
    val values =
      NDArray.fromSeq(Shape(4), Seq(2.5, 3.5, -2.5, -3.5))

    val converted = values.convert[Int]().toOption.get

    assertEquals(converted.elementsIterator.toList, List(2, 4, -2, -4))

    val overflow = NDArray
      .fromSeq(Shape(3), Seq(1.0, 128.0, 2.0))
      .convert[Byte]()
    assertEquals(
      overflow,
      Left(ConversionError.OutOfRange(1, "Double", "Byte"))
    )
  }

  test("checked conversion applies every rounding and overflow policy") {
    val values = NDArray.fromSeq(Shape(2), Seq(1.9, -1.9))

    assertEquals(
      values
        .convert[Int](ConversionPolicy(Rounding.TowardZero, Overflow.Reject))
        .toOption
        .get
        .elementsIterator
        .toList,
      List(1, -1)
    )
    assertEquals(
      values
        .convert[Int](ConversionPolicy(Rounding.Floor, Overflow.Reject))
        .toOption
        .get
        .elementsIterator
        .toList,
      List(1, -2)
    )
    assertEquals(
      values
        .convert[Int](ConversionPolicy(Rounding.Ceiling, Overflow.Reject))
        .toOption
        .get
        .elementsIterator
        .toList,
      List(2, -1)
    )

    val outOfRange = NDArray.fromSeq(Shape(3), Seq(-200.0, 5.6, 300.0))
    assertEquals(
      outOfRange
        .convert[Byte](ConversionPolicy(Rounding.NearestEven, Overflow.Clamp))
        .toOption
        .get
        .elementsIterator
        .toList,
      List(Byte.MinValue, 6.toByte, Byte.MaxValue)
    )
    assertEquals(
      NDArray
        .fromSeq(Shape(1), Seq(130.0))
        .convert[Byte](ConversionPolicy(Rounding.TowardZero, Overflow.Wrap))
        .toOption
        .get(0),
      (-126).toByte
    )
  }

  test("checked conversion preserves lossless integral widening round-trips") {
    val source =
      NDArray.fromSeq(Shape(4), Seq[Byte](Byte.MinValue, -1, 0, Byte.MaxValue))

    val roundTrip =
      source
        .convert[Int]()
        .toOption
        .get
        .convert[Byte]()
        .toOption
        .get

    assert(roundTrip.sameElements(source))
  }

  test("checked conversion rejects non-finite floating-to-integral values") {
    val values =
      NDArray.fromSeq(Shape(3), Seq(1.0, Double.NaN, 3.0))

    assertEquals(
      values.convert[Int](),
      Left(ConversionError.NonFiniteToIntegral(1, "Double", "Int"))
    )
  }

  test("unsigned casts widen by magnitude and wrap on narrowing") {
    val bytes =
      NDArray.fromSeq(Shape(3), Seq(UInt8.unsafe(255), UInt8.unsafe(128), UInt8.unsafe(0)))
    assertEquals(bytes.cast[Int].elementsIterator.toList, List(255, 128, 0))
    assertEquals(bytes.cast[Long].elementsIterator.toList, List(255L, 128L, 0L))

    val ints = NDArray.fromSeq(Shape(4), Seq(255, 256, -1, 40000))
    assertEquals(
      ints.cast[UInt8].elementsIterator.toList,
      List(UInt8.unsafe(255), UInt8.unsafe(0), UInt8.unsafe(255), UInt8.unsafe(64))
    )
    assertEquals(
      ints.cast[UInt16].elementsIterator.toList,
      List(UInt16.unsafe(255), UInt16.unsafe(256), UInt16.unsafe(65535), UInt16.unsafe(40000))
    )
  }

  test("checked conversion into unsigned applies Reject Clamp and Wrap") {
    val values = NDArray.fromSeq(Shape(3), Seq(-1.0, 200.0, 300.0))
    assertEquals(
      values.convert[UInt8](),
      Left(ConversionError.OutOfRange(0, "Double", "UInt8"))
    )
    assertEquals(
      values
        .convert[UInt8](ConversionPolicy(Rounding.NearestEven, Overflow.Clamp))
        .toOption
        .get
        .elementsIterator
        .toList,
      List(UInt8.unsafe(0), UInt8.unsafe(200), UInt8.unsafe(255))
    )
    assertEquals(
      NDArray
        .fromSeq(Shape(1), Seq(300.0))
        .convert[UInt8](ConversionPolicy(Rounding.TowardZero, Overflow.Wrap))
        .toOption
        .get(0),
      UInt8.unsafe(44)
    )
  }

  test("unsigned min max and comparisons use magnitude order") {
    val values =
      NDArray.fromSeq(
        Shape(4),
        Seq(UInt8.unsafe(1), UInt8.unsafe(255), UInt8.unsafe(0), UInt8.unsafe(128))
      )
    assertEquals(values.min, UInt8.unsafe(0))
    assertEquals(values.max, UInt8.unsafe(255))
    assertEquals(values.argMin, 2)
    assertEquals(values.argMax, 1)
    assertEquals(
      (values > UInt8.unsafe(128)).elementsIterator.toList,
      List(false, true, false, false)
    )
    assertEquals(
      values.clip(UInt8.unsafe(10), UInt8.unsafe(200)).elementsIterator.toList,
      List(UInt8.unsafe(10), UInt8.unsafe(200), UInt8.unsafe(10), UInt8.unsafe(128))
    )
  }
