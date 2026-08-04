package ravel

import munit.FunSuite
import ravel.DType.given

final class BorrowedEqualitySuite extends FunSuite:
  test("borrowed ownership cannot be erased by subtyping or implicit conversion") {
    val subtypeErrors = compileErrors("""
      import ravel.*
      val owned: NDArray[Double, Rank[1]] =
        null.asInstanceOf[BorrowedNDArray[Double, Rank[1]]]
    """)
    val conversionErrors = compileErrors("""
      import ravel.*
      val borrowed = null.asInstanceOf[BorrowedNDArray[Double, Rank[1]]]
      val owned: NDArray[Double, Rank[1]] = borrowed
    """)
    assert(subtypeErrors.nonEmpty)
    assert(conversionErrors.nonEmpty)
  }

  test("borrowed structural views retain borrowed provenance") {
    val borrowed = null.asInstanceOf[BorrowedNDArray[Double, Rank[2]]]
    compileErrors("""
      import ravel.*
      val borrowed = null.asInstanceOf[BorrowedNDArray[Double, Rank[2]]]
      val sliced: BorrowedNDArray[Double, Rank[2]] =
        borrowed.slice(1, Slice(0, 1))
      val row: BorrowedNDArray[Double, Rank[1]] =
        borrowed.select(0, 0)
      val transposed: BorrowedNDArray[Double, Rank[2]] =
        borrowed.transpose
    """) match
      case "" => ()
      case errors => fail(errors)
    assert(borrowed == null)
  }

  test("borrowed computations expose owned result types") {
    compileErrors("""
      import ravel.*
      import ravel.DType.given
      val borrowed = null.asInstanceOf[BorrowedNDArray[Double, Rank[2]]]
      val mapped: NDArray[Double, Rank[2]] = borrowed.map(identity)
      val added: NDArray[Double, Rank[2]] = borrowed + 1.0
      val reduced: NDArray[Double, Rank[1]] = borrowed.sum(0)
      val copied: NDArray[Double, Rank[2]] = borrowed.copy
    """) match
      case "" => ()
      case errors => fail(errors)
  }

  test("equals remains reference equality while explicit equality is extensional") {
    val left = NDArray.fromSeq(Shape(2), Seq(0.0, -0.0))
    val same = NDArray.fromSeq(Shape(2), Seq(0.0, -0.0))
    val signedZeroChanged = NDArray.fromSeq(Shape(2), Seq(-0.0, 0.0))
    assert(!(left == same))
    assert(left.sameElements(same))
    assert(left.sameElements(signedZeroChanged))
    assert(!left.sameElementsBits(signedZeroChanged))
  }

  test("bit equality can compare a stable NaN representation") {
    val nan1 = java.lang.Double.longBitsToDouble(0x7ff8000000000001L)
    val left = NDArray.fromSeq(Shape(1), Seq(nan1))
    val sameBits = NDArray.fromSeq(Shape(1), Seq(nan1))
    assert(!left.sameElements(sameBits))
    assert(left.sameElementsBits(sameBits))
  }

  test("allClose requires explicit nonnegative tolerances") {
    val left = NDArray.fromSeq(Shape(2), Seq(1.0, 100.0))
    val close = NDArray.fromSeq(Shape(2), Seq(1.00001, 100.01))
    assert(left.allClose(close, relativeTolerance = 0.001, absoluteTolerance = 0.0))
    assert(!left.allClose(close, relativeTolerance = 0.000001, absoluteTolerance = 0.0))
    intercept[IllegalArgumentException] {
      left.allClose(close, relativeTolerance = -1.0, absoluteTolerance = 0.0)
    }
  }

  test("preview is bounded by each axis and total emitted elements") {
    val array = NDArray.tabulate[Int](8, 8)((row, column) => row * 100 + column)
    val rendered = array.toString
    assert(rendered.startsWith("NDArray[Int](shape = (8, 8), contiguous = true"))
    assert(rendered.contains("…"))
    assert(!rendered.contains("707"))
    assert(rendered.length < 300)
  }

  test("dtype formatting renders logical unsigned magnitudes and floating edge values") {
    val uint8 = NDArray.fromSeq(
      Shape(3),
      Seq(UInt8.unsafe(0), UInt8.unsafe(128), UInt8.unsafe(255))
    )
    val uint16 = NDArray.scalar(UInt16.unsafe(65535))

    assert(uint8.toString.contains("values = [0, 128, 255]"))
    assert(uint16.toString.contains("values = 65535"))
    assertEquals(summon[DType[Float]].format(Float.NegativeInfinity), "-Infinity")
    assertEquals(summon[DType[Float]].format(-0.0f), "-0.0")
    assertEquals(summon[DType[Double]].format(Double.NaN), "NaN")
    assertEquals(summon[DType[Double]].format(Double.PositiveInfinity), "Infinity")
    assertEquals(summon[DType[Double]].format(-0.0), "-0.0")
  }

  test("unary plus always returns an isolated owned copy") {
    val owned = NDArray.fromSeq(Shape(3), Seq(1.0, 2.0, 3.0))
    val ownedCopy = +owned
    assert(!(ownedCopy eq owned))
    assert(ownedCopy.sameElements(owned))

    val mutable = owned.mutableCopy
    val mutableCopy = +mutable
    mutable.update(0, 99.0)
    assertEquals(mutableCopy.elementsIterator.toList, List(1.0, 2.0, 3.0))
  }
