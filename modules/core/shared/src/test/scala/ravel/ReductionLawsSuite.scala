package ravel

import munit.FunSuite
import ravel.DType.given
import ravel.SumAs.given
import ravel.internal.ReductionKernels

final class ReductionLawsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("sum and product identities and fixed-width overflow are deterministic") {
    val emptyInt = NDArray.zeros[Int](0)
    val emptyFloat = NDArray.zeros[Float](0)
    assertEquals(emptyInt.sum, 0)
    assertEquals(emptyInt.product, 1)
    assertEquals(
      java.lang.Float.floatToRawIntBits(emptyFloat.sum),
      java.lang.Float.floatToRawIntBits(0.0f)
    )
    assertEquals(emptyFloat.product, 1.0f)
    assert(emptyFloat.mean.isNaN)
    val overflow = NDArray.fromSeq(Shape(2), Seq(Int.MaxValue, 1))
    assertEquals(overflow.sum, Int.MinValue)
  }

  test("scalar and axis reductions preserve rank and logical order") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    assertEquals(source.sum, 36)
    assertEquals(source.product, 0)
    val columns: Array1[Int] = source.sum(0)
    val rows: Array1[Int] = source.sum(1)
    val kept: Array2[Int] = source.sumKeep(1)
    assertEquals(values(columns), List(10, 12, 14))
    assertEquals(values(rows), List(3, 33))
    assertEquals(kept.shape.toString, "(2, 1)")
    assertEquals(values(kept), List(3, 33))
    assertEquals(values(source.product(0)), List(0, 11, 24))
    assertEquals(values(source.sumAxes(0, 1)), List(36))
    assertEquals(source.sumAxes(0, 1).rank, 0)
  }

  test("ordered reductions define empty, tie, NaN, and signed-zero behavior") {
    val empty = NDArray.zeros[Double](0)
    intercept[EmptyReduction](empty.min)
    intercept[EmptyReduction](empty.max)
    intercept[EmptyReduction](empty.argMin)
    intercept[EmptyReduction](empty.argMax)

    val signed = NDArray.fromSeq(Shape(2), Seq(0.0, -0.0))
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.min),
      java.lang.Double.doubleToRawLongBits(-0.0)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.max),
      java.lang.Double.doubleToRawLongBits(0.0)
    )
    assertEquals(signed.argMin, 0)
    assertEquals(signed.argMax, 0)

    val nan = NDArray.fromSeq(Shape(5), Seq(3.0, Double.NaN, 1.0, Double.NaN, 1.0))
    assert(nan.min.isNaN)
    assert(nan.max.isNaN)
    assertEquals(nan.argMin, 1)
    assertEquals(nan.argMax, 1)

    val ties = NDArray.fromSeq(Shape(4), Seq(2, 1, 1, 3))
    assertEquals(ties.argMin, 1)
    assertEquals(ties.argMax, 3)
  }

  test("axis min/max/arg and empty fibers follow the scalar contract") {
    val source = NDArray.fromSeq(
      Shape(2, 4),
      Seq(3.0, Double.NaN, 1.0, 2.0, 4.0, 5.0, 5.0, -1.0)
    )
    val mins: Array1[Double] = source.min(1)
    val maxes: Array1[Double] = source.max(1)
    assert(values(mins).head.isNaN)
    assert(values(maxes).head.isNaN)
    assertEquals(values(mins).last, -1.0)
    assertEquals(values(maxes).last, 5.0)
    assertEquals(values(source.argMin(1)), List(1, 3))
    assertEquals(values(source.argMax(1)), List(1, 1))

    val emptyFibers = NDArray.zeros[Double](3, 0)
    assertEquals(values(emptyFibers.sum(1)), List(0.0, 0.0, 0.0))
    assert(values(emptyFibers.mean(1)).forall(_.isNaN))
    intercept[EmptyReduction](emptyFibers.min(1))
  }

  test("block-pairwise floating sums use the documented schedule") {
    val data = Vector.tabulate(513) { index =>
      if index % 3 == 0 then 1.0e10f
      else if index % 3 == 1 then 1.0f
      else -1.0e10f
    }
    val source = NDArray.fromSeq(Shape(data.size), data)
    assertEquals(
      java.lang.Float.floatToRawIntBits(source.sum),
      java.lang.Float.floatToRawIntBits(referenceFloatPairwise(data))
    )
    assertEqualsDouble(source.sumAs[Double], referenceDoublePairwise(data.map(_.toDouble)), 0.0)
    assertEquals(
      java.lang.Float.floatToRawIntBits(source.mean),
      java.lang.Float.floatToRawIntBits(
        (referenceDoublePairwise(data.map(_.toDouble)) / data.size.toDouble).toFloat
      )
    )
  }

  test("extensionally equal strided and contiguous views have identical floating schedules") {
    val source = NDArray.tabulate[Double](17, 19)((i, j) => math.sin(i * 0.2 + j))
    val strided = source.transpose.reverse(0)
    val contiguous = strided.contiguous
    assertEquals(
      java.lang.Double.doubleToRawLongBits(strided.sum),
      java.lang.Double.doubleToRawLongBits(contiguous.sum)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(strided.mean),
      java.lang.Double.doubleToRawLongBits(contiguous.mean)
    )
  }

  test("only Int-to-Long and Float-to-Double widening sums compile") {
    assertEquals(
      NDArray.fromSeq(Shape(2), Seq(Int.MaxValue, Int.MaxValue)).sumAs[Long],
      2L * Int.MaxValue.toLong
    )
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      import ravel.SumAs.given
      NDArray.zeros[Long](2).sumAs[Double]
    """).nonEmpty)
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Byte](2).sum
    """).nonEmpty)
  }

  test("rank-zero axis reductions cannot compile") {
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.scalar(1).sum(0)
    """).nonEmpty)
  }

  private def referenceFloatPairwise(values: Seq[Float]): Float =
    val blocks =
      values
        .grouped(ReductionKernels.PairwiseBlockSize)
        .map(_.foldLeft(0.0f)((sum, value) => (sum + value).toFloat))
        .toArray
    mergeFloat(blocks)

  private def mergeFloat(input: Array[Float]): Float =
    val values = input.clone()
    var count = values.length
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        values(write) = (values(read) + values(read + 1)).toFloat
        read += 2
        write += 1
      if read < count then
        values(write) = values(read)
        write += 1
      count = write
    if values.isEmpty then 0.0f else values(0)

  private def referenceDoublePairwise(values: Seq[Double]): Double =
    val blocks =
      values
        .grouped(ReductionKernels.PairwiseBlockSize)
        .map(_.sum)
        .toArray
    var count = blocks.length
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(write) = blocks(read) + blocks(read + 1)
        read += 2
        write += 1
      if read < count then
        blocks(write) = blocks(read)
        write += 1
      count = write
    if blocks.isEmpty then 0.0 else blocks(0)
