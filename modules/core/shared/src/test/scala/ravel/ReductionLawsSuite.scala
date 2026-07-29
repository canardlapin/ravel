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

  test("optimized extrema preserve NaN and signed-zero semantics across layouts and axes") {
    val signed = NDArray.fromSeq(
      Shape(4, 4),
      Seq(
        0.0, 7.0, -0.0, -3.0, -0.0, 2.0, 0.0, 9.0, 4.0, -5.0, 6.0, 0.0, 1.0, 8.0, -2.0, -0.0
      )
    )
    val expectedMinimum = values(signed).foldLeft(Double.PositiveInfinity)(math.min)
    val expectedMaximum = values(signed).foldLeft(Double.NegativeInfinity)(math.max)
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.min),
      java.lang.Double.doubleToRawLongBits(expectedMinimum)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.max),
      java.lang.Double.doubleToRawLongBits(expectedMaximum)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.transpose.min),
      java.lang.Double.doubleToRawLongBits(expectedMinimum)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(signed.transpose.max),
      java.lang.Double.doubleToRawLongBits(expectedMaximum)
    )

    val axis0Minimum = signed.min(0)
    val axis0Maximum = signed.max(0)
    val axis1Minimum = signed.min(1)
    val axis1Maximum = signed.max(1)
    var column = 0
    while column < signed.shape(1) do
      val fiber = Vector.tabulate(signed.shape(0))(row => signed(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis0Minimum(column)),
        java.lang.Double.doubleToRawLongBits(fiber.reduce(math.min))
      )
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis0Maximum(column)),
        java.lang.Double.doubleToRawLongBits(fiber.reduce(math.max))
      )
      column += 1
    var row = 0
    while row < signed.shape(0) do
      val fiber = Vector.tabulate(signed.shape(1))(column => signed(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis1Minimum(row)),
        java.lang.Double.doubleToRawLongBits(fiber.reduce(math.min))
      )
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis1Maximum(row)),
        java.lang.Double.doubleToRawLongBits(fiber.reduce(math.max))
      )
      row += 1

    val nan = NDArray.fromSeq(Shape(4, 4), values(signed).updated(9, Double.NaN))
    assert(nan.min.isNaN)
    assert(nan.max.isNaN)
    assert(nan.min(0)(1).isNaN)
    assert(nan.max(1)(2).isNaN)

    val packed3d =
      NDArray
        .fromSeq(
          Shape(2, 3, 4),
          Seq.tabulate(24)(index => if index % 3 == 0 then -0.0 else 0.0)
        )
        .permuteAxes(1, 2, 0)
        .reverse(0)
        .reverse(2)
    assert(packed3d.layout.isPhysicallyDense)
    assertEquals(
      java.lang.Double.doubleToRawLongBits(packed3d.min),
      java.lang.Double.doubleToRawLongBits(-0.0)
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(packed3d.max),
      java.lang.Double.doubleToRawLongBits(0.0)
    )

    val packedNaN =
      NDArray
        .fromSeq(
          Shape(2, 3, 4),
          Seq.tabulate(24)(index => if index == 17 then Double.NaN else index.toDouble)
        )
        .permuteAxes(2, 0, 1)
        .reverse(0)
    assert(packedNaN.min.isNaN)
    assert(packedNaN.max.isNaN)
  }

  test("contiguous integer fold specializations preserve wraparound and tails") {
    var length = 0
    while length <= 35 do
      val ints = Vector.tabulate(length) { index =>
        if index % 5 == 0 then Int.MaxValue else index * 17 - 23
      }
      val longs = Vector.tabulate(length) { index =>
        if index % 5 == 0 then Long.MaxValue else index.toLong * 1000000007L - 31L
      }
      val intArray = NDArray.fromSeq(Shape(length), ints)
      val longArray = NDArray.fromSeq(Shape(length), longs)
      assertEquals(intArray.sum, ints.foldLeft(0)(_ + _))
      assertEquals(intArray.product, ints.foldLeft(1)(_ * _))
      assertEquals(longArray.sum, longs.foldLeft(0L)(_ + _))
      assertEquals(longArray.product, longs.foldLeft(1L)(_ * _))
      length += 1

    val denseInts =
      NDArray
        .fromSeq(Shape(2, 3, 5), Seq.tabulate(30)(index => index * 17 - 211))
        .permuteAxes(1, 2, 0)
        .reverse(0)
        .reverse(2)
    val denseLongs =
      NDArray
        .fromSeq(
          Shape(2, 3, 5),
          Seq.tabulate(30)(index => index.toLong * 1000000007L - 31L)
        )
        .permuteAxes(2, 0, 1)
        .reverse(1)
    assert(denseInts.layout.isPhysicallyDense)
    assert(denseLongs.layout.isPhysicallyDense)
    assertEquals(denseInts.sum, values(denseInts).foldLeft(0)(_ + _))
    assertEquals(denseInts.product, values(denseInts).foldLeft(1)(_ * _))
    assertEquals(denseLongs.sum, values(denseLongs).foldLeft(0L)(_ + _))
    assertEquals(denseLongs.product, values(denseLongs).foldLeft(1L)(_ * _))
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

  test("specialized full and axis sums stay bit-identical to the pairwise reference") {
    val matrix = NDArray.tabulate[Double](257, 64)((i, j) => math.sin(i * 0.17 + j * 0.09))
    val flat = values(matrix)
    assertEquals(
      java.lang.Double.doubleToRawLongBits(matrix.sum),
      java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(flat))
    )
    val transposed = matrix.transpose
    assertEquals(
      java.lang.Double.doubleToRawLongBits(transposed.sum),
      java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(values(transposed)))
    )
    val axis0 = matrix.sum(0)
    val axis1 = matrix.sum(1)
    assertEquals(axis0.size, 64)
    assertEquals(axis1.size, 257)
    var column = 0
    while column < 64 do
      val fiber = Vector.tabulate(257)(row => matrix(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis0(column)),
        java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(fiber))
      )
      column += 1
    var row = 0
    while row < 257 do
      val fiber = Vector.tabulate(64)(column => matrix(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(axis1(row)),
        java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(fiber))
      )
      row += 1
  }

  test("adversarial float and NaN fibers stay bit-identical across specialized paths") {
    val floats = NDArray.tabulate[Float](129, 33)((i, j) =>
      if (i + j) % 17 == 0 then Float.NaN
      else math.sin(i * 0.31 + j * 0.11).toFloat
    )
    assertEquals(
      java.lang.Float.floatToRawIntBits(floats.sum),
      java.lang.Float.floatToRawIntBits(referenceFloatPairwise(values(floats)))
    )
    val axis0 = floats.sum(0)
    val axis1 = floats.sum(1)
    var column = 0
    while column < 33 do
      val fiber = Vector.tabulate(129)(row => floats(row, column))
      assertEquals(
        java.lang.Float.floatToRawIntBits(axis0(column)),
        java.lang.Float.floatToRawIntBits(referenceFloatPairwise(fiber))
      )
      column += 1
    var row = 0
    while row < 129 do
      val fiber = Vector.tabulate(33)(column => floats(row, column))
      assertEquals(
        java.lang.Float.floatToRawIntBits(axis1(row)),
        java.lang.Float.floatToRawIntBits(referenceFloatPairwise(fiber))
      )
      row += 1
    val emptyRows = NDArray.zeros[Double](0, 5)
    assertEquals(values(emptyRows.sum(0)), List(0.0, 0.0, 0.0, 0.0, 0.0))
    assertEquals(values(emptyRows.sum(1)), Nil)
  }

  test("exact reduction scheduling survives four-lane block, row, and tail shapes") {
    val doubles =
      NDArray.tabulate[Double](9, 513)((row, column) =>
        math.sin(row * 0.37 + column * 0.013) *
          (if column % 11 == 0 then 1.0e12 else 1.0)
      )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(doubles.sum),
      java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(values(doubles)))
    )
    assertEquals(
      java.lang.Double.doubleToRawLongBits(doubles.mean),
      java.lang.Double.doubleToRawLongBits(
        referenceDoublePairwise(values(doubles)) / doubles.size.toDouble
      )
    )
    val doubleRows = doubles.sum(1)
    val doubleRowMeans = doubles.mean(1)
    val doubleColumnMeans = doubles.mean(0)
    var row = 0
    while row < 9 do
      val fiber = Vector.tabulate(513)(column => doubles(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(doubleRows(row)),
        java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(fiber))
      )
      assertEquals(
        java.lang.Double.doubleToRawLongBits(doubleRowMeans(row)),
        java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(fiber) / fiber.size.toDouble)
      )
      row += 1
    var column = 0
    while column < 513 do
      val fiber = Vector.tabulate(9)(row => doubles(row, column))
      assertEquals(
        java.lang.Double.doubleToRawLongBits(doubleColumnMeans(column)),
        java.lang.Double.doubleToRawLongBits(referenceDoublePairwise(fiber) / fiber.size.toDouble)
      )
      column += 1

    val floats =
      NDArray.tabulate[Float](7, 259)((row, column) =>
        (
          math.cos(row * 0.19 + column * 0.021) *
            (if column % 7 == 0 then 1.0e6 else 1.0)
        ).toFloat
      )
    assertEquals(
      java.lang.Float.floatToRawIntBits(floats.sum),
      java.lang.Float.floatToRawIntBits(referenceFloatPairwise(values(floats)))
    )
    val floatRows = floats.sum(1)
    row = 0
    while row < 7 do
      val fiber = Vector.tabulate(259)(column => floats(row, column))
      assertEquals(
        java.lang.Float.floatToRawIntBits(floatRows(row)),
        java.lang.Float.floatToRawIntBits(referenceFloatPairwise(fiber))
      )
      row += 1
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
