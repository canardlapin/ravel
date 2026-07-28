package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import scala.compiletime.uninitialized

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class AccessPatternBenchmarks:
  @Param(Array("256", "1024"))
  var side: Int = 0

  private var fixture: AccessPatternFixture = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    fixture = new AccessPatternFixture(side)

  @Benchmark
  def contiguous_add(): Array2[Double] =
    AccessPatternWorkloads.contiguousAdd(fixture)

  @Benchmark
  def inner_stride_add(): Array2[Double] =
    AccessPatternWorkloads.innerStrideAdd(fixture)

  @Benchmark
  def outer_stride_add(): Array2[Double] =
    AccessPatternWorkloads.outerStrideAdd(fixture)

  @Benchmark
  def reverse_add(): Array2[Double] =
    AccessPatternWorkloads.reverseAdd(fixture)

  @Benchmark
  def transpose_add(): Array2[Double] =
    AccessPatternWorkloads.transposeAdd(fixture)

  @Benchmark
  def broadcast_row_add(): Array2[Double] =
    AccessPatternWorkloads.broadcastRowAdd(fixture)

  @Benchmark
  def full_sum_contiguous(): Double =
    AccessPatternWorkloads.fullSumContiguous(fixture)

  @Benchmark
  def full_sum_inner_stride(): Double =
    AccessPatternWorkloads.fullSumInnerStride(fixture)

  @Benchmark
  def axis0_sum(): Array1[Double] =
    AccessPatternWorkloads.axis0Sum(fixture)

  @Benchmark
  def axis1_sum(): Array1[Double] =
    AccessPatternWorkloads.axis1Sum(fixture)

  @Benchmark
  def copy_inner_stride(): Array2[Double] =
    AccessPatternWorkloads.copyInnerStride(fixture)

  @Benchmark
  def copy_transpose(): Array2[Double] =
    AccessPatternWorkloads.copyTranspose(fixture)

  @Benchmark
  def scalar_read_row_major(): Double =
    AccessPatternWorkloads.scalarReadRowMajor(fixture)

  @Benchmark
  def scalar_read_column_major(): Double =
    AccessPatternWorkloads.scalarReadColumnMajor(fixture)

  @Benchmark
  def scalar_write_row_major(): Double =
    AccessPatternWorkloads.scalarWriteRowMajor(fixture)

  @Benchmark
  def scalar_write_column_major(): Double =
    AccessPatternWorkloads.scalarWriteColumnMajor(fixture)

  @Benchmark
  def view_inner_stride_create(): Array2[Double] =
    AccessPatternWorkloads.viewInnerStrideCreate(fixture)

  @Benchmark
  def view_transpose_create(): Array2[Double] =
    AccessPatternWorkloads.viewTransposeCreate(fixture)

private[bench] final class AccessPatternFixture(val side: Int):
  require(side > 0 && side % 2 == 0, s"side must be a positive even integer, got $side")
  Math.multiplyExact(side, side)

  val left: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0
    }

  val right: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      ((row * 43 + column * 19) % 257 - 128).toDouble / 32.0
    }

  val row: Array1[Double] =
    NDArray.tabulate(side) { column =>
      ((column * 29) % 127 - 63).toDouble / 8.0
    }

  val innerStrideLeft: Array2[Double] =
    left.slice(axis = 1, Slice(0, side, 2))

  val innerStrideRight: Array2[Double] =
    right.slice(axis = 1, Slice(1, side, 2))

  val outerStrideLeft: Array2[Double] =
    left.slice(axis = 0, Slice(0, side, 2))

  val outerStrideRight: Array2[Double] =
    right.slice(axis = 0, Slice(1, side, 2))

  val reversedLeft: Array2[Double] =
    left.reverse(axis = 1)

  val transposedLeft: Array2[Double] =
    left.transpose

  val transposedRight: Array2[Double] =
    right.transpose

  val mutableOutput: MutableNDArray[Double, Rank[2]] =
    left.mutableCopy

private[bench] object AccessPatternWorkloads:
  def contiguousAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.left + fixture.right

  def innerStrideAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.innerStrideLeft + fixture.innerStrideRight

  def outerStrideAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.outerStrideLeft + fixture.outerStrideRight

  def reverseAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.reversedLeft + fixture.right

  def transposeAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.transposedLeft + fixture.transposedRight

  def broadcastRowAdd(fixture: AccessPatternFixture): Array2[Double] =
    fixture.left + fixture.row

  def fullSumContiguous(fixture: AccessPatternFixture): Double =
    fixture.left.sum

  def fullSumInnerStride(fixture: AccessPatternFixture): Double =
    fixture.innerStrideLeft.sum

  def axis0Sum(fixture: AccessPatternFixture): Array1[Double] =
    fixture.left.sum(axis = 0)

  def axis1Sum(fixture: AccessPatternFixture): Array1[Double] =
    fixture.left.sum(axis = 1)

  def copyInnerStride(fixture: AccessPatternFixture): Array2[Double] =
    fixture.innerStrideLeft.copy

  def copyTranspose(fixture: AccessPatternFixture): Array2[Double] =
    fixture.transposedLeft.copy

  def scalarReadRowMajor(fixture: AccessPatternFixture): Double =
    var total = 0.0
    var row = 0
    while row < fixture.side do
      var column = 0
      while column < fixture.side do
        total += fixture.left(row, column)
        column += 1
      row += 1
    total

  def scalarReadColumnMajor(fixture: AccessPatternFixture): Double =
    var total = 0.0
    var column = 0
    while column < fixture.side do
      var row = 0
      while row < fixture.side do
        total += fixture.left(row, column)
        row += 1
      column += 1
    total

  def scalarWriteRowMajor(fixture: AccessPatternFixture): Double =
    var row = 0
    while row < fixture.side do
      var column = 0
      while column < fixture.side do
        fixture.mutableOutput(row, column) = row.toDouble + column.toDouble
        column += 1
      row += 1
    fixture.mutableOutput(fixture.side - 1, fixture.side - 1)

  def scalarWriteColumnMajor(fixture: AccessPatternFixture): Double =
    var column = 0
    while column < fixture.side do
      var row = 0
      while row < fixture.side do
        fixture.mutableOutput(row, column) = row.toDouble + column.toDouble
        row += 1
      column += 1
    fixture.mutableOutput(fixture.side - 1, fixture.side - 1)

  def viewInnerStrideCreate(fixture: AccessPatternFixture): Array2[Double] =
    fixture.left.slice(axis = 1, Slice(0, fixture.side, 2))

  def viewTransposeCreate(fixture: AccessPatternFixture): Array2[Double] =
    fixture.left.transpose
