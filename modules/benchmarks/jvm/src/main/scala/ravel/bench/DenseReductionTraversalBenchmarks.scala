package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import scala.compiletime.uninitialized

/** Full reductions across equivalent packed layouts.
  *
  * Floating extrema may reorder because NaN propagation and signed-zero selection are invariant
  * under regrouping. Fixed-width sums and products may reorder because their wraparound arithmetic
  * is associative modulo the dtype width. Floating sums and products are deliberately absent: their
  * public schedules remain order-sensitive.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class DenseReductionTraversalBenchmarks:
  @Param(
    Array(
      "contiguous",
      "permute",
      "reverse",
      "permute_reverse"
    )
  )
  var layoutName: String = ""

  private var doubles: NDArray[Double, Rank[3]] = uninitialized
  private var ints: NDArray[Int, Rank[3]] = uninitialized
  private var longs: NDArray[Long, Rank[3]] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val doubleSource =
      NDArray.tabulate(64, 128, 128) { (plane, row, column) =>
        math.sin(plane * 0.031 + row * 0.017 + column * 0.013)
      }
    val intSource =
      NDArray.tabulate(64, 128, 128) { (plane, row, column) =>
        if (plane + row + column) % 3 == 0 then -1 else 1
      }
    val longSource =
      NDArray.tabulate(64, 128, 128) { (plane, row, column) =>
        if (plane + row + column) % 5 == 0 then -1L else 1L
      }
    doubles = selectLayout(doubleSource)
    ints = selectLayout(intSource)
    longs = selectLayout(longSource)

  private def selectLayout[A](source: NDArray[A, Rank[3]]): NDArray[A, Rank[3]] =
    layoutName match
      case "contiguous" => source
      case "permute" => source.permuteAxes(1, 2, 0)
      case "reverse" => source.reverse(0).reverse(2)
      case "permute_reverse" =>
        source.permuteAxes(2, 0, 1).reverse(0).reverse(2)
      case other => throw new IllegalArgumentException(s"unknown dense layout: $other")

  @Benchmark
  def minimum_double(): Double =
    doubles.min

  @Benchmark
  def maximum_double(): Double =
    doubles.max

  @Benchmark
  def sum_int(): Int =
    ints.sum

  @Benchmark
  def product_int(): Int =
    ints.product

  @Benchmark
  def sum_long(): Long =
    longs.sum

  @Benchmark
  def product_long(): Long =
    longs.product
