package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import scala.compiletime.uninitialized

/** Focused witness for physical-order mutation across equivalent packed layouts.
  *
  * Correctness belongs to the shared JVM/Scala.js layout and mutable-law suites. This benchmark
  * checks that the general packed-layout predicate gives permutations and reversals the same
  * allocation-free primitive loop as canonical C-order storage.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class DenseMutableTraversalBenchmarks:
  @Param(Array("1024"))
  var side: Int = 0

  @Param(
    Array(
      "contiguous",
      "transpose",
      "reverse_axis0",
      "reverse_axis1",
      "reverse_both",
      "transpose_reverse"
    )
  )
  var layoutName: String = ""

  private var target: MutableNDArray[Double, Rank[2]] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val source =
      NDArray.tabulate(side, side) { (row, column) =>
        ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0
      }
    val mutable = source.mutableCopy
    target = layoutName match
      case "contiguous" => mutable
      case "transpose" => mutable.transpose
      case "reverse_axis0" => mutable.reverse(0)
      case "reverse_axis1" => mutable.reverse(1)
      case "reverse_both" => mutable.reverse(0).reverse(1)
      case "transpose_reverse" => mutable.transpose.reverse(0).reverse(1)
      case other => throw new IllegalArgumentException(s"unknown dense layout: $other")

  @Benchmark
  def add_in_place(): Double =
    target.addInPlace(0.25)
    target(0, 0)
