package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import scala.compiletime.uninitialized

/** Allocation evidence for the fixed-rank reads and consuming builder used by external kernels.
  *
  * Run with:
  * {{{
  * sbt 'representationProbeJVM/Jmh/run -prof gc -wi 5 -i 7 -f 2 -p edge=8,16 .*KernelApiBenchmarks.*'
  * }}}
  *
  * The checksum benchmarks must report no allocation proportional to element count. Builder
  * allocation may contain one primitive output buffer plus constant-sized wrapper, layout,
  * callback, and builder objects; it must not contain a second output-sized buffer.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class KernelApiBenchmarks:
  @Param(Array("8", "16"))
  var edge: Int = 0

  private var rank3: Array3[Double] = uninitialized
  private var rank4: Array4[Double] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    rank3 = NDArray.tabulate(edge, edge, edge) { (i, j, k) =>
      i.toDouble * 0.5 + j.toDouble * 0.25 + k.toDouble
    }
    rank4 = NDArray.tabulate(edge, edge, edge, edge) { (i, j, k, l) =>
      i.toDouble * 0.5 + j.toDouble * 0.25 + k.toDouble * 0.125 + l.toDouble
    }

  @Benchmark
  def rank3_read_checksum(): Double =
    var checksum = 0.0
    var i = 0
    while i < edge do
      var j = 0
      while j < edge do
        var k = 0
        while k < edge do
          checksum += rank3(i, j, k)
          k += 1
        j += 1
      i += 1
    checksum

  @Benchmark
  def rank4_read_checksum(): Double =
    var checksum = 0.0
    var i = 0
    while i < edge do
      var j = 0
      while j < edge do
        var k = 0
        while k < edge do
          var l = 0
          while l < edge do
            checksum += rank4(i, j, k, l)
            l += 1
          k += 1
        j += 1
      i += 1
    checksum

  @Benchmark
  def rank3_read_build_rank3(): Array3[Double] =
    NDArray.build[Double, Rank[3]](rank3.shape) { builder =>
      var out = 0
      var i = 0
      while i < edge do
        var j = 0
        while j < edge do
          var k = 0
          while k < edge do
            builder.writeLinear(out, rank3(i, j, k) * 2.0)
            out += 1
            k += 1
          j += 1
        i += 1
    }

  @Benchmark
  def rank4_read_build_linear(): Array1[Double] =
    NDArray.build[Double, Rank[1]](Shape(rank4.size)) { builder =>
      var out = 0
      var i = 0
      while i < edge do
        var j = 0
        while j < edge do
          var k = 0
          while k < edge do
            var l = 0
            while l < edge do
              builder.writeLinear(out, rank4(i, j, k, l) + 1.0)
              out += 1
              l += 1
            k += 1
          j += 1
        i += 1
    }
