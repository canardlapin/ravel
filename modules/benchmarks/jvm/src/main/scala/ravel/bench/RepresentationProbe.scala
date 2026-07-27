package ravel.bench

import org.openjdk.jmh.annotations.*
import ravel.DType.given
import ravel.internal.ProbeApi
import ravel.*
import java.util.concurrent.TimeUnit

@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Thread)
class RepresentationProbe:
  private val size = 65536
  private val left = ProbeApi.allocate[Double](size)
  private val right = ProbeApi.allocate[Double](size)
  private val out = ProbeApi.allocate[Double](size)
  private val publicLeft = NDArray.fill(Shape(size), 1.25)
  private val publicRight = NDArray.fill(Shape(size), 2.5)
  private val publicStridedLeft =
    NDArray.fill(Shape(size * 2), 1.25).slice(0, Slice(0, size * 2, 2))
  private val publicStridedRight =
    NDArray.fill(Shape(size * 2), 2.5).slice(0, Slice(1, size * 2, 2)).reverse(0)
  ProbeApi.fill(left, 1.25)
  ProbeApi.fill(right, 2.5)

  @Benchmark
  def linearAddDouble(): Double =
    ProbeApi.add(left, right, out, size)
    ProbeApi.get(out, size - 1)

  @Benchmark
  def generalStridedAddDouble(): Double =
    ProbeApi.addStrided(left, 0, 2, right, 1, 2, out, size / 2)
    ProbeApi.get(out, size / 2 - 1)

  @Benchmark
  def publicLinearAddDouble(): Double =
    (publicLeft + publicRight)(size - 1)

  @Benchmark
  def publicStridedAddDouble(): Double =
    (publicStridedLeft + publicStridedRight)(size - 1)
