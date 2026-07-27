package ravel.bench

import ravel.DType.given
import ravel.internal.ProbeApi
import ravel.*
import scala.scalajs.js

object RepresentationProbe:
  def main(args: Array[String]): Unit =
    val size = 65536
    val iterations = 2000
    val left = ProbeApi.allocate[Double](size)
    val right = ProbeApi.allocate[Double](size)
    val out = ProbeApi.allocate[Double](size)
    ProbeApi.fill(left, 1.25)
    ProbeApi.fill(right, 2.5)

    var warmup = 0
    while warmup < 200 do
      ProbeApi.add(left, right, out, size)
      warmup += 1

    val started = js.Date.now()
    var i = 0
    while i < iterations do
      ProbeApi.add(left, right, out, size)
      i += 1
    val elapsedMs = js.Date.now() - started
    val elementsPerSecond = iterations.toDouble * size.toDouble / (elapsedMs / 1000.0)
    println(f"linearAddDouble: $elementsPerSecond%.0f elements/s")
    println(s"checksum: ${ProbeApi.get(out, size - 1)}")

    val publicLeft = NDArray.fill(Shape(size), 1.25)
    val publicRight = NDArray.fill(Shape(size), 2.5)
    var publicResult = publicLeft + publicRight
    warmup = 0
    while warmup < 200 do
      publicResult = publicLeft + publicRight
      warmup += 1
    val publicStarted = js.Date.now()
    i = 0
    while i < iterations do
      publicResult = publicLeft + publicRight
      i += 1
    val publicElapsedMs = js.Date.now() - publicStarted
    val publicElementsPerSecond =
      iterations.toDouble * size.toDouble / (publicElapsedMs / 1000.0)
    println(f"publicLinearAddDouble: $publicElementsPerSecond%.0f elements/s")
    println(s"public checksum: ${publicResult(size - 1)}")

    val publicStridedLeft =
      NDArray.fill(Shape(size * 2), 1.25).slice(0, Slice(0, size * 2, 2))
    val publicStridedRight =
      NDArray.fill(Shape(size * 2), 2.5).slice(0, Slice(1, size * 2, 2)).reverse(0)
    var publicStridedResult = publicStridedLeft + publicStridedRight
    warmup = 0
    while warmup < 200 do
      publicStridedResult = publicStridedLeft + publicStridedRight
      warmup += 1
    val stridedStarted = js.Date.now()
    i = 0
    while i < iterations do
      publicStridedResult = publicStridedLeft + publicStridedRight
      i += 1
    val stridedElapsedMs = js.Date.now() - stridedStarted
    val stridedElementsPerSecond =
      iterations.toDouble * size.toDouble / (stridedElapsedMs / 1000.0)
    println(f"publicStridedAddDouble: $stridedElementsPerSecond%.0f elements/s")
    println(s"public strided checksum: ${publicStridedResult(size - 1)}")
