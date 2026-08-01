package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given
import ravel.jvm.JvmInterop

/** Allocation court for fixed-rank unsigned access and conversion used by image4s. */
final class UnsignedAllocationSuite extends FunSuite:
  private var sink: Any = null

  test("canonical UInt8 sequential access allocates only noise"):
    val edge = 256
    val values = Array.tabulate(edge * edge)(i => (i & 0xff).toByte)
    val array = JvmInterop.unsafeBorrowUInt8(values, Shape(edge, edge))

    var warmup = 0
    var acc = 0
    while warmup < 20 do
      var row = 0
      while row < edge do
        var column = 0
        while column < edge do
          acc += array(row, column).toInt
          column += 1
        row += 1
      warmup += 1
    sink = acc

    val allocated = medianAllocation {
      var total = 0
      var row = 0
      while row < edge do
        var column = 0
        while column < edge do
          total += array(row, column).toInt
          column += 1
        row += 1
      sink = total
    }

    assert(sink != null)
    assert(
      allocated <= 256L,
      s"canonical UInt8 traversal allocated $allocated B; expected profiler noise only"
    )

  test("checked Double-to-UInt8 conversion stays near the output buffer"):
    val edge = 256
    val source =
      NDArray.tabulate[Double](edge, edge)((row, column) => ((row + column) % 256).toDouble)
    val expectedOutputBytes = edge.toLong * edge.toLong

    var warmup = 0
    while warmup < 20 do
      sink = source.convert[UInt8]()
      warmup += 1

    val allocated = medianAllocation {
      sink = source.convert[UInt8]()
    }

    assert(sink != null)
    assert(
      allocated <= expectedOutputBytes + 8192L,
      s"allocated $allocated B for a $expectedOutputBytes B UInt8 output; " +
        "conversion must not materialize a boxed intermediate"
    )

  private def medianAllocation(run: => Unit): Long =
    Vector
      .tabulate(7) { _ =>
        val bean =
          ManagementFactory.getThreadMXBean match
            case candidate: ThreadMXBean if candidate.isThreadAllocatedMemorySupported =>
              if !candidate.isThreadAllocatedMemoryEnabled then
                candidate.setThreadAllocatedMemoryEnabled(true)
              candidate
            case _ =>
              fail("this JVM does not expose per-thread allocation accounting")
        val thread = Thread.currentThread().threadId()
        val before = bean.getThreadAllocatedBytes(thread)
        run
        bean.getThreadAllocatedBytes(thread) - before
      }
      .sorted
      .apply(3)
