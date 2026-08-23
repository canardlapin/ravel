package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given

final class MapAllocationSuite extends FunSuite:
  private var sink: Any = null

  test("Double map allocates only its primitive output and constant overhead"):
    val edge = 256
    val source =
      NDArray.tabulate[Double](edge, edge)((row, column) => row.toDouble + column.toDouble)
    val expectedOutputBytes = edge.toLong * edge.toLong * 8L

    var warmup = 0
    while warmup < 40 do
      sink = source.map(_ + 1.0)
      warmup += 1

    val allocated = medianAllocation {
      sink = source.map(_ + 1.0)
    }

    assert(sink != null)
    assert(
      allocated <= expectedOutputBytes + 8192L,
      s"allocated $allocated B for a $expectedOutputBytes B Double output; " +
        "map must not box its primitive input and output per element"
    )
    println(
      s"RAVEL-MAP JVM baseline: allocated=$allocated B, output=$expectedOutputBytes B"
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
