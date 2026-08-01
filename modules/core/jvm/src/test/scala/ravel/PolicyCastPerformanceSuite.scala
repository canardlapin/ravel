package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given

/** Ensures checked conversion allocates only its final primitive output buffer. */
final class PolicyCastPerformanceSuite extends FunSuite:
  private var sink: Any = null

  test("checked Double-to-Float conversion has no output-sized intermediates"):
    val edge = 512
    val source =
      NDArray.tabulate[Double](edge, edge)((row, column) => row.toDouble + column.toDouble * 0.25)
    val expectedOutputBytes = edge.toLong * edge.toLong * 4L

    var warmup = 0
    while warmup < 20 do
      sink = source.convert[Float]()
      warmup += 1

    val allocated = medianAllocation {
      sink = source.convert[Float]()
    }

    assert(sink != null)
    assert(
      allocated <= expectedOutputBytes + 8192L,
      s"allocated $allocated B for a $expectedOutputBytes B Float output; " +
        "conversion must not materialize a boxed intermediate"
    )
    println(
      s"RAVEL-CONVERSION JVM baseline: allocated=$allocated B, " +
        s"output=$expectedOutputBytes B"
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
