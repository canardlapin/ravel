package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite

/** Rejects implementations that materialize one output for each selected axis. */
final class MultiAxisReductionAllocationSuite extends FunSuite:
  private var sink: Any = null

  test("multi-axis sum allocates the final output but no sequential intermediate"):
    val edge = 64
    val source = NDArray.tabulate[Double](edge, edge, edge)((i, j, k) => i + j + k)
    val axes = Axes.from(source.rank, 0, 2).toOption.get
    val expectedOutputBytes = edge.toLong * 8L

    var warmup = 0
    while warmup < 20 do
      sink = source.sum(axes)
      warmup += 1

    val allocated = medianAllocation {
      sink = source.sum(axes)
    }
    assert(sink != null)
    assert(
      allocated <= expectedOutputBytes + 8192L,
      s"allocated $allocated B for a $expectedOutputBytes B multi-axis sum output; " +
        "the reduction must not materialize a sequential intermediate"
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
