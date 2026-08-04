package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given

/** Guards both mutable copying reshape paths against a second output-sized materialization. */
final class MutableReshapeAllocationSuite extends FunSuite:
  private var sink: Any = null

  test("canonical reshapeCopy allocates one primitive output"):
    val edge = 512
    val source = NDArray.zeros[Int](edge, edge).mutableCopy
    assertSingleOutput(edge.toLong * edge.toLong * 4L) {
      source.reshapeCopy(Shape(edge * edge))
    }

  test("noncontiguous reshape fallback allocates one primitive output"):
    val edge = 512
    val source = NDArray.zeros[Int](edge, edge).mutableCopy.transpose
    assertSingleOutput(edge.toLong * edge.toLong * 4L) {
      source.reshape(Shape(edge * edge))
    }

  private def assertSingleOutput(expectedOutputBytes: Long)(run: => Any): Unit =
    var warmup = 0
    while warmup < 20 do
      sink = run
      warmup += 1

    val allocated = medianAllocation {
      sink = run
    }
    assert(sink != null)
    assert(
      allocated <= expectedOutputBytes + 8192L,
      s"allocated $allocated B for a $expectedOutputBytes B mutable reshape output; " +
        "reshape must not allocate a second output-sized buffer"
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
