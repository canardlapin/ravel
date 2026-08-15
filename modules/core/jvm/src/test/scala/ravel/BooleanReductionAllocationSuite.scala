package ravel

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given

final class BooleanReductionAllocationSuite extends FunSuite:
  private var booleanSink = false
  private var intSink = 0
  private var arraySink: Any = null

  test("whole-array Boolean reductions allocate only profiler noise"):
    val source = NDArray.tabulate[Boolean](256, 256)((row, column) => (row + column) % 3 != 0)

    var warmup = 0
    while warmup < 20 do
      booleanSink = source.all
      booleanSink = source.any
      intSink = source.countTrue
      warmup += 1

    val allocated = medianAllocation {
      booleanSink = source.all
      booleanSink = source.any
      intSink = source.countTrue
    }
    assert(booleanSink || intSink > 0)
    assert(
      allocated <= 256L,
      s"whole Boolean reductions allocated $allocated B; expected profiler noise only"
    )

  test("multi-axis Boolean reductions allocate one result and bounded plans"):
    val edge = 64
    val source = NDArray.tabulate[Boolean](edge, edge, edge) { (i, j, k) =>
      (i + j + k) % 5 != 0
    }
    val axes = Axes.from(source.rank, 0, 2).toOption.get

    var warmup = 0
    while warmup < 20 do
      arraySink = source.all(axes)
      arraySink = source.countTrue(axes)
      warmup += 1

    val allAllocated = medianAllocation {
      arraySink = source.all(axes)
    }
    val countAllocated = medianAllocation {
      arraySink = source.countTrue(axes)
    }
    assert(arraySink != null)
    assert(
      allAllocated <= edge + 8192L,
      s"multi-axis all allocated $allAllocated B for a $edge-element Boolean output"
    )
    assert(
      countAllocated <= edge.toLong * 4L + 8192L,
      s"multi-axis countTrue allocated $countAllocated B for a $edge-element Int output"
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
