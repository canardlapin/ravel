package ravel.stencil

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.DType.given
import ravel.MutableNDArray
import ravel.NDArray
import ravel.Rank
import ravel.Shape

/** Allocation court for the direct primitive Double executor.
  *
  * The reference engine deliberately allocates index arrays as its clarity baseline. The direct
  * path must not allocate proportionally to the number of output samples or support offsets.
  */
final class DirectNeighborhoodExecutorPerformanceSuite extends FunSuite:
  test("prepared direct Double execution reuses all workspace"):
    val nx = 192
    val ny = 128
    val source =
      NDArray.tabulate[Double](nx, ny)((i, j) => i.toDouble * 0.25 + j.toDouble * 0.5)
    val directDestination =
      MutableNDArray.zeros[Double, Rank[2]](Shape(nx, ny))
    val referenceDestination =
      MutableNDArray.zeros[Double, Rank[2]](Shape(nx, ny))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(
          Vector(-1, -1),
          Vector(-1, 0),
          Vector(-1, 1),
          Vector(0, -1),
          Vector(0, 0),
          Vector(0, 1),
          Vector(1, -1),
          Vector(1, 0),
          Vector(1, 1)
        ),
        border = BorderMode.ReflectWithoutEdge,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(nx, ny)
      )
    val primitive =
      new DoubleNeighborhoodReducer:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value
        def finish(acc: Double): Double = acc
    val generic =
      new NeighborhoodReducer[Double, Double, Double]:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value
        def finish(acc: Double): Double = acc
    val plan =
      DirectNeighborhoodExecutor.prepare(source, directDestination, spec)

    var warmup = 0
    while warmup < 20 do
      plan.runDouble(
        source,
        directDestination,
        primitive,
        constant = 0.0
      )
      warmup += 1

    val directBytes =
      medianAllocation {
        plan.runDouble(
          source,
          directDestination,
          primitive,
          constant = 0.0
        )
      }
    val referenceBytes =
      medianAllocation {
        ReferenceNeighborhoodExecutor.run(
          source,
          referenceDestination,
          spec,
          generic,
          constant = 0.0
        )
      }

    assert(
      directDestination
        .freezeCopy()
        .sameElements(
          referenceDestination.freezeCopy()
        ),
      "direct and reference outputs diverged"
    )
    assert(
      directBytes <= 256L,
      s"prepared direct run allocated $directBytes B; expected reusable workspace"
    )
    assert(
      referenceBytes > directBytes * 20L,
      s"reference allocation $referenceBytes B was not materially larger " +
        s"than direct allocation $directBytes B"
    )
    println(
      s"RAVEL-STENCIL JVM baseline: directAllocated=$directBytes B, " +
        s"referenceAllocated=$referenceBytes B, samples=${nx * ny}, " +
        s"support=${spec.offsets.size}"
    )

  test("prepared direct Boolean execution reuses all workspace"):
    val nx = 192
    val ny = 128
    val source =
      NDArray.tabulate[Boolean](nx, ny)((row, column) =>
        (row + 3 * column) % 7 == 0
      )
    val destination =
      MutableNDArray.zeros[Boolean, Rank[2]](Shape(nx, ny))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(
          Vector(-1, 0),
          Vector(0, -1),
          Vector(0, 0),
          Vector(0, 1),
          Vector(1, 0)
        ),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(nx, ny)
      )
    val reducer =
      new BooleanNeighborhoodReducer:
        def zero: Boolean = false
        def accumulate(
            accumulator: Boolean,
            value: Boolean,
            offsetIndex: Int
        ): Boolean =
          accumulator || value
        def finish(accumulator: Boolean): Boolean = accumulator
    val plan =
      DirectNeighborhoodExecutor.prepare(source, destination, spec)

    var warmup = 0
    while warmup < 20 do
      plan.runBoolean(source, destination, reducer, constant = false)
      warmup += 1

    val allocated =
      medianAllocation {
        plan.runBoolean(source, destination, reducer, constant = false)
      }

    assert(
      allocated <= 256L,
      s"prepared Boolean run allocated $allocated B; expected reusable workspace"
    )
    println(
      s"RAVEL-STENCIL JVM baseline: booleanAllocated=$allocated B, " +
        s"samples=${nx * ny}, support=${spec.offsets.size}"
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
