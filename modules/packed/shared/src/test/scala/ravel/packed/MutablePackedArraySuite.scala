package ravel.packed

import munit.FunSuite
import ravel.Shape

final class MutablePackedArraySuite extends FunSuite:
  private val widths = Vector(PackedBits.B1, PackedBits.B2, PackedBits.B4)

  test("generated lifecycle sequences enforce consuming freeze and snapshot isolation"):
    widths.foreach { bits =>
      lifecycleSequences(bits).zipWithIndex.foreach { (actions, scenario) =>
        val workspace = MutablePackedArray.allocate(Shape(9), bits)
        var expected = Vector.fill(9)(0)
        var open = true
        var snapshots = Vector.empty[(PackedArray[?], Vector[Int])]

        actions.foreach { action =>
          action match
            case Action.Set(index, code) =>
              if open then
                workspace.setCode(index, code)
                expected = expected.updated(index, code)
              else assertConsumed(workspace.setCode(index, code))
            case Action.Read(index) =>
              if open then assertEquals(workspace.codeAt(index), expected(index))
              else assertConsumed(workspace.codeAt(index))
            case Action.FreezeCopy =>
              if open then
                val copied = workspace.freezeCopy
                assertEquals(copied.codeVector, expected)
                snapshots :+= ((copied, expected))
              else assertConsumed(workspace.freezeCopy)
            case Action.Freeze =>
              if open then
                val frozen = workspace.freeze
                assertEquals(frozen.codeVector, expected)
                snapshots :+= ((frozen, expected))
                open = false
              else assertConsumed(workspace.freeze)

          snapshots.foreach { (snapshot, snapshotValues) =>
            assertEquals(
              snapshot.codeVector,
              snapshotValues,
              s"width=$bits scenario=$scenario action=$action"
            )
          }
        }

        assert(!open, s"generated scenario $scenario must consume its workspace")
      }
    }

  private def lifecycleSequences(bits: PackedBits): Vector[Vector[Action]] =
    Vector.tabulate(32) { seed =>
      val beforeFreeze = Vector.tabulate(12) { step =>
        val mixed = seed.toLong * 1103515245L + step.toLong * 12345L
        val index = ((mixed >>> 8) % 9L).toInt
        val code = ((mixed >>> 16) % (bits.maxCode + 1).toLong).toInt
        (mixed & 3L).toInt match
          case 0 | 1 => Action.Set(index, code)
          case 2 => Action.Read(index)
          case _ => Action.FreezeCopy
      }
      beforeFreeze ++ Vector(
        Action.FreezeCopy,
        Action.Set(seed % 9, seed % (bits.maxCode + 1)),
        Action.Freeze,
        Action.Read(seed % 9),
        Action.Set((seed + 1) % 9, (seed + 1) % (bits.maxCode + 1)),
        Action.FreezeCopy,
        Action.Freeze
      )
    }

  private def assertConsumed[A](operation: => A): Unit =
    val _ = intercept[PackedWorkspaceConsumedException](operation)

  private enum Action derives CanEqual:
    case Set(index: Int, code: Int)
    case Read(index: Int)
    case FreezeCopy
    case Freeze
