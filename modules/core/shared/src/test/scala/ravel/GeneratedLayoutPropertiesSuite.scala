package ravel

import munit.ScalaCheckSuite
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll
import ravel.DType.given

final class GeneratedLayoutPropertiesSuite extends ScalaCheckSuite:
  override def scalaCheckTestParameters =
    super.scalaCheckTestParameters
      .withMinSuccessfulTests(250)
      .withMaxDiscardRatio(5)
      .withWorkers(1)

  private val smallDimension = Gen.choose(0, 8)
  private val nonemptyDimension = Gen.choose(1, 8)

  property("generated shapes preserve checked rank and product") {
    val dimensions =
      Gen.choose(0, 5).flatMap(rank => Gen.listOfN(rank, smallDimension))
    forAll(dimensions) { dims =>
      val shape = Shape.from(dims).toOption.get
      val expected = dims.foldLeft(1L)(_ * _.toLong)
      shape.rank == dims.length &&
      shape.size.toLong == expected &&
      dims.indices.forall(axis => shape(axis) == dims(axis))
    }
  }

  property("generated transpose, reversal, and slicing match coordinates") {
    forAll(nonemptyDimension, nonemptyDimension, Gen.choose(1, 3)) { (rows, columns, step) =>
      val source =
        NDArray.tabulate[Int](rows, columns)((row, column) => row * 100 + column)
      val actual =
        source.transpose
          .reverse(0)
          .slice(1, Slice(0, rows, step))
          .elementsIterator
          .toList
      val expected =
        (columns - 1 to 0 by -1).flatMap { column =>
          (0 until rows by step).map(row => row * 100 + column)
        }.toList
      actual == expected
    }
  }

  property("generated broadcast addition matches a coordinate oracle") {
    forAll(smallDimension, smallDimension, Gen.choose(-100, 100)) { (rows, columns, shift) =>
      val left =
        NDArray.tabulate[Int](rows, columns)((row, column) => row * 100 + column)
      val right = NDArray.tabulate[Int](columns)(column => shift - column)
      val actual = (left + right).elementsIterator.toList
      val expected =
        (0 until rows).flatMap { row =>
          (0 until columns).map { column =>
            row * 100 + column + shift - column
          }
        }.toList
      actual == expected
    }
  }

  property("generated mutable view updates remain local") {
    forAll(nonemptyDimension, nonemptyDimension) { (rows, columns) =>
      val mutable =
        NDArray.tabulate[Int](rows, columns)((row, column) => row * columns + column).mutableCopy
      val view = mutable.transpose.reverse(0)
      val targetRow = columns / 2
      val targetColumn = rows / 2
      val before = mutable.freezeCopy().elementsIterator.toVector
      view(targetRow, targetColumn) = Int.MinValue
      val after = mutable.freezeCopy().elementsIterator.toVector
      val changed = before.indices.filter(index => before(index) != after(index))
      changed.size == 1 &&
      after(targetColumn * columns + (columns - 1 - targetRow)) == Int.MinValue
    }
  }

  property("generated packed mutable permutations and reversals update every element once") {
    forAll(
      nonemptyDimension,
      nonemptyDimension,
      Gen.oneOf(true, false),
      Gen.oneOf(true, false),
      Gen.oneOf(true, false),
      Gen.choose(-100, 100)
    ) { (rows, columns, transpose, reverse0, reverse1, shift) =>
      val source =
        NDArray.tabulate[Int](rows, columns)((row, column) => row * 100 + column)
      val mutable = source.mutableCopy
      val permuted = if transpose then mutable.transpose else mutable
      val first = if reverse0 then permuted.reverse(0) else permuted
      val target = if reverse1 then first.reverse(1) else first
      target.addInPlace(shift)
      target.layout.isPhysicallyDense &&
      mutable.freezeCopy().elementsIterator.toList ==
        source.elementsIterator.map(_ + shift).toList
    }
  }
