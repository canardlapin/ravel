package ravel

import munit.FunSuite

final class AxesSuite extends FunSuite:
  test("construction normalizes once and records one canonical sorted order"):
    val axes = Axes.from(4, -1, 0, -2).toOption.get
    assertEquals(IArray.genericWrapArray(axes.normalized).toList, List(0, 2, 3))
    val reordered = Axes.from(4, 2, 3, 0).toOption.get
    assertEquals(axes, reordered)
    assertEquals(axes.hashCode, reordered.hashCode)
    assertEquals(axes.rank, 4)

  test("construction returns precise pure invalid-axis and duplicate errors"):
    val invalid = InvalidReductionAxis(-4, 3)
    val duplicate = DuplicateReductionAxis(-3, 0)
    assertEquals(Axes.from(3, -4), Left(invalid))
    assertEquals(Axes.from(3, 0, -3), Left(duplicate))
    assertEquals(Axes.from(-1), Left(InvalidAxesRank(-1)))
    assert(!(invalid: Any).isInstanceOf[Throwable])
    assert(!(duplicate: Any).isInstanceOf[Throwable])

  test("empty and all-axis values cover scalar and higher ranks"):
    assertEquals(Axes.from(0).map(_.size), Right(0))
    assertEquals(
      Axes.all(3).map(axes => IArray.genericWrapArray(axes.normalized).toList),
      Right(List(0, 1, 2))
    )

  test("applying axes validated for another rank throws a typed wrapper"):
    val axes = Axes.from(3, 0, 2).toOption.get
    val failure = intercept[InvalidAxesException] {
      NDArray.zeros[Int](2, 3).sum(axes)
    }
    assertEquals(failure.error, AxesRankMismatch(expectedRank = 2, axesRank = 3))

  test("varargs reduction conveniences wrap the precise checked axes error"):
    val source = NDArray.zeros[Int](2, 3)
    val failure = intercept[InvalidAxesException] {
      source.sumAxes(0, -2)
    }
    assertEquals(failure.error, DuplicateReductionAxis(axis = -2, normalizedAxis = 0))
