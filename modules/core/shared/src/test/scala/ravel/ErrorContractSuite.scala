package ravel

import munit.FunSuite

final class ErrorContractSuite extends FunSuite:
  private def assertPure(error: RavelError): Unit =
    assert(!(error: Any).isInstanceOf[Throwable])

  test("checked constructors and refinements return pure error data"):
    val shapeError = ShapeError("axis 1 has negative dimension -1")
    val sliceError = SliceError("step must not be zero")
    val rankError = RankMismatch(2, 1)
    val canonicalError = CanonicalLayoutError(
      "canonical linear access requires a whole canonical array"
    )

    assertEquals(Shape.from(Seq(2, -1)), Left(shapeError))
    assertEquals(Slice.from(0, 1, 0), Left(sliceError))
    assertEquals(
      NDArray.zeros[Int](3).transpose2D,
      Left(rankError)
    )
    assertEquals(
      CanonicalArray.from(NDArray.zeros[Int](2, 3).transpose),
      Left(canonicalError)
    )

    List(shapeError, sliceError, rankError, canonicalError).foreach(assertPure)

  test("checked permutations distinguish arity, invalid axes, and duplicates"):
    val source = NDArray.tabulate[Int](2, 3)((row, column) => row * 10 + column)

    assertEquals(
      source.permuteAxesChecked(0),
      Left(InvalidPermutation(expectedRank = 2, receivedAxes = 1))
    )
    assertEquals(
      source.permuteAxesChecked(0, 2),
      Left(InvalidPermutationAxis(axis = 2, rank = 2))
    )
    assertEquals(
      source.permuteAxesChecked(0, -2),
      Left(DuplicateAxis(axis = -2, normalizedAxis = 0))
    )
    assertEquals(source.permuteAxesChecked(1, 0).map(_.shape), Right(Shape(3, 2)))

    val mutable = source.mutableCopy
    assertEquals(
      mutable.permuteAxesChecked(0, 0),
      Left(DuplicateAxis(axis = 0, normalizedAxis = 0))
    )

  test("throwing permutation convenience preserves the precise checked error"):
    val source = NDArray.zeros[Int](2, 3)
    val duplicate = intercept[InvalidPermutationException] {
      source.permuteAxes(0, 0)
    }
    assertEquals(duplicate.error, DuplicateAxis(axis = 0, normalizedAxis = 0))

    val wrongArity = intercept[InvalidPermutationException] {
      source.permuteAxes(0)
    }
    assertEquals(wrongArity.error, InvalidPermutation(expectedRank = 2, receivedAxes = 1))

  test("checked narrow reports its full operation and throwing narrow wraps it"):
    val source = NDArray.tabulate[Int](5)(identity)
    val error = InvalidNarrow(
      axis = 0,
      from = -6,
      length = 1,
      reason = "start -6 is outside [-5, 5]"
    )

    assertEquals(source.narrowChecked(0, -6, 1), Left(error))
    assertPure(error)
    assertEquals(
      source.narrowChecked(-1, -1, 1).map(_.elementsIterator.toList),
      Right(List(4))
    )

    val thrown = intercept[InvalidNarrowException] {
      source.narrow(0, -6, 1)
    }
    assertEquals(thrown.error, error)

  test("throwing constructors use documented typed exceptions"):
    intercept[InvalidShape](Shape(-1))
    intercept[InvalidSlice](Slice(0, 1, 0))
    intercept[NonContiguousLayout] {
      CanonicalArray.require(NDArray.zeros[Int](2, 3).transpose)
    }
