package ravel.packed

import munit.FunSuite
import ravel.*
import ravel.internal.{Layout, NarrowPlan}

final class PackedArithmeticSuite extends FunSuite:
  test("packed construction reuses checked core Shape limits"):
    val oneAxis = shapeRight(Shape.from(Vector(Int.MaxValue)))
    assertEquals(oneAxis.size, Int.MaxValue)

    assert(Shape.from(Vector(Int.MaxValue, 2)).isLeft)
    assert(Shape.from(Vector(46341, 46341)).isLeft)

    val zeroDominated = shapeRight(Shape.from(Vector(0, Int.MaxValue, Int.MaxValue)))
    intercept[LayoutOverflow](PackedArray.zeros(zeroDominated, PackedBits.B1))
    intercept[LayoutOverflow](MutablePackedArray.allocate(zeroDominated, PackedBits.B4))

  test("word-count rounding stays checked at Int.MaxValue"):
    assertEquals(PackedArray.wordCount(0, PackedBits.B1), 0)
    assertEquals(PackedArray.wordCount(Int.MaxValue, PackedBits.B1), 67108864)
    assertEquals(PackedArray.wordCount(Int.MaxValue, PackedBits.B2), 134217728)
    assertEquals(PackedArray.wordCount(Int.MaxValue, PackedBits.B4), 268435456)
    intercept[InvalidShape](PackedArray.wordCount(-1, PackedBits.B1))

  test("core layout and narrow planners widen boundary arithmetic"):
    assertEquals(Layout.checkedInt(Int.MaxValue.toLong, "test"), Int.MaxValue)
    intercept[LayoutOverflow](Layout.checkedInt(Int.MaxValue.toLong + 1L, "test"))

    assertEquals(
      NarrowPlan.from(IArray(Int.MaxValue), 0, Int.MaxValue - 1, 1).map(_.stopExclusive),
      Right(Int.MaxValue)
    )
    assert(NarrowPlan.from(IArray(Int.MaxValue), 0, Int.MaxValue, 1).isLeft)

  private def shapeRight(value: Either[ShapeError, Shape[AnyRank]]): Shape[AnyRank] =
    value match
      case Right(shape) => shape
      case Left(error) => fail(error.reason)
