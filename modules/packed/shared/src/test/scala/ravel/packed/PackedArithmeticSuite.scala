package ravel.packed

import munit.FunSuite

final class PackedArithmeticSuite extends FunSuite:
  test("checked shape planning accepts the Int limit and rejects overflowing products"):
    val oneAxis = planRight(PackedLayoutPlan.from(Vector(Int.MaxValue)))
    assertEquals(oneAxis.size, Int.MaxValue)
    assertEquals(oneAxis.rowMajorStrides, Vector(1))

    val leadingSingleton = planRight(PackedLayoutPlan.from(Vector(1, Int.MaxValue)))
    assertEquals(leadingSingleton.size, Int.MaxValue)
    assertEquals(leadingSingleton.rowMajorStrides, Vector(Int.MaxValue, 1))

    assertInvalidShape(PackedLayoutPlan.from(Vector(Int.MaxValue, 2)))
    assertInvalidShape(PackedLayoutPlan.from(Vector(46341, 46341)))
    assertInvalidShape(PackedArray.zeros(Vector(Int.MaxValue, 2), PackedBits.B1))
    assertInvalidShape(MutablePackedArray.allocate(Vector(Int.MaxValue, 2), PackedBits.B4))

  test("word-count rounding stays checked at Int.MaxValue"):
    assertEquals(PackedLayoutPlan.wordCount(0, PackedBits.B1), Right(0))
    assertEquals(
      PackedLayoutPlan.wordCount(Int.MaxValue, PackedBits.B1),
      Right(67108864)
    )
    assertEquals(
      PackedLayoutPlan.wordCount(Int.MaxValue, PackedBits.B2),
      Right(134217728)
    )
    assertEquals(
      PackedLayoutPlan.wordCount(Int.MaxValue, PackedBits.B4),
      Right(268435456)
    )
    assert(PackedLayoutPlan.wordCount(-1, PackedBits.B1).isLeft)

  test("offset transforms detect overflow without allocating backing storage"):
    assertEquals(
      PackedLayoutPlan.sampleOffset(Int.MaxValue - 1, 1, 1),
      Right(Int.MaxValue)
    )
    PackedLayoutPlan.sampleOffset(Int.MaxValue, 1, 1) match
      case Left(PackedError.AddressOverflow(base, index, stride)) =>
        assertEquals(base, Int.MaxValue)
        assertEquals(index, 1)
        assertEquals(stride, 1)
      case other => fail(s"expected AddressOverflow, got $other")
    assert(PackedLayoutPlan.sampleOffset(0, Int.MaxValue, Int.MaxValue).isLeft)

  test("narrow endpoints use widened arithmetic"):
    assertEquals(
      PackedLayoutPlan.narrowEnd(0, Int.MaxValue - 1, 1, Int.MaxValue),
      Right(Int.MaxValue)
    )
    PackedLayoutPlan.narrowEnd(0, Int.MaxValue, 1, Int.MaxValue) match
      case Left(error @ PackedError.InvalidRange(_, _, _, _)) =>
        assert(error.message.contains("2147483648"))
      case other => fail(s"expected InvalidRange, got $other")

  private def planRight(
      value: Either[PackedError, PackedLayoutPlan]
  ): PackedLayoutPlan =
    value match
      case Right(plan) => plan
      case Left(error) => fail(error.message)

  private def assertInvalidShape[A](value: Either[PackedError, A]): Unit =
    value match
      case Left(PackedError.InvalidShape(_)) => ()
      case other => fail(s"expected InvalidShape, got $other")
