package ravel.packed

import munit.FunSuite
import ravel.Shape

final class PackedBitOpsSuite extends FunSuite:
  private val sizes = Vector(31, 32, 33, 100)

  test("wordwise algebra agrees with the Boolean reference at word boundaries"):
    sizes.foreach { size =>
      val leftBooleans = Vector.tabulate(size)(index => (index * 5) % 7 < 3)
      val rightBooleans = Vector.tabulate(size)(index => (index * 3) % 5 < 2)
      val left = mask(size, leftBooleans)
      val right = mask(size, rightBooleans)

      assertEquals(
        codes(packedRight(PackedBitOps.union(left, right))),
        leftBooleans.zip(rightBooleans).map(_ || _),
        s"union at size $size"
      )
      assertEquals(
        codes(packedRight(PackedBitOps.intersection(left, right))),
        leftBooleans.zip(rightBooleans).map(_ && _),
        s"intersection at size $size"
      )
      assertEquals(
        codes(packedRight(PackedBitOps.difference(left, right))),
        leftBooleans.zip(rightBooleans).map((l, r) => l && !r),
        s"difference at size $size"
      )
      assertEquals(
        codes(packedRight(PackedBitOps.symmetricDifference(left, right))),
        leftBooleans.zip(rightBooleans).map(_ != _),
        s"symmetric difference at size $size"
      )
      assertEquals(
        codes(packedRight(PackedBitOps.complement(left))),
        leftBooleans.map(!_),
        s"complement at size $size"
      )
      assertEquals(
        packedRight(PackedBitOps.countTrue(left)),
        leftBooleans.count(identity).toLong,
        s"countTrue at size $size"
      )
    }

  test("results keep the canonical zero tail invariant"):
    val size = 33
    val everything =
      packedRight(
        PackedBitOps.complement(
          PackedArray.zeros(Shape(size), PackedBits.B1)
        )
      )

    assertEquals(everything.codeVector, Vector.fill(size)(1))
    assertEquals(everything.wordVector(1), 1)
    assertEquals(packedRight(PackedBitOps.countTrue(everything)), size.toLong)
    val roundTrip =
      PackedArray.fromWords(Shape(size), PackedBits.B1, everything.wordVector)
    assert(roundTrip.isRight, "complement output must stay serializable")

  test("views are canonicalized before wordwise combination"):
    val base =
      packedRight(
        PackedArray.fromCodes(
          Shape(4, 10),
          PackedBits.B1,
          Vector.tabulate(40)(index => index % 3 == 0).map(if _ then 1 else 0)
        )
      )
    val view = base.narrow(0, 1, 2)
    val other =
      packedRight(
        PackedArray.fromCodes(
          Shape(2, 10),
          PackedBits.B1,
          Vector.tabulate(20)(index => index % 2)
        )
      )
    val union = packedRight(PackedBitOps.union(view, other))

    assertEquals(
      union.codeVector,
      view.codeVector.zip(other.codeVector).map(_ | _)
    )

  test("set algebra rejects incompatible operands"):
    val oneBit = PackedArray.zeros(Shape(8), PackedBits.B1)
    val twoBit = PackedArray.zeros(Shape(8), PackedBits.B2)
    val shorter = PackedArray.zeros(Shape(4), PackedBits.B1)

    PackedBitOps.union(oneBit, twoBit) match
      case Left(PackedError.BitsMismatch(_, _)) => ()
      case other => fail(s"expected BitsMismatch, got $other")
    PackedBitOps.union(oneBit, shorter) match
      case Left(PackedError.ShapeMismatch(_, _)) => ()
      case other => fail(s"expected ShapeMismatch, got $other")
    PackedBitOps.complement(twoBit) match
      case Left(PackedError.NotOneBit(PackedBits.B2)) => ()
      case other => fail(s"expected NotOneBit, got $other")

  private def mask(size: Int, values: Vector[Boolean]): PackedArray[?] =
    packedRight(
      PackedArray.fromCodes(
        Shape(size),
        PackedBits.B1,
        values.map(if _ then 1 else 0)
      )
    )

  private def codes(packed: PackedArray[?]): Vector[Boolean] =
    packed.codeVector.map(_ == 1)

  private def packedRight[A](value: Either[PackedError, A]): A =
    value match
      case Right(result) => result
      case Left(error) => fail(error.message)
