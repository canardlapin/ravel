package ravel.packed

import munit.FunSuite
import ravel.{Shape, Slice}

final class PackedByteFormatSuite extends FunSuite:
  private val widths = Vector(PackedBits.B1, PackedBits.B2, PackedBits.B4)

  test("portable bytes round-trip every width, views, scalars, and empty shapes"):
    widths.foreach { bits =>
      val shape = Shape(3, 5)
      val codes = Vector.tabulate(shape.size)(index => (index * 3 + 1) & bits.mask)
      val packed = packedRight(PackedArray.fromCodes(shape, bits, codes))
      val view = packed.reverse(0).slice(1, Slice.reverse)
      val restored = packedRight(PackedArray.fromBytes(bytesRight(view.toBytes)))

      assertEquals(restored.shape, view.shape, s"view shape at $bits")
      assertEquals(restored.bits, bits, s"view width at $bits")
      assertEquals(restored.codeVector, view.codeVector, s"view codes at $bits")
      assert(restored.isCanonical, s"view must serialize canonically at $bits")

      val scalar = packedRight(PackedArray.scalar(bits.maxCode, bits))
      val scalarRestored = packedRight(PackedArray.fromBytes(bytesRight(scalar.toBytes)))
      assertEquals(scalarRestored.shape, Shape.scalar, s"scalar shape at $bits")
      assertEquals(scalarRestored.item, bits.maxCode, s"scalar code at $bits")

      val empty = PackedArray.zeros(Shape(2, 0, 3), bits)
      val emptyRestored = packedRight(PackedArray.fromBytes(bytesRight(empty.toBytes)))
      assertEquals(emptyRestored.shape, empty.shape, s"empty shape at $bits")
      assertEquals(emptyRestored.codeVector, Vector.empty, s"empty codes at $bits")
    }

  test("version 1 bytes have an exact cross-platform representation"):
    val packed =
      packedRight(
        PackedArray.fromCodes(
          Shape(2, 3),
          PackedBits.B2,
          Vector(0, 1, 2, 3, 1, 0)
        )
      )
    val actual = IArray.genericWrapArray(bytesRight(packed.toBytes)).toVector.map(_ & 0xff)

    assertEquals(
      actual,
      Vector(
        0x52, 0x56, 0x50, 0x4b, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x01, 0xe4
      )
    )

  test("decoder rejects malformed headers, lengths, shapes, and tail bits"):
    val packed = packedRight(PackedArray.fromCodes(Shape(5), PackedBits.B2, Vector(0, 1, 2, 3, 0)))
    val valid = mutableBytes(bytesRight(packed.toBytes))

    assertInvalid(valid.take(4), "truncated header")
    assertInvalid(valid.updated(0, 0), "magic")
    assertInvalid(valid.updated(4, 2), "version")
    assertInvalid(valid.updated(5, 3), "width")
    assertInvalid(valid.updated(6, 1), "reserved")
    assertInvalid(valid.updated(11, 3), "truncated dimensions")
    assertInvalid(valid.updated(12, 0x80.toByte), "negative dimension")
    assertInvalid(valid :+ 0, "extra payload byte")

    widths.foreach { bits =>
      val tailSource =
        packedRight(
          PackedArray.fromCodes(
            Shape(5),
            bits,
            Vector.tabulate(5)(index => index & bits.mask)
          )
        )
      val tailBytes = mutableBytes(bytesRight(tailSource.toBytes))
      val nonCanonicalTail =
        tailBytes.updated(
          tailBytes.length - 4,
          (tailBytes(tailBytes.length - 4) | 0x80).toByte
        )
      PackedArray.fromBytes(immutableBytes(nonCanonicalTail)) match
        case Left(PackedError.NonCanonicalTail(_)) => ()
        case other => fail(s"expected NonCanonicalTail for $bits, got $other")
    }

  private def assertInvalid(bytes: Array[Byte], clue: String): Unit =
    PackedArray.fromBytes(immutableBytes(bytes)) match
      case Left(PackedError.InvalidByteFormat(_)) => ()
      case other => fail(s"expected InvalidByteFormat for $clue, got $other")

  private def mutableBytes(bytes: IArray[Byte]): Array[Byte] =
    IArray.genericWrapArray(bytes).toArray

  private def immutableBytes(bytes: Array[Byte]): IArray[Byte] =
    IArray.unsafeFromArray(bytes)

  private def bytesRight(value: Either[PackedError, IArray[Byte]]): IArray[Byte] =
    packedRight(value)

  private def packedRight[A](value: Either[PackedError, A]): A =
    value match
      case Right(result) => result
      case Left(error) => fail(error.message)
