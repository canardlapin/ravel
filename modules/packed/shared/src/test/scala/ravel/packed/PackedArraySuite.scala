package ravel.packed

import munit.FunSuite

final class PackedArraySuite extends FunSuite:
  private val widths = Vector(PackedBits.B1, PackedBits.B2, PackedBits.B4)

  test("codes round-trip at sizes that do not fill the final word"):
    widths.foreach { bits =>
      val shape = Vector(5, 7)
      val codes = Vector.tabulate(35)(index => (index * 3 + 1) % (bits.maxCode + 1))
      val packed = packedRight(PackedArray.fromCodes(shape, bits, codes))

      assertEquals(packed.codeVector, codes, s"width $bits")
      assertEquals(packed.size, 35)
      assert(packed.isCanonical)
      var linear = 0
      while linear < 35 do
        assertEquals(packed.codeAt(linear), codes(linear))
        assertEquals(packed(linear / 7, linear % 7), codes(linear))
        linear += 1
    }

  test("construction rejects out-of-range codes and wrong code counts"):
    PackedArray.fromCodes(Vector(4), PackedBits.B2, Vector(0, 1, 4, 0)) match
      case Left(PackedError.InvalidCode(index, code, maxCode)) =>
        assertEquals(index, 2)
        assertEquals(code, 4)
        assertEquals(maxCode, 3)
      case other => fail(s"expected InvalidCode, got $other")
    assert(PackedArray.fromCodes(Vector(4), PackedBits.B1, Vector(1, 0, 1)).isLeft)
    assert(PackedArray.fromCodes(Vector(2), PackedBits.B1, Vector(1, 0, 1)).isLeft)
    assert(PackedArray.fromCodes(Vector(0), PackedBits.B1, Vector.empty).isLeft)

  test("word layout is least-significant-bit first and fixed across platforms"):
    val oneBit =
      packedRight(
        PackedArray.fromCodes(
          Vector(8),
          PackedBits.B1,
          Vector(1, 0, 1, 0, 1, 0, 1, 0)
        )
      )
    assertEquals(oneBit.wordVector, Vector(0x55))

    val fourBit =
      packedRight(
        PackedArray.fromCodes(
          Vector(6),
          PackedBits.B4,
          Vector(0x1, 0x2, 0x3, 0x4, 0x5, 0x6)
        )
      )
    assertEquals(fourBit.wordVector, Vector(0x654321))

    val twoBit =
      packedRight(
        PackedArray.fromCodes(
          Vector(4),
          PackedBits.B2,
          Vector(3, 0, 2, 1)
        )
      )
    assertEquals(twoBit.wordVector, Vector((1 << 6) | (2 << 4) | 3))

  test("serialized words round-trip and reject non-canonical input"):
    val shape = Vector(3, 11)
    val codes = Vector.tabulate(33)(index => index % 4)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B2, codes))
    val restored =
      packedRight(PackedArray.fromWords(shape, PackedBits.B2, packed.wordVector))

    assertEquals(restored.codeVector, codes)
    PackedArray.fromWords(shape, PackedBits.B2, packed.wordVector.dropRight(1)) match
      case Left(PackedError.WordLengthMismatch(expected, actual)) =>
        assertEquals(expected, 3)
        assertEquals(actual, 2)
      case other => fail(s"expected WordLengthMismatch, got $other")
    val corrupted =
      packed.wordVector.updated(2, packed.wordVector(2) | (1 << 5))
    PackedArray.fromWords(shape, PackedBits.B2, corrupted) match
      case Left(PackedError.NonCanonicalTail(_)) => ()
      case other => fail(s"expected NonCanonicalTail, got $other")

  test("slice and narrow views agree with naive extraction and re-canonicalize"):
    val shape = Vector(4, 6)
    val codes = Vector.tabulate(24)(index => index % 16)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B4, codes))

    val row = packedRight(packed.slice(0, 2))
    assertEquals(row.shape, Vector(6))
    assertEquals(row.codeVector, Vector.tabulate(6)(column => codes(2 * 6 + column)))
    assert(!row.isCanonical)

    val block = packedRight(packedRight(packed.narrow(0, 1, 2)).narrow(1, 2, 3))
    assertEquals(block.shape, Vector(2, 3))
    assertEquals(
      block.codeVector,
      Vector.tabulate(6)(linear => codes((1 + linear / 3) * 6 + 2 + linear % 3))
    )

    val copied = block.copy
    assert(copied.isCanonical)
    assertEquals(copied.codeVector, block.codeVector)
    assertEquals(
      copied.wordVector.length,
      PackedArray.wordCount(copied.size, PackedBits.B4)
    )

    assert(packed.slice(2, 0).isLeft)
    assert(packed.narrow(1, 4, 3).isLeft)

  test("sumCodes agrees between canonical fast path and view fallback"):
    val shape = Vector(9, 5)
    val codes = Vector.tabulate(45)(index => (index * 7) % 4)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B2, codes))
    val view = packedRight(packed.narrow(0, 1, 7))

    assertEquals(packed.sumCodes, codes.map(_.toLong).sum)
    assertEquals(view.sumCodes, view.codeVector.map(_.toLong).sum)

  test("mutable workspace freezes by ownership transfer and by copy"):
    val workspace =
      packedRight(MutablePackedArray.allocate(Vector(10), PackedBits.B2))
    var linear = 0
    while linear < 10 do
      workspace.setCode(linear, linear % 4)
      linear += 1
    val copied = workspace.freezeCopy
    workspace.setCode(0, 3)
    val frozen = workspace.freeze

    assertEquals(copied.codeAt(0), 0)
    assertEquals(frozen.codeAt(0), 3)
    assertEquals(copied.codeVector.drop(1), frozen.codeVector.drop(1))

  private def packedRight[A](value: Either[PackedError, A]): A =
    value match
      case Right(result) => result
      case Left(error) => fail(error.message)
