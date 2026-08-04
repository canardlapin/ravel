package ravel.packed

import ravel.Shape

/** Closed family of supported sub-byte code widths.
  *
  * Thirty-two is divisible by every member, so a code never straddles a word boundary and every
  * read is one shift and one mask.
  */
enum PackedBits(val bits: Int) derives CanEqual:
  case B1 extends PackedBits(1)
  case B2 extends PackedBits(2)
  case B4 extends PackedBits(4)

  /** Codes stored in one 32-bit word. */
  def codesPerWord: Int =
    32 / bits

  /** Inclusive maximum code value. */
  def maxCode: Int =
    (1 << bits) - 1

  /** All-ones mask of `bits` width. */
  def mask: Int =
    (1 << bits) - 1

/** Errors raised while constructing or combining packed arrays. */
enum PackedError derives CanEqual:
  case InvalidShape(detail: String)
  case InvalidCode(index: Int, code: Int, maxCode: Int)
  case WordLengthMismatch(expected: Int, actual: Int)
  case NonCanonicalTail(word: Int)
  case ShapeMismatch(left: Shape[?], right: Shape[?])
  case BitsMismatch(left: PackedBits, right: PackedBits)
  case NotOneBit(bits: PackedBits)

  def message: String =
    this match
      case InvalidShape(detail) =>
        detail
      case InvalidCode(index, code, maxCode) =>
        s"code $code at linear index $index exceeds maximum $maxCode"
      case WordLengthMismatch(expected, actual) =>
        s"expected $expected backing words, got $actual"
      case NonCanonicalTail(word) =>
        s"unused tail bits must be zero, got 0x${word.toHexString}"
      case ShapeMismatch(left, right) =>
        s"shapes $left and $right differ"
      case BitsMismatch(left, right) =>
        s"code widths $left and $right differ"
      case NotOneBit(bits) =>
        s"wordwise set algebra requires one-bit codes, got $bits"
