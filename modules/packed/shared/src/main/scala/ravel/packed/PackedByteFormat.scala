package ravel.packed

import ravel.{AnyRank, Shape}

/** Version 1 portable packed representation.
  *
  * Integers are big-endian. Canonical words retain packed storage's logical order and put the
  * earliest code in the least-significant slot of each word. Unused high tail bits are zero.
  */
private[packed] object PackedByteFormat:
  private val Magic0 = 'R'.toByte
  private val Magic1 = 'V'.toByte
  private val Magic2 = 'P'.toByte
  private val Magic3 = 'K'.toByte
  private val Version = 1
  private val HeaderBytes = 12

  def encode(input: PackedArray[?]): Either[PackedError, IArray[Byte]] =
    val canonical = if input.isCanonical then input else input.copy
    val required =
      HeaderBytes.toLong + canonical.rank.toLong * 4L + canonical.words.length.toLong * 4L
    if required > Int.MaxValue.toLong then Left(PackedError.ByteLengthOverflow(required))
    else
      val output = new Array[Byte](required.toInt)
      output(0) = Magic0
      output(1) = Magic1
      output(2) = Magic2
      output(3) = Magic3
      output(4) = Version.toByte
      output(5) = canonical.bits.bits.toByte
      output(6) = 0
      output(7) = 0
      writeInt(output, 8, canonical.rank)
      var cursor = HeaderBytes
      var axis = 0
      while axis < canonical.rank do
        writeInt(output, cursor, canonical.shape(axis))
        cursor += 4
        axis += 1
      var word = 0
      while word < canonical.words.length do
        writeInt(output, cursor, canonical.words(word))
        cursor += 4
        word += 1
      Right(IArray.unsafeFromArray(output))

  def decode(bytes: IArray[Byte]): Either[PackedError, PackedArray[AnyRank]] =
    if bytes.length < HeaderBytes then invalid(s"truncated header: ${bytes.length} bytes")
    else if bytes(0) != Magic0 || bytes(1) != Magic1 || bytes(2) != Magic2 || bytes(3) != Magic3
    then invalid("magic must be RVPK")
    else if unsigned(bytes(4)) != Version then invalid(s"unsupported version ${unsigned(bytes(4))}")
    else if bytes(6) != 0 || bytes(7) != 0 then invalid("reserved header bytes must be zero")
    else
      val bitsResult =
        unsigned(bytes(5)) match
          case 1 => Right(PackedBits.B1)
          case 2 => Right(PackedBits.B2)
          case 4 => Right(PackedBits.B4)
          case other => invalid(s"unsupported code width $other")
      bitsResult.flatMap { bits =>
        val rank = readInt(bytes, 8)
        if rank < 0 then invalid(s"rank must be nonnegative, got $rank")
        else if rank > (bytes.length - HeaderBytes) / 4 then invalid("truncated dimensions")
        else
          val dimensions = new Array[Int](rank)
          var cursor = HeaderBytes
          var axis = 0
          while axis < rank do
            dimensions(axis) = readInt(bytes, cursor)
            cursor += 4
            axis += 1
          Shape.from(IArray.unsafeFromArray(dimensions)) match
            case Left(error) => invalid(error.reason)
            case Right(shape) =>
              val words = PackedArray.wordCount(shape.size, bits)
              val expectedLength = cursor.toLong + words.toLong * 4L
              if expectedLength != bytes.length.toLong then
                invalid(s"expected $expectedLength bytes, got ${bytes.length}")
              else
                val decoded = new Array[Int](words)
                var word = 0
                while word < words do
                  decoded(word) = readInt(bytes, cursor)
                  cursor += 4
                  word += 1
                PackedArray.fromWords(shape, bits, IArray.unsafeFromArray(decoded))
      }

  private def invalid[A](detail: String): Either[PackedError, A] =
    Left(PackedError.InvalidByteFormat(detail))

  private def unsigned(value: Byte): Int =
    value & 0xff

  private def writeInt(output: Array[Byte], offset: Int, value: Int): Unit =
    output(offset) = (value >>> 24).toByte
    output(offset + 1) = (value >>> 16).toByte
    output(offset + 2) = (value >>> 8).toByte
    output(offset + 3) = value.toByte

  private def readInt(input: IArray[Byte], offset: Int): Int =
    unsigned(input(offset)) << 24 |
      unsigned(input(offset + 1)) << 16 |
      unsigned(input(offset + 2)) << 8 |
      unsigned(input(offset + 3))
