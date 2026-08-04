package ravel.packed

/** Wordwise set algebra over one-bit packed arrays.
  *
  * Every operation processes thirty-two samples per instruction and never expands codes to
  * `Boolean` values. Inputs are canonicalized on entry if they are views; canonical inputs are
  * combined without any per-sample work.
  */
object PackedBitOps:
  def union[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, PackedArray[R]] =
    canonicalPair(left, right).map { (leftCanonical, rightCanonical) =>
      val output = new Array[Int](leftCanonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = leftCanonical.words(index) | rightCanonical.words(index)
        index += 1
      finish(leftCanonical, output)
    }

  def intersection[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, PackedArray[R]] =
    canonicalPair(left, right).map { (leftCanonical, rightCanonical) =>
      val output = new Array[Int](leftCanonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = leftCanonical.words(index) & rightCanonical.words(index)
        index += 1
      finish(leftCanonical, output)
    }

  /** Samples present in `left` and absent from `right`. */
  def difference[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, PackedArray[R]] =
    canonicalPair(left, right).map { (leftCanonical, rightCanonical) =>
      val output = new Array[Int](leftCanonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = leftCanonical.words(index) & ~rightCanonical.words(index)
        index += 1
      finish(leftCanonical, output)
    }

  def xor[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, PackedArray[R]] =
    canonicalPair(left, right).map { (leftCanonical, rightCanonical) =>
      val output = new Array[Int](leftCanonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = leftCanonical.words(index) ^ rightCanonical.words(index)
        index += 1
      finish(leftCanonical, output)
    }

  def symmetricDifference[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, PackedArray[R]] =
    xor(left, right)

  def complement[R <: ravel.AnyRank](
      input: PackedArray[R]
  ): Either[PackedError, PackedArray[R]] =
    requireOneBit(input).map { canonical =>
      val output = new Array[Int](canonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = ~canonical.words(index)
        index += 1
      zeroTail(output, canonical.size)
      PackedArray.canonical(canonical.shape, PackedBits.B1, output)
    }

  /** Number of set samples, thirty-two samples per popcount. */
  def countTrue(input: PackedArray[?]): Either[PackedError, Long] =
    requireOneBit(input).map { canonical =>
      var total = 0L
      var index = 0
      while index < canonical.words.length do
        total += Integer.bitCount(canonical.words(index)).toLong
        index += 1
      total
    }

  private def canonicalPair[R <: ravel.AnyRank](
      left: PackedArray[R],
      right: PackedArray[?]
  ): Either[PackedError, (PackedArray[R], PackedArray[?])] =
    if left.bits != right.bits then Left(PackedError.BitsMismatch(left.bits, right.bits))
    else if left.shape != right.shape then Left(PackedError.ShapeMismatch(left.shape, right.shape))
    else
      for
        leftCanonical <- requireOneBit(left)
        rightCanonical <- requireOneBit(right)
      yield (leftCanonical, rightCanonical)

  private def finish[R <: ravel.AnyRank](
      left: PackedArray[R],
      output: Array[Int]
  ): PackedArray[R] =
    zeroTail(output, left.size)
    PackedArray.canonical(left.shape, PackedBits.B1, output)

  private def requireOneBit[R <: ravel.AnyRank](
      input: PackedArray[R]
  ): Either[PackedError, PackedArray[R]] =
    if input.bits != PackedBits.B1 then Left(PackedError.NotOneBit(input.bits))
    else if input.isCanonical then Right(input)
    else Right(input.copy)

  private def zeroTail(words: Array[Int], samples: Int): Unit =
    val tail = samples & 31
    if tail != 0 then words(words.length - 1) &= ~(-1 << tail)
