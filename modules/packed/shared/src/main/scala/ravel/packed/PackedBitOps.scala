package ravel.packed

/** Wordwise set algebra over one-bit packed arrays.
  *
  * Every operation processes thirty-two samples per instruction and never expands codes to
  * `Boolean` values. Inputs are canonicalized on entry if they are views; canonical inputs are
  * combined without any per-sample work.
  */
object PackedBitOps:
  def union(
      left: PackedArray,
      right: PackedArray
  ): Either[PackedError, PackedArray] =
    combine(left, right)((a, b) => a | b)

  def intersection(
      left: PackedArray,
      right: PackedArray
  ): Either[PackedError, PackedArray] =
    combine(left, right)((a, b) => a & b)

  /** Samples present in `left` and absent from `right`. */
  def difference(
      left: PackedArray,
      right: PackedArray
  ): Either[PackedError, PackedArray] =
    combine(left, right)((a, b) => a & ~b)

  def symmetricDifference(
      left: PackedArray,
      right: PackedArray
  ): Either[PackedError, PackedArray] =
    combine(left, right)((a, b) => a ^ b)

  def complement(input: PackedArray): Either[PackedError, PackedArray] =
    requireOneBit(input).map { canonical =>
      val output = new Array[Int](canonical.words.length)
      var index = 0
      while index < output.length do
        output(index) = ~canonical.words(index)
        index += 1
      zeroTail(output, canonical.size)
      fromCanonicalWords(canonical.layout, output)
    }

  /** Number of set samples, thirty-two samples per popcount. */
  def countTrue(input: PackedArray): Either[PackedError, Long] =
    requireOneBit(input).map { canonical =>
      var total = 0L
      var index = 0
      while index < canonical.words.length do
        total += Integer.bitCount(canonical.words(index)).toLong
        index += 1
      total
    }

  private def combine(
      left: PackedArray,
      right: PackedArray
  )(op: (Int, Int) => Int): Either[PackedError, PackedArray] =
    if left.bits != right.bits then Left(PackedError.BitsMismatch(left.bits, right.bits))
    else if left.shape != right.shape then Left(PackedError.ShapeMismatch(left.shape, right.shape))
    else
      for
        leftCanonical <- requireOneBit(left)
        rightCanonical <- requireOneBit(right)
      yield
        val output = new Array[Int](leftCanonical.words.length)
        var index = 0
        while index < output.length do
          output(index) = op(leftCanonical.words(index), rightCanonical.words(index))
          index += 1
        zeroTail(output, leftCanonical.size)
        fromCanonicalWords(leftCanonical.layout, output)

  private def requireOneBit(
      input: PackedArray
  ): Either[PackedError, PackedArray] =
    if input.bits != PackedBits.B1 then Left(PackedError.NotOneBit(input.bits))
    else if input.isCanonical then Right(input)
    else Right(input.copy)

  private def zeroTail(words: Array[Int], samples: Int): Unit =
    val tail = samples & 31
    if tail != 0 then words(words.length - 1) &= ~(-1 << tail)

  private def fromCanonicalWords(
      layout: PackedLayoutPlan,
      words: Array[Int]
  ): PackedArray =
    new PackedArray(
      layout,
      PackedBits.B1,
      words,
      layout.rowMajorStrides,
      sampleOffset = 0
    )
