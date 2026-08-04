package ravel.packed

/** Immutable N-dimensional array of sub-byte integer codes.
  *
  * Storage layout is canonical row-major over 32-bit words: the code at canonical linear index `k`
  * occupies bits `[(k * bits) mod 32, ...)` of word `(k * bits) / 32`, least-significant bits
  * first. Words are logical `Int` values, so serialized word sequences are endian-independent by
  * construction. Unused bits in the final word of a canonical array are always zero.
  *
  * Storage is one shared `Array[Int]` implementation: the JVM uses a primitive int array and
  * Scala.js compiles the same field to an `Int32Array`, so both platforms get flat word storage
  * without platform-specific source.
  *
  * Views created by [[slice]] and [[narrow]] share backing words and address samples through
  * logical sample strides; [[copy]] re-canonicalizes any view into minimal zero-tailed storage.
  *
  * This is deliberately not a member of the `DType`/`NDArray` family: sub-byte codes are storage
  * codes, not primitive element values.
  */
final class PackedArray private[packed] (
    private[packed] val layout: PackedLayoutPlan,
    val bits: PackedBits,
    private[packed] val words: Array[Int],
    private[packed] val sampleStrides: Vector[Int],
    private[packed] val sampleOffset: Int
):
  val shape: Vector[Int] =
    layout.shape

  /** Number of logical samples. */
  val size: Int =
    layout.size

  def rank: Int =
    shape.length

  /** True when this array owns dense row-major storage with a zeroed tail. */
  def isCanonical: Boolean =
    sampleOffset == 0 &&
      sampleStrides == layout.rowMajorStrides &&
      words.length == PackedArray.wordCount(size, bits)

  /** Read the code at a canonical row-major linear index. */
  def codeAt(linear: Int): Int =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    var remaining = linear
    var axis = rank - 1
    var sample = sampleOffset
    while axis >= 0 do
      val extent = shape(axis)
      sample = PackedLayoutPlan.requireSampleOffset(
        sample,
        remaining % extent,
        sampleStrides(axis)
      )
      remaining /= extent
      axis -= 1
    readSample(sample)

  /** Read the code at explicit indices. */
  def apply(indices: Int*): Int =
    require(indices.length == rank, s"expected $rank indices, got ${indices.length}")
    var axis = 0
    var sample = sampleOffset
    while axis < rank do
      val index = indices(axis)
      require(
        index >= 0 && index < shape(axis),
        s"index $index outside axis $axis extent ${shape(axis)}"
      )
      sample = PackedLayoutPlan.requireSampleOffset(sample, index, sampleStrides(axis))
      axis += 1
    readSample(sample)

  /** Zero-copy view fixing one axis at one index (drops the axis). */
  def slice(axis: Int, index: Int): Either[PackedError, PackedArray] =
    if axis < 0 || axis >= rank then Left(PackedError.InvalidAxis(axis, rank))
    else if index < 0 || index >= shape(axis) then
      Left(PackedError.InvalidRange(axis, index, 1, shape(axis)))
    else if rank == 1 then Left(PackedError.InvalidShape("cannot slice a rank-1 array to rank 0"))
    else
      val nextShape = shape.patch(axis, Nil, 1)
      for
        nextLayout <- PackedLayoutPlan.from(nextShape)
        nextOffset <- PackedLayoutPlan.sampleOffset(sampleOffset, index, sampleStrides(axis))
      yield new PackedArray(
        nextLayout,
        bits,
        words,
        sampleStrides.patch(axis, Nil, 1),
        nextOffset
      )

  /** Zero-copy view restricting one axis to `[start, start + length)`. */
  def narrow(axis: Int, start: Int, length: Int): Either[PackedError, PackedArray] =
    if axis < 0 || axis >= rank then Left(PackedError.InvalidAxis(axis, rank))
    else
      val nextShape = shape.updated(axis, length)
      for
        _ <- PackedLayoutPlan.narrowEnd(axis, start, length, shape(axis))
        nextLayout <- PackedLayoutPlan.from(nextShape)
        nextOffset <- PackedLayoutPlan.sampleOffset(sampleOffset, start, sampleStrides(axis))
      yield new PackedArray(
        nextLayout,
        bits,
        words,
        sampleStrides,
        nextOffset
      )

  /** Materialize into canonical minimal storage with a zeroed tail. */
  def copy: PackedArray =
    val output = MutablePackedArray.zeros(layout, bits)
    var linear = 0
    while linear < size do
      output.setCode(linear, codeAt(linear))
      linear += 1
    output.freeze

  /** Canonical backing words. Copies views into canonical form first. */
  def wordVector: Vector[Int] =
    if isCanonical then words.toVector else copy.wordVector

  /** All codes in canonical row-major order. */
  def codeVector: Vector[Int] =
    Vector.tabulate(size)(codeAt)

  /** Sum of all codes; canonical arrays read each word exactly once. */
  def sumCodes: Long =
    if isCanonical then
      val perWord = bits.codesPerWord
      val mask = bits.mask
      var total = 0L
      var wordIndex = 0
      while wordIndex < words.length do
        var word = words(wordIndex)
        var slot = 0
        while slot < perWord && word != 0 do
          total += (word & mask).toLong
          word = word >>> bits.bits
          slot += 1
        wordIndex += 1
      total
    else
      var total = 0L
      var linear = 0
      while linear < size do
        total += codeAt(linear).toLong
        linear += 1
      total

  private def readSample(sample: Int): Int =
    val wordIndex = sample / bits.codesPerWord
    val slot = sample % bits.codesPerWord
    val shift = slot * bits.bits
    (words(wordIndex) >>> shift) & bits.mask

object PackedArray:
  /** Words required for `samples` codes of the given width. */
  def wordCount(samples: Int, bits: PackedBits): Int =
    PackedLayoutPlan.wordCount(samples, bits) match
      case Right(count) => count
      case Left(error) => throw new IllegalArgumentException(error.message)

  private[packed] def validateShape(
      shape: Vector[Int]
  ): Either[PackedError, Unit] =
    PackedLayoutPlan.from(shape).map(_ => ())

  /** Pack explicit codes in row-major order. */
  def fromCodes(
      shape: Vector[Int],
      bits: PackedBits,
      codes: IterableOnce[Int]
  ): Either[PackedError, PackedArray] =
    PackedLayoutPlan.from(shape).flatMap { layout =>
      val size = layout.size
      val output = MutablePackedArray.zeros(layout, bits)
      val iterator = codes.iterator
      var linear = 0
      var error: Option[PackedError] = None
      while linear < size && error.isEmpty && iterator.hasNext do
        val code = iterator.next()
        if code < 0 || code > bits.maxCode then
          error = Some(PackedError.InvalidCode(linear, code, bits.maxCode))
        else
          output.setCode(linear, code)
          linear += 1
      error match
        case Some(err) => Left(err)
        case None if linear < size || iterator.hasNext =>
          Left(
            PackedError.InvalidShape(
              s"expected $size codes for shape ${shape.mkString("(", ",", ")")}"
            )
          )
        case None =>
          Right(output.freeze)
    }

  /** Tabulate by canonical row-major linear index. Codes are masked to width. */
  def tabulate(
      shape: Vector[Int],
      bits: PackedBits
  )(code: Int => Int): Either[PackedError, PackedArray] =
    PackedLayoutPlan.from(shape).map { layout =>
      val output = MutablePackedArray.zeros(layout, bits)
      var linear = 0
      while linear < output.size do
        output.setCode(linear, code(linear) & bits.mask)
        linear += 1
      output.freeze
    }

  /** All-zero canonical array. */
  def zeros(
      shape: Vector[Int],
      bits: PackedBits
  ): Either[PackedError, PackedArray] =
    PackedLayoutPlan.from(shape).map(layout => MutablePackedArray.zeros(layout, bits).freeze)

  /** Reconstruct from serialized canonical words.
    *
    * Word order and intra-word bit order are part of the format, so a word sequence written on one
    * platform decodes identically on any other. Rejects wrong word counts and nonzero tail bits.
    */
  def fromWords(
      shape: Vector[Int],
      bits: PackedBits,
      words: IterableOnce[Int]
  ): Either[PackedError, PackedArray] =
    PackedLayoutPlan.from(shape).flatMap { layout =>
      val size = layout.size
      val expected = wordCount(size, bits)
      val copied = words.iterator.toArray
      if copied.length != expected then
        Left(PackedError.WordLengthMismatch(expected, copied.length))
      else
        val tailCodes = size % bits.codesPerWord
        val tailWord = if expected == 0 then 0 else copied(expected - 1)
        val unusedMask =
          if tailCodes == 0 then 0 else -1 << (tailCodes * bits.bits)
        if (tailWord & unusedMask) != 0 then Left(PackedError.NonCanonicalTail(tailWord))
        else
          Right(
            new PackedArray(
              layout,
              bits,
              copied,
              layout.rowMajorStrides,
              sampleOffset = 0
            )
          )
    }
