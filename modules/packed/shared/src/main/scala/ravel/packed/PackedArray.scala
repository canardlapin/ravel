package ravel.packed

import ravel.*
import ravel.internal.{Layout, NarrowPlan, PermutationPlan, ViewLayout}

/** Immutable N-dimensional array of sub-byte integer codes.
  *
  * Packed arrays reuse core [[Shape]], [[Slice]], index normalization, exact narrow planning, and
  * axis-permutation validation. Only the storage representation is separate: logical samples are
  * stored least-significant slot first in 32-bit words, with a zeroed unused tail.
  */
final class PackedArray[+R <: AnyRank] private[packed] (
    private[packed] val layout: Layout,
    val bits: PackedBits,
    private[packed] val words: Array[Int],
    private[packed] val sampleCapacity: Int
):
  val shape: Shape[R] =
    Shape.retag[R](layout.shapeValue)

  val size: Int =
    layout.size

  def rank: Int =
    layout.rank

  /** True when this value covers its complete canonical sample buffer. */
  def isCanonical: Boolean =
    layout.isCanonicalLayout &&
      layout.isWholeBuffer(sampleCapacity) &&
      words.length == PackedArray.wordCount(sampleCapacity, bits)

  /** Read a logical C-order linear index in `[0, size)`. */
  def codeAt(linear: Int): Int =
    if linear < 0 || linear >= size then throw InvalidIndex.LinearOutOfBounds(linear, size)
    readSample(physicalIndexAtLinear(linear))

  /** Read fixed-rank logical coordinates. Negative indices follow core normalization. */
  def apply(i: Int)(using R <:< Rank[1]): Int =
    readSample(layout.physicalIndex1(i))

  def apply(i: Int, j: Int)(using R <:< Rank[2]): Int =
    readSample(layout.physicalIndex2(i, j))

  def apply(i: Int, j: Int, k: Int)(using R <:< Rank[3]): Int =
    readSample(layout.physicalIndex3(i, j, k))

  def apply(i: Int, j: Int, k: Int, l: Int)(using R <:< Rank[4]): Int =
    readSample(layout.physicalIndex4(i, j, k, l))

  /** Dynamic-rank coordinate access. */
  def at(indices: IArray[Int]): Int =
    readSample(layout.physicalIndex(indices))

  /** Single code when `size == 1`; otherwise throws [[InvalidShape]]. */
  def item: Int =
    if size != 1 then throw InvalidShape(s"item requires size 1, found size $size")
    readSample(layout.offset)

  /** Zero-copy axis-dropping selection. */
  def select(axis: Int, index: Int): PackedArray[AnyRank] =
    new PackedArray[AnyRank](
      ViewLayout.select(layout, axis, index, sampleCapacity),
      bits,
      words,
      sampleCapacity
    )

  /** Zero-copy rank-preserving slice. */
  def slice(axis: Int, slice: Slice): PackedArray[R] =
    new PackedArray[R](
      ViewLayout.slice(layout, axis, slice, sampleCapacity),
      bits,
      words,
      sampleCapacity
    )

  def slice(axis: Int, range: Range): PackedArray[R] =
    val checked = Slice.from(range).fold(error => throw InvalidSlice(error.reason), identity)
    slice(axis, checked)

  /** Checked exact, non-clipping narrow with negative-start normalization. */
  def narrowChecked(
      axis: Int,
      from: Int,
      length: Int
  ): Either[InvalidNarrow, PackedArray[R]] =
    NarrowPlan.from(layout.shape, axis, from, length).map { plan =>
      new PackedArray[R](
        ViewLayout.narrow(layout, plan, sampleCapacity),
        bits,
        words,
        sampleCapacity
      )
    }

  def narrow(axis: Int, from: Int, length: Int): PackedArray[R] =
    narrowChecked(axis, from, length).fold(
      error => throw InvalidNarrowException(error),
      identity
    )

  def reverse(axis: Int): PackedArray[R] =
    new PackedArray[R](
      ViewLayout.reverse(layout, axis, sampleCapacity),
      bits,
      words,
      sampleCapacity
    )

  def swapAxes(first: Int, second: Int): PackedArray[R] =
    val left = layout.normalizedAxis(first)
    val right = layout.normalizedAxis(second)
    val order = Array.tabulate(rank)(identity)
    val temporary = order(left)
    order(left) = order(right)
    order(right) = temporary
    permuteAxes(order.toSeq*)

  def permuteAxesChecked(order: Int*): Either[PermutationError, PackedArray[R]] =
    PermutationPlan.from(rank, order).map { plan =>
      new PackedArray[R](
        ViewLayout.permute(layout, plan, sampleCapacity),
        bits,
        words,
        sampleCapacity
      )
    }

  def permuteAxes(order: Int*): PackedArray[R] =
    permuteAxesChecked(order*).fold(
      error => throw InvalidPermutationException(error),
      identity
    )

  /** Materialize into canonical minimal storage with a zeroed tail. */
  def copy: PackedArray[R] =
    val output = MutablePackedArray.zeros(shape, bits)
    var linear = 0
    while linear < size do
      output.setCodeDuringBuild(linear, codeAt(linear))
      linear += 1
    output.freeze

  /** Canonical backing words. Views are copied into canonical form first. */
  def wordVector: Vector[Int] =
    if isCanonical then words.toVector else copy.wordVector

  /** All codes in logical C order. */
  def codeVector: Vector[Int] =
    Vector.tabulate(size)(codeAt)

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

  private def physicalIndexAtLinear(linear: Int): Int =
    if rank == 0 then layout.offset
    else
      var remaining = linear
      var axis = rank - 1
      var sample = layout.offset.toLong
      while axis >= 0 do
        val extent = layout.shape(axis)
        sample = Layout.checkedAdd(
          sample,
          Layout.checkedMultiply(
            (remaining % extent).toLong,
            layout.strides(axis).toLong,
            s"packed logical linear axis $axis"
          ),
          s"packed logical linear axis $axis"
        )
        remaining /= extent
        axis -= 1
      Layout.checkedInt(sample, "packed logical linear index")

  private def readSample(sample: Int): Int =
    val wordIndex = sample / bits.codesPerWord
    val slot = sample % bits.codesPerWord
    val shift = slot * bits.bits
    (words(wordIndex) >>> shift) & bits.mask

object PackedArray:
  private[packed] def wordCount(samples: Int, bits: PackedBits): Int =
    if samples < 0 then throw InvalidShape(s"sample count must be nonnegative, got $samples")
    else if samples == 0 then 0
    else 1 + (samples - 1) / bits.codesPerWord

  private[packed] def canonical[R <: AnyRank](
      shape: Shape[R],
      bits: PackedBits,
      words: Array[Int]
  ): PackedArray[R] =
    new PackedArray[R](
      Layout.contiguous(shape, shape.size),
      bits,
      words,
      shape.size
    )

  def scalar(code: Int, bits: PackedBits): Either[PackedError, PackedArray[Rank[0]]] =
    fromCodes(Shape.scalar, bits, Iterator.single(code))

  /** Pack explicit codes in logical C order. */
  def fromCodes[R <: AnyRank](
      shape: Shape[R],
      bits: PackedBits,
      codes: IterableOnce[Int]
  ): Either[PackedError, PackedArray[R]] =
    val output = MutablePackedArray.zeros(shape, bits)
    val iterator = codes.iterator
    var linear = 0
    var error: Option[PackedError] = None
    while linear < shape.size && error.isEmpty && iterator.hasNext do
      val code = iterator.next()
      if code < 0 || code > bits.maxCode then
        error = Some(PackedError.InvalidCode(linear, code, bits.maxCode))
      else
        output.setCodeDuringBuild(linear, code)
        linear += 1
    error match
      case Some(value) => Left(value)
      case None if linear < shape.size || iterator.hasNext =>
        Left(PackedError.InvalidShape(s"expected ${shape.size} codes for shape $shape"))
      case None => Right(output.freeze)

  /** Tabulate by logical C-order linear index. Codes are masked to width. */
  def tabulate[R <: AnyRank](
      shape: Shape[R],
      bits: PackedBits
  )(code: Int => Int): Either[PackedError, PackedArray[R]] =
    val output = MutablePackedArray.zeros(shape, bits)
    var linear = 0
    while linear < output.size do
      output.setCodeDuringBuild(linear, code(linear) & bits.mask)
      linear += 1
    Right(output.freeze)

  def zeros[R <: AnyRank](shape: Shape[R], bits: PackedBits): PackedArray[R] =
    MutablePackedArray.zeros(shape, bits).freeze

  /** Reconstruct from canonical logical words. */
  def fromWords[R <: AnyRank](
      shape: Shape[R],
      bits: PackedBits,
      words: IterableOnce[Int]
  ): Either[PackedError, PackedArray[R]] =
    val expected = wordCount(shape.size, bits)
    val copied = words.iterator.toArray
    if copied.length != expected then Left(PackedError.WordLengthMismatch(expected, copied.length))
    else
      val tailCodes = shape.size % bits.codesPerWord
      val tailWord = if expected == 0 then 0 else copied(expected - 1)
      val unusedMask =
        if tailCodes == 0 then 0 else -1 << (tailCodes * bits.bits)
      if (tailWord & unusedMask) != 0 then Left(PackedError.NonCanonicalTail(tailWord))
      else Right(canonical(shape, bits, copied))

type PackedArray0 = PackedArray[Rank[0]]
type PackedArray1 = PackedArray[Rank[1]]
type PackedArray2 = PackedArray[Rank[2]]
type PackedArray3 = PackedArray[Rank[3]]
type PackedArray4 = PackedArray[Rank[4]]
