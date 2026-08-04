package ravel.packed

/** Mutable canonical packed workspace.
  *
  * Always dense row-major with a zeroed tail; there are no mutable views. [[freeze]] transfers
  * ownership of the backing words to an immutable [[PackedArray]] without copying, after which this
  * workspace must not be written again. [[freezeCopy]] leaves the workspace reusable.
  */
final class MutablePackedArray private (
    private[packed] val layout: PackedLayoutPlan,
    val bits: PackedBits,
    private[packed] val words: Array[Int]
):
  val shape: Vector[Int] =
    layout.shape

  val size: Int =
    layout.size

  def codeAt(linear: Int): Int =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    val wordIndex = linear / bits.codesPerWord
    val slot = linear % bits.codesPerWord
    val shift = slot * bits.bits
    (words(wordIndex) >>> shift) & bits.mask

  def setCode(linear: Int, code: Int): Unit =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    require(
      code >= 0 && code <= bits.maxCode,
      s"code $code exceeds maximum ${bits.maxCode}"
    )
    val wordIndex = linear / bits.codesPerWord
    val slot = linear % bits.codesPerWord
    val shift = slot * bits.bits
    words(wordIndex) = (words(wordIndex) & ~(bits.mask << shift)) | (code << shift)

  /** Ownership-transferring freeze; do not mutate this workspace afterwards. */
  def freeze: PackedArray =
    new PackedArray(
      layout,
      bits,
      words,
      layout.rowMajorStrides,
      sampleOffset = 0
    )

  /** Copying freeze; the workspace stays writable. */
  def freezeCopy: PackedArray =
    new PackedArray(
      layout,
      bits,
      words.clone(),
      layout.rowMajorStrides,
      sampleOffset = 0
    )

object MutablePackedArray:
  /** Zeroed canonical workspace. Shape must already be validated. */
  private[packed] def zeros(
      layout: PackedLayoutPlan,
      bits: PackedBits
  ): MutablePackedArray =
    new MutablePackedArray(
      layout,
      bits,
      new Array[Int](PackedArray.wordCount(layout.size, bits))
    )

  def allocate(
      shape: Vector[Int],
      bits: PackedBits
  ): Either[PackedError, MutablePackedArray] =
    PackedLayoutPlan.from(shape).map(layout => zeros(layout, bits))
