package ravel.packed

/** Mutable canonical packed workspace.
  *
  * Always dense row-major with a zeroed tail; there are no mutable views. [[freeze]] transfers
  * ownership of the backing words to an immutable [[PackedArray]] without copying, after which this
  * workspace must not be written again. [[freezeCopy]] leaves the workspace reusable.
  */
final class MutablePackedArray private (
    val shape: Vector[Int],
    val bits: PackedBits,
    private[packed] val words: Array[Int]
):
  val size: Int =
    shape.product

  def codeAt(linear: Int): Int =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    val bitIndex = linear * bits.bits
    (words(bitIndex >>> 5) >>> (bitIndex & 31)) & bits.mask

  def setCode(linear: Int, code: Int): Unit =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    require(
      code >= 0 && code <= bits.maxCode,
      s"code $code exceeds maximum ${bits.maxCode}"
    )
    val bitIndex = linear * bits.bits
    val wordIndex = bitIndex >>> 5
    val shift = bitIndex & 31
    words(wordIndex) = (words(wordIndex) & ~(bits.mask << shift)) | (code << shift)

  /** Ownership-transferring freeze; do not mutate this workspace afterwards. */
  def freeze: PackedArray =
    new PackedArray(
      shape,
      bits,
      words,
      PackedArray.rowMajorStrides(shape),
      sampleOffset = 0
    )

  /** Copying freeze; the workspace stays writable. */
  def freezeCopy: PackedArray =
    new PackedArray(
      shape,
      bits,
      words.clone(),
      PackedArray.rowMajorStrides(shape),
      sampleOffset = 0
    )

object MutablePackedArray:
  /** Zeroed canonical workspace. Shape must already be validated. */
  private[packed] def zeros(
      shape: Vector[Int],
      bits: PackedBits
  ): MutablePackedArray =
    val samples = shape.product
    new MutablePackedArray(
      shape,
      bits,
      new Array[Int](PackedArray.wordCount(samples, bits))
    )

  def allocate(
      shape: Vector[Int],
      bits: PackedBits
  ): Either[PackedError, MutablePackedArray] =
    PackedArray.validateShape(shape).map(_ => zeros(shape, bits))
