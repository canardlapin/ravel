package ravel.packed

/** Mutable canonical packed workspace.
  *
  * Always dense row-major with a zeroed tail; there are no mutable views. [[freeze]] transfers
  * ownership of the backing words to an immutable [[PackedArray]] without copying and consumes this
  * workspace. Every subsequent data read, write, copying freeze, or ownership-transferring freeze
  * fails with [[PackedWorkspaceConsumedException]]. [[freezeCopy]] leaves the workspace reusable.
  */
final class MutablePackedArray private (
    private[packed] val layout: PackedLayoutPlan,
    val bits: PackedBits,
    initialWords: Array[Int]
):
  private var openBacking: Option[Array[Int]] =
    Some(initialWords)

  val shape: Vector[Int] =
    layout.shape

  val size: Int =
    layout.size

  def codeAt(linear: Int): Int =
    this.synchronized {
      val words = requireOpenBacking()
      require(linear >= 0 && linear < size, s"linear index $linear outside $size")
      readCode(words, linear)
    }

  def setCode(linear: Int, code: Int): Unit =
    this.synchronized {
      val words = requireOpenBacking()
      validateCode(linear, code)
      writeCode(words, linear, code)
    }

  /** Ownership-transferring freeze. This workspace is consumed atomically. */
  def freeze: PackedArray =
    this.synchronized {
      val transferred = requireOpenBacking()
      openBacking = None
      new PackedArray(
        layout,
        bits,
        transferred,
        layout.rowMajorStrides,
        sampleOffset = 0
      )
    }

  /** Copying freeze; the workspace stays writable. */
  def freezeCopy: PackedArray =
    this.synchronized {
      val copied = requireOpenBacking().clone()
      new PackedArray(
        layout,
        bits,
        copied,
        layout.rowMajorStrides,
        sampleOffset = 0
      )
    }

  /** Lock-free write for a freshly allocated workspace that has not escaped its builder. */
  private[packed] def setCodeDuringBuild(linear: Int, code: Int): Unit =
    val words = requireOpenBacking()
    validateCode(linear, code)
    writeCode(words, linear, code)

  private def validateCode(linear: Int, code: Int): Unit =
    require(linear >= 0 && linear < size, s"linear index $linear outside $size")
    require(
      code >= 0 && code <= bits.maxCode,
      s"code $code exceeds maximum ${bits.maxCode}"
    )

  private def readCode(words: Array[Int], linear: Int): Int =
    val wordIndex = linear / bits.codesPerWord
    val slot = linear % bits.codesPerWord
    val shift = slot * bits.bits
    (words(wordIndex) >>> shift) & bits.mask

  private def writeCode(words: Array[Int], linear: Int, code: Int): Unit =
    val wordIndex = linear / bits.codesPerWord
    val slot = linear % bits.codesPerWord
    val shift = slot * bits.bits
    words(wordIndex) = (words(wordIndex) & ~(bits.mask << shift)) | (code << shift)

  private def requireOpenBacking(): Array[Int] =
    openBacking match
      case Some(words) => words
      case None => throw new PackedWorkspaceConsumedException()

/** Raised when a data operation is attempted after [[MutablePackedArray.freeze]]. */
final class PackedWorkspaceConsumedException()
    extends IllegalStateException("mutable packed workspace was consumed by freeze")

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
