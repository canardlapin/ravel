package ravel.packed

import ravel.*
import ravel.internal.Layout

/** Mutable canonical packed workspace.
  *
  * [[freeze]] transfers its backing words to an immutable [[PackedArray]] and consumes the
  * workspace. Every later data operation fails with [[PackedWorkspaceConsumedException]].
  * [[freezeCopy]] leaves the workspace reusable.
  */
final class MutablePackedArray[R <: AnyRank] private (
    private[packed] val layout: Layout,
    val bits: PackedBits,
    initialWords: Array[Int]
):
  private var openBacking: Option[Array[Int]] =
    Some(initialWords)

  val shape: Shape[R] =
    Shape.retag[R](layout.shapeValue)

  val size: Int =
    layout.size

  def codeAt(linear: Int): Int =
    this.synchronized {
      val words = requireOpenBacking()
      checkedLinearIndex(linear)
      readCode(words, linear)
    }

  def apply(i: Int)(using R <:< Rank[1]): Int =
    readPhysical(layout.physicalIndex1(i))

  def apply(i: Int, j: Int)(using R <:< Rank[2]): Int =
    readPhysical(layout.physicalIndex2(i, j))

  def apply(i: Int, j: Int, k: Int)(using R <:< Rank[3]): Int =
    readPhysical(layout.physicalIndex3(i, j, k))

  def apply(i: Int, j: Int, k: Int, l: Int)(using R <:< Rank[4]): Int =
    readPhysical(layout.physicalIndex4(i, j, k, l))

  def at(indices: IArray[Int]): Int =
    readPhysical(layout.physicalIndex(indices))

  def setCode(linear: Int, code: Int): Unit =
    this.synchronized {
      val words = requireOpenBacking()
      checkedLinearIndex(linear)
      validateCode(code)
      writeCode(words, linear, code)
    }

  def updateAt(indices: IArray[Int], code: Int): Unit =
    this.synchronized {
      val words = requireOpenBacking()
      validateCode(code)
      writeCode(words, layout.physicalIndex(indices), code)
    }

  private def readPhysical(index: Int): Int =
    this.synchronized {
      readCode(requireOpenBacking(), index)
    }

  /** Ownership-transferring freeze. This workspace is consumed atomically. */
  def freeze: PackedArray[R] =
    this.synchronized {
      val transferred = requireOpenBacking()
      openBacking = None
      new PackedArray[R](layout, bits, transferred, size)
    }

  /** Copying freeze; the workspace stays writable. */
  def freezeCopy: PackedArray[R] =
    this.synchronized {
      new PackedArray[R](layout, bits, requireOpenBacking().clone(), size)
    }

  /** Lock-free write for a fresh workspace that has not escaped its builder. */
  private[packed] def setCodeDuringBuild(linear: Int, code: Int): Unit =
    val words = requireOpenBacking()
    checkedLinearIndex(linear)
    validateCode(code)
    writeCode(words, linear, code)

  private def checkedLinearIndex(linear: Int): Unit =
    if linear < 0 || linear >= size then throw InvalidIndex.LinearOutOfBounds(linear, size)

  private def validateCode(code: Int): Unit =
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

final class PackedWorkspaceConsumedException()
    extends IllegalStateException("mutable packed workspace was consumed by freeze")

object MutablePackedArray:
  private[packed] def zeros[R <: AnyRank](
      shape: Shape[R],
      bits: PackedBits
  ): MutablePackedArray[R] =
    val layout = Layout.contiguous(shape, shape.size)
    new MutablePackedArray[R](
      layout,
      bits,
      new Array[Int](PackedArray.wordCount(shape.size, bits))
    )

  def allocate[R <: AnyRank](shape: Shape[R], bits: PackedBits): MutablePackedArray[R] =
    zeros(shape, bits)
