package ravel

import ravel.internal.*

/** A read-only array view whose storage remains externally mutable.
  *
  * This type is deliberately not a subtype of [[NDArray]]. Structural views retain the borrowed
  * marker; computations and materialization return owned `NDArray` values via [[ReadableArray]]
  * operations.
  */
final class BorrowedNDArray[A, R <: AnyRank] private[ravel] (
    private[ravel] val underlying: NDArray[A, R]
) extends ReadableArray[A, R]:
  val dtype: DType[A] = underlying.dtype
  val shape: Shape[R] = underlying.shape
  def rank: Int = underlying.rank
  def size: Int = underlying.size
  def isContiguous: Boolean = underlying.isContiguous
  def isCanonicalLayout: Boolean = underlying.isCanonicalLayout
  def isWholeBuffer: Boolean = underlying.isWholeBuffer

  private[ravel] def storage: Storage[A] = underlying.storage
  private[ravel] def layout: Layout = underlying.layout

  inline def apply(i: Int)(using R <:< Rank[1]): A = underlying(i)
  inline def apply(i: Int, j: Int)(using R <:< Rank[2]): A = underlying(i, j)
  inline def apply(i: Int, j: Int, k: Int)(using R <:< Rank[3]): A = underlying(i, j, k)
  inline def apply(i: Int, j: Int, k: Int, l: Int)(using R <:< Rank[4]): A =
    underlying(i, j, k, l)
  def at(indices: IArray[Int]): A = underlying.at(indices)
  def foreachElement(f: A => Unit): Unit = underlying.foreachElement(f)
  def foreachIndex(f: IArray[Int] => Unit): Unit = underlying.foreachIndex(f)
  def elementsIterator: Iterator[A] = underlying.elementsIterator
  def iterator: Iterator[A] = underlying.iterator

  def requireRank[N <: Int](using
      expected: ValueOf[N]
  ): Either[RankMismatch, BorrowedNDArray[A, Rank[N]]] =
    if rank == expected.value then Right(this.asInstanceOf[BorrowedNDArray[A, Rank[N]]])
    else Left(RankMismatch(expected.value, rank))

  def select(axis: Int, index: Int)(using
      CanDropAxis[R]
  ): BorrowedNDArray[A, DropAxis[R]] =
    new BorrowedNDArray(underlying.select(axis, index))

  def slice(axis: Int, slice: Slice): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.slice(axis, slice))

  def slice(axis: Int, range: Range): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.slice(axis, range))

  /** Checked exact narrow that retains borrowed provenance. */
  def narrowChecked(
      axis: Int,
      from: Int,
      length: Int
  ): Either[InvalidNarrow, BorrowedNDArray[A, R]] =
    underlying.narrowChecked(axis, from, length).map(new BorrowedNDArray(_))

  def narrow(axis: Int, from: Int, length: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.narrow(axis, from, length))

  def reverse(axis: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.reverse(axis))

  def swapAxes(first: Int, second: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.swapAxes(first, second))

  /** Checked axis permutation that retains borrowed provenance. */
  def permuteAxesChecked(
      order: Int*
  ): Either[PermutationError, BorrowedNDArray[A, R]] =
    underlying.permuteAxesChecked(order*).map(new BorrowedNDArray(_))

  def permuteAxes(order: Int*): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.permuteAxes(order*))

  def transpose(using R <:< Rank[2]): BorrowedNDArray[A, Rank[2]] =
    new BorrowedNDArray(underlying.transpose)

  /** Checked rank-two transpose that retains borrowed provenance. */
  def transpose2D: Either[RankMismatch, BorrowedNDArray[A, Rank[2]]] =
    underlying.transpose2D.map(new BorrowedNDArray(_))

  def newAxis(axis: Int): BorrowedNDArray[A, AddAxis[R]] =
    new BorrowedNDArray(underlying.newAxis(axis))

  def squeeze(axis: Int)(using
      CanDropAxis[R]
  ): BorrowedNDArray[A, DropAxis[R]] =
    new BorrowedNDArray(underlying.squeeze(axis))

  def broadcastTo[S <: AnyRank](
      target: Shape[S]
  ): BorrowedNDArray[A, S] =
    new BorrowedNDArray(underlying.broadcastTo(target))

  /** Reshapes without copying and retains borrowed provenance. */
  def reshapeView[S <: AnyRank](
      target: Shape[S]
  ): BorrowedNDArray[A, S] =
    new BorrowedNDArray(underlying.reshapeView(target))

  /** Reshapes through an owned logical-order copy. */
  def reshapeCopy[S <: AnyRank](target: Shape[S]): NDArray[A, S] =
    underlying.reshapeCopy(target)

  /** Always materializes owned storage. */
  def copy: NDArray[A, R] = underlying.copy

  /** Borrowed contiguity cannot erase ownership provenance, so this copies. */
  def contiguous: NDArray[A, R] = underlying.copy

  def flattenCopy: Array1[A] = underlying.flattenCopy

  def cast[B](using
      source: NumericDType[A],
      target: NumericDType[B]
  ): NDArray[B, R] =
    underlying.cast[B]

  def convert[B](
      policy: ConversionPolicy = ConversionPolicy()
  )(using
      source: NumericDType[A],
      target: NumericDType[B]
  ): Either[ConversionError, NDArray[B, R]] =
    underlying.convert[B](policy)

  def item: A = underlying.item

  def sameElements(other: ReadableArray[A, ?]): Boolean =
    EqualityApi.sameElements(this, other)

  def sameElementsBits(other: ReadableArray[A, ?]): Boolean =
    EqualityApi.sameElementsBits(this, other)

  override def toString: String =
    s"Borrowed${underlying.toString}"
