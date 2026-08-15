package ravel

import ravel.internal.*

/** Private read source for kernels. It carries no ownership claim. */
private[ravel] trait ArraySource[A, +R <: AnyRank]:
  private[ravel] def storage: Storage[A]
  private[ravel] def layout: Layout
  def dtype: DType[A]
  def shape: Shape[R]
  def rank: Int
  def size: Int

/** Readable owned, borrowed, or explicitly mutable array.
  *
  * Structural operations keep their concrete ownership type. Numerical and extensional operations
  * accept every kind and return owned [[NDArray]] values.
  */
trait ReadableArray[A, +R <: AnyRank] extends ArraySource[A, R]:
  def dtype: DType[A]
  def shape: Shape[R]
  def rank: Int
  def size: Int
  def isContiguous: Boolean
  def isCanonicalLayout: Boolean
  def isWholeBuffer: Boolean
