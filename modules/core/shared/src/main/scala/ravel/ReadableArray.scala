package ravel

/** Readable owned or borrowed array.
  *
  * Structural operations keep their concrete ownership type. Numerical and extensional operations
  * accept either kind and return owned [[NDArray]] values.
  */
trait ReadableArray[A, +R <: AnyRank]:
  def dtype: DType[A]
  def shape: Shape[R]
  def rank: Int
  def size: Int
  def isContiguous: Boolean
  def isCanonicalLayout: Boolean
  def isWholeBuffer: Boolean

  private[ravel] def toNDArray: NDArray[A, R]
