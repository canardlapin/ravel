package ravel

import ravel.internal.EqualityApi
import scala.annotation.unused

extension [A, R <: AnyRank](array: NDArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def allClose(
      other: NDArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  ): Boolean =
    EqualityApi.allClose(
      array,
      other,
      relativeTolerance,
      absoluteTolerance
    )

  def allClose(
      other: BorrowedNDArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  ): Boolean =
    EqualityApi.allClose(
      array,
      other.underlying,
      relativeTolerance,
      absoluteTolerance
    )
