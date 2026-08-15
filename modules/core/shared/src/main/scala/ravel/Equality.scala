package ravel

import ravel.internal.EqualityApi
import scala.annotation.unused

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def allClose(
      other: ReadableArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  ): Boolean =
    EqualityApi.allClose(
      array,
      other,
      relativeTolerance,
      absoluteTolerance
    )
