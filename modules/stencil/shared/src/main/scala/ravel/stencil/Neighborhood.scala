package ravel.stencil

import ravel.AnyRank
import ravel.MutableNDArray
import ravel.NDArray

/** Accumulator contract for one destination sample's neighborhood. */
trait NeighborhoodReducer[A, Acc, B]:
  def zero: Acc
  def accumulate(acc: Acc, value: A, offsetIndex: Int): Acc
  def finish(acc: Acc): B

/** Primitive-double reducer used by direct floating-point executors.
  *
  * Its `Double` methods keep the numeric accumulator out of `Any`-typed generic reducer slots. This
  * is the production surface for floating filters; [[NeighborhoodReducer]] remains the generic
  * reference surface.
  */
trait DoubleNeighborhoodReducer:
  def zero: Double
  def accumulate(acc: Double, value: Double, offsetIndex: Int): Double
  def finish(acc: Double): Double

/** Primitive-float reducer used by direct floating-point executors. */
trait FloatNeighborhoodReducer:
  def zero: Float
  def accumulate(acc: Float, value: Float, offsetIndex: Int): Float
  def finish(acc: Float): Float

/** Primitive-Boolean reducer used by direct binary morphology executors. */
trait BooleanNeighborhoodReducer:
  def zero: Boolean
  def accumulate(acc: Boolean, value: Boolean, offsetIndex: Int): Boolean
  def finish(acc: Boolean): Boolean

  /** Return true when the accumulator is final and remaining offsets are irrelevant. */
  def isTerminal(acc: Boolean): Boolean = false

/** Primitive-byte reducer used by direct morphology-style executors. */
trait ByteNeighborhoodReducer:
  def zero: Byte
  def accumulate(acc: Byte, value: Byte, offsetIndex: Int): Byte
  def finish(acc: Byte): Byte

/** Primitive-short reducer used by direct morphology-style executors. */
trait ShortNeighborhoodReducer:
  def zero: Short
  def accumulate(acc: Short, value: Short, offsetIndex: Int): Short
  def finish(acc: Short): Short

/** Description of one neighborhood pass over leading spatial axes.
  *
  * Non-spatial trailing axes are treated as an independent batch: each batch member is processed
  * with the same spatial neighborhood.
  *
  * @param spatialAxes
  *   number of leading axes that participate in the neighborhood
  * @param offsets
  *   each offset has length `spatialAxes`
  * @param outputOrigin
  *   source-space coordinate of destination index 0 on each spatial axis (0 for Same, left
  *   footprint for Valid, `-left` for Full)
  * @param outputSpatialShape
  *   spatial shape of the destination
  */
final case class NeighborhoodSpec(
    spatialAxes: Int,
    offsets: Vector[Vector[Int]],
    border: BorderMode,
    outputOrigin: Vector[Int],
    outputSpatialShape: Vector[Int]
):
  def validate(): Unit =
    if spatialAxes < 1 then
      throw IllegalArgumentException(
        s"spatialAxes must be >= 1, got $spatialAxes"
      )
    if offsets.isEmpty then throw IllegalArgumentException("offsets must be non-empty")
    if outputOrigin.length != spatialAxes then
      throw IllegalArgumentException(
        s"outputOrigin rank ${outputOrigin.length} != spatialAxes $spatialAxes"
      )
    if outputSpatialShape.length != spatialAxes then
      throw IllegalArgumentException(
        s"outputSpatialShape rank ${outputSpatialShape.length} != spatialAxes $spatialAxes"
      )
    if outputSpatialShape.exists(_ < 0) then
      throw IllegalArgumentException(
        s"outputSpatialShape must be nonnegative, got $outputSpatialShape"
      )
    offsets.foreach { offset =>
      if offset.length != spatialAxes then
        throw IllegalArgumentException(
          s"offset rank ${offset.length} != spatialAxes $spatialAxes"
        )
    }

/** Generic neighborhood executor over Ravel arrays.
  *
  * Implementations must understand arbitrary positive, negative, broadcast, and permuted strides.
  * They must not interpret image roles, physical spacing, or display semantics.
  */
trait NeighborhoodExecutor:
  def run[A, Acc, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      reducer: NeighborhoodReducer[A, Acc, B],
      constant: A
  ): Unit
