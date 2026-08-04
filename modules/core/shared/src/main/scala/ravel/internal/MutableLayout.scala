package ravel.internal

import ravel.*

/** Provenance wrapper for layouts known to be injective.
  *
  * It has no constructor from arbitrary strides. Instances originate from a canonical owned layout
  * or from transformations that preserve injectivity.
  */
private[ravel] final class MutableLayout private (
    val underlying: Layout
)

private[ravel] object MutableLayout:
  def owned[R <: AnyRank](shape: Shape[R], bufferLength: Int): MutableLayout =
    new MutableLayout(Layout.contiguous(shape, bufferLength))

  def select(
      layout: MutableLayout,
      axis: Int,
      index: Int,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.select(layout.underlying, axis, index, bufferLength))

  def slice(
      layout: MutableLayout,
      axis: Int,
      slice: Slice,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.slice(layout.underlying, axis, slice, bufferLength))

  def narrow(
      layout: MutableLayout,
      plan: NarrowPlan,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.narrow(layout.underlying, plan, bufferLength))

  def reverse(
      layout: MutableLayout,
      axis: Int,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.reverse(layout.underlying, axis, bufferLength))

  def permute(
      layout: MutableLayout,
      plan: PermutationPlan,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.permute(layout.underlying, plan, bufferLength))

  def newAxis(
      layout: MutableLayout,
      axis: Int,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.newAxis(layout.underlying, axis, bufferLength))

  def squeeze(
      layout: MutableLayout,
      axis: Int,
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.squeeze(layout.underlying, axis, bufferLength))

  def reshape[S <: AnyRank](
      layout: MutableLayout,
      target: Shape[S],
      bufferLength: Int
  ): MutableLayout =
    proven(ViewLayout.reshape(layout.underlying, target, bufferLength))

  private def proven(layout: Layout): MutableLayout =
    new MutableLayout(layout)
