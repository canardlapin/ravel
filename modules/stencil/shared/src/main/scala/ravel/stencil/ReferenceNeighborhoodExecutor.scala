package ravel.stencil

import ravel.AnyRank
import ravel.MutableNDArray
import ravel.NDArray

/** Clarity-first neighborhood executor; the conformance oracle for optimized engines.
  *
  * Traverses destination samples in C-order over the full rank. Leading `spatialAxes` participate
  * in the neighborhood; trailing axes are held fixed (batch axes).
  */
object ReferenceNeighborhoodExecutor extends NeighborhoodExecutor:
  def run[A, Acc, B, R <: AnyRank](
      source: NDArray[A, R],
      destination: MutableNDArray[B, R],
      spec: NeighborhoodSpec,
      reducer: NeighborhoodReducer[A, Acc, B],
      constant: A,
      policy: StencilExecutionPolicy = StencilExecutionPolicy()
  ): Unit =
    val _ = policy
    spec.validate()
    val rank = source.rank
    if destination.rank != rank then
      throw IllegalArgumentException(
        s"source rank $rank != destination rank ${destination.rank}"
      )
    if spec.spatialAxes > rank then
      throw IllegalArgumentException(
        s"spatialAxes ${spec.spatialAxes} exceeds array rank $rank"
      )
    val batchAxes = rank - spec.spatialAxes
    var axis = 0
    while axis < spec.spatialAxes do
      if destination.shape(axis) != spec.outputSpatialShape(axis) then
        throw IllegalArgumentException(
          s"destination spatial shape mismatch on axis $axis"
        )
      axis += 1
    while axis < rank do
      if destination.shape(axis) != source.shape(axis) then
        throw IllegalArgumentException(
          s"batch axis $axis shape mismatch between source and destination"
        )
      axis += 1

    val sourceExtents =
      Vector.tabulate(spec.spatialAxes)(source.shape(_))
    val destIndices = new Array[Int](rank)
    val sourceIndices = new Array[Int](rank)
    val total = destination.size
    var linear = 0
    while linear < total do
      unravel(linear, destination, destIndices)
      var acc = reducer.zero
      var offsetIndex = 0
      while offsetIndex < spec.offsets.length do
        val offset = spec.offsets(offsetIndex)
        var spatial = 0
        var outside = false
        while spatial < spec.spatialAxes && !outside do
          val logical =
            spec.outputOrigin(spatial) + destIndices(spatial) + offset(spatial)
          BorderIndex.map(logical, sourceExtents(spatial), spec.border) match
            case MappedIndex.Inside(mapped) =>
              sourceIndices(spatial) = mapped
            case MappedIndex.Outside =>
              outside = true
          spatial += 1
        var batch = 0
        while batch < batchAxes do
          sourceIndices(spec.spatialAxes + batch) = destIndices(spec.spatialAxes + batch)
          batch += 1
        val sample =
          if outside then constant
          else source.at(IArray.unsafeFromArray(sourceIndices.clone()))
        acc = reducer.accumulate(acc, sample, offsetIndex)
        offsetIndex += 1
      destination.updateAt(
        IArray.unsafeFromArray(destIndices.clone()),
        reducer.finish(acc)
      )
      linear += 1

  private def unravel[A, R <: AnyRank](
      linear: Int,
      array: MutableNDArray[A, R],
      out: Array[Int]
  ): Unit =
    var rem = linear
    var axis = array.rank - 1
    while axis >= 0 do
      val extent = array.shape(axis)
      out(axis) = rem % extent
      rem /= extent
      axis -= 1
