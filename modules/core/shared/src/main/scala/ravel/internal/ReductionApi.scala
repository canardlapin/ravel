package ravel.internal

import ravel.*

private[ravel] object ReductionApi:
  def sumAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxis(array, axis, keep, ReductionKernels.sumAxis)

  def productAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxis(array, axis, keep, ReductionKernels.productAxis)

  def minimumAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxis(array, axis, keep, ReductionKernels.minimumAxis)

  def maximumAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxis(array, axis, keep, ReductionKernels.maximumAxis)

  def meanAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxis(array, axis, keep, ReductionKernels.meanAxis)

  def argMinimumAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[Int, Out] =
    argAxis(array, axis, keep, ReductionKernels.argMinimumAxis)

  def argMaximumAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean
  ): NDArray[Int, Out] =
    argAxis(array, axis, keep, ReductionKernels.argMaximumAxis)

  def sumAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxes(array, axes, keep, ReductionKernels.sumAxes)

  def productAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxes(array, axes, keep, ReductionKernels.productAxes)

  def minimumAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxes(array, axes, keep, ReductionKernels.minimumAxes)

  def maximumAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxes(array, axes, keep, ReductionKernels.maximumAxes)

  def meanAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean
  ): NDArray[A, Out] =
    sameDTypeAxes(array, axes, keep, ReductionKernels.meanAxes)

  private def sameDTypeAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean,
      run: (Storage[A], Storage[A], ReductionPlan) => Unit
  ): NDArray[A, Out] =
    val plan = ReductionPlan(array.layout, axis, keep)
    val shape = Shape.validated[Out](plan.outputShape)
    val output = ProbeApi.allocate[A](plan.outputSize)(using array.dtype)
    run(array.storage, output, plan)
    new NDArray(output, Layout.contiguous(shape, plan.outputSize), array.dtype)

  private def argAxis[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axis: Int,
      keep: Boolean,
      run: (Storage[A], Storage[Int], ReductionPlan) => Unit
  ): NDArray[Int, Out] =
    val plan = ReductionPlan(array.layout, axis, keep)
    val shape = Shape.validated[Out](plan.outputShape)
    val output = ProbeApi.allocate[Int](plan.outputSize)(using DType.intDType)
    run(array.storage, output, plan)
    new NDArray(output, Layout.contiguous(shape, plan.outputSize), DType.intDType)

  private def sameDTypeAxes[A, R <: AnyRank, Out <: AnyRank](
      array: ArraySource[A, R],
      axes: Axes,
      keep: Boolean,
      run: (Storage[A], Storage[A], MultiReductionPlan) => Unit
  ): NDArray[A, Out] =
    val plan = MultiReductionPlan(array.layout, axes, keep)
    val shape = Shape.validated[Out](plan.outputShape)
    val output = ProbeApi.allocate[A](plan.outputSize)(using array.dtype)
    run(array.storage, output, plan)
    new NDArray(output, Layout.contiguous(shape, plan.outputSize), array.dtype)
