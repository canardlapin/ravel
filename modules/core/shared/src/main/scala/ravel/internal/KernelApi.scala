package ravel.internal

import ravel.*

private[ravel] object KernelApi:
  def copy[A, R <: AnyRank](source: ArraySource[A, R]): NDArray[A, R] =
    val output = ProbeApi.allocate[A](source.size)(using source.dtype)
    CopyKernels.logical(source.storage, source.layout, output)
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      source.dtype
    )

  def unary[A, R <: AnyRank](
      operation: Byte,
      source: ArraySource[A, R]
  ): NDArray[A, R] =
    val plan = LoopPlan.unary(source.layout)
    val output = ProbeApi.allocate[A](source.size)(using source.dtype)
    ElementKernels.unary(operation, source.storage, output, plan)
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      source.dtype
    )

  def scalar[A, R <: AnyRank](
      operation: Byte,
      source: ArraySource[A, R],
      value: A
  ): NDArray[A, R] =
    val plan = LoopPlan.unary(source.layout)
    val output = ProbeApi.allocate[A](source.size)(using source.dtype)
    ElementKernels.scalar(operation, source.storage, value, output, plan)
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      source.dtype
    )

  def clip[A, R <: AnyRank](
      source: ArraySource[A, R],
      lower: A,
      upper: A
  ): NDArray[A, R] =
    val plan = LoopPlan.unary(source.layout)
    val output = ProbeApi.allocate[A](source.size)(using source.dtype)
    ElementKernels.clip(source.storage, lower, upper, output, plan)
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      source.dtype
    )

  def binary[A, RX <: AnyRank, RY <: AnyRank](
      operation: Byte,
      left: ArraySource[A, RX],
      right: ArraySource[A, RY]
  ): NDArray[A, BroadcastRank[RX, RY]] =
    if left.layout.isCContiguous &&
      right.layout.isCContiguous &&
      Layout.sameShape(left.layout.shape, right.layout.shape)
    then
      val output = ProbeApi.allocate[A](left.size)(using left.dtype)
      if operation == KernelOp.Add then
        if left.layout.offset == 0 && right.layout.offset == 0 then
          ProbeKernels.add(left.storage, right.storage, output, left.size)
        else
          ProbeKernels.addLinear(
            left.storage,
            left.layout.offset,
            right.storage,
            right.layout.offset,
            output,
            left.size
          )
      else
        val plan = LoopPlan.linearSameShape(left.layout, right.layout)
        ElementKernels.binary(operation, left.storage, right.storage, output, plan)
      new NDArray(
        output,
        Layout.contiguous(Shape.retag[BroadcastRank[RX, RY]](left.shape), left.size),
        left.dtype
      )
    else
      val plan = LoopPlan.broadcast(left.layout, right.layout)
      val shape =
        Shape.validated[BroadcastRank[RX, RY]](plan.resultShape)
      val output = ProbeApi.allocate[A](plan.size)(using left.dtype)
      ElementKernels.binary(operation, left.storage, right.storage, output, plan)
      new NDArray(output, Layout.contiguous(shape, plan.size), left.dtype)

  def compare[A, RX <: AnyRank, RY <: AnyRank](
      operation: Byte,
      left: ArraySource[A, RX],
      right: ArraySource[A, RY]
  ): NDArray[Boolean, BroadcastRank[RX, RY]] =
    val plan = LoopPlan.broadcast(left.layout, right.layout)
    val shape = Shape.validated[BroadcastRank[RX, RY]](plan.resultShape)
    val output = ProbeApi.allocate[Boolean](plan.size)(using DType.booleanDType)
    ElementKernels.compare(operation, left.storage, right.storage, output, plan)
    new NDArray(output, Layout.contiguous(shape, plan.size), DType.booleanDType)

  def orderedCompareScalar[A, R <: AnyRank](
      operation: Byte,
      source: ArraySource[A, R],
      value: A,
      scalarLeft: Boolean
  ): NDArray[Boolean, R] =
    val primitiveFloating =
      source.layout.isCContiguous &&
        (source.storage match
          case _: FloatStorage => true
          case _: DoubleStorage => true
          case _ => false)
    if primitiveFloating then
      val output = ProbeApi.allocate[Boolean](source.size)(using DType.booleanDType)
      ElementKernels.orderedCompareContiguousScalar(
        operation,
        source.storage,
        value,
        output,
        source.layout.offset,
        source.size,
        scalarLeft
      )
      new NDArray(
        output,
        Layout.contiguous(source.shape, source.size),
        DType.booleanDType
      )
    else
      val scalarArray = NDArray.scalar(value)(using source.dtype)
      val result =
        if scalarLeft then compare(operation, scalarArray, source)
        else compare(operation, source, scalarArray)
      result.asInstanceOf[NDArray[Boolean, R]]

  def floatingPredicate[A, R <: AnyRank](
      operation: Byte,
      source: ArraySource[A, R]
  ): NDArray[Boolean, R] =
    val plan = LoopPlan.unary(source.layout)
    val output = ProbeApi.allocate[Boolean](source.size)(using DType.booleanDType)
    ElementKernels.floatingPredicate(operation, source.storage, output, plan)
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      DType.booleanDType
    )

  def map[A, B, R <: AnyRank](
      source: ArraySource[A, R],
      f: A => B
  )(using target: DType[B]): NDArray[B, R] =
    val output = ProbeApi.allocate[B](source.size)
    if source.layout.isCContiguous then
      var read = source.layout.offset
      var write = 0
      while write < source.size do
        val value = f(ProbeApi.get(source.storage, read))
        ProbeApi.set(output, write, value)
        read += 1
        write += 1
    else
      var write = 0
      source.layout.foreachPhysicalIndex { physical =>
        val value = f(ProbeApi.get(source.storage, physical))
        ProbeApi.set(output, write, value)
        write += 1
      }
    new NDArray(
      output,
      Layout.contiguous(source.shape, source.size),
      target
    )

  def zipMap[A, B, RX <: AnyRank, RY <: AnyRank](
      left: ArraySource[A, RX],
      right: ArraySource[A, RY],
      exact: Boolean,
      f: (A, A) => B
  )(using target: DType[B]): NDArray[B, BroadcastRank[RX, RY]] =
    val plan =
      if exact then LoopPlan.exact(left.layout, right.layout)
      else LoopPlan.broadcast(left.layout, right.layout)
    val shape = Shape.validated[BroadcastRank[RX, RY]](plan.resultShape)
    val output = ProbeApi.allocate[B](plan.size)
    plan.foreachOffset { (leftIndex, rightIndex, outputIndex) =>
      val value = f(
        ProbeApi.get(left.storage, leftIndex),
        ProbeApi.get(right.storage, rightIndex)
      )
      ProbeApi.set(output, outputIndex, value)
    }
    new NDArray(output, Layout.contiguous(shape, plan.size), target)
