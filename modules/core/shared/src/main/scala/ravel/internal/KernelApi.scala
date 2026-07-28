package ravel.internal

import ravel.*

private[ravel] object KernelApi:
  def unary[A, R <: AnyRank](
      operation: Byte,
      source: NDArray[A, R]
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
      source: NDArray[A, R],
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
      source: NDArray[A, R],
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
      left: NDArray[A, RX],
      right: NDArray[A, RY]
  ): NDArray[A, BroadcastRank[RX, RY]] =
    if operation == KernelOp.Add &&
        left.layout.isCContiguous &&
        right.layout.isCContiguous &&
        Layout.sameShape(left.layout.shape, right.layout.shape)
    then
      val shape =
        Shape.unsafeRanked[BroadcastRank[RX, RY]](left.layout.shape)
      val output = ProbeApi.allocate[A](left.size)(using left.dtype)
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
      new NDArray(output, Layout.contiguous(shape, left.size), left.dtype)
    else
      val plan = LoopPlan.broadcast(left.layout, right.layout)
      val shape =
        Shape.unsafeRanked[BroadcastRank[RX, RY]](plan.resultShape)
      val output = ProbeApi.allocate[A](plan.size)(using left.dtype)
      ElementKernels.binary(operation, left.storage, right.storage, output, plan)
      new NDArray(output, Layout.contiguous(shape, plan.size), left.dtype)

  def operand[A, R <: AnyRank, B <: A | NDArray[A, ? <: AnyRank]](
      operation: Byte,
      left: NDArray[A, R],
      right: B
  ): NDArray[A, OperandRank[A, R, B]] =
    val result =
      right match
        case array: NDArray[?, ?] =>
          binary(
            operation,
            left,
            array.asInstanceOf[NDArray[A, AnyRank]]
          )
        case scalar =>
          KernelApi.scalar(operation, left, scalar.asInstanceOf[A])
    result.asInstanceOf[NDArray[A, OperandRank[A, R, B]]]

  def compare[A, RX <: AnyRank, RY <: AnyRank](
      operation: Byte,
      left: NDArray[A, RX],
      right: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[RX, RY]] =
    val plan = LoopPlan.broadcast(left.layout, right.layout)
    val shape = Shape.unsafeRanked[BroadcastRank[RX, RY]](plan.resultShape)
    val output = ProbeApi.allocate[Boolean](plan.size)(using DType.booleanDType)
    ElementKernels.compare(operation, left.storage, right.storage, output, plan)
    new NDArray(output, Layout.contiguous(shape, plan.size), DType.booleanDType)

  def floatingPredicate[A, R <: AnyRank](
      operation: Byte,
      source: NDArray[A, R]
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
      source: NDArray[A, R],
      f: A => B
  )(using target: DType[B]): NDArray[B, R] =
    val output = ProbeApi.allocate[B](source.size)
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
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      exact: Boolean,
      f: (A, A) => B
  )(using target: DType[B]): NDArray[B, BroadcastRank[RX, RY]] =
    val plan =
      if exact then LoopPlan.exact(left.layout, right.layout)
      else LoopPlan.broadcast(left.layout, right.layout)
    val shape = Shape.unsafeRanked[BroadcastRank[RX, RY]](plan.resultShape)
    val output = ProbeApi.allocate[B](plan.size)
    plan.foreachOffset { (leftIndex, rightIndex, outputIndex) =>
      val value = f(
        ProbeApi.get(left.storage, leftIndex),
        ProbeApi.get(right.storage, rightIndex)
      )
      ProbeApi.set(output, outputIndex, value)
    }
    new NDArray(output, Layout.contiguous(shape, plan.size), target)
