package ravel

import ravel.internal.*
import scala.annotation.unused

/** Expert reusable-output kernels. Destinations must be whole contiguous buffers. */
object kernel:
  def addInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    binaryInto(KernelOp.Add, left, right, output)

  def subtractInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    binaryInto(KernelOp.Subtract, left, right, output)

  def multiplyInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    binaryInto(KernelOp.Multiply, left, right, output)

  def quotInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused integral: IntegralArithmeticDType[A]): Unit =
    binaryInto(KernelOp.Divide, left, right, output)

  def divideInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused floating: FloatingDType[A]): Unit =
    binaryInto(KernelOp.Divide, left, right, output)

  def minimumInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused ordered: OrderedDType[A]): Unit =
    binaryInto(KernelOp.Minimum, left, right, output)

  def maximumInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused ordered: OrderedDType[A]): Unit =
    binaryInto(KernelOp.Maximum, left, right, output)

  def negateInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    unaryInto(KernelOp.Negate, source, output)

  def absInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    unaryInto(KernelOp.Absolute, source, output)

  def addScalarInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    scalarInto(KernelOp.Add, source, value, output)

  def subtractScalarInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    scalarInto(KernelOp.Subtract, source, value, output)

  def multiplyScalarInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    scalarInto(KernelOp.Multiply, source, value, output)

  def quotScalarInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  )(using @unused integral: IntegralArithmeticDType[A]): Unit =
    scalarInto(KernelOp.Divide, source, value, output)

  def divideScalarInto[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  )(using @unused floating: FloatingDType[A]): Unit =
    scalarInto(KernelOp.Divide, source, value, output)

  def mapInto[A, B, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      output: MutableNDArray[B, RO]
  )(f: A => B): Unit =
    MutableNDArray.requireSameShape(source.layout.shape, output.layout.shape)
    requireWholeOutput(output)
    var write = 0
    source.layout.foreachPhysicalIndex { physical =>
      ProbeApi.set(output.storage, write, f(ProbeApi.get(source.storage, physical)))
      write += 1
    }

  def zipMapInto[A, B, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[B, RO]
  )(f: (A, A) => B): Unit =
    val plan = LoopPlan.broadcast(left.layout, right.layout)
    requireResultShape(plan, output)
    requireWholeOutput(output)
    plan.foreachOffset { (leftIndex, rightIndex, outputIndex) =>
      ProbeApi.set(
        output.storage,
        outputIndex,
        f(
          ProbeApi.get(left.storage, leftIndex),
          ProbeApi.get(right.storage, rightIndex)
        )
      )
    }

  private def binaryInto[
      A,
      RX <: AnyRank,
      RY <: AnyRank,
      RO <: AnyRank
  ](
      operation: Byte,
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  ): Unit =
    if left.layout.isCContiguous &&
      right.layout.isCContiguous &&
      Layout.sameShape(left.layout.shape, right.layout.shape)
    then
      MutableNDArray.requireSameShape(left.layout.shape, output.layout.shape)
      requireWholeOutput(output)
      requireNoAlias(left, right, output)
      if operation == KernelOp.Add then
        if left.layout.offset == 0 && right.layout.offset == 0 then
          ProbeKernels.add(left.storage, right.storage, output.storage, left.size)
        else
          ProbeKernels.addLinear(
            left.storage,
            left.layout.offset,
            right.storage,
            right.layout.offset,
            output.storage,
            left.size
          )
      else
        val plan = LoopPlan.linearSameShape(left.layout, right.layout)
        ElementKernels.binary(operation, left.storage, right.storage, output.storage, plan)
    else
      val plan = LoopPlan.broadcast(left.layout, right.layout)
      requireResultShape(plan, output)
      requireWholeOutput(output)
      requireNoAlias(left, right, output)
      ElementKernels.binary(operation, left.storage, right.storage, output.storage, plan)

  private def unaryInto[A, RX <: AnyRank, RO <: AnyRank](
      operation: Byte,
      source: NDArray[A, RX],
      output: MutableNDArray[A, RO]
  ): Unit =
    MutableNDArray.requireSameShape(source.layout.shape, output.layout.shape)
    requireWholeOutput(output)
    requireNoAliasUnary(source, output)
    val plan = LoopPlan.unary(source.layout)
    ElementKernels.unary(operation, source.storage, output.storage, plan)

  private def scalarInto[A, RX <: AnyRank, RO <: AnyRank](
      operation: Byte,
      source: NDArray[A, RX],
      value: A,
      output: MutableNDArray[A, RO]
  ): Unit =
    MutableNDArray.requireSameShape(source.layout.shape, output.layout.shape)
    requireWholeOutput(output)
    requireNoAliasUnary(source, output)
    val plan = LoopPlan.unary(source.layout)
    ElementKernels.scalar(operation, source.storage, value, output.storage, plan)

  private def requireNoAlias[
      A,
      RX <: AnyRank,
      RY <: AnyRank,
      RO <: AnyRank
  ](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  ): Unit =
    if (output.storage.asInstanceOf[AnyRef] eq left.storage.asInstanceOf[AnyRef]) ||
      (output.storage.asInstanceOf[AnyRef] eq right.storage.asInstanceOf[AnyRef])
    then throw new IllegalArgumentException("output must not alias an input")

  private def requireNoAliasUnary[A, RX <: AnyRank, RO <: AnyRank](
      source: NDArray[A, RX],
      output: MutableNDArray[A, RO]
  ): Unit =
    if output.storage.asInstanceOf[AnyRef] eq source.storage.asInstanceOf[AnyRef] then
      throw new IllegalArgumentException("output must not alias an input")

  private def requireResultShape[A, R <: AnyRank](
      plan: LoopPlan,
      output: MutableNDArray[A, R]
  ): Unit =
    MutableNDArray.requireSameShape(plan.resultShape, output.layout.shape)

  private def requireWholeOutput[A, R <: AnyRank](
      output: MutableNDArray[A, R]
  ): Unit =
    if !output.layout.isCContiguous ||
      !output.layout.isWholeBuffer(output.storage.length)
    then
      throw NonContiguousLayout(
        "reusable-output kernels require a whole contiguous destination"
      )
