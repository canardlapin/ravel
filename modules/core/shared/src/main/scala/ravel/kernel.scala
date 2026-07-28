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

  def multiplyInto[A, RX <: AnyRank, RY <: AnyRank, RO <: AnyRank](
      left: NDArray[A, RX],
      right: NDArray[A, RY],
      output: MutableNDArray[A, RO]
  )(using @unused arithmetic: ArithmeticDType[A]): Unit =
    binaryInto(KernelOp.Multiply, left, right, output)

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
    if operation == KernelOp.Add &&
        left.layout.isCContiguous &&
        right.layout.isCContiguous &&
        Layout.sameShape(left.layout.shape, right.layout.shape)
    then
      MutableNDArray.requireSameShape(left.layout.shape, output.layout.shape)
      requireWholeOutput(output)
      requireNoAlias(left, right, output)
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
      val plan = LoopPlan.broadcast(left.layout, right.layout)
      requireResultShape(plan, output)
      requireWholeOutput(output)
      requireNoAlias(left, right, output)
      ElementKernels.binary(operation, left.storage, right.storage, output.storage, plan)

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

  private def requireResultShape[A, R <: AnyRank](
      plan: LoopPlan,
      output: MutableNDArray[A, R]
  ): Unit =
    MutableNDArray.requireSameShape(plan.resultShape, output.layout.shape)

  private def requireWholeOutput[A, R <: AnyRank](
      output: MutableNDArray[A, R]
  ): Unit =
    if !output.layout.isCContiguous ||
        output.layout.offset != 0 ||
        output.size != output.storage.length
    then
      throw NonContiguousLayout(
        "reusable-output kernels require a whole canonical contiguous destination"
      )
