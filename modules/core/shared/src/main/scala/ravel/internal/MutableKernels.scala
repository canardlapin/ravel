package ravel.internal

import ravel.*

/** Mutable assign and scalar in-place updates over injective layouts. */
private[ravel] object MutableKernels:
  def assign[A](
      source: Storage[A],
      sourceLayout: Layout,
      destination: Storage[A],
      destinationLayout: Layout
  ): Unit =
    if sourceLayout.size == 0 then ()
    else if sourceLayout.isCContiguous &&
      destinationLayout.isCContiguous &&
      destinationLayout.isWholeBuffer(destination.length) &&
      sourceLayout.offset == 0 &&
      sourceLayout.size == source.length &&
      sourceLayout.size == destinationLayout.size
    then ProbeKernels.copy(source, 0, destination, 0, sourceLayout.size)
    else if sourceLayout.isCContiguous && destinationLayout.isCContiguous then
      var write = 0
      var read = sourceLayout.offset
      while write < destinationLayout.size do
        ProbeApi.set(destination, destinationLayout.offset + write, ProbeApi.get(source, read))
        write += 1
        read += 1
    else zipSet(source, sourceLayout, destination, destinationLayout)

  def scalarInPlace[A](
      operation: Byte,
      storage: Storage[A],
      layout: Layout,
      value: A
  ): Unit =
    if layout.size == 0 then ()
    else if layout.isCContiguous && layout.isWholeBuffer(storage.length) then
      contiguousScalar(operation, storage, value, layout.size)
    else
      layout.foreachPhysicalIndex { index =>
        val current = ProbeApi.get(storage, index)
        ProbeApi.set(storage, index, applyScalar(operation, current, value))
      }

  private def zipSet[A](
      source: Storage[A],
      sourceLayout: Layout,
      destination: Storage[A],
      destinationLayout: Layout
  ): Unit =
    if sourceLayout.rank == 0 then
      ProbeApi.set(destination, destinationLayout.offset, ProbeApi.get(source, sourceLayout.offset))
    else
      val counters = new Array[Int](sourceLayout.rank)
      var sourceAddress = sourceLayout.offset.toLong
      var destinationAddress = destinationLayout.offset.toLong
      var visited = 0
      while visited < sourceLayout.size do
        ProbeApi.set(
          destination,
          Layout.checkedInt(destinationAddress, "assign destination"),
          ProbeApi.get(source, Layout.checkedInt(sourceAddress, "assign source"))
        )
        visited += 1
        if visited < sourceLayout.size then
          var axis = sourceLayout.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            counters(axis) += 1
            sourceAddress = Layout.checkedAdd(
              sourceAddress,
              sourceLayout.strides(axis).toLong,
              "assign source"
            )
            destinationAddress = Layout.checkedAdd(
              destinationAddress,
              destinationLayout.strides(axis).toLong,
              "assign destination"
            )
            if counters(axis) < sourceLayout.shape(axis) then advanced = true
            else
              sourceAddress = Layout.checkedAdd(
                sourceAddress,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  sourceLayout.strides(axis).toLong,
                  "assign source rewind"
                ),
                "assign source rewind"
              )
              destinationAddress = Layout.checkedAdd(
                destinationAddress,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  destinationLayout.strides(axis).toLong,
                  "assign destination rewind"
                ),
                "assign destination rewind"
              )
              counters(axis) = 0
              axis -= 1

  private def contiguousScalar[A](
      operation: Byte,
      storage: Storage[A],
      value: A,
      size: Int
  ): Unit =
    var index = 0
    while index < size do
      ProbeApi.set(storage, index, applyScalar(operation, ProbeApi.get(storage, index), value))
      index += 1

  private def applyScalar[A](operation: Byte, left: A, right: A): A =
    (left, right) match
      case (x: Int, y: Int) =>
        (operation match
          case KernelOp.Add => x + y
          case KernelOp.Subtract => x - y
          case KernelOp.Multiply => x * y
          case KernelOp.Divide => x / y
          case _ => throw new MatchError(operation)
        ).asInstanceOf[A]
      case (x: Long, y: Long) =>
        (operation match
          case KernelOp.Add => x + y
          case KernelOp.Subtract => x - y
          case KernelOp.Multiply => x * y
          case KernelOp.Divide => x / y
          case _ => throw new MatchError(operation)
        ).asInstanceOf[A]
      case (x: Float, y: Float) =>
        (operation match
          case KernelOp.Add => x + y
          case KernelOp.Subtract => x - y
          case KernelOp.Multiply => x * y
          case KernelOp.Divide => x / y
          case _ => throw new MatchError(operation)
        ).asInstanceOf[A]
      case (x: Double, y: Double) =>
        (operation match
          case KernelOp.Add => x + y
          case KernelOp.Subtract => x - y
          case KernelOp.Multiply => x * y
          case KernelOp.Divide => x / y
          case _ => throw new MatchError(operation)
        ).asInstanceOf[A]
      case _ => throw new IllegalArgumentException("unsupported in-place scalar dtype")
