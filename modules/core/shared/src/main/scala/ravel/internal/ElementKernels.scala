package ravel.internal

private[ravel] object ElementKernels:
  def binary[A](
      operation: Byte,
      left: Storage[A],
      right: Storage[A],
      output: Storage[A],
      plan: LoopPlan
  ): Unit =
    (left, right, output) match
      case (x: ByteStorage, y: ByteStorage, z: ByteStorage) =>
        binaryByte(operation, x, y, z, plan)
      case (x: ShortStorage, y: ShortStorage, z: ShortStorage) =>
        binaryShort(operation, x, y, z, plan)
      case (x: IntStorage, y: IntStorage, z: IntStorage) =>
        binaryInt(operation, x, y, z, plan)
      case (x: LongStorage, y: LongStorage, z: LongStorage) =>
        binaryLong(operation, x, y, z, plan)
      case (x: FloatStorage, y: FloatStorage, z: FloatStorage) =>
        binaryFloat(operation, x, y, z, plan)
      case (x: DoubleStorage, y: DoubleStorage, z: DoubleStorage) =>
        binaryDouble(operation, x, y, z, plan)
      case _ => throw new IllegalArgumentException("kernel dtype mismatch")

  def unary[A](
      operation: Byte,
      source: Storage[A],
      output: Storage[A],
      plan: LoopPlan
  ): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        unaryInt(operation, x, z, plan)
      case (x: LongStorage, z: LongStorage) =>
        unaryLong(operation, x, z, plan)
      case (x: FloatStorage, z: FloatStorage) =>
        unaryFloat(operation, x, z, plan)
      case (x: DoubleStorage, z: DoubleStorage) =>
        unaryDouble(operation, x, z, plan)
      case _ => throw new UnsupportedOperationException("unary arithmetic requires Int, Long, Float, or Double")

  def scalar[A](
      operation: Byte,
      source: Storage[A],
      scalar: A,
      output: Storage[A],
      plan: LoopPlan
  ): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        scalarInt(operation, x, scalar.asInstanceOf[Int], z, plan)
      case (x: LongStorage, z: LongStorage) =>
        scalarLong(operation, x, scalar.asInstanceOf[Long], z, plan)
      case (x: FloatStorage, z: FloatStorage) =>
        scalarFloat(operation, x, scalar.asInstanceOf[Float], z, plan)
      case (x: DoubleStorage, z: DoubleStorage) =>
        scalarDouble(operation, x, scalar.asInstanceOf[Double], z, plan)
      case (x: ByteStorage, z: ByteStorage) =>
        scalarByte(operation, x, scalar.asInstanceOf[Byte], z, plan)
      case (x: ShortStorage, z: ShortStorage) =>
        scalarShort(operation, x, scalar.asInstanceOf[Short], z, plan)
      case _ => throw new IllegalArgumentException("scalar kernel dtype mismatch")

  private def binaryByte(
      operation: Byte,
      x: ByteStorage,
      y: ByteStorage,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum =>
        binaryByteMinimum(x, y, z, plan)
      case KernelOp.Maximum =>
        binaryByteMaximum(x, y, z, plan)
      case _ => unsupportedArithmetic("Byte")

  private def binaryShort(
      operation: Byte,
      x: ShortStorage,
      y: ShortStorage,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum =>
        binaryShortMinimum(x, y, z, plan)
      case KernelOp.Maximum =>
        binaryShortMaximum(x, y, z, plan)
      case _ => unsupportedArithmetic("Short")

  private def binaryInt(
      operation: Byte,
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryIntAdd(x, y, z, plan)
      case KernelOp.Subtract => binaryIntSubtract(x, y, z, plan)
      case KernelOp.Multiply => binaryIntMultiply(x, y, z, plan)
      case KernelOp.Divide   => binaryIntDivide(x, y, z, plan)
      case KernelOp.Minimum  => binaryIntMinimum(x, y, z, plan)
      case KernelOp.Maximum  => binaryIntMaximum(x, y, z, plan)

  private def binaryLong(
      operation: Byte,
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryLongAdd(x, y, z, plan)
      case KernelOp.Subtract => binaryLongSubtract(x, y, z, plan)
      case KernelOp.Multiply => binaryLongMultiply(x, y, z, plan)
      case KernelOp.Divide   => binaryLongDivide(x, y, z, plan)
      case KernelOp.Minimum  => binaryLongMinimum(x, y, z, plan)
      case KernelOp.Maximum  => binaryLongMaximum(x, y, z, plan)

  private def binaryFloat(
      operation: Byte,
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryFloatAdd(x, y, z, plan)
      case KernelOp.Subtract => binaryFloatSubtract(x, y, z, plan)
      case KernelOp.Multiply => binaryFloatMultiply(x, y, z, plan)
      case KernelOp.Divide   => binaryFloatDivide(x, y, z, plan)
      case KernelOp.Minimum  => binaryFloatMinimum(x, y, z, plan)
      case KernelOp.Maximum  => binaryFloatMaximum(x, y, z, plan)

  private def binaryDouble(
      operation: Byte,
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryDoubleAdd(x, y, z, plan)
      case KernelOp.Subtract => binaryDoubleSubtract(x, y, z, plan)
      case KernelOp.Multiply => binaryDoubleMultiply(x, y, z, plan)
      case KernelOp.Divide   => binaryDoubleDivide(x, y, z, plan)
      case KernelOp.Minimum  => binaryDoubleMinimum(x, y, z, plan)
      case KernelOp.Maximum  => binaryDoubleMaximum(x, y, z, plan)

  private def binaryByteMinimum(
      x: ByteStorage,
      y: ByteStorage,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.min(a, b).toByte
    )

  private def binaryByteMaximum(
      x: ByteStorage,
      y: ByteStorage,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.max(a, b).toByte
    )

  private def binaryShortMinimum(
      x: ShortStorage,
      y: ShortStorage,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.min(a, b).toShort
    )

  private def binaryShortMaximum(
      x: ShortStorage,
      y: ShortStorage,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.max(a, b).toShort
    )

  private def binaryIntAdd(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    if plan.kind == LoopKind.LinearContiguous then addLinear(plan, x, y, z)
    else if plan.rank <= 1 then addStrided(plan, x, y, z)
    else binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)

  private def binaryIntSubtract(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)

  private def binaryIntMultiply(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)

  private def binaryIntDivide(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)

  private def binaryIntMinimum(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)

  private def binaryIntMaximum(
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

  private def binaryLongAdd(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    if plan.kind == LoopKind.LinearContiguous then addLinear(plan, x, y, z)
    else if plan.rank <= 1 then addStrided(plan, x, y, z)
    else binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)

  private def binaryLongSubtract(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)

  private def binaryLongMultiply(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)

  private def binaryLongDivide(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)

  private def binaryLongMinimum(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)

  private def binaryLongMaximum(
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

  private def binaryFloatAdd(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    if plan.kind == LoopKind.LinearContiguous then addLinear(plan, x, y, z)
    else if plan.rank <= 1 then addStrided(plan, x, y, z)
    else
      binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
        (a + b).toFloat
      )

  private def binaryFloatSubtract(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      (a - b).toFloat
    )

  private def binaryFloatMultiply(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      (a * b).toFloat
    )

  private def binaryFloatDivide(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      (a / b).toFloat
    )

  private def binaryFloatMinimum(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.min(a, b).toFloat
    )

  private def binaryFloatMaximum(
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
      math.max(a, b).toFloat
    )

  private def binaryDoubleAdd(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    if plan.kind == LoopKind.LinearContiguous then addLinear(plan, x, y, z)
    else if plan.rank <= 1 then addStrided(plan, x, y, z)
    else binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)

  private def binaryDoubleSubtract(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)

  private def binaryDoubleMultiply(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)

  private def binaryDoubleDivide(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)

  private def binaryDoubleMinimum(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)

  private def binaryDoubleMaximum(
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

  private def unaryInt(
      operation: Byte,
      x: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Negate   => unaryLoop(plan, x.raw.apply, z.raw.update)(a => -a)
      case KernelOp.Absolute => unaryLoop(plan, x.raw.apply, z.raw.update)(math.abs)
      case _                 => unsupportedUnary("Int")

  private def unaryLong(
      operation: Byte,
      x: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Negate   => unaryLoop(plan, x.raw.apply, z.raw.update)(a => -a)
      case KernelOp.Absolute => unaryLoop(plan, x.raw.apply, z.raw.update)(math.abs)
      case _                 => unsupportedUnary("Long")

  private def unaryFloat(
      operation: Byte,
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Negate   => unaryFloatNegate(x, z, plan)
      case KernelOp.Absolute => unaryFloatAbsolute(x, z, plan)
      case KernelOp.Sqrt     => unaryFloatSqrt(x, z, plan)
      case KernelOp.Exp      => unaryFloatExp(x, z, plan)
      case KernelOp.Log      => unaryFloatLog(x, z, plan)
      case KernelOp.Sin      => unaryFloatSin(x, z, plan)
      case KernelOp.Cos      => unaryFloatCos(x, z, plan)
      case KernelOp.Tan      => unaryFloatTan(x, z, plan)
      case KernelOp.Floor    => unaryFloatFloor(x, z, plan)
      case KernelOp.Ceil     => unaryFloatCeil(x, z, plan)

  private def unaryDouble(
      operation: Byte,
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Negate   => unaryDoubleNegate(x, z, plan)
      case KernelOp.Absolute => unaryDoubleAbsolute(x, z, plan)
      case KernelOp.Sqrt     => unaryDoubleSqrt(x, z, plan)
      case KernelOp.Exp      => unaryDoubleExp(x, z, plan)
      case KernelOp.Log      => unaryDoubleLog(x, z, plan)
      case KernelOp.Sin      => unaryDoubleSin(x, z, plan)
      case KernelOp.Cos      => unaryDoubleCos(x, z, plan)
      case KernelOp.Tan      => unaryDoubleTan(x, z, plan)
      case KernelOp.Floor    => unaryDoubleFloor(x, z, plan)
      case KernelOp.Ceil     => unaryDoubleCeil(x, z, plan)

  private def unaryFloatNegate(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => (-a).toFloat)

  private def unaryFloatAbsolute(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.abs(a).toFloat)

  private def unaryFloatSqrt(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.sqrt(a.toDouble).toFloat)

  private def unaryFloatExp(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.exp(a.toDouble).toFloat)

  private def unaryFloatLog(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.log(a.toDouble).toFloat)

  private def unaryFloatSin(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.sin(a.toDouble).toFloat)

  private def unaryFloatCos(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.cos(a.toDouble).toFloat)

  private def unaryFloatTan(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.tan(a.toDouble).toFloat)

  private def unaryFloatFloor(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.floor(a.toDouble).toFloat)

  private def unaryFloatCeil(
      x: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.ceil(a.toDouble).toFloat)

  private def unaryDoubleNegate(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => -a)

  private def unaryDoubleAbsolute(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.abs)

  private def unaryDoubleSqrt(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.sqrt)

  private def unaryDoubleExp(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.exp)

  private def unaryDoubleLog(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.log)

  private def unaryDoubleSin(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.sin)

  private def unaryDoubleCos(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.cos)

  private def unaryDoubleTan(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.tan)

  private def unaryDoubleFloor(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.floor)

  private def unaryDoubleCeil(
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.ceil)

  private def scalarByte(
      operation: Byte,
      x: ByteStorage,
      value: Byte,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum => scalarByteMinimum(x, value, z, plan)
      case KernelOp.Maximum => scalarByteMaximum(x, value, z, plan)
      case _ => unsupportedArithmetic("Byte")

  private def scalarShort(
      operation: Byte,
      x: ShortStorage,
      value: Short,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum => scalarShortMinimum(x, value, z, plan)
      case KernelOp.Maximum => scalarShortMaximum(x, value, z, plan)
      case _ => unsupportedArithmetic("Short")

  private def scalarInt(
      operation: Byte,
      x: IntStorage,
      value: Int,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => scalarIntAdd(x, value, z, plan)
      case KernelOp.Subtract => scalarIntSubtract(x, value, z, plan)
      case KernelOp.Multiply => scalarIntMultiply(x, value, z, plan)
      case KernelOp.Divide   => scalarIntDivide(x, value, z, plan)
      case KernelOp.Minimum  => scalarIntMinimum(x, value, z, plan)
      case KernelOp.Maximum  => scalarIntMaximum(x, value, z, plan)

  private def scalarLong(
      operation: Byte,
      x: LongStorage,
      value: Long,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => scalarLongAdd(x, value, z, plan)
      case KernelOp.Subtract => scalarLongSubtract(x, value, z, plan)
      case KernelOp.Multiply => scalarLongMultiply(x, value, z, plan)
      case KernelOp.Divide   => scalarLongDivide(x, value, z, plan)
      case KernelOp.Minimum  => scalarLongMinimum(x, value, z, plan)
      case KernelOp.Maximum  => scalarLongMaximum(x, value, z, plan)

  private def scalarFloat(
      operation: Byte,
      x: FloatStorage,
      value: Float,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => scalarFloatAdd(x, value, z, plan)
      case KernelOp.Subtract => scalarFloatSubtract(x, value, z, plan)
      case KernelOp.Multiply => scalarFloatMultiply(x, value, z, plan)
      case KernelOp.Divide   => scalarFloatDivide(x, value, z, plan)
      case KernelOp.Minimum  => scalarFloatMinimum(x, value, z, plan)
      case KernelOp.Maximum  => scalarFloatMaximum(x, value, z, plan)

  private def scalarDouble(
      operation: Byte,
      x: DoubleStorage,
      value: Double,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => scalarDoubleAdd(x, value, z, plan)
      case KernelOp.Subtract => scalarDoubleSubtract(x, value, z, plan)
      case KernelOp.Multiply => scalarDoubleMultiply(x, value, z, plan)
      case KernelOp.Divide   => scalarDoubleDivide(x, value, z, plan)
      case KernelOp.Minimum  => scalarDoubleMinimum(x, value, z, plan)
      case KernelOp.Maximum  => scalarDoubleMaximum(x, value, z, plan)

  private def scalarByteMinimum(
      x: ByteStorage,
      value: Byte,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toByte)

  private def scalarByteMaximum(
      x: ByteStorage,
      value: Byte,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toByte)

  private def scalarShortMinimum(
      x: ShortStorage,
      value: Short,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toShort)

  private def scalarShortMaximum(
      x: ShortStorage,
      value: Short,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toShort)

  private def scalarIntAdd(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)

  private def scalarIntSubtract(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)

  private def scalarIntMultiply(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)

  private def scalarIntDivide(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)

  private def scalarIntMinimum(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))

  private def scalarIntMaximum(x: IntStorage, value: Int, z: IntStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  private def scalarLongAdd(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)

  private def scalarLongSubtract(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)

  private def scalarLongMultiply(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)

  private def scalarLongDivide(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)

  private def scalarLongMinimum(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))

  private def scalarLongMaximum(x: LongStorage, value: Long, z: LongStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  private def scalarFloatAdd(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a + value).toFloat)

  private def scalarFloatSubtract(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a - value).toFloat)

  private def scalarFloatMultiply(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a * value).toFloat)

  private def scalarFloatDivide(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a / value).toFloat)

  private def scalarFloatMinimum(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toFloat)

  private def scalarFloatMaximum(x: FloatStorage, value: Float, z: FloatStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toFloat)

  private def scalarDoubleAdd(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)

  private def scalarDoubleSubtract(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)

  private def scalarDoubleMultiply(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)

  private def scalarDoubleDivide(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)

  private def scalarDoubleMinimum(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))

  private def scalarDoubleMaximum(x: DoubleStorage, value: Double, z: DoubleStorage, plan: LoopPlan): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  def clip[A](
      source: Storage[A],
      lower: A,
      upper: A,
      output: Storage[A],
      plan: LoopPlan
  ): Unit =
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        clipByte(x, lower.asInstanceOf[Byte], upper.asInstanceOf[Byte], z, plan)
      case (x: ShortStorage, z: ShortStorage) =>
        clipShort(x, lower.asInstanceOf[Short], upper.asInstanceOf[Short], z, plan)
      case (x: IntStorage, z: IntStorage) =>
        clipInt(x, lower.asInstanceOf[Int], upper.asInstanceOf[Int], z, plan)
      case (x: LongStorage, z: LongStorage) =>
        clipLong(x, lower.asInstanceOf[Long], upper.asInstanceOf[Long], z, plan)
      case (x: FloatStorage, z: FloatStorage) =>
        clipFloat(x, lower.asInstanceOf[Float], upper.asInstanceOf[Float], z, plan)
      case (x: DoubleStorage, z: DoubleStorage) =>
        clipDouble(x, lower.asInstanceOf[Double], upper.asInstanceOf[Double], z, plan)
      case _ => throw new IllegalArgumentException("clip kernel dtype mismatch")

  private def clipByte(
      x: ByteStorage,
      lower: Byte,
      upper: Byte,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper).toByte
    )

  private def clipShort(
      x: ShortStorage,
      lower: Short,
      upper: Short,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper).toShort
    )

  private def clipInt(
      x: IntStorage,
      lower: Int,
      upper: Int,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper)
    )

  private def clipLong(
      x: LongStorage,
      lower: Long,
      upper: Long,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper)
    )

  private def clipFloat(
      x: FloatStorage,
      lower: Float,
      upper: Float,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper).toFloat
    )

  private def clipDouble(
      x: DoubleStorage,
      lower: Double,
      upper: Double,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    unaryLoop(plan, x.raw.apply, z.raw.update)(a =>
      math.min(math.max(a, lower), upper)
    )

  def compare[A](
      operation: Byte,
      left: Storage[A],
      right: Storage[A],
      output: Storage[Boolean],
      plan: LoopPlan
  ): Unit =
    val boolean = output.asInstanceOf[BooleanStorage]
    (left, right) match
      case (x: BooleanStorage, y: BooleanStorage) =>
        compareBoolean(operation, plan, boolean, x, y)
      case (x: ByteStorage, y: ByteStorage) =>
        compareByte(operation, plan, boolean, x, y)
      case (x: ShortStorage, y: ShortStorage) =>
        compareShort(operation, plan, boolean, x, y)
      case (x: IntStorage, y: IntStorage) =>
        compareInt(operation, plan, boolean, x, y)
      case (x: LongStorage, y: LongStorage) =>
        compareLong(operation, plan, boolean, x, y)
      case (x: FloatStorage, y: FloatStorage) =>
        compareFloat(operation, plan, boolean, x, y)
      case (x: DoubleStorage, y: DoubleStorage) =>
        compareDouble(operation, plan, boolean, x, y)
      case _ => throw new IllegalArgumentException("comparison dtype mismatch")

  private def compareBoolean(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: BooleanStorage,
      y: BooleanStorage
  ): Unit =
    operation match
      case KernelOp.Equal    => compareBooleanEqual(plan, output, x, y)
      case KernelOp.NotEqual => compareBooleanNotEqual(plan, output, x, y)
      case _ => unsupportedComparison("Boolean")

  private def compareByte(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: ByteStorage,
      y: ByteStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareByteEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareByteNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareByteLess(plan, output, x, y)
      case KernelOp.LessEqual => compareByteLessEqual(plan, output, x, y)

  private def compareShort(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: ShortStorage,
      y: ShortStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareShortEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareShortNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareShortLess(plan, output, x, y)
      case KernelOp.LessEqual => compareShortLessEqual(plan, output, x, y)

  private def compareInt(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: IntStorage,
      y: IntStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareIntEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareIntNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareIntLess(plan, output, x, y)
      case KernelOp.LessEqual => compareIntLessEqual(plan, output, x, y)

  private def compareLong(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: LongStorage,
      y: LongStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareLongEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareLongNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareLongLess(plan, output, x, y)
      case KernelOp.LessEqual => compareLongLessEqual(plan, output, x, y)

  private def compareFloat(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: FloatStorage,
      y: FloatStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareFloatEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareFloatNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareFloatLess(plan, output, x, y)
      case KernelOp.LessEqual => compareFloatLessEqual(plan, output, x, y)

  private def compareDouble(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: DoubleStorage,
      y: DoubleStorage
  ): Unit =
    operation match
      case KernelOp.Equal     => compareDoubleEqual(plan, output, x, y)
      case KernelOp.NotEqual  => compareDoubleNotEqual(plan, output, x, y)
      case KernelOp.Less      => compareDoubleLess(plan, output, x, y)
      case KernelOp.LessEqual => compareDoubleLessEqual(plan, output, x, y)

  private def compareBooleanEqual(plan: LoopPlan, output: BooleanStorage, x: BooleanStorage, y: BooleanStorage): Unit =
    compareLoop(plan, output, i => x.raw(i), i => y.raw(i))(_ == _)

  private def compareBooleanNotEqual(plan: LoopPlan, output: BooleanStorage, x: BooleanStorage, y: BooleanStorage): Unit =
    compareLoop(plan, output, i => x.raw(i), i => y.raw(i))(_ != _)

  private def compareByteEqual(plan: LoopPlan, output: BooleanStorage, x: ByteStorage, y: ByteStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareByteNotEqual(plan: LoopPlan, output: BooleanStorage, x: ByteStorage, y: ByteStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareByteLess(plan: LoopPlan, output: BooleanStorage, x: ByteStorage, y: ByteStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareByteLessEqual(plan: LoopPlan, output: BooleanStorage, x: ByteStorage, y: ByteStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  private def compareShortEqual(plan: LoopPlan, output: BooleanStorage, x: ShortStorage, y: ShortStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareShortNotEqual(plan: LoopPlan, output: BooleanStorage, x: ShortStorage, y: ShortStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareShortLess(plan: LoopPlan, output: BooleanStorage, x: ShortStorage, y: ShortStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareShortLessEqual(plan: LoopPlan, output: BooleanStorage, x: ShortStorage, y: ShortStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  private def compareIntEqual(plan: LoopPlan, output: BooleanStorage, x: IntStorage, y: IntStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareIntNotEqual(plan: LoopPlan, output: BooleanStorage, x: IntStorage, y: IntStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareIntLess(plan: LoopPlan, output: BooleanStorage, x: IntStorage, y: IntStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareIntLessEqual(plan: LoopPlan, output: BooleanStorage, x: IntStorage, y: IntStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  private def compareLongEqual(plan: LoopPlan, output: BooleanStorage, x: LongStorage, y: LongStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareLongNotEqual(plan: LoopPlan, output: BooleanStorage, x: LongStorage, y: LongStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareLongLess(plan: LoopPlan, output: BooleanStorage, x: LongStorage, y: LongStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareLongLessEqual(plan: LoopPlan, output: BooleanStorage, x: LongStorage, y: LongStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  private def compareFloatEqual(plan: LoopPlan, output: BooleanStorage, x: FloatStorage, y: FloatStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareFloatNotEqual(plan: LoopPlan, output: BooleanStorage, x: FloatStorage, y: FloatStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareFloatLess(plan: LoopPlan, output: BooleanStorage, x: FloatStorage, y: FloatStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareFloatLessEqual(plan: LoopPlan, output: BooleanStorage, x: FloatStorage, y: FloatStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  private def compareDoubleEqual(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage, y: DoubleStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ == _)

  private def compareDoubleNotEqual(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage, y: DoubleStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ != _)

  private def compareDoubleLess(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage, y: DoubleStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ < _)

  private def compareDoubleLessEqual(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage, y: DoubleStorage): Unit =
    compareLoop(plan, output, x.raw.apply, y.raw.apply)(_ <= _)

  def floatingPredicate[A](
      operation: Byte,
      source: Storage[A],
      output: Storage[Boolean],
      plan: LoopPlan
  ): Unit =
    val boolean = output.asInstanceOf[BooleanStorage]
    source match
      case x: FloatStorage =>
        operation match
          case KernelOp.IsNaN    => predicateFloatNaN(plan, boolean, x)
          case KernelOp.IsFinite => predicateFloatFinite(plan, boolean, x)
      case x: DoubleStorage =>
        operation match
          case KernelOp.IsNaN    => predicateDoubleNaN(plan, boolean, x)
          case KernelOp.IsFinite => predicateDoubleFinite(plan, boolean, x)
      case _ => throw new UnsupportedOperationException("floating predicate requires Float or Double")

  private def predicateFloatNaN(plan: LoopPlan, output: BooleanStorage, x: FloatStorage): Unit =
    predicateLoop(plan, output, x.raw.apply)(_.isNaN)

  private def predicateFloatFinite(plan: LoopPlan, output: BooleanStorage, x: FloatStorage): Unit =
    predicateLoop(plan, output, x.raw.apply)(_.isFinite)

  private def predicateDoubleNaN(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage): Unit =
    predicateLoop(plan, output, x.raw.apply)(_.isNaN)

  private def predicateDoubleFinite(plan: LoopPlan, output: BooleanStorage, x: DoubleStorage): Unit =
    predicateLoop(plan, output, x.raw.apply)(_.isFinite)

  private inline def binaryLoop[T](
      plan: LoopPlan,
      inline left: Int => T,
      inline right: Int => T,
      inline output: (Int, T) => Unit
  )(inline operation: (T, T) => T): Unit =
    loopOffsets(plan) { (leftIndex, rightIndex, outputIndex) =>
      output(outputIndex, operation(left(leftIndex), right(rightIndex)))
    }

  private def addLinear[A](
      plan: LoopPlan,
      left: Storage[A],
      right: Storage[A],
      output: Storage[A]
  ): Unit =
    if plan.leftOffset == 0 && plan.rightOffset == 0 then
      ProbeKernels.add(left, right, output, plan.size)
    else
      ProbeKernels.addLinear(
        left,
        plan.leftOffset,
        right,
        plan.rightOffset,
        output,
        plan.size
      )

  private def addStrided[A](
      plan: LoopPlan,
      left: Storage[A],
      right: Storage[A],
      output: Storage[A]
  ): Unit =
    val leftStride = if plan.rank == 0 then 0 else plan.leftStrides(0)
    val rightStride = if plan.rank == 0 then 0 else plan.rightStrides(0)
    ProbeKernels.addStrided(
      left,
      plan.leftOffset,
      leftStride,
      right,
      plan.rightOffset,
      rightStride,
      output,
      plan.size
    )

  private inline def unaryLoop[T](
      plan: LoopPlan,
      inline source: Int => T,
      inline output: (Int, T) => Unit
  )(inline operation: T => T): Unit =
    loopOffsets(plan) { (sourceIndex, _, outputIndex) =>
      output(outputIndex, operation(source(sourceIndex)))
    }

  private inline def compareLoop[T](
      plan: LoopPlan,
      output: BooleanStorage,
      inline left: Int => T,
      inline right: Int => T
  )(inline predicate: (T, T) => Boolean): Unit =
    loopOffsets(plan) { (leftIndex, rightIndex, outputIndex) =>
      PlatformBoolean.set(output, outputIndex, predicate(left(leftIndex), right(rightIndex)))
    }

  private inline def predicateLoop[T](
      plan: LoopPlan,
      output: BooleanStorage,
      inline source: Int => T
  )(inline predicate: T => Boolean): Unit =
    loopOffsets(plan) { (sourceIndex, _, outputIndex) =>
      PlatformBoolean.set(output, outputIndex, predicate(source(sourceIndex)))
    }

  private inline def loopOffsets(
      plan: LoopPlan
  )(inline body: (Int, Int, Int) => Unit): Unit =
    plan.kind match
      case LoopKind.LinearContiguous =>
        loopLinear(plan, body)
      case LoopKind.ScalarBroadcast | LoopKind.InnerStrided =>
        if plan.rank <= 1 then loopRankOne(plan, body)
        else if plan.rank == 2 then loopRankTwo(plan, body)
        else loopGeneral(plan, body)
      case LoopKind.GeneralStrided =>
        loopGeneral(plan, body)

  private inline def loopLinear(
      plan: LoopPlan,
      inline body: (Int, Int, Int) => Unit
  ): Unit =
    var output = 0
    while output < plan.size do
      body(
        plan.leftOffset + output,
        plan.rightOffset + output,
        output
      )
      output += 1

  private inline def loopRankOne(
      plan: LoopPlan,
      inline body: (Int, Int, Int) => Unit
  ): Unit =
    if plan.size > 0 then
      val leftStride = if plan.rank == 0 then 0 else plan.leftStrides(0)
      val rightStride = if plan.rank == 0 then 0 else plan.rightStrides(0)
      var left = plan.leftOffset
      var right = plan.rightOffset
      var output = 0
      while output < plan.size do
        body(left, right, output)
        left += leftStride
        right += rightStride
        output += 1

  private inline def loopRankTwo(
      plan: LoopPlan,
      inline body: (Int, Int, Int) => Unit
  ): Unit =
    val outerSize = plan.shape(0)
    val innerSize = plan.shape(1)
    val leftOuterStride = plan.leftStrides(0)
    val leftInnerStride = plan.leftStrides(1)
    val rightOuterStride = plan.rightStrides(0)
    val rightInnerStride = plan.rightStrides(1)
    var leftOuter = plan.leftOffset
    var rightOuter = plan.rightOffset
    var output = 0
    var outer = 0
    while outer < outerSize do
      var left = leftOuter
      var right = rightOuter
      var inner = 0
      while inner < innerSize do
        body(left, right, output)
        left += leftInnerStride
        right += rightInnerStride
        output += 1
        inner += 1
      leftOuter += leftOuterStride
      rightOuter += rightOuterStride
      outer += 1

  private inline def loopGeneral(
      plan: LoopPlan,
      inline body: (Int, Int, Int) => Unit
  ): Unit =
    if plan.size > 0 then
      if plan.rank == 0 then
        body(plan.leftOffset, plan.rightOffset, 0)
      else
        val counters = new Array[Int](plan.rank)
        var left = plan.leftOffset
        var right = plan.rightOffset
        var output = 0
        while output < plan.size do
          body(left, right, output)
          output += 1
          if output < plan.size then
            var axis = plan.rank - 1
            var advanced = false
            while axis >= 0 && !advanced do
              if counters(axis) + 1 < plan.shape(axis) then
                counters(axis) += 1
                left += plan.leftStrides(axis)
                right += plan.rightStrides(axis)
                advanced = true
              else
                left -= counters(axis) * plan.leftStrides(axis)
                right -= counters(axis) * plan.rightStrides(axis)
                counters(axis) = 0
                axis -= 1

  private def unsupportedArithmetic(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support arithmetic")

  private def unsupportedUnary(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support this unary operation")

  private def unsupportedComparison(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support ordering comparisons")
