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
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.min(a, b).toByte
        )
      case KernelOp.Maximum =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.max(a, b).toByte
        )
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
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.min(a, b).toShort
        )
      case KernelOp.Maximum =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.max(a, b).toShort
        )
      case _ => unsupportedArithmetic("Short")

  private def binaryInt(
      operation: Byte,
      x: IntStorage,
      y: IntStorage,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)
      case KernelOp.Subtract => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)
      case KernelOp.Multiply => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)
      case KernelOp.Divide   => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)
      case KernelOp.Minimum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)
      case KernelOp.Maximum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

  private def binaryLong(
      operation: Byte,
      x: LongStorage,
      y: LongStorage,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)
      case KernelOp.Subtract => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)
      case KernelOp.Multiply => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)
      case KernelOp.Divide   => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)
      case KernelOp.Minimum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)
      case KernelOp.Maximum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

  private def binaryFloat(
      operation: Byte,
      x: FloatStorage,
      y: FloatStorage,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          (a + b).toFloat
        )
      case KernelOp.Subtract =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          (a - b).toFloat
        )
      case KernelOp.Multiply =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          (a * b).toFloat
        )
      case KernelOp.Divide =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          (a / b).toFloat
        )
      case KernelOp.Minimum =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.min(a, b).toFloat
        )
      case KernelOp.Maximum =>
        binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)((a, b) =>
          math.max(a, b).toFloat
        )

  private def binaryDouble(
      operation: Byte,
      x: DoubleStorage,
      y: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ + _)
      case KernelOp.Subtract => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ - _)
      case KernelOp.Multiply => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ * _)
      case KernelOp.Divide   => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(_ / _)
      case KernelOp.Minimum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.min)
      case KernelOp.Maximum  => binaryLoop(plan, x.raw.apply, y.raw.apply, z.raw.update)(math.max)

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
      case KernelOp.Negate =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => (-a).toFloat)
      case KernelOp.Absolute =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.abs(a).toFloat)
      case KernelOp.Sqrt =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.sqrt(a.toDouble).toFloat)
      case KernelOp.Exp =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.exp(a.toDouble).toFloat)
      case KernelOp.Log =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.log(a.toDouble).toFloat)
      case KernelOp.Sin =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.sin(a.toDouble).toFloat)
      case KernelOp.Cos =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.cos(a.toDouble).toFloat)
      case KernelOp.Tan =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.tan(a.toDouble).toFloat)
      case KernelOp.Floor =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.floor(a.toDouble).toFloat)
      case KernelOp.Ceil =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.ceil(a.toDouble).toFloat)

  private def unaryDouble(
      operation: Byte,
      x: DoubleStorage,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Negate   => unaryLoop(plan, x.raw.apply, z.raw.update)(a => -a)
      case KernelOp.Absolute => unaryLoop(plan, x.raw.apply, z.raw.update)(math.abs)
      case KernelOp.Sqrt     => unaryLoop(plan, x.raw.apply, z.raw.update)(math.sqrt)
      case KernelOp.Exp      => unaryLoop(plan, x.raw.apply, z.raw.update)(math.exp)
      case KernelOp.Log      => unaryLoop(plan, x.raw.apply, z.raw.update)(math.log)
      case KernelOp.Sin      => unaryLoop(plan, x.raw.apply, z.raw.update)(math.sin)
      case KernelOp.Cos      => unaryLoop(plan, x.raw.apply, z.raw.update)(math.cos)
      case KernelOp.Tan      => unaryLoop(plan, x.raw.apply, z.raw.update)(math.tan)
      case KernelOp.Floor    => unaryLoop(plan, x.raw.apply, z.raw.update)(math.floor)
      case KernelOp.Ceil     => unaryLoop(plan, x.raw.apply, z.raw.update)(math.ceil)

  private def scalarByte(
      operation: Byte,
      x: ByteStorage,
      value: Byte,
      z: ByteStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toByte)
      case KernelOp.Maximum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toByte)
      case _ => unsupportedArithmetic("Byte")

  private def scalarShort(
      operation: Byte,
      x: ShortStorage,
      value: Short,
      z: ShortStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Minimum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toShort)
      case KernelOp.Maximum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toShort)
      case _ => unsupportedArithmetic("Short")

  private def scalarInt(
      operation: Byte,
      x: IntStorage,
      value: Int,
      z: IntStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)
      case KernelOp.Subtract => unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)
      case KernelOp.Multiply => unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)
      case KernelOp.Divide   => unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)
      case KernelOp.Minimum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))
      case KernelOp.Maximum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  private def scalarLong(
      operation: Byte,
      x: LongStorage,
      value: Long,
      z: LongStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)
      case KernelOp.Subtract => unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)
      case KernelOp.Multiply => unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)
      case KernelOp.Divide   => unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)
      case KernelOp.Minimum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))
      case KernelOp.Maximum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  private def scalarFloat(
      operation: Byte,
      x: FloatStorage,
      value: Float,
      z: FloatStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a + value).toFloat)
      case KernelOp.Subtract =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a - value).toFloat)
      case KernelOp.Multiply =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a * value).toFloat)
      case KernelOp.Divide =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => (a / value).toFloat)
      case KernelOp.Minimum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(a, value).toFloat)
      case KernelOp.Maximum =>
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.max(a, value).toFloat)

  private def scalarDouble(
      operation: Byte,
      x: DoubleStorage,
      value: Double,
      z: DoubleStorage,
      plan: LoopPlan
  ): Unit =
    operation match
      case KernelOp.Add      => unaryLoop(plan, x.raw.apply, z.raw.update)(_ + value)
      case KernelOp.Subtract => unaryLoop(plan, x.raw.apply, z.raw.update)(_ - value)
      case KernelOp.Multiply => unaryLoop(plan, x.raw.apply, z.raw.update)(_ * value)
      case KernelOp.Divide   => unaryLoop(plan, x.raw.apply, z.raw.update)(_ / value)
      case KernelOp.Minimum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.min(_, value))
      case KernelOp.Maximum  => unaryLoop(plan, x.raw.apply, z.raw.update)(math.max(_, value))

  def clip[A](
      source: Storage[A],
      lower: A,
      upper: A,
      output: Storage[A],
      plan: LoopPlan
  ): Unit =
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        val lo = lower.asInstanceOf[Byte]
        val hi = upper.asInstanceOf[Byte]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi).toByte)
      case (x: ShortStorage, z: ShortStorage) =>
        val lo = lower.asInstanceOf[Short]
        val hi = upper.asInstanceOf[Short]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi).toShort)
      case (x: IntStorage, z: IntStorage) =>
        val lo = lower.asInstanceOf[Int]
        val hi = upper.asInstanceOf[Int]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi))
      case (x: LongStorage, z: LongStorage) =>
        val lo = lower.asInstanceOf[Long]
        val hi = upper.asInstanceOf[Long]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi))
      case (x: FloatStorage, z: FloatStorage) =>
        val lo = lower.asInstanceOf[Float]
        val hi = upper.asInstanceOf[Float]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        val lo = lower.asInstanceOf[Double]
        val hi = upper.asInstanceOf[Double]
        unaryLoop(plan, x.raw.apply, z.raw.update)(a => math.min(math.max(a, lo), hi))
      case _ => throw new IllegalArgumentException("clip kernel dtype mismatch")

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
      case KernelOp.Equal =>
        compareLoop(plan, output, i => x.raw(i), i => y.raw(i))(_ == _)
      case KernelOp.NotEqual =>
        compareLoop(plan, output, i => x.raw(i), i => y.raw(i))(_ != _)
      case _ => unsupportedComparison("Boolean")

  private def compareByte(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: ByteStorage,
      y: ByteStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

  private def compareShort(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: ShortStorage,
      y: ShortStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

  private def compareInt(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: IntStorage,
      y: IntStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

  private def compareLong(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: LongStorage,
      y: LongStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

  private def compareFloat(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: FloatStorage,
      y: FloatStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

  private def compareDouble(
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      x: DoubleStorage,
      y: DoubleStorage
  ): Unit =
    orderedCompare(operation, plan, output, x.raw.apply, y.raw.apply)(
      _ == _,
      _ < _,
      _ <= _
    )

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
          case KernelOp.IsNaN   => predicateLoop(plan, boolean, x.raw.apply)(_.isNaN)
          case KernelOp.IsFinite => predicateLoop(plan, boolean, x.raw.apply)(_.isFinite)
      case x: DoubleStorage =>
        operation match
          case KernelOp.IsNaN   => predicateLoop(plan, boolean, x.raw.apply)(_.isNaN)
          case KernelOp.IsFinite => predicateLoop(plan, boolean, x.raw.apply)(_.isFinite)
      case _ => throw new UnsupportedOperationException("floating predicate requires Float or Double")

  private inline def binaryLoop[T](
      plan: LoopPlan,
      inline left: Int => T,
      inline right: Int => T,
      inline output: (Int, T) => Unit
  )(inline operation: (T, T) => T): Unit =
    loopOffsets(plan) { (leftIndex, rightIndex, outputIndex) =>
      output(outputIndex, operation(left(leftIndex), right(rightIndex)))
    }

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

  private inline def orderedCompare[T](
      operation: Byte,
      plan: LoopPlan,
      output: BooleanStorage,
      inline left: Int => T,
      inline right: Int => T
  )(
      inline equal: (T, T) => Boolean,
      inline less: (T, T) => Boolean,
      inline lessEqual: (T, T) => Boolean
  ): Unit =
    operation match
      case KernelOp.Equal     => compareLoop(plan, output, left, right)(equal)
      case KernelOp.NotEqual  => compareLoop(plan, output, left, right)((a, b) => !equal(a, b))
      case KernelOp.Less      => compareLoop(plan, output, left, right)(less)
      case KernelOp.LessEqual => compareLoop(plan, output, left, right)(lessEqual)

  private def unsupportedArithmetic(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support arithmetic")

  private def unsupportedUnary(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support this unary operation")

  private def unsupportedComparison(dtype: String): Nothing =
    throw new UnsupportedOperationException(s"$dtype does not support ordering comparisons")
