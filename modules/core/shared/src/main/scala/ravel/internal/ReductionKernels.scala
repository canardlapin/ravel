package ravel.internal

import ravel.*

private[ravel] object ReductionKernels:
  val PairwiseBlockSize: Int = 128

  def sum[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage    => logicalFold(layout, x.raw.apply, 0)(_ + _)
      case x: LongStorage   => logicalFold(layout, x.raw.apply, 0L)(_ + _)
      case x: FloatStorage  => pairwiseFloat(layout, x.raw.apply)
      case x: DoubleStorage => pairwiseDouble(layout, x.raw.apply)
      case _ => throw new UnsupportedOperationException("sum requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def product[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage    => logicalFold(layout, x.raw.apply, 1)(_ * _)
      case x: LongStorage   => logicalFold(layout, x.raw.apply, 1L)(_ * _)
      case x: FloatStorage  => logicalFold(layout, x.raw.apply, 1.0f)((a, b) => (a * b).toFloat)
      case x: DoubleStorage => logicalFold(layout, x.raw.apply, 1.0)(_ * _)
      case _ => throw new UnsupportedOperationException("product requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def minimum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("min")
    (source match
      case x: ByteStorage   => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toByte)
      case x: ShortStorage  => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toShort)
      case x: IntStorage    => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: LongStorage   => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: FloatStorage  => logicalExtremum(layout, x.raw.apply)((a, b) => math.min(a, b).toFloat)
      case x: DoubleStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case _ => throw new UnsupportedOperationException("min requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def maximum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("max")
    (source match
      case x: ByteStorage   => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toByte)
      case x: ShortStorage  => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toShort)
      case x: IntStorage    => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: LongStorage   => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: FloatStorage  => logicalExtremum(layout, x.raw.apply)((a, b) => math.max(a, b).toFloat)
      case x: DoubleStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case _ => throw new UnsupportedOperationException("max requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def argMinimum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMin")
    source match
      case x: ByteStorage   => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: ShortStorage  => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: IntStorage    => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: LongStorage   => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: FloatStorage  => logicalArgFloat(layout, x.raw.apply, minimum = true)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = true)
      case _ => throw new UnsupportedOperationException("argMin requires an ordered numeric dtype")

  def argMaximum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMax")
    source match
      case x: ByteStorage   => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: ShortStorage  => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: IntStorage    => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: LongStorage   => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: FloatStorage  => logicalArgFloat(layout, x.raw.apply, minimum = false)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = false)
      case _ => throw new UnsupportedOperationException("argMax requires an ordered numeric dtype")

  def mean[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: FloatStorage =>
        if layout.size == 0 then Float.NaN
        else (pairwiseDouble(layout, index => x.raw(index).toDouble) / layout.size.toDouble).toFloat
      case x: DoubleStorage =>
        if layout.size == 0 then Double.NaN
        else pairwiseDouble(layout, x.raw.apply) / layout.size.toDouble
      case _ => throw new UnsupportedOperationException("mean requires Float or Double")
    ).asInstanceOf[A]

  def sumAsLong(source: Storage[Int], layout: Layout): Long =
    val raw = source.asInstanceOf[IntStorage].raw
    logicalFold(layout, raw.apply, 0L)((acc, value) => acc + value.toLong)

  def sumAsDouble(source: Storage[Float], layout: Layout): Double =
    val raw = source.asInstanceOf[FloatStorage].raw
    pairwiseDouble(layout, index => raw(index).toDouble)

  def sumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 0)(_ + _)
      case (x: LongStorage, z: LongStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 0L)(_ + _)
      case (x: FloatStorage, z: FloatStorage) =>
        axisPairwiseFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisPairwiseDouble(plan, x.raw.apply, z.raw.update)
      case _ => throw new UnsupportedOperationException("sum requires matching arithmetic storage")

  def productAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1)(_ * _)
      case (x: LongStorage, z: LongStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1L)(_ * _)
      case (x: FloatStorage, z: FloatStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1.0f)((a, b) => (a * b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1.0)(_ * _)
      case _ => throw new UnsupportedOperationException("product requires matching arithmetic storage")

  def minimumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("min")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toByte)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toShort)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.min(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case _ => throw new UnsupportedOperationException("min requires matching ordered storage")

  def maximumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("max")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toByte)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toShort)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.max(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case _ => throw new UnsupportedOperationException("max requires matching ordered storage")

  def meanAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: FloatStorage, z: FloatStorage) =>
        axisPairwiseMeanFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisPairwiseMeanDouble(plan, x.raw.apply, z.raw.update)
      case _ => throw new UnsupportedOperationException("mean requires matching floating storage")

  def argMinimumAxis[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan
  ): Unit =
    axisArgDispatch(source, output, plan, minimum = true)

  def argMaximumAxis[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan
  ): Unit =
    axisArgDispatch(source, output, plan, minimum = false)

  private def axisArgDispatch[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan,
      minimum: Boolean
  ): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then
      throw EmptyReduction(if minimum then "argMin" else "argMax")
    val target = output.asInstanceOf[IntStorage]
    source match
      case x: ByteStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: ShortStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: IntStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: LongStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: FloatStorage =>
        axisArgFloat(plan, x.raw.apply, target.raw.update, minimum)
      case x: DoubleStorage =>
        axisArgDouble(plan, x.raw.apply, target.raw.update, minimum)
      case _ => throw new UnsupportedOperationException("arg reduction requires ordered storage")

  private inline def logicalFold[T, U](
      layout: Layout,
      inline read: Int => T,
      initial: U
  )(inline combine: (U, T) => U): U =
    var result = initial
    foreachLogical(layout) { physical =>
      result = combine(result, read(physical))
    }
    result

  private inline def logicalExtremum[T](
      layout: Layout,
      inline read: Int => T
  )(inline combine: (T, T) => T): T =
    var initialized = false
    var result: T = null.asInstanceOf[T]
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        result = value
        initialized = true
      else result = combine(result, value)
    }
    result

  private inline def logicalArg[T](
      layout: Layout,
      inline read: Int => T,
      inline less: (T, T) => Boolean
  ): Int =
    var initialized = false
    var best: T = null.asInstanceOf[T]
    var bestIndex = 0
    var logical = 0
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
      else if less(value, best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private inline def logicalArgFloat(
      layout: Layout,
      inline read: Int => Float,
      minimum: Boolean
  ): Int =
    var initialized = false
    var best = 0.0f
    var bestIndex = 0
    var logical = 0
    var foundNaN = false
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
        foundNaN = value.isNaN
      else if !foundNaN && value.isNaN then
        best = value
        bestIndex = logical
        foundNaN = true
      else if !foundNaN && (if minimum then value < best else value > best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private inline def logicalArgDouble(
      layout: Layout,
      inline read: Int => Double,
      minimum: Boolean
  ): Int =
    var initialized = false
    var best = 0.0
    var bestIndex = 0
    var logical = 0
    var foundNaN = false
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
        foundNaN = value.isNaN
      else if !foundNaN && value.isNaN then
        best = value
        bestIndex = logical
        foundNaN = true
      else if !foundNaN && (if minimum then value < best else value > best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private inline def pairwiseFloat(
      layout: Layout,
      inline read: Int => Float
  ): Float =
    if layout.size == 0 then 0.0f
    else
      val blocks = new Array[Float]((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
      var block = 0
      var within = 0
      var sum = 0.0f
      foreachLogical(layout) { physical =>
        sum = (sum + read(physical)).toFloat
        within += 1
        if within == PairwiseBlockSize then
          blocks(block) = sum
          block += 1
          within = 0
          sum = 0.0f
      }
      if within > 0 then
        blocks(block) = sum
        block += 1
      mergeFloatBlocks(blocks, block)

  private inline def pairwiseDouble(
      layout: Layout,
      inline read: Int => Double
  ): Double =
    if layout.size == 0 then 0.0
    else
      val blocks = new Array[Double]((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
      var block = 0
      var within = 0
      var sum = 0.0
      foreachLogical(layout) { physical =>
        sum += read(physical)
        within += 1
        if within == PairwiseBlockSize then
          blocks(block) = sum
          block += 1
          within = 0
          sum = 0.0
      }
      if within > 0 then
        blocks(block) = sum
        block += 1
      mergeDoubleBlocks(blocks, block)

  private def mergeFloatBlocks(blocks: Array[Float], initialCount: Int): Float =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(write) = (blocks(read) + blocks(read + 1)).toFloat
        read += 2
        write += 1
      if read < count then
        blocks(write) = blocks(read)
        write += 1
      count = write
    if initialCount == 0 then 0.0f else blocks(0)

  private def mergeDoubleBlocks(blocks: Array[Double], initialCount: Int): Double =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(write) = blocks(read) + blocks(read + 1)
        read += 2
        write += 1
      if read < count then
        blocks(write) = blocks(read)
        write += 1
      count = write
    if initialCount == 0 then 0.0 else blocks(0)

  private inline def axisFold[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit,
      identity: T
  )(inline combine: (T, T) => T): Unit =
    foreachAxisBase(plan) { (base, output) =>
      var result = identity
      var index = 0
      var physical = base
      while index < plan.reducedLength do
        result = combine(result, read(physical))
        physical += plan.reducedStride
        index += 1
      write(output, result)
    }

  private inline def axisExtremum[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  )(inline combine: (T, T) => T): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var result = read(base)
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          result = combine(result, read(physical))
          physical += plan.reducedStride
          index += 1
        write(output, result)
    }

  private inline def axisArg[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, Int) => Unit
  )(inline better: (T, T) => Boolean): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if better(value, best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private inline def axisArgFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Int) => Unit,
      minimum: Boolean
  ): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var foundNaN = best.isNaN
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if !foundNaN && value.isNaN then
            best = value
            bestIndex = index
            foundNaN = true
          else if !foundNaN && (if minimum then value < best else value > best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private inline def axisArgDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Int) => Unit,
      minimum: Boolean
  ): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var foundNaN = best.isNaN
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if !foundNaN && value.isNaN then
            best = value
            bestIndex = index
            foundNaN = true
          else if !foundNaN && (if minimum then value < best else value > best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private inline def axisPairwiseFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = new Array[Float](blockCount)
    foreachAxisBase(plan) { (base, output) =>
      var block = 0
      var index = 0
      var physical = base
      while index < plan.reducedLength do
        var sum = 0.0f
        val until = math.min(index + PairwiseBlockSize, plan.reducedLength)
        while index < until do
          sum = (sum + read(physical)).toFloat
          physical += plan.reducedStride
          index += 1
        blocks(block) = sum
        block += 1
      write(output, mergeFloatBlocks(blocks, block))
    }

  private inline def axisPairwiseDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = new Array[Double](blockCount)
    foreachAxisBase(plan) { (base, output) =>
      var block = 0
      var index = 0
      var physical = base
      while index < plan.reducedLength do
        var sum = 0.0
        val until = math.min(index + PairwiseBlockSize, plan.reducedLength)
        while index < until do
          sum += read(physical)
          physical += plan.reducedStride
          index += 1
        blocks(block) = sum
        block += 1
      write(output, mergeDoubleBlocks(blocks, block))
    }

  private inline def axisPairwiseMeanFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = new Array[Double](blockCount)
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength == 0 then write(output, Float.NaN)
      else
        fillDoubleFiberBlocks(plan, base, blocks, index => read(index).toDouble)
        write(output, (mergeDoubleBlocks(blocks, blockCount) / plan.reducedLength.toDouble).toFloat)
    }

  private inline def axisPairwiseMeanDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = new Array[Double](blockCount)
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength == 0 then write(output, Double.NaN)
      else
        fillDoubleFiberBlocks(plan, base, blocks, read)
        write(output, mergeDoubleBlocks(blocks, blockCount) / plan.reducedLength.toDouble)
    }

  private inline def fillDoubleFiberBlocks(
      plan: ReductionPlan,
      base: Int,
      blocks: Array[Double],
      inline read: Int => Double
  ): Unit =
    var block = 0
    var index = 0
    var physical = base
    while index < plan.reducedLength do
      var sum = 0.0
      val until = math.min(index + PairwiseBlockSize, plan.reducedLength)
      while index < until do
        sum += read(physical)
        physical += plan.reducedStride
        index += 1
      blocks(block) = sum
      block += 1

  private inline def foreachLogical(
      layout: Layout
  )(inline body: Int => Unit): Unit =
    if layout.size > 0 then
      if layout.rank == 0 then body(layout.offset)
      else
        val counters = new Array[Int](layout.rank)
        var physical = layout.offset
        var visited = 0
        while visited < layout.size do
          body(physical)
          visited += 1
          if visited < layout.size then
            var axis = layout.rank - 1
            var advanced = false
            while axis >= 0 && !advanced do
              if counters(axis) + 1 < layout.shape(axis) then
                counters(axis) += 1
                physical += layout.strides(axis)
                advanced = true
              else
                physical -= counters(axis) * layout.strides(axis)
                counters(axis) = 0
                axis -= 1

  private inline def foreachAxisBase(
      plan: ReductionPlan
  )(inline body: (Int, Int) => Unit): Unit =
    if plan.outputSize > 0 then
      if plan.outerShape.isEmpty then body(plan.source.offset, 0)
      else
        val counters = new Array[Int](plan.outerShape.length)
        var base = plan.source.offset
        var output = 0
        while output < plan.outputSize do
          body(base, output)
          output += 1
          if output < plan.outputSize then
            var axis = plan.outerShape.length - 1
            var advanced = false
            while axis >= 0 && !advanced do
              if counters(axis) + 1 < plan.outerShape(axis) then
                counters(axis) += 1
                base += plan.outerStrides(axis)
                advanced = true
              else
                base -= counters(axis) * plan.outerStrides(axis)
                counters(axis) = 0
                axis -= 1
