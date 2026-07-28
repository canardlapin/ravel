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
    else
      // Storage dispatch happens once. Each selected loop reads and writes its primitive
      // representation directly; there is no generic get/set dispatch per element.
      (source, destination) match
        case (x: BooleanStorage, z: BooleanStorage) =>
          assignBoolean(x, sourceLayout, z, destinationLayout)
        case (x: ByteStorage, z: ByteStorage) =>
          assignByte(x, sourceLayout, z, destinationLayout)
        case (x: ShortStorage, z: ShortStorage) =>
          assignShort(x, sourceLayout, z, destinationLayout)
        case (x: IntStorage, z: IntStorage) =>
          assignInt(x, sourceLayout, z, destinationLayout)
        case (x: LongStorage, z: LongStorage) =>
          assignLong(x, sourceLayout, z, destinationLayout)
        case (x: FloatStorage, z: FloatStorage) =>
          assignFloat(x, sourceLayout, z, destinationLayout)
        case (x: DoubleStorage, z: DoubleStorage) =>
          assignDouble(x, sourceLayout, z, destinationLayout)
        case _ => throw new IllegalArgumentException("assign requires matching storage dtypes")

  def scalarInPlace[A](
      operation: Byte,
      storage: Storage[A],
      layout: Layout,
      value: A
  ): Unit =
    if layout.size == 0 then ()
    else
      // The tuple is a one-time erased-type boundary. Dtype and operation matches both
      // precede the primitive loop, so neither boxes values nor redispatches per element.
      (storage, value) match
        case (x: IntStorage, y: Int) => scalarInt(operation, x, layout, y)
        case (x: LongStorage, y: Long) => scalarLong(operation, x, layout, y)
        case (x: FloatStorage, y: Float) => scalarFloat(operation, x, layout, y)
        case (x: DoubleStorage, y: Double) => scalarDouble(operation, x, layout, y)
        case _ =>
          throw new IllegalArgumentException("unsupported in-place scalar dtype")

  private def assignBoolean(
      source: BooleanStorage,
      sourceLayout: Layout,
      destination: BooleanStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignByte(
      source: ByteStorage,
      sourceLayout: Layout,
      destination: ByteStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignShort(
      source: ShortStorage,
      sourceLayout: Layout,
      destination: ShortStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignInt(
      source: IntStorage,
      sourceLayout: Layout,
      destination: IntStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignLong(
      source: LongStorage,
      sourceLayout: Layout,
      destination: LongStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignFloat(
      source: FloatStorage,
      sourceLayout: Layout,
      destination: FloatStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def assignDouble(
      source: DoubleStorage,
      sourceLayout: Layout,
      destination: DoubleStorage,
      destinationLayout: Layout
  ): Unit =
    val x = source.raw
    val z = destination.raw
    foreachPhysicalIndexPair(sourceLayout, destinationLayout) { (read, write) =>
      z(write) = x(read)
    }

  private def scalarInt(
      operation: Byte,
      storage: IntStorage,
      layout: Layout,
      value: Int
  ): Unit =
    operation match
      case KernelOp.Add => addInt(storage, layout, value)
      case KernelOp.Subtract => subtractInt(storage, layout, value)
      case KernelOp.Multiply => multiplyInt(storage, layout, value)
      case KernelOp.Divide => divideInt(storage, layout, value)
      case _ => throw new MatchError(operation)

  private def scalarLong(
      operation: Byte,
      storage: LongStorage,
      layout: Layout,
      value: Long
  ): Unit =
    operation match
      case KernelOp.Add => addLong(storage, layout, value)
      case KernelOp.Subtract => subtractLong(storage, layout, value)
      case KernelOp.Multiply => multiplyLong(storage, layout, value)
      case KernelOp.Divide => divideLong(storage, layout, value)
      case _ => throw new MatchError(operation)

  private def scalarFloat(
      operation: Byte,
      storage: FloatStorage,
      layout: Layout,
      value: Float
  ): Unit =
    operation match
      case KernelOp.Add => addFloat(storage, layout, value)
      case KernelOp.Subtract => subtractFloat(storage, layout, value)
      case KernelOp.Multiply => multiplyFloat(storage, layout, value)
      case KernelOp.Divide => divideFloat(storage, layout, value)
      case _ => throw new MatchError(operation)

  private def scalarDouble(
      operation: Byte,
      storage: DoubleStorage,
      layout: Layout,
      value: Double
  ): Unit =
    operation match
      case KernelOp.Add => addDouble(storage, layout, value)
      case KernelOp.Subtract => subtractDouble(storage, layout, value)
      case KernelOp.Multiply => multiplyDouble(storage, layout, value)
      case KernelOp.Divide => divideDouble(storage, layout, value)
      case _ => throw new MatchError(operation)

  private def addInt(storage: IntStorage, layout: Layout, value: Int): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) + value
    }

  private def subtractInt(storage: IntStorage, layout: Layout, value: Int): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) - value
    }

  private def multiplyInt(storage: IntStorage, layout: Layout, value: Int): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) * value
    }

  private def divideInt(storage: IntStorage, layout: Layout, value: Int): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) / value
    }

  private def addLong(storage: LongStorage, layout: Layout, value: Long): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) + value
    }

  private def subtractLong(storage: LongStorage, layout: Layout, value: Long): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) - value
    }

  private def multiplyLong(storage: LongStorage, layout: Layout, value: Long): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) * value
    }

  private def divideLong(storage: LongStorage, layout: Layout, value: Long): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) / value
    }

  private def addFloat(storage: FloatStorage, layout: Layout, value: Float): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = (raw(index) + value).toFloat
    }

  private def subtractFloat(storage: FloatStorage, layout: Layout, value: Float): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = (raw(index) - value).toFloat
    }

  private def multiplyFloat(storage: FloatStorage, layout: Layout, value: Float): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = (raw(index) * value).toFloat
    }

  private def divideFloat(storage: FloatStorage, layout: Layout, value: Float): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = (raw(index) / value).toFloat
    }

  private def addDouble(storage: DoubleStorage, layout: Layout, value: Double): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) + value
    }

  private def subtractDouble(storage: DoubleStorage, layout: Layout, value: Double): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) - value
    }

  private def multiplyDouble(storage: DoubleStorage, layout: Layout, value: Double): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) * value
    }

  private def divideDouble(storage: DoubleStorage, layout: Layout, value: Double): Unit =
    val raw = storage.raw
    foreachPhysicalIndex(layout) { index =>
      raw(index) = raw(index) / value
    }

  /** Inline address traversal keeps the selected primitive operation inside the loop body. */
  private inline def foreachPhysicalIndex(
      layout: Layout
  )(inline visit: Int => Unit): Unit =
    if layout.isCContiguous then
      var index = 0
      var address = layout.offset
      while index < layout.size do
        visit(address)
        index += 1
        address += 1
    else if layout.rank == 0 then visit(layout.offset)
    else if layout.rank == 1 then
      var index = 0
      var address = layout.offset
      while index < layout.size do
        visit(address)
        index += 1
        if index < layout.size then address += layout.strides(0)
    else if layout.rank == 2 then
      val rows = layout.shape(0)
      val columns = layout.shape(1)
      val rowStride = layout.strides(0)
      val columnStride = layout.strides(1)
      var row = 0
      var rowAddress = layout.offset
      while row < rows do
        var column = 0
        var address = rowAddress
        while column < columns do
          visit(address)
          column += 1
          if column < columns then address += columnStride
        row += 1
        if row < rows then rowAddress += rowStride
    else
      val counters = new Array[Int](layout.rank)
      var address = layout.offset.toLong
      var visited = 0
      while visited < layout.size do
        visit(Layout.checkedInt(address, "mutable iteration address"))
        visited += 1
        if visited < layout.size then
          var axis = layout.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            counters(axis) += 1
            address = Layout.checkedAdd(
              address,
              layout.strides(axis).toLong,
              s"mutable advance axis $axis"
            )
            if counters(axis) < layout.shape(axis) then advanced = true
            else
              address = Layout.checkedAdd(
                address,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  layout.strides(axis).toLong,
                  s"mutable rewind axis $axis"
                ),
                s"mutable rewind axis $axis"
              )
              counters(axis) = 0
              axis -= 1

  /** Inline paired traversal preserves the existing logical-order alias semantics. */
  private inline def foreachPhysicalIndexPair(
      source: Layout,
      destination: Layout
  )(inline visit: (Int, Int) => Unit): Unit =
    if source.isCContiguous && destination.isCContiguous then
      var visited = 0
      var read = source.offset
      var write = destination.offset
      while visited < source.size do
        visit(read, write)
        visited += 1
        read += 1
        write += 1
    else if source.rank == 0 then visit(source.offset, destination.offset)
    else if source.rank == 1 then
      var visited = 0
      var read = source.offset
      var write = destination.offset
      while visited < source.size do
        visit(read, write)
        visited += 1
        if visited < source.size then
          read += source.strides(0)
          write += destination.strides(0)
    else if source.rank == 2 then
      val rows = source.shape(0)
      val columns = source.shape(1)
      val sourceRowStride = source.strides(0)
      val sourceColumnStride = source.strides(1)
      val destinationRowStride = destination.strides(0)
      val destinationColumnStride = destination.strides(1)
      var row = 0
      var sourceRowAddress = source.offset
      var destinationRowAddress = destination.offset
      while row < rows do
        var column = 0
        var read = sourceRowAddress
        var write = destinationRowAddress
        while column < columns do
          visit(read, write)
          column += 1
          if column < columns then
            read += sourceColumnStride
            write += destinationColumnStride
        row += 1
        if row < rows then
          sourceRowAddress += sourceRowStride
          destinationRowAddress += destinationRowStride
    else
      val counters = new Array[Int](source.rank)
      var read = source.offset.toLong
      var write = destination.offset.toLong
      var visited = 0
      while visited < source.size do
        visit(
          Layout.checkedInt(read, "assign source"),
          Layout.checkedInt(write, "assign destination")
        )
        visited += 1
        if visited < source.size then
          var axis = source.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            counters(axis) += 1
            read = Layout.checkedAdd(read, source.strides(axis).toLong, "assign source")
            write = Layout.checkedAdd(write, destination.strides(axis).toLong, "assign destination")
            if counters(axis) < source.shape(axis) then advanced = true
            else
              read = Layout.checkedAdd(
                read,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  source.strides(axis).toLong,
                  "assign source rewind"
                ),
                "assign source rewind"
              )
              write = Layout.checkedAdd(
                write,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  destination.strides(axis).toLong,
                  "assign destination rewind"
                ),
                "assign destination rewind"
              )
              counters(axis) = 0
              axis -= 1
