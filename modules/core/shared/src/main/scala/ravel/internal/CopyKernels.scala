package ravel.internal

/** Logical row-major materialization without per-element storage dispatch.
  *
  * The storage family is matched once. Contiguous sources use the platform bulk copy, rank-one and
  * rank-two views use small direct loops, and only higher ranks allocate a counter vector.
  */
private[ravel] object CopyKernels:
  private val TileSize = 8

  def logical[A](
      source: Storage[A],
      layout: Layout,
      target: Storage[A]
  ): Unit =
    if layout.size == 0 then ()
    else if layout.isCContiguous then
      ProbeKernels.copy(source, layout.offset, target, 0, layout.size)
    else
      (source, target) match
        case (x: BooleanStorage, z: BooleanStorage) =>
          copyBoolean(layout, x, z)
        case (x: ByteStorage, z: ByteStorage) =>
          copyByte(layout, x, z)
        case (x: ShortStorage, z: ShortStorage) =>
          copyShort(layout, x, z)
        case (x: IntStorage, z: IntStorage) =>
          copyInt(layout, x, z)
        case (x: LongStorage, z: LongStorage) =>
          copyLong(layout, x, z)
        case (x: FloatStorage, z: FloatStorage) =>
          copyFloat(layout, x, z)
        case (x: DoubleStorage, z: DoubleStorage) =>
          copyDouble(layout, x, z)
        case _ => throw new IllegalArgumentException("copy dtype mismatch")

  private def copyBoolean(
      layout: Layout,
      source: BooleanStorage,
      target: BooleanStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyByte(
      layout: Layout,
      source: ByteStorage,
      target: ByteStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyShort(
      layout: Layout,
      source: ShortStorage,
      target: ShortStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyInt(
      layout: Layout,
      source: IntStorage,
      target: IntStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyLong(
      layout: Layout,
      source: LongStorage,
      target: LongStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyFloat(
      layout: Layout,
      source: FloatStorage,
      target: FloatStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private def copyDouble(
      layout: Layout,
      source: DoubleStorage,
      target: DoubleStorage
  ): Unit =
    copyLogical(layout, source.raw.apply, target.raw.update)

  private inline def copyLogical[T](
      layout: Layout,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  ): Unit =
    layout.rank match
      case 0 =>
        write(0, read(layout.offset))
      case 1 =>
        copyRankOne(layout, read, write)
      case 2 =>
        if (layout.strides(0) == 1 || layout.strides(0) == -1) &&
          layout.shape(0) > TileSize &&
          layout.shape(1) > TileSize
        then copyRankTwoTiled(layout, read, write)
        else copyRankTwo(layout, read, write)
      case _ =>
        copyGeneral(layout, read, write)

  private inline def copyRankOne[T](
      layout: Layout,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  ): Unit =
    var physical = layout.offset
    var output = 0
    val stride = layout.strides(0)
    while output < layout.size do
      write(output, read(physical))
      physical += stride
      output += 1

  private inline def copyRankTwo[T](
      layout: Layout,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  ): Unit =
    val rows = layout.shape(0)
    val columns = layout.shape(1)
    val rowStride = layout.strides(0)
    val columnStride = layout.strides(1)
    var rowBase = layout.offset
    var output = 0
    var row = 0
    while row < rows do
      var physical = rowBase
      var column = 0
      while column < columns do
        write(output, read(physical))
        physical += columnStride
        output += 1
        column += 1
      rowBase += rowStride
      row += 1

  /** Balance source and destination locality for transpose-like layouts.
    *
    * The row stride is unit, so the inner loop reads adjacent source values while writing one
    * destination column. Bounding both dimensions to a small tile keeps those strided destination
    * cache lines live.
    */
  private inline def copyRankTwoTiled[T](
      layout: Layout,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  ): Unit =
    val rows = layout.shape(0)
    val columns = layout.shape(1)
    val rowStride = layout.strides(0)
    val columnStride = layout.strides(1)
    var rowStart = 0
    while rowStart < rows do
      val rowEnd = math.min(rowStart + TileSize, rows)
      var columnStart = 0
      while columnStart < columns do
        val columnEnd = math.min(columnStart + TileSize, columns)
        var row = rowStart
        while row < rowEnd do
          var physical =
            layout.offset + row * rowStride + columnStart * columnStride
          var output = row * columns + columnStart
          if columnEnd - columnStart == TileSize then
            write(output, read(physical))
            write(output + 1, read(physical + columnStride))
            write(output + 2, read(physical + columnStride * 2))
            write(output + 3, read(physical + columnStride * 3))
            write(output + 4, read(physical + columnStride * 4))
            write(output + 5, read(physical + columnStride * 5))
            write(output + 6, read(physical + columnStride * 6))
            write(output + 7, read(physical + columnStride * 7))
          else
            var column = columnStart
            while column < columnEnd do
              write(output, read(physical))
              physical += columnStride
              output += 1
              column += 1
          row += 1
        columnStart = columnEnd
      rowStart = rowEnd

  private inline def copyGeneral[T](
      layout: Layout,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  ): Unit =
    val counters = new Array[Int](layout.rank)
    var physical = layout.offset
    var output = 0
    while output < layout.size do
      write(output, read(physical))
      output += 1
      if output < layout.size then
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
