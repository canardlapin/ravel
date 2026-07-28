package ravel.internal

private[ravel] object ProbeKernels:
  def get[A](storage: Storage[A], index: Int): A =
    (storage match
      case x: BooleanStorage => x.raw(index)
      case x: ByteStorage    => x.raw(index)
      case x: ShortStorage   => x.raw(index)
      case x: IntStorage     => x.raw(index)
      case x: LongStorage    => x.raw(index)
      case x: FloatStorage   => x.raw(index)
      case x: DoubleStorage  => x.raw(index)
    ).asInstanceOf[A]

  def set[A](storage: Storage[A], index: Int, value: A): Unit =
    storage match
      case x: BooleanStorage => x.raw(index) = value.asInstanceOf[Boolean]
      case x: ByteStorage    => x.raw(index) = value.asInstanceOf[Byte]
      case x: ShortStorage   => x.raw(index) = value.asInstanceOf[Short]
      case x: IntStorage     => x.raw(index) = value.asInstanceOf[Int]
      case x: LongStorage    => x.raw(index) = value.asInstanceOf[Long]
      case x: FloatStorage   => x.raw(index) = value.asInstanceOf[Float]
      case x: DoubleStorage  => x.raw(index) = value.asInstanceOf[Double]

  def fill[A](storage: Storage[A], value: A): Unit =
    storage match
      case x: BooleanStorage => java.util.Arrays.fill(x.raw, value.asInstanceOf[Boolean])
      case x: ByteStorage    => java.util.Arrays.fill(x.raw, value.asInstanceOf[Byte])
      case x: ShortStorage   => java.util.Arrays.fill(x.raw, value.asInstanceOf[Short])
      case x: IntStorage     => java.util.Arrays.fill(x.raw, value.asInstanceOf[Int])
      case x: LongStorage    => java.util.Arrays.fill(x.raw, value.asInstanceOf[Long])
      case x: FloatStorage   => java.util.Arrays.fill(x.raw, value.asInstanceOf[Float])
      case x: DoubleStorage  => java.util.Arrays.fill(x.raw, value.asInstanceOf[Double])

  def copy[A](
      source: Storage[A],
      sourceOffset: Int,
      target: Storage[A],
      targetOffset: Int,
      length: Int
  ): Unit =
    (source, target) match
      case (x: BooleanStorage, y: BooleanStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: ByteStorage, y: ByteStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: ShortStorage, y: ShortStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: IntStorage, y: IntStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: LongStorage, y: LongStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: FloatStorage, y: FloatStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case (x: DoubleStorage, y: DoubleStorage) =>
        System.arraycopy(x.raw, sourceOffset, y.raw, targetOffset, length)
      case _ => throw new IllegalArgumentException("dtype mismatch")

  def add[A](left: Storage[A], right: Storage[A], out: Storage[A], size: Int): Unit =
    (left, right, out) match
      case (x: IntStorage, y: IntStorage, z: IntStorage)       => addInt(x.raw, y.raw, z.raw, size)
      case (x: LongStorage, y: LongStorage, z: LongStorage)    => addLong(x.raw, y.raw, z.raw, size)
      case (x: FloatStorage, y: FloatStorage, z: FloatStorage) => addFloat(x.raw, y.raw, z.raw, size)
      case (x: DoubleStorage, y: DoubleStorage, z: DoubleStorage) =>
        addDouble(x.raw, y.raw, z.raw, size)
      case _ => throw new UnsupportedOperationException("arithmetic requires matching Int, Long, Float, or Double storage")

  def addLinear[A](
      left: Storage[A],
      leftOffset: Int,
      right: Storage[A],
      rightOffset: Int,
      out: Storage[A],
      size: Int
  ): Unit =
    (left, right, out) match
      case (x: IntStorage, y: IntStorage, z: IntStorage) =>
        addIntLinear(x.raw, leftOffset, y.raw, rightOffset, z.raw, size)
      case (x: LongStorage, y: LongStorage, z: LongStorage) =>
        addLongLinear(x.raw, leftOffset, y.raw, rightOffset, z.raw, size)
      case (x: FloatStorage, y: FloatStorage, z: FloatStorage) =>
        addFloatLinear(x.raw, leftOffset, y.raw, rightOffset, z.raw, size)
      case (x: DoubleStorage, y: DoubleStorage, z: DoubleStorage) =>
        addDoubleLinear(x.raw, leftOffset, y.raw, rightOffset, z.raw, size)
      case _ => throw new UnsupportedOperationException("arithmetic requires matching Int, Long, Float, or Double storage")

  def negate[A](source: Storage[A], out: Storage[A], size: Int): Unit =
    (source, out) match
      case (x: IntStorage, z: IntStorage)       => negateInt(x.raw, z.raw, size)
      case (x: LongStorage, z: LongStorage)     => negateLong(x.raw, z.raw, size)
      case (x: FloatStorage, z: FloatStorage)   => negateFloat(x.raw, z.raw, size)
      case (x: DoubleStorage, z: DoubleStorage) => negateDouble(x.raw, z.raw, size)
      case _ => throw new UnsupportedOperationException("arithmetic requires matching Int, Long, Float, or Double storage")

  def addStrided[A](
      left: Storage[A],
      leftOffset: Int,
      leftStride: Int,
      right: Storage[A],
      rightOffset: Int,
      rightStride: Int,
      out: Storage[A],
      size: Int
  ): Unit =
    (left, right, out) match
      case (x: IntStorage, y: IntStorage, z: IntStorage) =>
        addIntStrided(x.raw, leftOffset, leftStride, y.raw, rightOffset, rightStride, z.raw, size)
      case (x: LongStorage, y: LongStorage, z: LongStorage) =>
        addLongStrided(x.raw, leftOffset, leftStride, y.raw, rightOffset, rightStride, z.raw, size)
      case (x: FloatStorage, y: FloatStorage, z: FloatStorage) =>
        addFloatStrided(x.raw, leftOffset, leftStride, y.raw, rightOffset, rightStride, z.raw, size)
      case (x: DoubleStorage, y: DoubleStorage, z: DoubleStorage) =>
        addDoubleStrided(x.raw, leftOffset, leftStride, y.raw, rightOffset, rightStride, z.raw, size)
      case _ => throw new UnsupportedOperationException("arithmetic requires matching Int, Long, Float, or Double storage")

  private def addInt(x: Array[Int], y: Array[Int], out: Array[Int], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addLong(x: Array[Long], y: Array[Long], out: Array[Long], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addFloat(x: Array[Float], y: Array[Float], out: Array[Float], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addDouble(x: Array[Double], y: Array[Double], out: Array[Double], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addIntLinear(
      x: Array[Int],
      xo: Int,
      y: Array[Int],
      yo: Int,
      out: Array[Int],
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = x(xo + i) + y(yo + i)
      i += 1

  private def addLongLinear(
      x: Array[Long],
      xo: Int,
      y: Array[Long],
      yo: Int,
      out: Array[Long],
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = x(xo + i) + y(yo + i)
      i += 1

  private def addFloatLinear(
      x: Array[Float],
      xo: Int,
      y: Array[Float],
      yo: Int,
      out: Array[Float],
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = x(xo + i) + y(yo + i)
      i += 1

  private def addDoubleLinear(
      x: Array[Double],
      xo: Int,
      y: Array[Double],
      yo: Int,
      out: Array[Double],
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = x(xo + i) + y(yo + i)
      i += 1

  private def negateInt(x: Array[Int], out: Array[Int], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def negateLong(x: Array[Long], out: Array[Long], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def negateFloat(x: Array[Float], out: Array[Float], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def negateDouble(x: Array[Double], out: Array[Double], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def addIntStrided(x: Array[Int], xo: Int, xs: Int, y: Array[Int], yo: Int, ys: Int, out: Array[Int], size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = x(xi) + y(yi)
      xi += xs
      yi += ys
      i += 1

  private def addLongStrided(x: Array[Long], xo: Int, xs: Int, y: Array[Long], yo: Int, ys: Int, out: Array[Long], size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = x(xi) + y(yi)
      xi += xs
      yi += ys
      i += 1

  private def addFloatStrided(x: Array[Float], xo: Int, xs: Int, y: Array[Float], yo: Int, ys: Int, out: Array[Float], size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = x(xi) + y(yi)
      xi += xs
      yi += ys
      i += 1

  private def addDoubleStrided(x: Array[Double], xo: Int, xs: Int, y: Array[Double], yo: Int, ys: Int, out: Array[Double], size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = x(xi) + y(yi)
      xi += xs
      yi += ys
      i += 1
