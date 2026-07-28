package ravel.internal

import scala.scalajs.js.typedarray.*

private[ravel] object ProbeKernels:
  def get[A](storage: Storage[A], index: Int): A =
    (storage match
      case x: BooleanStorage => x.raw(index) != 0
      case x: ByteStorage    => x.raw(index).toByte
      case x: ShortStorage   => x.raw(index).toShort
      case x: IntStorage     => x.raw(index)
      case x: LongStorage    => x.raw(index)
      case x: FloatStorage   => x.raw(index).toFloat
      case x: DoubleStorage  => x.raw(index)
    ).asInstanceOf[A]

  def set[A](storage: Storage[A], index: Int, value: A): Unit =
    storage match
      case x: BooleanStorage => x.raw(index) = (if value.asInstanceOf[Boolean] then 1 else 0)
      case x: ByteStorage    => x.raw(index) = value.asInstanceOf[Byte]
      case x: ShortStorage   => x.raw(index) = value.asInstanceOf[Short]
      case x: IntStorage     => x.raw(index) = value.asInstanceOf[Int]
      case x: LongStorage    => x.raw(index) = value.asInstanceOf[Long]
      case x: FloatStorage   => x.raw(index) = value.asInstanceOf[Float]
      case x: DoubleStorage  => x.raw(index) = value.asInstanceOf[Double]

  def fill[A](storage: Storage[A], value: A): Unit =
    var i = 0
    while i < storage.length do
      set(storage, i, value)
      i += 1

  def copy[A](
      source: Storage[A],
      sourceOffset: Int,
      target: Storage[A],
      targetOffset: Int,
      length: Int
  ): Unit =
    (source, target) match
      case (x: BooleanStorage, y: BooleanStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: ByteStorage, y: ByteStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: ShortStorage, y: ShortStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: IntStorage, y: IntStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: FloatStorage, y: FloatStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: DoubleStorage, y: DoubleStorage) =>
        y.raw.set(x.raw.subarray(sourceOffset, sourceOffset + length), targetOffset)
      case (x: LongStorage, y: LongStorage) =>
        var i = 0
        while i < length do
          y.raw(targetOffset + i) = x.raw(sourceOffset + i)
          i += 1
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

  private def addInt(x: Int32Array, y: Int32Array, out: Int32Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addLong(x: Array[Long], y: Array[Long], out: Array[Long], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addFloat(x: Float32Array, y: Float32Array, out: Float32Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = (x(i) + y(i)).toFloat
      i += 1

  private def addDouble(x: Float64Array, y: Float64Array, out: Float64Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = x(i) + y(i)
      i += 1

  private def addIntLinear(
      x: Int32Array,
      xo: Int,
      y: Int32Array,
      yo: Int,
      out: Int32Array,
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
      x: Float32Array,
      xo: Int,
      y: Float32Array,
      yo: Int,
      out: Float32Array,
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = (x(xo + i) + y(yo + i)).toFloat
      i += 1

  private def addDoubleLinear(
      x: Float64Array,
      xo: Int,
      y: Float64Array,
      yo: Int,
      out: Float64Array,
      size: Int
  ): Unit =
    var i = 0
    while i < size do
      out(i) = x(xo + i) + y(yo + i)
      i += 1

  private def negateInt(x: Int32Array, out: Int32Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def negateLong(x: Array[Long], out: Array[Long], size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def negateFloat(x: Float32Array, out: Float32Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = (-x(i)).toFloat
      i += 1

  private def negateDouble(x: Float64Array, out: Float64Array, size: Int): Unit =
    var i = 0
    while i < size do
      out(i) = -x(i)
      i += 1

  private def addIntStrided(x: Int32Array, xo: Int, xs: Int, y: Int32Array, yo: Int, ys: Int, out: Int32Array, size: Int): Unit =
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

  private def addFloatStrided(x: Float32Array, xo: Int, xs: Int, y: Float32Array, yo: Int, ys: Int, out: Float32Array, size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = (x(xi) + y(yi)).toFloat
      xi += xs
      yi += ys
      i += 1

  private def addDoubleStrided(x: Float64Array, xo: Int, xs: Int, y: Float64Array, yo: Int, ys: Int, out: Float64Array, size: Int): Unit =
    var i = 0
    var xi = xo
    var yi = yo
    while i < size do
      out(i) = x(xi) + y(yi)
      xi += xs
      yi += ys
      i += 1
