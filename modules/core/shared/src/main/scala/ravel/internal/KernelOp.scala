package ravel.internal

private[ravel] object KernelOp:
  val Add: Byte = 0
  val Subtract: Byte = 1
  val Multiply: Byte = 2
  val Divide: Byte = 3
  val Minimum: Byte = 4
  val Maximum: Byte = 5

  val Negate: Byte = 10
  val Absolute: Byte = 11
  val Sqrt: Byte = 12
  val Exp: Byte = 13
  val Log: Byte = 14
  val Sin: Byte = 15
  val Cos: Byte = 16
  val Tan: Byte = 17
  val Floor: Byte = 18
  val Ceil: Byte = 19

  val Equal: Byte = 20
  val NotEqual: Byte = 21
  val Less: Byte = 22
  val LessEqual: Byte = 23
  val IsNaN: Byte = 24
  val IsFinite: Byte = 25
  // Ordering.compare semantics, including total ordering for floating NaNs
  // and signed zero. The ordinary Less/LessEqual operations retain IEEE
  // primitive comparison semantics.
  val OrderedLess: Byte = 26
  val OrderedLessEqual: Byte = 27
