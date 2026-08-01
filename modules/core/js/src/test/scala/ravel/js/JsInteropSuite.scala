package ravel.js

import munit.FunSuite
import ravel.*
import scala.scalajs.js.typedarray.*

final class JsInteropSuite extends FunSuite:
  test("unsafeBorrow exposes typed-array mutation through borrowed views") {
    val values = new Float64Array(6)
    var i = 0
    while i < values.length do
      values(i) = i.toDouble
      i += 1
    val borrowed = JsInterop.unsafeBorrow(values, Shape(2, 3))
    val reversed = borrowed.reverse(1)
    values(2) = 99.0
    assertEquals(borrowed(0, 2), 99.0)
    assertEquals(reversed(0, 0), 99.0)
  }

  test("descriptor retains raw strided layout and copies metadata") {
    val values = new Float64Array(6)
    val borrowed = JsInterop.unsafeBorrow(values, Shape(2, 3))
    val view = borrowed.reverse(1).transpose
    val descriptor = JsInterop.describeDouble(view)
    assert(descriptor.buffer eq values)
    assertEquals(descriptor.offset, 2)
    assertEquals((0 until descriptor.shape.length).map(descriptor.shape(_)), Seq(3, 2))
    assertEquals((0 until descriptor.strides.length).map(descriptor.strides(_)), Seq(-1, 3))
    descriptor.shape(0) = 100
    descriptor.strides(0) = 100
    assertEquals(view.shape(0), 3)
    assertEquals(view(0, 0), 0.0)
  }

  test("copyToFloat64Array materializes logical order and isolates storage") {
    val values = new Float64Array(6)
    var i = 0
    while i < values.length do
      values(i) = i.toDouble
      i += 1
    val borrowed = JsInterop.unsafeBorrow(values, Shape(2, 3))
    val copied = JsInterop.copyToFloat64Array(borrowed.transpose)
    assertEquals((0 until copied.length).map(copied(_)), Seq(0.0, 3.0, 1.0, 4.0, 2.0, 5.0))
    copied(0) = 999.0
    assertEquals(borrowed(0, 0), 0.0)
  }

  test("Boolean borrowing validates the 0/1 encoding") {
    val invalid = new Uint8Array(2)
    invalid(1) = 2
    intercept[IllegalArgumentException] {
      JsInterop.unsafeBorrow(invalid, Shape(2))
    }
  }

  test("Long has no native zero-copy JavaScript descriptor") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.js.*
      import ravel.DType.given
      val values = NDArray.zeros[Long](2)
      JsInterop.describeLong(values)
    """)
    assert(errors.nonEmpty)
  }

  test("unsafeBorrowUInt8 exposes Uint8Array high bits without Boolean 0/1 validation") {
    val values = new Uint8Array(4)
    values(0) = 0
    values(1) = 255.toShort
    values(2) = 128.toShort
    values(3) = 1
    val borrowed = JsInterop.unsafeBorrowUInt8(values, Shape(2, 2))
    assertEquals(borrowed(0, 1).toInt, 255)
    assertEquals(borrowed(1, 0).toInt, 128)
    values(1) = 10
    assertEquals(borrowed(0, 1).toInt, 10)
    val descriptor = JsInterop.describeUInt8(borrowed)
    assert(descriptor.buffer eq values)
    val copied = JsInterop.copyToUInt8Array(borrowed.transpose)
    assertEquals((0 until copied.length).map(i => copied(i).toInt), Seq(0, 128, 10, 1))
  }

  test("unsafeBorrowUInt16 exposes Uint16Array magnitudes") {
    val values = new Uint16Array(3)
    values(0) = 0
    values(1) = 65535
    values(2) = 40000
    val borrowed = JsInterop.unsafeBorrowUInt16(values, Shape(3))
    assertEquals(borrowed(1).toInt, 65535)
    assertEquals(borrowed(2).toInt, 40000)
    val copied = JsInterop.copyToUInt16Array(borrowed)
    assertEquals(copied(1), 65535)
  }
