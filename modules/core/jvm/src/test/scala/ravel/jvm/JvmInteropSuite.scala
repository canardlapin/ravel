package ravel.jvm

import munit.FunSuite
import ravel.*
import ravel.DType.given

final class JvmInteropSuite extends FunSuite:
  test("unsafeBorrow exposes external mutation through borrowed views") {
    val values = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    val borrowed = JvmInterop.unsafeBorrow(values, Shape(2, 3))
    val reversed = borrowed.reverse(1)
    values(2) = 99.0
    assertEquals(borrowed(0, 2), 99.0)
    assertEquals(reversed(0, 0), 99.0)
  }

  test("borrowed computations own and isolate their results") {
    val values = Array(1.0, 2.0, 3.0, 4.0)
    val borrowed = JvmInterop.unsafeBorrow(values, Shape(2, 2))
    val copied = borrowed.copy
    val mapped = borrowed.map(_ * 2.0)
    val reduced = borrowed.sum(0)
    values(0) = 100.0
    assertEquals(copied(0, 0), 1.0)
    assertEquals(mapped(0, 0), 2.0)
    assertEquals(reduced(0), 4.0)
  }

  test("copyToArray follows logical row-major order and always copies") {
    val owned = NDArray.tabulate[Int](2, 3)((row, column) => row * 10 + column)
    val copied = JvmInterop.copyToArray(owned.transpose)
    assertEquals(copied.toSeq, Seq(0, 10, 1, 11, 2, 12))
    copied(0) = 999
    assertEquals(owned(0, 0), 0)
  }

  test("unsafeBorrow rejects mismatched shape") {
    intercept[ShapeMismatch] {
      JvmInterop.unsafeBorrow(Array(1, 2, 3), Shape(2, 2))
    }
  }

  test("borrowed elementsIterator observes live external storage") {
    val values = Array(1, 2, 3, 4)
    val borrowed = JvmInterop.unsafeBorrow(values, Shape(2, 2))
    val iterator = borrowed.elementsIterator
    assert(iterator.hasNext)
    values(0) = 99
    assertEquals(iterator.next(), 99)
    assertEquals(iterator.next(), 2)
  }

  test("owned and borrowed arithmetic are symmetric on the JVM") {
    val owned = NDArray.fromSeq(Shape(2), Seq(1.0, 2.0))
    val buffer = Array(10.0, 20.0)
    val borrowed = JvmInterop.unsafeBorrow(buffer, Shape(2))
    assertEquals(owned.elementsIterator.toList, List(1.0, 2.0))
    assertEquals((owned + borrowed).elementsIterator.toList, List(11.0, 22.0))
    assertEquals((borrowed + owned).elementsIterator.toList, List(11.0, 22.0))
    assertEquals((borrowed.sqrt).elementsIterator.toList, List(math.sqrt(10.0), math.sqrt(20.0)))
    assertEquals((borrowed > 15.0).elementsIterator.toList, List(false, true))
  }

  test("JVM bit equality distinguishes raw NaN payloads") {
    val nan1 = java.lang.Double.longBitsToDouble(0x7ff8000000000001L)
    val nan2 = java.lang.Double.longBitsToDouble(0x7ff8000000000002L)
    val left = NDArray.fromSeq(Shape(1), Seq(nan1))
    val other = NDArray.fromSeq(Shape(1), Seq(nan2))
    assert(!left.sameElementsBits(other))
  }

  test("unsafeBorrowUInt8 exposes NIfTI-style byte buffers as unsigned values") {
    val values = Array[Byte](0, -1, 127, -128)
    val borrowed = JvmInterop.unsafeBorrowUInt8(values, Shape(2, 2))
    assertEquals(borrowed(0, 0).toInt, 0)
    assertEquals(borrowed(0, 1).toInt, 255)
    assertEquals(borrowed(1, 0).toInt, 127)
    assertEquals(borrowed(1, 1).toInt, 128)
    values(1) = 10
    assertEquals(borrowed(0, 1).toInt, 10)
    val copied = JvmInterop.copyToArray(borrowed)
    assertEquals(copied.toSeq.map(_.toInt), Seq(0, 10, 127, 128))
  }

  test("unsafeBorrowUInt16 exposes short buffers as unsigned magnitudes") {
    val values = Array[Short](0, -1, 40000.toShort)
    val borrowed = JvmInterop.unsafeBorrowUInt16(values, Shape(3))
    assertEquals(borrowed(0).toInt, 0)
    assertEquals(borrowed(1).toInt, 65535)
    assertEquals(borrowed(2).toInt, 40000)
  }
