package ravel.internal

import munit.FunSuite
import ravel.DType.given

final class RepresentationSuite extends FunSuite:
  test("allocate, set, get, fill, and copy every storage dtype") {
    checkStorage[Boolean](true, false)
    checkStorage[Byte](12.toByte, (-2).toByte)
    checkStorage[Short](1234.toShort, (-3).toShort)
    checkStorage[Int](123456, -4)
    checkStorage[Long](1234567890123L, -5L)
    checkStorage[Float](1.25f, -6.5f)
    checkStorage[Double](1.25, -7.5)
  }

  test("linear unary, binary, and strided kernels agree with scalar access") {
    val x = ProbeApi.allocate[Double](6)
    val y = ProbeApi.allocate[Double](6)
    val out = ProbeApi.allocate[Double](3)
    var i = 0
    while i < 6 do
      ProbeApi.set(x, i, i.toDouble)
      ProbeApi.set(y, i, i.toDouble * 10.0)
      i += 1

    ProbeApi.addStrided(x, 0, 2, y, 1, 2, out, 3)
    assertEquals(
      List.tabulate(3)(i => ProbeApi.get(out, i)),
      List(10.0, 32.0, 54.0)
    )

    val negated = ProbeApi.allocate[Double](3)
    ProbeApi.negate(out, negated, 3)
    assertEquals(
      List.tabulate(3)(i => ProbeApi.get(negated, i)),
      List(-10.0, -32.0, -54.0)
    )
  }

  test("Byte and Short are not admitted to arithmetic kernels") {
    val bytes = ProbeApi.allocate[Byte](1)
    val byteOut = ProbeApi.allocate[Byte](1)
    intercept[UnsupportedOperationException] {
      ProbeApi.add(bytes, bytes, byteOut, 1)
    }

    val shorts = ProbeApi.allocate[Short](1)
    val shortOut = ProbeApi.allocate[Short](1)
    intercept[UnsupportedOperationException] {
      ProbeApi.negate(shorts, shortOut, 1)
    }
  }

  private def checkStorage[A: ravel.DType](first: A, second: A): Unit =
    val source = ProbeApi.allocate[A](4)
    val target = ProbeApi.allocate[A](4)
    ProbeApi.fill(source, first)
    ProbeApi.set(source, 2, second)
    ProbeApi.copy(source, target, 4)
    assertEquals(ProbeApi.get(target, 0), first)
    assertEquals(ProbeApi.get(target, 2), second)
