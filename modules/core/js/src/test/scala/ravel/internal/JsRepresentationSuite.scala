package ravel.internal

import munit.FunSuite
import ravel.UInt16
import ravel.UInt8
import ravel.DType.given
import scala.scalajs.js.typedarray.*

final class JsRepresentationSuite extends FunSuite:
  test("Boolean storage is Uint8Array and uses exactly 0/1") {
    val storage = ProbeApi.allocate[Boolean](2).asInstanceOf[BooleanStorage]
    ProbeApi.set(storage, 0, false)
    ProbeApi.set(storage, 1, true)
    assert(storage.raw.isInstanceOf[Uint8Array])
    assertEquals(storage.raw(0), 0.toShort)
    assertEquals(storage.raw(1), 1.toShort)
  }

  test("fast storage cases wrap the promised typed arrays") {
    assert(ProbeApi.allocate[Byte](1).asInstanceOf[ByteStorage].raw.isInstanceOf[Int8Array])
    assert(ProbeApi.allocate[UInt8](1).asInstanceOf[UInt8Storage].raw.isInstanceOf[Uint8Array])
    assert(ProbeApi.allocate[Short](1).asInstanceOf[ShortStorage].raw.isInstanceOf[Int16Array])
    assert(ProbeApi.allocate[UInt16](1).asInstanceOf[UInt16Storage].raw.isInstanceOf[Uint16Array])
    assert(ProbeApi.allocate[Int](1).asInstanceOf[IntStorage].raw.isInstanceOf[Int32Array])
    assert(ProbeApi.allocate[Float](1).asInstanceOf[FloatStorage].raw.isInstanceOf[Float32Array])
    assert(ProbeApi.allocate[Double](1).asInstanceOf[DoubleStorage].raw.isInstanceOf[Float64Array])
  }

  test("unsigned storage preserves high-bit values through typed-array raw access") {
    val u8 = ProbeApi.allocate[UInt8](2).asInstanceOf[UInt8Storage]
    ProbeApi.set(u8, 0, UInt8.unsafe(255))
    ProbeApi.set(u8, 1, UInt8.unsafe(128))
    assertEquals(u8.raw(0), 255.toShort)
    assertEquals(u8.raw(1), 128.toShort)
    assertEquals(ProbeApi.get(u8, 0), UInt8.unsafe(255))

    val u16 = ProbeApi.allocate[UInt16](2).asInstanceOf[UInt16Storage]
    ProbeApi.set(u16, 0, UInt16.unsafe(65535))
    ProbeApi.set(u16, 1, UInt16.unsafe(40000))
    assertEquals(u16.raw(0), 65535)
    assertEquals(u16.raw(1), 40000)
    assertEquals(ProbeApi.get(u16, 0), UInt16.unsafe(65535))
  }

  test("Long uses the documented Scala Array fallback") {
    val storage = ProbeApi.allocate[Long](2).asInstanceOf[LongStorage]
    assert(storage.raw.isInstanceOf[Array[Long]])
    ProbeApi.set(storage, 0, Long.MaxValue)
    assertEquals(ProbeApi.get(storage, 0), Long.MaxValue)
  }
