package ravel.internal

import munit.FunSuite
import ravel.DType.given
import scala.scalajs.js.typedarray.*

final class JsRepresentationSuite extends FunSuite:
  test("Boolean storage is Uint8Array and uses exactly 0/1") {
    val storage = ProbeApi.allocate[Boolean](2)
    storage match
      case x: BooleanStorage =>
        ProbeApi.set(storage, 0, false)
        ProbeApi.set(storage, 1, true)
        assert(x.raw.isInstanceOf[Uint8Array])
        assertEquals(x.raw(0), 0.toShort)
        assertEquals(x.raw(1), 1.toShort)
  }

  test("fast storage cases wrap the promised typed arrays") {
    assert(ProbeApi.allocate[Byte](1).asInstanceOf[ByteStorage].raw.isInstanceOf[Int8Array])
    assert(ProbeApi.allocate[Short](1).asInstanceOf[ShortStorage].raw.isInstanceOf[Int16Array])
    assert(ProbeApi.allocate[Int](1).asInstanceOf[IntStorage].raw.isInstanceOf[Int32Array])
    assert(ProbeApi.allocate[Float](1).asInstanceOf[FloatStorage].raw.isInstanceOf[Float32Array])
    assert(ProbeApi.allocate[Double](1).asInstanceOf[DoubleStorage].raw.isInstanceOf[Float64Array])
  }

  test("Long uses the documented Scala Array fallback") {
    val storage = ProbeApi.allocate[Long](2).asInstanceOf[LongStorage]
    assert(storage.raw.isInstanceOf[Array[Long]])
    ProbeApi.set(storage, 0, Long.MaxValue)
    assertEquals(ProbeApi.get(storage, 0), Long.MaxValue)
  }
