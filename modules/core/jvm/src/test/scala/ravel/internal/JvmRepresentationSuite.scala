package ravel.internal

import munit.FunSuite
import ravel.DType.given

final class JvmRepresentationSuite extends FunSuite:
  test("storage cases wrap primitive JVM arrays") {
    assert(ProbeApi.allocate[Boolean](1).asInstanceOf[BooleanStorage].raw.getClass == classOf[Array[Boolean]])
    assert(ProbeApi.allocate[Byte](1).asInstanceOf[ByteStorage].raw.getClass == classOf[Array[Byte]])
    assert(ProbeApi.allocate[Short](1).asInstanceOf[ShortStorage].raw.getClass == classOf[Array[Short]])
    assert(ProbeApi.allocate[Int](1).asInstanceOf[IntStorage].raw.getClass == classOf[Array[Int]])
    assert(ProbeApi.allocate[Long](1).asInstanceOf[LongStorage].raw.getClass == classOf[Array[Long]])
    assert(ProbeApi.allocate[Float](1).asInstanceOf[FloatStorage].raw.getClass == classOf[Array[Float]])
    assert(ProbeApi.allocate[Double](1).asInstanceOf[DoubleStorage].raw.getClass == classOf[Array[Double]])
  }
