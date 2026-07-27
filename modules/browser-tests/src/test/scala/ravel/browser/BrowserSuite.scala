package ravel.browser

import munit.FunSuite
import ravel.*
import ravel.DType.given
import ravel.js.JsInterop
import scala.scalajs.js
import scala.scalajs.js.typedarray.Float64Array

final class BrowserSuite extends FunSuite:
  test("suite is executing in a browser") {
    assert(js.typeOf(js.Dynamic.global.window) != "undefined")
    assert(js.typeOf(js.Dynamic.global.document) != "undefined")
  }

  test("browser typed-array borrowing preserves aliasing and view layout") {
    val raw = new Float64Array(6)
    var index = 0
    while index < raw.length do
      raw(index) = index.toDouble
      index += 1
    val borrowed = JsInterop.unsafeBorrow(raw, Shape(2, 3))
    val view = borrowed.reverse(1).transpose
    raw(2) = 99.0
    assertEquals(view(0, 0), 99.0)
    val descriptor = JsInterop.describeDouble(view)
    assert(descriptor.buffer eq raw)
    assertEquals(descriptor.offset, 2)
  }

  test("browser execution agrees for broadcast, views, and reductions") {
    val array = NDArray.tabulate[Double](2, 3)((row, column) => row * 10.0 + column)
    val bias = NDArray.fromSeq(Shape(3), Seq(1.0, 2.0, 3.0))
    val result = (array.reverse(1) + bias).sum(0)
    assert(result.sameElements(NDArray.fromSeq(Shape(3), Seq(16.0, 16.0, 16.0))))
  }
