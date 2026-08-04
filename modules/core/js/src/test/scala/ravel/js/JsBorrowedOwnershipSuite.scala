package ravel.js

import ravel.*
import scala.scalajs.js.typedarray.Float64Array

final class JsBorrowedOwnershipSuite extends BorrowedOwnershipContract:
  protected def borrow(values: Seq[Double]): BorrowedDoubleFixture =
    val external = new Float64Array(values.length)
    var index = 0
    while index < values.length do
      external(index) = values(index)
      index += 1
    new BorrowedDoubleFixture(
      JsInterop.unsafeBorrow(external, Shape(2, 3)),
      (index, value) => external(index) = value
    )
