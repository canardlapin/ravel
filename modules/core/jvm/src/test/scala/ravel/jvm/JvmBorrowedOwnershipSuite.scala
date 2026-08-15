package ravel.jvm

import ravel.*
import ravel.DType.given

final class JvmBorrowedOwnershipSuite extends BorrowedOwnershipContract:
  protected def borrow(values: Seq[Double]): BorrowedDoubleFixture =
    val external = values.toArray
    new BorrowedDoubleFixture(
      JvmInterop.unsafeBorrow(external, Shape(2, 3)),
      (index, value) => external(index) = value
    )
