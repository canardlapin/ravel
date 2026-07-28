package ravel
package overloadrepro

/** Documents a Scala 3.7.4 extension-overload limitation that forced NumericOps to keep a
  * union/`OperandRank` encoding instead of scalar vs array overloads.
  *
  * Symptom: for `left + right` with both sides `NDArray[Int, _]`, the compiler selects
  * `+(scalar: A)` and reports `Required: Int` for the array argument. `@targetName` and concreting
  * the array parameter to `NDArray` do not help. A lone array overload, or the production union
  * encoding, works.
  */
final class ExtensionOverloadReproSuite extends munit.FunSuite:
  test("scalar vs ReadableArray + selects scalar and rejects NDArray arg") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given
      import scala.annotation.targetName

      extension [A, R <: AnyRank](array: ReadableArray[A, R])(using ArithmeticDType[A])
        def +(scalar: A): NDArray[A, R] = array.toNDArray
        @targetName("addArray")
        def +[S <: AnyRank](other: ReadableArray[A, S]): NDArray[A, BroadcastRank[R, S]] =
          array.toNDArray.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

      val left = NDArray.zeros[Int](2, 1, 3)
      val right = NDArray.zeros[Int](3)
      val result: Array3[Int] = left + right
    """)
    assert(errors.contains("Required: Int"), clues(errors))
  }

  test("scalar vs concrete NDArray + has the same failure") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given
      import scala.annotation.targetName

      extension [A, R <: AnyRank](array: ReadableArray[A, R])(using ArithmeticDType[A])
        def +(scalar: A): NDArray[A, R] = array.toNDArray
        @targetName("addNDArray")
        def +[S <: AnyRank](other: NDArray[A, S]): NDArray[A, BroadcastRank[R, S]] =
          array.toNDArray.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

      val left = NDArray.zeros[Int](2, 1, 3)
      val right = NDArray.zeros[Int](3)
      val result: Array3[Int] = left + right
    """)
    assert(errors.contains("Required: Int"), clues(errors))
  }

  test("array-only extension overload compiles") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given

      extension [A, R <: AnyRank](array: ReadableArray[A, R])(using ArithmeticDType[A])
        def +[S <: AnyRank](other: ReadableArray[A, S]): NDArray[A, BroadcastRank[R, S]] =
          array.toNDArray.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

      val left = NDArray.zeros[Int](2, 1, 3)
      val right = NDArray.zeros[Int](3)
      val result: Array3[Int] = left + right
    """)
    assertEquals(errors, "")
  }

  test("production union OperandRank form compiles for array and scalar") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given

      val left = NDArray.zeros[Int](2, 1, 3)
      val right = NDArray.zeros[Int](3)
      val result: Array3[Int] = left + right
      val scaled: Array3[Int] = left + 1
    """)
    assertEquals(errors, "")
  }
