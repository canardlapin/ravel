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
  test("extension overload prototype fails, while evidence-gated members compile") {
    val extensionErrors = compileErrors("""
      sealed trait ProbeRank
      sealed trait ProbeRank1 extends ProbeRank
      sealed trait ProbeRank2 extends ProbeRank
      final class ProbeArray[R <: ProbeRank]
      import scala.annotation.targetName

      extension [R <: ProbeRank](array: ProbeArray[R])
        @targetName("probeApply1")
        def apply(i: Int)(using R =:= ProbeRank1): Int = i
        @targetName("probeApply2")
        def apply(i: Int, j: Int)(using R =:= ProbeRank2): Int = i + j
        def transpose(using R =:= ProbeRank2): ProbeArray[ProbeRank2] =
          array.asInstanceOf[ProbeArray[ProbeRank2]]
    """)
    assert(extensionErrors.contains("apply is already defined"), clues(extensionErrors))

    val valid = compileErrors("""
      sealed trait ProbeRank
      sealed trait ProbeRank1 extends ProbeRank
      sealed trait ProbeRank2 extends ProbeRank
      final class ProbeArray[R <: ProbeRank]:
        def apply(i: Int)(using R <:< ProbeRank1): Int = i
        def apply(i: Int, j: Int)(using R <:< ProbeRank2): Int = i + j
        def transpose(using R <:< ProbeRank2): ProbeArray[ProbeRank2] =
          this.asInstanceOf[ProbeArray[ProbeRank2]]
      final class ProbeMutable[R <: ProbeRank]:
        def update(i: Int, value: Int)(using R <:< ProbeRank1): Unit = ()
        def update(i: Int, j: Int, value: Int)(using R <:< ProbeRank2): Unit = ()

      val one = new ProbeArray[ProbeRank1]
      val two = new ProbeArray[ProbeRank2]
      val mutableOne = new ProbeMutable[ProbeRank1]
      val mutableTwo = new ProbeMutable[ProbeRank2]
      val readOne = one(0)
      val readTwo = two(0, 1)
      val transposed = two.transpose
      mutableOne(0) = 1
      mutableTwo(0, 1) = 2
    """)
    assertEquals(valid, "")

    val wrongArity = compileErrors("""
      sealed trait ProbeRank
      sealed trait ProbeRank1 extends ProbeRank
      sealed trait ProbeRank2 extends ProbeRank
      final class ProbeArray[R <: ProbeRank]:
        def apply(i: Int)(using R =:= ProbeRank1): Int = i
        def apply(i: Int, j: Int)(using R =:= ProbeRank2): Int = i + j
        def transpose(using R =:= ProbeRank2): ProbeArray[ProbeRank2] =
          this.asInstanceOf[ProbeArray[ProbeRank2]]
      val one = new ProbeArray[ProbeRank1]
      one(0, 1)
      one.transpose
    """)
    assert(wrongArity.nonEmpty)
  }

  test("scalar vs ReadableArray + selects scalar and rejects NDArray arg") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given
      import scala.annotation.targetName

      extension [A, R <: AnyRank](array: ReadableArray[A, R])(using ArithmeticDType[A])
        def +(scalar: A): NDArray[A, R] = null.asInstanceOf[NDArray[A, R]]
        @targetName("addArray")
        def +[S <: AnyRank](other: ReadableArray[A, S]): NDArray[A, BroadcastRank[R, S]] =
          null.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

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
        def +(scalar: A): NDArray[A, R] = null.asInstanceOf[NDArray[A, R]]
        @targetName("addNDArray")
        def +[S <: AnyRank](other: NDArray[A, S]): NDArray[A, BroadcastRank[R, S]] =
          null.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

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
          null.asInstanceOf[NDArray[A, BroadcastRank[R, S]]]

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
