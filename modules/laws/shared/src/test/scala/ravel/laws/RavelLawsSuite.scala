package ravel.laws

import munit.FunSuite
import org.scalacheck.Test
import ravel.*
import ravel.DType.given

final class RavelLawsSuite extends FunSuite:
  test("reusable core law bundles execute") {
    val array = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    ShapeLaws.valid(array.shape)
    LayoutLaws.materialization(array.transpose)
    ViewLaws.reverseInvolution(array, 1)
    ViewLaws.swapInvolution(array, 0, 1)
    ViewLaws.pureMapCommutesWithSlice(
      array,
      1,
      Slice(0, 3, 2),
      (value: Int) => value + 1
    )
    BroadcastLaws.commutativeShape(array, NDArray.zeros[Int](3))
    KernelLaws.additionMatchesCallback(array, NDArray.zeros[Int](3), _ + _)
    DTypeLaws.named(summon[DType[Double]], "Double")
    CastLaws.exactRoundTrip[Int, Double](array)

    val mutable = array.mutableCopy
    MutableLaws.freezeIsolation(mutable, value => value(0, 0) = 99)

    val copied = 1
    InteropLaws.copyIsolation(
      readCopied = () => copied,
      mutateOriginal = () => (),
      expected = 1
    )

    var external = 1
    BorrowedInteropLaws.externalMutationObserved(
      borrowedRead = () => external,
      mutateExternal = () => external = 2,
      expectedAfter = 2
    )
  }

  test("Discipline ruleset is reusable") {
    val rules = RavelDiscipline.core
    assertEquals(rules.name, "ravel-core")
    rules.all.properties.foreach { (name, property) =>
      val result =
        Test.check(
          Test.Parameters.default
            .withMinSuccessfulTests(100)
            .withWorkers(1),
          property
        )
      assert(result.passed, s"$name failed: $result")
    }
  }
