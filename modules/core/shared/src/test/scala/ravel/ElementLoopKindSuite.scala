package ravel

import munit.FunSuite
import ravel.DType.given
import ravel.internal.*

final class ElementLoopKindSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("linear and scalar-broadcast loops preserve elementwise semantics") {
    val base = NDArray.tabulate[Double](3, 4)((i, j) => i * 10.0 + j + 1.0)
    val other = NDArray.fill(Shape(3, 4), 2.0)

    assertEquals(LoopPlan.broadcast(base.layout, other.layout).kind, LoopKind.LinearContiguous)
    assertEquals(values(base + other), values(base).map(_ + 2.0))
    assertEquals(values(base * other), values(base).map(_ * 2.0))

    val scalar = NDArray.scalar(2.0)
    assertEquals(LoopPlan.broadcast(scalar.layout, base.layout).kind, LoopKind.ScalarBroadcast)
    assertEquals(values(scalar + base), values(base).map(_ + 2.0))
    assertEquals(values(scalar < base), values(base).map(_ > 2.0))
  }

  test("rank-two inner-strided loops agree with a logical oracle") {
    val base = NDArray.tabulate[Double](3, 4)((i, j) => i * 10.0 + j - 8.0)
    val left = base.reverse(1)
    val right = base.reverse(0)
    assertEquals(LoopPlan.broadcast(left.layout, right.layout).kind, LoopKind.InnerStrided)

    val expectedLeft = values(left)
    val expectedRight = values(right)
    assertEquals(values(left + right), expectedLeft.zip(expectedRight).map(_ + _))
    assertEquals(values(left.abs), expectedLeft.map(math.abs))
    assertEquals(values(left <= right), expectedLeft.zip(expectedRight).map(_ <= _))
    assertEquals(values(left.isFinite), List.fill(left.size)(true))
  }

  test("general-rank fallback preserves binary, unary, comparison, and predicates") {
    val source =
      NDArray
        .tabulate[Double](2, 3, 4)((i, j, k) => i * 100.0 + j * 10.0 + k + 1.0)
        .permuteAxes(2, 1, 0)
        .reverse(1)
    val other = source.contiguous
    assertEquals(LoopPlan.broadcast(source.layout, other.layout).kind, LoopKind.GeneralStrided)

    val expected = values(source)
    assertEquals(values(source + other), expected.map(_ * 2.0))
    assertEquals(values(-source), expected.map(-_))
    assertEquals(values(source === other), List.fill(source.size)(true))
    assertEquals(values(source.isNaN), List.fill(source.size)(false))
  }
