package ravel

import munit.FunSuite
import ravel.DType.given
import ravel.internal.*

final class ShapeLayoutSuite extends FunSuite:
  test("shape factories preserve rank and scalar product semantics") {
    val scalar = Shape.scalar
    val matrix = Shape(3, 4)
    assertEquals(scalar.rank, 0)
    assertEquals(scalar.size, 1)
    assertEquals(matrix.rank, 2)
    assertEquals(matrix.size, 12)
    val typed: Shape[Rank[2]] = matrix
    assertEquals(typed(0), 3)
    assertEquals(typed(-1), 4)
  }

  test("shape equality is structural by dimensions") {
    assertEquals(Shape(2, 3), Shape(2, 3))
    assertEquals(Shape(2, 3).hashCode, Shape(2, 3).hashCode)
    assert(!(Shape(2, 3) == Shape(3, 2)))
    assertEquals(Shape.scalar, Shape.scalar)
    assert(Shape.from(Seq(2, 3)).exists(_ == Shape(2, 3)))
  }

  test("newAxis keeps logical contiguity while losing canonical strides") {
    val matrix = NDArray.zeros[Int](2, 3)
    assert(matrix.isContiguous)
    assert(matrix.isCanonicalLayout)
    assert(matrix.isWholeBuffer)
    val inserted = matrix.newAxis(-1)
    assertEquals(inserted.shape, Shape(2, 3, 1))
    assert(inserted.isContiguous)
    assert(!inserted.isCanonicalLayout)
    assert(inserted.contiguous eq inserted)
  }

  test("negative dimensions and element-count overflow are rejected") {
    assert(Shape.from(Seq(2, -1)).isLeft)
    assert(Shape.from(Seq(Int.MaxValue, 2)).isLeft)
  }

  test("canonical stride construction checks overflow even for empty shapes") {
    val emptyHuge = Shape.from(Seq(0, Int.MaxValue, Int.MaxValue)).toOption.get
    intercept[LayoutOverflow] {
      NDArray.zeros[Int, AnyRank](emptyHuge)
    }
  }

  test("negative-stride offset denotes logical index zero and double reversal restores mapping") {
    val storage = ProbeApi.allocate[Int](4)
    var i = 0
    while i < 4 do
      ProbeApi.set(storage, i, i + 1)
      i += 1
    val reversed = Layout.view(IArray(4), IArray(-1), 3, 4)
    assertEquals(
      List.tabulate(4)(i => ProbeApi.get(storage, reversed.physicalIndex(IArray(i)))),
      List(4, 3, 2, 1)
    )
    val restored = Layout.view(IArray(4), IArray(1), 0, 4)
    assertEquals(
      List.tabulate(4)(i => ProbeApi.get(storage, restored.physicalIndex(IArray(i)))),
      List(1, 2, 3, 4)
    )
  }

  test("broadcast flag ignores singleton zero strides") {
    val singleton = Layout.view(IArray(1), IArray(0), 0, 1)
    val repeated = Layout.view(IArray(2), IArray(0), 0, 1)
    assert(!singleton.hasBroadcastStride)
    assert(repeated.hasBroadcastStride)
  }

  test("physical density recognizes packed permutations and reversals without admitting holes") {
    val reversed = Layout.view(IArray(4), IArray(-1), 3, 4)
    val permutedAndReversed =
      Layout.view(IArray(2, 3, 4), IArray(-4, 8, -1), 7, 24)
    val withSingleton =
      Layout.view(IArray(1, 2, 3), IArray(0, -3, 1), 3, 6)
    val stridedWithHoles =
      Layout.view(IArray(2, 3), IArray(4, 1), 0, 7)
    val overlapping =
      Layout.view(IArray(2, 3), IArray(1, 1), 0, 4)
    val broadcast =
      Layout.view(IArray(2, 3), IArray(0, 1), 0, 3)
    val empty =
      Layout.view(IArray(0, 3), IArray(3, 0), 0, 0)

    assert(reversed.isPhysicallyDense)
    assertEquals(reversed.minimumPhysicalAddress, 0)
    assert(permutedAndReversed.isPhysicallyDense)
    assertEquals(permutedAndReversed.minimumPhysicalAddress, 0)
    assert(withSingleton.isPhysicallyDense)
    assertEquals(withSingleton.minimumPhysicalAddress, 0)
    assert(!stridedWithHoles.isPhysicallyDense)
    assert(!overlapping.isPhysicallyDense)
    assert(!broadcast.isPhysicallyDense)
    assert(!empty.isPhysicallyDense)
  }

  test("reachable-address validation rejects out-of-buffer layouts") {
    intercept[LayoutOverflow] {
      Layout.view(IArray(3), IArray(2), 0, 4)
    }
    intercept[LayoutOverflow] {
      Layout.view(IArray(3), IArray(-1), 1, 3)
    }
  }

  test("narrow planning normalizes negative starts and checks exact bounds") {
    val extentFive = IArray(5)
    assertEquals(NarrowPlan(extentFive, 0, -1, 1), NarrowPlan(0, 4, 5, 1))
    assertEquals(NarrowPlan(extentFive, -1, -5, 5), NarrowPlan(0, 0, 5, 5))
    assertEquals(NarrowPlan(extentFive, 0, 5, 0), NarrowPlan(0, 5, 5, 0))
    intercept[InvalidNarrowException](NarrowPlan(extentFive, 0, 5, 1))
    intercept[InvalidNarrowException](NarrowPlan(extentFive, 0, -6, 1))
    intercept[InvalidNarrowException](NarrowPlan(extentFive, 0, 0, -1))

    val empty = IArray(0)
    assertEquals(NarrowPlan(empty, 0, 0, 0), NarrowPlan(0, 0, 0, 0))
    intercept[InvalidNarrowException](NarrowPlan(empty, 0, -1, 0))

    val maximumExtent = IArray(Int.MaxValue)
    assertEquals(
      NarrowPlan(maximumExtent, 0, Int.MaxValue, 0),
      NarrowPlan(0, Int.MaxValue, Int.MaxValue, 0)
    )
    assertEquals(
      NarrowPlan(maximumExtent, -1, -1, 1),
      NarrowPlan(0, Int.MaxValue - 1, Int.MaxValue, 1)
    )
    intercept[InvalidNarrowException](NarrowPlan(maximumExtent, 0, Int.MinValue, 0))
    intercept[InvalidNarrowException](NarrowPlan(extentFive, Int.MinValue, 0, 0))
    intercept[InvalidNarrowException](NarrowPlan(extentFive, Int.MaxValue, 0, 0))
  }
