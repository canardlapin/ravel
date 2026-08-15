package ravel

import munit.FunSuite
import ravel.DType.given
import ravel.internal.ProbeApi
import scala.compiletime.testing.typeCheckErrors

final class BooleanReductionSuite extends FunSuite:
  test("whole-array Boolean reductions use explicit empty identities"):
    val empty = NDArray.zeros[Boolean](0)
    assert(empty.all)
    assert(!empty.any)
    assertEquals(empty.countTrue, 0)

    val values = NDArray.fromSeq(Shape(5), Seq(true, false, true, true, false))
    assert(!values.all)
    assert(values.any)
    assertEquals(values.countTrue, 3)
    assert(NDArray.scalar(true).all)
    assert(!NDArray.scalar(false).any)

  test("single-axis all any and countTrue share keep-dimensions semantics"):
    val source = NDArray.fromSeq(
      Shape(2, 3),
      Seq(true, true, false, true, true, true)
    )
    assertEquals(values(source.all(0)), Vector(true, true, false))
    assertEquals(values(source.all(1)), Vector(false, true))
    assertEquals(values(source.any(-2)), Vector(true, true, true))
    assertEquals(values(source.any(-1)), Vector(true, true))
    assertEquals(values(source.countTrue(0)), Vector(2, 2, 1))
    assertEquals(values(source.countTrue(1)), Vector(2, 3))
    assertEquals(source.allKeep(1).shape, Shape(2, 1))
    assertEquals(source.anyKeepDims(0).shape, Shape(1, 3))
    assertEquals(source.countTrueKeep(-1).shape, Shape(2, 1))

  test("validated multi-axis Boolean reductions plan once and own their outputs"):
    val source = NDArray.tabulate[Boolean](2, 2, 2) { (i, j, k) =>
      i == 0 || (j == 1 && k == 1)
    }
    val axes = Axes.from(source.rank, 0, -1).toOption.get
    assertEquals(values(source.all(axes)), Vector(false, false))
    assertEquals(values(source.any(axes)), Vector(true, true))
    assertEquals(values(source.countTrue(axes)), Vector(2, 3))
    assertEquals(source.allKeep(axes).shape, Shape(1, 2, 1))
    assertEquals(source.anyKeepDims(axes).shape, Shape(1, 2, 1))
    assertEquals(source.countTrueKeep(axes).shape, Shape(1, 2, 1))

    val noAxes = Axes.from(source.rank).toOption.get
    val allIdentity = source.all(noAxes)
    val anyIdentity = source.any(noAxes)
    val counts = source.countTrue(noAxes)
    assertEquals(values(allIdentity), values(source))
    assertEquals(values(anyIdentity), values(source))
    assertEquals(values(counts), values(source).map(if _ then 1 else 0))
    assert(!(allIdentity.storage eq source.storage))
    assert(!(anyIdentity.storage eq source.storage))

  test("empty fibers and zero-sized outputs retain Boolean identities"):
    val emptyFibers = NDArray.zeros[Boolean](2, 0, 3)
    assertEquals(values(emptyFibers.all(1)), Vector.fill(6)(true))
    assertEquals(values(emptyFibers.any(1)), Vector.fill(6)(false))
    assertEquals(values(emptyFibers.countTrue(1)), Vector.fill(6)(0))
    assertEquals(emptyFibers.all(0).shape, Shape(0, 3))
    assertEquals(emptyFibers.any(0).size, 0)
    assertEquals(emptyFibers.countTrue(0).size, 0)

  test("broadcast sliced reversed and permuted views reduce logical values"):
    val source = NDArray
      .fromSeq(Shape(1, 4), Seq(true, false, true, false))
      .broadcastTo(Shape(3, 4))
      .reverse(0)
    assertEquals(source.countTrue, 6)
    assert(source.any)
    assert(!source.all)
    assertEquals(values(source.countTrue(0)), Vector(3, 0, 3, 0))

    val view = source.permuteAxes(1, 0).slice(0, Slice.reverse)
    assertEquals(view.countTrue, 6)
    assertEquals(values(view.all(1)), Vector(false, true, false, true))

  test("owned borrowed and mutable readable sources share Boolean semantics"):
    val owned = NDArray.fromSeq(Shape(2, 2), Seq(true, true, true, true))
    val borrowed = new BorrowedNDArray(owned)
    val mutable = owned.mutableCopy
    val axes = Axes.all(2).toOption.get

    val borrowedAll = borrowed.all(axes)
    val borrowedCount = borrowed.countTrue(axes)
    assertEquals(values(borrowedAll), Vector(true))
    assertEquals(values(borrowedCount), Vector(4))
    assertEquals(values(mutable.any(axes)), Vector(true))
    assertEquals(values(mutable.countTrue(axes)), Vector(4))

    ProbeApi.setBoolean(owned.storage, 0, false)
    assert(!borrowed.all)
    assertEquals(values(borrowedAll), Vector(true))
    assertEquals(values(borrowedCount), Vector(4))

  test("rank zero through six reduce through the same validated Axes contract"):
    var rank = 0
    while rank <= 6 do
      val dimensions = Vector.fill(rank)(2)
      val dynamicShape = Shape.from(dimensions).toOption.get
      val logicalSize = dynamicShape.size
      val source = NDArray.fromSeq(
        dynamicShape,
        Vector.tabulate(logicalSize)(index => index != logicalSize - 1)
      )
      val axes = Axes.all(rank).toOption.get
      assert(!source.all)
      assertEquals(source.any, logicalSize > 1)
      assertEquals(source.countTrue, logicalSize - 1)
      assertEquals(source.all(axes).rank, 0)
      assertEquals(values(source.any(axes)), Vector(logicalSize > 1))
      assertEquals(values(source.countTrue(axes)), Vector(logicalSize - 1))
      rank += 1

  test("invalid axes and unsupported dtypes fail before execution or compilation"):
    val source = NDArray.fromSeq(Shape(2), Seq(true, false))
    val wrongRank = Axes.all(2).toOption.get
    intercept[InvalidAxesException](source.all(wrongRank))
    intercept[InvalidAxesException](source.anyAxes(0, -1))

    val numericErrors = typeCheckErrors("""
      import ravel.DType.given
      val numeric = NDArray.zeros[Int](2)
      numeric.all
      numeric.any
      numeric.countTrue
    """)
    assert(numericErrors.nonEmpty)

    val rankZeroErrors = typeCheckErrors("""
      import ravel.DType.given
      val scalar = NDArray.scalar(true)
      scalar.all(0)
    """)
    assert(rankZeroErrors.nonEmpty)

  private def values[A](array: ReadableArray[A, ?]): Vector[A] =
    array match
      case owned: NDArray[A @unchecked, ?] => owned.elementsIterator.toVector
      case borrowed: BorrowedNDArray[A @unchecked, ?] => borrowed.elementsIterator.toVector
      case mutable: MutableNDArray[A @unchecked, ?] =>
        mutable.freezeCopy().elementsIterator.toVector
