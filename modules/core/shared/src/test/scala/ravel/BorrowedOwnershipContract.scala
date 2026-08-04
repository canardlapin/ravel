package ravel

import munit.FunSuite
import ravel.DType.given

private[ravel] final class BorrowedDoubleFixture(
    val array: BorrowedNDArray[Double, Rank[2]],
    val updateExternal: (Int, Double) => Unit
)

/** Cross-platform ownership contract. Platform suites provide only the external-buffer adapter. */
abstract class BorrowedOwnershipContract extends FunSuite:
  protected def borrow(values: Seq[Double]): BorrowedDoubleFixture

  private val initialValues = Seq(0.5, 1.5, 2.5, 3.5, 4.5, 5.5)

  private def fixture(): BorrowedDoubleFixture =
    borrow(initialValues)

  private def assertBorrowedView(
      name: String
  )(
      build: BorrowedNDArray[Double, Rank[2]] => BorrowedNDArray[Double, ?]
  ): Unit =
    val external = fixture()
    val view = build(external.array)
    val before = view.elementsIterator.toList
    external.updateExternal(0, 99.0)
    val after = view.elementsIterator.toList
    assert(before != after, clues(name, before, after))
    assert(after.contains(99.0), clues(name, after))

  private def assertOwnedResult[A](
      name: String
  )(
      build: BorrowedNDArray[Double, Rank[2]] => NDArray[A, ?]
  ): Unit =
    val external = fixture()
    val owned = build(external.array)
    val before = owned.elementsIterator.toList
    external.updateExternal(0, 99.0)
    val after = owned.elementsIterator.toList
    assertEquals(after, before, clues(name))

  test("every structural view preserves borrowed provenance and observes external mutation") {
    assertBorrowedView("select")(_.select(0, 0))
    assertBorrowedView("slice")(_.slice(1, Slice(0, 3, 2)))
    assertBorrowedView("range slice")(_.slice(1, 0 until 3 by 2))
    assertBorrowedView("narrow")(_.narrow(-1, -3, 2))
    assertBorrowedView("reverse")(_.reverse(1))
    assertBorrowedView("swapAxes")(_.swapAxes(0, 1))
    assertBorrowedView("permuteAxes")(_.permuteAxes(1, 0))
    assertBorrowedView("transpose")(_.transpose)
    assertBorrowedView("newAxis")(_.newAxis(0))
    assertBorrowedView("squeeze")(_.newAxis(0).squeeze(0))
    assertBorrowedView("broadcastTo")(
      _.select(0, 0).newAxis(0).broadcastTo(Shape(2, 3))
    )
    assertBorrowedView("reshapeView")(_.reshapeView(Shape(3, 2)))
    assertBorrowedView("requireRank")(
      _.requireRank[2].fold(error => fail(error.toString), identity)
    )
  }

  test("every materialization and numerical array result owns and isolates its storage") {
    assertOwnedResult("reshapeCopy")(_.reshapeCopy(Shape(3, 2)))
    assertOwnedResult("contiguous")(_.contiguous)
    assertOwnedResult("copy")(_.copy)
    assertOwnedResult("flattenCopy")(_.flattenCopy)
    assertOwnedResult("cast")(_.cast[Float])
    assertOwnedResult("convert")(
      _.convert[Float]().fold(error => fail(error.toString), identity)
    )
    assertOwnedResult("map")(_.map(_ * 2.0))
    assertOwnedResult("arithmetic")(_ + 1.0)
    assertOwnedResult("comparison")(_ > 2.5)
    assertOwnedResult("sum(axis)")(_.sum(0))
    assertOwnedResult("sumKeep(axis)")(_.sumKeep(0))
    assertOwnedResult("product(axis)")(_.product(0))
    assertOwnedResult("productKeep(axis)")(_.productKeep(0))
    assertOwnedResult("min(axis)")(_.min(0))
    assertOwnedResult("minKeep(axis)")(_.minKeep(0))
    assertOwnedResult("max(axis)")(_.max(0))
    assertOwnedResult("maxKeep(axis)")(_.maxKeep(0))
    assertOwnedResult("argMin(axis)")(_.argMin(0))
    assertOwnedResult("argMinKeep(axis)")(_.argMinKeep(0))
    assertOwnedResult("argMax(axis)")(_.argMax(0))
    assertOwnedResult("argMaxKeep(axis)")(_.argMaxKeep(0))
    assertOwnedResult("mean(axis)")(_.mean(0))
    assertOwnedResult("meanKeep(axis)")(_.meanKeep(0))
    assertOwnedResult("sumAxes(empty)")(_.sumAxes())
    assertOwnedResult("sumAxes(multiple)")(_.sumAxes(0, 1))
    val both = Axes.from(2, 0, 1).toOption.get
    assertOwnedResult("sum(Axes)")(_.sum(both))
    assertOwnedResult("product(Axes)")(_.product(both))
    assertOwnedResult("min(Axes)")(_.min(both))
    assertOwnedResult("max(Axes)")(_.max(both))
    assertOwnedResult("mean(Axes)")(_.mean(both))
  }

  test("scalar observations and iterators read live external storage") {
    val external = fixture()
    val iterator = external.array.elementsIterator
    external.updateExternal(0, 99.0)
    assertEquals(external.array(0, 0), 99.0)
    assertEquals(external.array.at(IArray(0, 0)), 99.0)
    assertEquals(iterator.next(), 99.0)

    val visited = collection.mutable.ArrayBuffer.empty[Double]
    external.array.foreachElement(visited += _)
    assertEquals(visited.head, 99.0)
    val indices = collection.mutable.ArrayBuffer.empty[List[Int]]
    external.array.foreachIndex(index => indices += IArray.genericWrapArray(index).toList)
    assertEquals(indices.head, List(0, 0))

    val scalar = external.array.select(0, 0).select(0, 0)
    assertEquals(scalar.item, 99.0)
    val ownedSnapshot = external.array.copy
    assert(external.array.sameElements(ownedSnapshot))
    assert(external.array.sameElementsBits(ownedSnapshot))

    val before = external.array.sum
    val renderedBefore = external.array.toString
    external.updateExternal(1, 101.0)
    assertNotEquals(external.array.sum, before)
    assert(!external.array.sameElements(ownedSnapshot))
    assert(!external.array.sameElementsBits(ownedSnapshot))
    assertNotEquals(external.array.toString, renderedBefore)
  }

  test("ambiguous borrowed reshape is unavailable") {
    val errors = compileErrors("""
      import ravel.*
      val borrowed = null.asInstanceOf[BorrowedNDArray[Double, Rank[2]]]
      borrowed.reshape(Shape(4))
    """)
    assert(errors.nonEmpty)
  }

  test("expert kernels accept borrowed read sources without weakening destination ownership") {
    val external = fixture()
    val right = NDArray.fill(Shape(2, 3), 1.0)
    val output = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 3))
    kernel.addInto(external.array, right, output)
    assertEquals(output.freezeCopy().elementsIterator.toList, initialValues.map(_ + 1.0).toList)
  }
