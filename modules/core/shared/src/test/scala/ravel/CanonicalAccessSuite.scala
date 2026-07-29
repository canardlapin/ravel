package ravel

import munit.FunSuite

final class CanonicalAccessSuite extends FunSuite:
  test("canonical immutable refinement is allocation-free logical C-order access"):
    val source =
      NDArray.tabulate[Int](2, 3, 4)((i, j, k) => 100 * i + 10 * j + k)
    val canonical = CanonicalArray.require(source)

    assert(canonical.asInstanceOf[AnyRef] eq source)
    assertEquals(canonical.size, source.size)
    var linear = 0
    while linear < canonical.size do
      val i = linear / 12
      val rest = linear % 12
      val j = rest / 4
      val k = rest % 4
      assertEquals(canonical.readLinear(linear), source(i, j, k))
      linear += 1

  test("canonical refinement preserves the complete coordinate API"):
    val source =
      NDArray.tabulate[Int](2, 3)((row, column) => row * 10 + column)
    val canonical = CanonicalArray.require(source)

    assertEquals(canonical(1, -1), 12)
    assertEquals(canonical.at(IArray(0, -1)), 2)
    assertEquals(canonical.shape, Shape(2, 3))
    val transposed: Array2[Int] = canonical.transpose
    assertEquals(transposed(2, 1), 12)

  test("canonical refinement rejects noncanonical and partial views"):
    val source = NDArray.tabulate[Double](3, 4)((i, j) => i * 10.0 + j)

    assert(CanonicalArray.from(source.transpose).isLeft)
    assert(CanonicalArray.from(source.reverse(1)).isLeft)
    assert(CanonicalArray.from(source.narrow(0, 0, 2)).isLeft)
    assert(CanonicalArray.from(source.reshapeView(Shape(12))).isRight)

  test("canonical immutable access checks linear bounds"):
    val source = CanonicalArray.require(NDArray.fromSeq(Shape(2), Seq(1, 2)))

    assertEquals(source(-1), 2)
    intercept[InvalidIndex.LinearOutOfBounds](source.readLinear(-1))
    intercept[InvalidIndex.LinearOutOfBounds](source.readLinear(2))

  test("canonical mutable refinement reads and writes primitive values"):
    val ints = MutableNDArray.zeros[Int, Rank[1]](Shape(3))
    val intAccess = MutableCanonicalArray.require(ints)
    intAccess.writeLinear(0, 10)
    intAccess.writeLinear(1, 20)
    intAccess(-1) = 30
    assertEquals(intAccess.readLinear(1), 20)
    assertEquals(intAccess(-1), 30)
    assertEquals(
      ints.freezeCopy().elementsIterator.toList,
      List(10, 20, 30)
    )

    val booleans = MutableNDArray.zeros[Boolean, Rank[1]](Shape(2))
    val booleanAccess = MutableCanonicalArray.require(booleans)
    booleanAccess.writeLinear(1, true)
    assert(!booleanAccess.readLinear(0))
    assert(booleanAccess(1))

    val doubles = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 2))
    val doubleAccess = MutableCanonicalArray.require(doubles)
    doubleAccess(1, -1) = 4.5
    assertEqualsDouble(doubleAccess.readLinear(3), 4.5, 0.0)

  test("canonical mutable refinement is the original owner and rejects views"):
    val source = MutableNDArray.zeros[Int, Rank[2]](Shape(2, 3))
    val canonical = MutableCanonicalArray.require(source)

    assert(canonical.asInstanceOf[AnyRef] eq source)
    assert(MutableCanonicalArray.from(source.transpose).isLeft)
    assert(MutableCanonicalArray.from(source.reverse(0)).isLeft)
    assert(
      MutableCanonicalArray.from(source.narrow(1, 0, 2)).isLeft
    )

  test("immutable canonical access exposes no update operation"):
    assert(
      compileErrors("""
        import ravel.*
        val source = CanonicalArray.require(NDArray.zeros[Int](4))
        source(0) = 1
      """).nonEmpty
    )
