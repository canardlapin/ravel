package ravel

import munit.FunSuite
import ravel.DType.given
import scala.collection.mutable.ArrayBuffer

final class RankIndexOwnershipSuite extends FunSuite:
  test("known-arity constructors infer refined ranks") {
    val vector: Array1[Double] = NDArray.zeros[Double](3)
    val matrix: Array2[Double] = NDArray.zeros[Double](3, 4)
    val cube: Array3[Double] = NDArray.zeros[Double](2, 3, 4)
    val hyper: Array4[Double] = NDArray.zeros[Double](2, 2, 2, 2)
    assertEquals(vector.rank, 1)
    assertEquals(matrix.rank, 2)
    assertEquals(cube.rank, 3)
    assertEquals(hyper.rank, 4)
  }

  test("dynamic shapes refine through requireRank") {
    val shape = Shape.from(Seq(2, 3)).toOption.get
    val dynamic: AnyNDArray[Int] = NDArray.zeros[Int, AnyRank](shape)
    assert(dynamic.requireRank[2].isRight)
    assertEquals(dynamic.requireRank[3], Left(RankMismatch(3, 2)))
  }

  test("known rank zero has no CanDropAxis evidence") {
    val errors = compileErrors("""
      import ravel.*
      summon[CanDropAxis[Rank[0]]]
    """)
    assert(errors.nonEmpty)
    summon[CanDropAxis[Rank[1]]]
    summon[CanDropAxis[AnyRank]]
  }

  test("rank-one through rank-four indexing and arbitrary-rank fallback agree") {
    val one = NDArray.tabulate[Int](4)(identity)
    val two = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val three = NDArray.tabulate[Int](2, 2, 2)((i, j, k) => i * 100 + j * 10 + k)
    val four = NDArray.tabulate[Int](2, 2, 2, 2)((i, j, k, l) => i * 1000 + j * 100 + k * 10 + l)
    assertEquals(one(3), 3)
    assertEquals(two(1, 2), 12)
    assertEquals(three(1, 0, 1), 101)
    assertEquals(four(1, 0, 1, 0), 1010)
    assertEquals(four.at(IArray(1, 0, 1, 0)), four(1, 0, 1, 0))
    intercept[NDArrayException](two(2, 0))
    intercept[NDArrayException](two.at(IArray(0)))
  }

  test("owned construction isolates a mutable caller sequence") {
    val values = ArrayBuffer(1, 2, 3, 4)
    val array = NDArray.fromSeq(Shape(2, 2), values)
    values(0) = 99
    assertEquals(array(0, 0), 1)
  }

  test("explicit traversal preserves logical row-major order") {
    val array = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    assertEquals(array.elementsIterator.toList, List(0, 1, 2, 10, 11, 12))
    val indices = ArrayBuffer.empty[List[Int]]
    array.foreachIndex(index => indices += IArray.genericWrapArray(index).toList)
    assertEquals(
      indices.toList,
      List(List(0, 0), List(0, 1), List(0, 2), List(1, 0), List(1, 1), List(1, 2))
    )
  }

  test("NDArray is not a standard collection") {
    val errors = compileErrors("""
      import ravel.*
      import ravel.DType.given
      val x: Iterable[Int] = NDArray.zeros[Int](3)
    """)
    assert(errors.nonEmpty)
  }
