package ravel

import munit.FunSuite
import ravel.DType.given

final class ScalarIndexingSuite extends FunSuite:
  test("rank-specific reads agree with arbitrary-rank indexing") {
    val one = NDArray.tabulate[Int](4)(identity)
    val two = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val three =
      NDArray.tabulate[Int](2, 2, 2)((i, j, k) => i * 100 + j * 10 + k)
    val four =
      NDArray.tabulate[Int](2, 2, 2, 2) { (i, j, k, l) =>
        i * 1000 + j * 100 + k * 10 + l
      }

    assertEquals(one(3), one.at(IArray(3)))
    assertEquals(two(1, 2), two.at(IArray(1, 2)))
    assertEquals(three(1, 0, 1), three.at(IArray(1, 0, 1)))
    assertEquals(four(1, 0, 1, 0), four.at(IArray(1, 0, 1, 0)))
  }

  test("rank-specific reads preserve offset, negative, permuted, and broadcast layouts") {
    val source = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j)
    val offset = source.slice(0, Slice(1, 3))
    val reversed = source.reverse(1)
    val transposed = source.transpose
    val broadcast =
      NDArray.fromSeq(Shape(1, 4), Seq(7, 8, 9, 10)).broadcastTo(Shape(3, 4))

    assertEquals(offset(0, 0), 10)
    assertEquals(offset(1, 3), 23)
    assertEquals(reversed(0, 0), 3)
    assertEquals(reversed(2, 3), 20)
    assertEquals(transposed(3, 2), 23)
    assertEquals(broadcast(2, 1), 8)
  }

  test("rank-1 reads cover offset, sliced, negative-stride, and broadcast views") {
    val source = NDArray.tabulate[Int](6)(identity)
    val offset = source.slice(0, Slice(1, 5))
    val sliced = source.slice(0, Slice(0, 6, 2))
    val reversed = source.reverse(0)
    val broadcast = NDArray.fromSeq(Shape(1), Seq(7)).broadcastTo(Shape(4))

    assertEquals(source(4), 4)
    assertEquals(offset(0), 1)
    assertEquals(sliced(2), 4)
    assertEquals(reversed(1), 4)
    assertEquals(broadcast(3), 7)
  }

  test("rank-3 reads cover offset, sliced, negative-stride, permuted, and broadcast views") {
    val source =
      NDArray.tabulate[Int](3, 4, 5)((i, j, k) => i * 100 + j * 10 + k)
    val offset = source.slice(0, Slice(1, 3))
    val sliced = source.slice(2, Slice(0, 5, 2))
    val reversed = source.reverse(1)
    val permuted = source.permuteAxes(2, 0, 1)
    val broadcast =
      NDArray
        .tabulate[Int](1, 4, 1)((_, j, _) => j * 10)
        .broadcastTo(Shape(3, 4, 5))

    assertEquals(source(2, 3, 4), 234)
    assertEquals(offset(0, 2, 4), 124)
    assertEquals(sliced(2, 3, 2), 234)
    assertEquals(reversed(2, 0, 4), 234)
    assertEquals(permuted(4, 2, 3), 234)
    assertEquals(broadcast(2, 3, 4), 30)
  }

  test("rank-4 reads cover offset, sliced, negative-stride, permuted, and broadcast views") {
    val source =
      NDArray.tabulate[Int](2, 3, 4, 5) { (i, j, k, l) =>
        i * 1000 + j * 100 + k * 10 + l
      }
    val offset = source.slice(1, Slice(1, 3))
    val sliced = source.slice(3, Slice(0, 5, 2))
    val reversed = source.reverse(2)
    val permuted = source.permuteAxes(3, 1, 0, 2)
    val broadcast =
      NDArray
        .tabulate[Int](2, 1, 4, 1)((i, _, k, _) => i * 1000 + k * 10)
        .broadcastTo(Shape(2, 3, 4, 5))

    assertEquals(source(1, 2, 3, 4), 1234)
    assertEquals(offset(1, 1, 3, 4), 1234)
    assertEquals(sliced(1, 2, 3, 2), 1234)
    assertEquals(reversed(1, 2, 0, 4), 1234)
    assertEquals(permuted(4, 2, 1, 3), 1234)
    assertEquals(broadcast(1, 2, 3, 4), 1030)
  }

  test("rank-specific reads preserve precise bounds failures") {
    val vector = NDArray.tabulate[Int](2)(identity)
    val matrix = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    val cube = NDArray.zeros[Int](2, 3, 4)
    val hypercube = NDArray.zeros[Int](2, 3, 4, 5)
    val empty = NDArray.zeros[Int](0)

    assertSameFailure(vector(2), vector.at(IArray(2)))
    assertSameFailure(matrix(2, 0), matrix.at(IArray(2, 0)))
    assertSameFailure(matrix(0, 3), matrix.at(IArray(0, 3)))
    assertSameFailure(cube(0, 0, 4), cube.at(IArray(0, 0, 4)))
    assertSameFailure(hypercube(0, 3, 0, 0), hypercube.at(IArray(0, 3, 0, 0)))
    assertSameFailure(empty(0), empty.at(IArray(0)))

    val dynamic: AnyNDArray[Int] = matrix
    val arity = intercept[InvalidIndex.ArityMismatch](dynamic.at(IArray(0)))
    assertEquals(arity, InvalidIndex.ArityMismatch(2, 1))
  }

  test("negative element indices normalize like axes") {
    val matrix = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j)
    assertEquals(matrix(0, -1), 2)
    assertEquals(matrix(-1, -2), 11)
    assertEquals(matrix.select(0, -1).at(IArray(1)), 11)
    assertSameFailure(matrix(0, -4), matrix.at(IArray(0, -4)))
  }

  test("rank-specific mutable reads and updates preserve view locality") {
    val mutable = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j).mutableCopy
    val reversed = mutable.reverse(1)
    val transposed = mutable.transpose

    assertEquals(reversed(0, 0), 3)
    reversed(0, 0) = 99
    assertEquals(mutable(0, 3), 99)
    transposed(2, 1) = 77
    assertEquals(mutable(1, 2), 77)
    assertEquals(mutable.at(IArray(1, 2)), 77)
  }

  test("rank-1, rank-3, and rank-4 mutable view writes remain local") {
    val vector = NDArray.tabulate[Int](6)(identity).mutableCopy
    vector.slice(0, Slice(0, 6, 2))(1) = 70
    assertEquals(vector(2), 70)

    val cube =
      NDArray.tabulate[Int](3, 4, 5)((i, j, k) => i * 100 + j * 10 + k).mutableCopy
    cube.reverse(2)(1, 2, 0) = 999
    assertEquals(cube(1, 2, 4), 999)
    cube.permuteAxes(2, 0, 1)(3, 2, 1) = 888
    assertEquals(cube(2, 1, 3), 888)

    val hypercube =
      NDArray
        .tabulate[Int](2, 3, 4, 5) { (i, j, k, l) =>
          i * 1000 + j * 100 + k * 10 + l
        }
        .mutableCopy
    hypercube.slice(1, Slice(1, 3))(1, 0, 3, 4) = 777
    assertEquals(hypercube(1, 1, 3, 4), 777)
    hypercube.permuteAxes(3, 1, 0, 2)(4, 2, 1, 3) = 666
    assertEquals(hypercube(1, 2, 3, 4), 666)
  }

  test("rank-specific mutable updates preserve precise bounds failures") {
    val matrix = NDArray.zeros[Int](2, 3).mutableCopy

    assertSameFailure(matrix(2, 0) = 1, matrix.updateAt(IArray(2, 0), 1))
    assertSameFailure(matrix(0, 3) = 1, matrix.updateAt(IArray(0, 3), 1))

    val dynamic = matrix.asInstanceOf[MutableNDArray[Int, AnyRank]]
    val arity = intercept[InvalidIndex.ArityMismatch](dynamic.updateAt(IArray(0), 1))
    assertEquals(arity, InvalidIndex.ArityMismatch(2, 1))
  }

  private def assertSameFailure(direct: => Any, generic: => Any): Unit =
    val directFailure = intercept[InvalidIndex](direct)
    val genericFailure = intercept[InvalidIndex](generic)
    assertEquals(directFailure.getMessage, genericFailure.getMessage)
