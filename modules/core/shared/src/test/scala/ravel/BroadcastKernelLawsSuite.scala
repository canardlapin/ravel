package ravel

import munit.FunSuite
import ravel.DType.given
import ravel.internal.*
import scala.collection.mutable.ArrayBuffer

final class BroadcastKernelLawsSuite extends FunSuite:
  private def values[A](array: NDArray[A, ?]): List[A] =
    array.elementsIterator.toList

  test("broadcast rules align trailing axes and treat 0 with 1 as 0") {
    val left = NDArray.tabulate[Int](2, 1, 3)((i, _, k) => i * 10 + k)
    val right = NDArray.fromSeq(Shape(3), Seq(100, 200, 300))
    val result: Array3[Int] = left + right
    assertEquals(result.shape.toString, "(2, 1, 3)")
    assertEquals(values(result), List(100, 201, 302, 110, 211, 312))

    val empty = NDArray.zeros[Int](0) + NDArray.fromSeq(Shape(1), Seq(7))
    assertEquals(empty.shape.toString, "(0)")
    assertEquals(empty.size, 0)

    val mismatch = intercept[BroadcastMismatch] {
      NDArray.zeros[Int](2, 3) + NDArray.zeros[Int](4)
    }
    assertEquals(mismatch.alignedAxis, -1)
    assert(mismatch.getMessage.contains("(2, 3)"))
    assert(mismatch.getMessage.contains("(4)"))
  }

  test("static broadcast rank is the maximum known rank") {
    val matrix = NDArray.zeros[Double](2, 3)
    val vector = NDArray.zeros[Double](3)
    val result: Array2[Double] = matrix + vector
    assertEquals(result.rank, 2)
  }

  test("arithmetic surface is eager and agrees on contiguous and adversarial views") {
    val base = NDArray.tabulate[Int](3, 4)((i, j) => i * 10 + j)
    val left = base.transpose.reverse(0).slice(1, Slice(0, 3, 2))
    val right = NDArray.fromSeq(Shape(1, 2), Seq(2, 3))
    val expectedLeft = values(left)
    val expectedRight = List(2, 3, 2, 3, 2, 3, 2, 3)
    assertEquals(values(left + right), expectedLeft.zip(expectedRight).map(_ + _))
    assertEquals(values(left - right), expectedLeft.zip(expectedRight).map(_ - _))
    assertEquals(values(left * right), expectedLeft.zip(expectedRight).map(_ * _))
    assertEquals(values(left.quot(right)), expectedLeft.zip(expectedRight).map(_ / _))
    assertEquals(values(-left), expectedLeft.map(-_))
    assertEquals(values(left.abs), expectedLeft.map(math.abs))
    assert(!(left + right).storage.eq(left.storage))
  }

  test("scalar arithmetic and fixed-width edge behavior are explicit") {
    val ints = NDArray.fromSeq(Shape(3), Seq(Int.MaxValue, Int.MinValue, 6))
    assertEquals(values(ints + 1), List(Int.MinValue, Int.MinValue + 1, 7))
    assertEquals(values(ints.abs), List(Int.MaxValue, Int.MinValue, 6))
    assertEquals(values(ints.quot(2)), List(Int.MaxValue / 2, Int.MinValue / 2, 3))
    intercept[ArithmeticException](ints.quot(0))
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Int](2) / 2
    """).nonEmpty)
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Byte](2) + NDArray.zeros[Byte](2)
    """).nonEmpty)
    assert(compileErrors("""
      import ravel.*
      import ravel.DType.given
      NDArray.zeros[Int](2) + NDArray.zeros[Double](2)
    """).nonEmpty)
  }

  test("ordered kernels, explicit equality, and IEEE comparisons allocate Boolean arrays") {
    val x = NDArray.fromSeq(Shape(5), Seq(-2.0, -0.0, 1.0, Double.NaN, 5.0))
    val y = NDArray.fromSeq(Shape(5), Seq(-1.0, 0.0, 1.0, 4.0, Double.NaN))
    assertEquals(values(x.minimum(y)).take(3), List(-2.0, -0.0, 1.0))
    assert(values(x.minimum(y))(3).isNaN)
    assert(values(x.maximum(y))(4).isNaN)
    assertEquals(values(x.clip(-1.0, 2.0)).take(3), List(-1.0, -0.0, 1.0))
    assertEquals(values(x < y), List(true, false, false, false, false))
    assertEquals(values(x <= y), List(true, true, true, false, false))
    assertEquals(values(x === y), List(false, true, true, false, false))
    assertEquals(values(x =!= y), List(true, false, false, true, true))
    val booleanDType: DType[Boolean] = (x < y).dtype
    assertEquals(booleanDType.name, "Boolean")
  }

  test("Byte and Short retain ordered kernels without arithmetic") {
    val bytes = NDArray.fromSeq(Shape(3), Seq[Byte](3, -1, 7))
    val other = NDArray.fromSeq(Shape(3), Seq[Byte](2, 4, 7))
    assertEquals(values(bytes.minimum(other)), List[Byte](2, -1, 7))
    assertEquals(values(bytes.maximum(0.toByte)), List[Byte](3, 0, 7))
    assertEquals(values(bytes < other), List(false, true, false))
  }

  test("floating kernels follow primitive functions and predicates") {
    val x = NDArray.fromSeq(
      Shape(5),
      Seq(0.0, 0.5, 1.0, Double.NaN, Double.PositiveInfinity)
    )
    def close(actual: List[Double], expected: List[Double]): Unit =
      actual.zip(expected).foreach { (a, e) =>
        if e.isNaN then assert(a.isNaN)
        else if e.isInfinite then assertEquals(a, e)
        else assertEqualsDouble(a, e, 1e-12)
      }
    close(values(x.sqrt), values(x).map(math.sqrt))
    close(values(x.exp), values(x).map(math.exp))
    close(values(x.log), values(x).map(math.log))
    close(values(x.sin), values(x).map(math.sin))
    close(values(x.cos), values(x).map(math.cos))
    close(values(x.tan), values(x).map(math.tan))
    close(values(x.floor), values(x).map(math.floor))
    close(values(x.ceil), values(x).map(math.ceil))
    assertEquals(values(x.isNaN), List(false, false, false, true, false))
    assertEquals(values(x.isFinite), List(true, true, true, false, false))
  }

  test("planner exposes all four validated classifications") {
    val linear = NDArray.zeros[Int](2, 3)
    assertEquals(LoopPlan.broadcast(linear.layout, linear.layout).kind, LoopKind.LinearContiguous)

    val scalar = NDArray.scalar(1)
    assertEquals(LoopPlan.broadcast(scalar.layout, linear.layout).kind, LoopKind.ScalarBroadcast)

    val inner = linear.reverse(1)
    assertEquals(LoopPlan.broadcast(inner.layout, linear.layout).kind, LoopKind.InnerStrided)

    val generalBase = NDArray.zeros[Int](2, 3, 4)
    val general = generalBase.permuteAxes(2, 1, 0)
    assertEquals(
      LoopPlan.broadcast(general.layout, general.contiguous.layout).kind,
      LoopKind.GeneralStrided
    )
  }

  test("generic callbacks run exactly once in logical row-major order") {
    val source = NDArray.tabulate[Int](2, 3)((i, j) => i * 10 + j).transpose.reverse(0)
    val observed = ArrayBuffer.empty[Int]
    val mapped = source.map { value =>
      observed += value
      value * 2
    }
    assertEquals(observed.toList, values(source))
    assertEquals(values(mapped), values(source).map(_ * 2))

    val calls = ArrayBuffer.empty[(Int, Int)]
    val zipped = source.zipMap(NDArray.fromSeq(Shape(1), Seq(100))) { (a, b) =>
      calls += ((a, b))
      a + b
    }
    assertEquals(calls.toList, values(source).map(_ -> 100))
    assertEquals(values(zipped), values(source).map(_ + 100))
  }

  test("callback exceptions stop evaluation and empty callbacks are not invoked") {
    val source = NDArray.tabulate[Int](5)(identity)
    var calls = 0
    intercept[IllegalStateException] {
      source.map { value =>
        calls += 1
        if value == 2 then throw new IllegalStateException("stop")
        value
      }
    }
    assertEquals(calls, 3)

    var emptyCalls = 0
    val empty = NDArray.zeros[Int](0).map { value =>
      emptyCalls += 1
      value
    }
    assertEquals(emptyCalls, 0)
    assertEquals(empty.size, 0)
  }

  test("pure total map commutes extensionally with reindexing") {
    val source = NDArray.tabulate[Int](4, 5)((i, j) => i * 10 + j)
    val function: Int => Int = value => value * value + 1
    val left = source.slice(1, Slice(0, 5, 2)).map(function)
    val right = source.map(function).slice(1, Slice(0, 5, 2))
    assertEquals(values(left), values(right))
  }

  test("zipMapExact rejects shapes that broadcasting would accept") {
    val matrix = NDArray.zeros[Int](2, 3)
    val row = NDArray.zeros[Int](3)
    assertEquals(matrix.zipMap(row)(_ + _).shape.toString, "(2, 3)")
    intercept[ShapeMismatch] {
      matrix.zipMapExact(row)(_ + _)
    }
  }
