package ravel.consumer

import munit.FunSuite
import ravel.*
import ravel.DType.given
import scala.compiletime.testing.typeCheckErrors

final class ArrayBuilderSuite extends FunSuite:
  test("ordered and unordered writes build one immutable logical result") {
    val ordered = NDArray.build[Int, Rank[2]](Shape(2, 3)) { builder =>
      var index = 0
      while index < builder.size do
        builder.writeLinear(index, index + 1)
        index += 1
    }
    assertEquals(ordered.elementsIterator.toList, List(1, 2, 3, 4, 5, 6))

    val unordered = NDArray.build[Int, Rank[1]](Shape(5)) { builder =>
      builder(4) = 40
      builder(1) = 10
      builder(3) = 30
      builder(1) = 11
    }
    assertEquals(unordered.elementsIterator.toList, List(0, 11, 0, 30, 40))
  }

  test("builder bounds failures are public and seal the failed builder") {
    var escaped = Option.empty[ArrayBuilder[Int]]
    intercept[InvalidIndex] {
      NDArray.build[Int, Rank[1]](Shape(2)) { builder =>
        escaped = Some(builder)
        builder(-1) = 1
      }
    }
    intercept[BuilderClosed] {
      escaped.get(0) = 1
    }

    intercept[InvalidIndex] {
      NDArray.build[Int, Rank[1]](Shape(2)) { builder =>
        builder.writeLinear(2, 1)
      }
    }
  }

  test("builder is sealed after a successful callback") {
    var escaped = Option.empty[ArrayBuilder[Int]]
    val result = NDArray.build[Int, Rank[1]](Shape(1)) { builder =>
      escaped = Some(builder)
      builder(0) = 42
    }
    assertEquals(result(0), 42)
    intercept[BuilderClosed] {
      escaped.get(0) = 7
    }
  }

  test("callback failure publishes no result and seals the builder") {
    var escaped = Option.empty[ArrayBuilder[Double]]
    val sentinel = new RuntimeException("stop construction")
    val observed = intercept[RuntimeException] {
      NDArray.build[Double, Rank[1]](Shape(4)) { builder =>
        escaped = Some(builder)
        builder(0) = 1.0
        throw sentinel
      }
    }
    assert(observed eq sentinel)
    intercept[BuilderClosed] {
      escaped.get(1) = 2.0
    }
  }

  test("zero-length construction is valid") {
    var escaped = Option.empty[ArrayBuilder[Long]]
    val empty = NDArray.build[Long, Rank[3]](Shape(2, 0, 3)) { builder =>
      escaped = Some(builder)
      assertEquals(builder.size, 0)
    }
    assertEquals(empty.shape.toIArray.toList, List(2, 0, 3))
    assertEquals(empty.size, 0)
    assertEquals(empty.elementsIterator.toList, Nil)
    intercept[BuilderClosed] {
      escaped.get(0) = 1L
    }
  }

  test("construction supports every primitive dtype family") {
    val booleans = NDArray.build[Boolean, Rank[1]](Shape(2)) { builder =>
      builder(0) = true
      builder(1) = false
    }
    val bytes = NDArray.build[Byte, Rank[1]](Shape(1))(_(0) = 7.toByte)
    val shorts = NDArray.build[Short, Rank[1]](Shape(1))(_(0) = 8.toShort)
    val ints = NDArray.build[Int, Rank[1]](Shape(1))(_(0) = 9)
    val longs = NDArray.build[Long, Rank[1]](Shape(1))(_(0) = 10L)
    val floats = NDArray.build[Float, Rank[1]](Shape(1))(_(0) = 11.5f)
    val doubles = NDArray.build[Double, Rank[1]](Shape(1))(_(0) = 12.5)

    assertEquals(booleans.elementsIterator.toList, List(true, false))
    assertEquals(bytes(0), 7.toByte)
    assertEquals(shorts(0), 8.toShort)
    assertEquals(ints(0), 9)
    assertEquals(longs(0), 10L)
    assertEquals(floats(0), 11.5f)
    assertEquals(doubles(0), 12.5)
  }

  test("builder element type rejects mismatched writes at compile time") {
    val errors = typeCheckErrors(
      """
        import ravel.*
        import ravel.DType.given
        NDArray.build[Double, Rank[1]](Shape(1)) { builder =>
          builder(0) = "not a double"
        }
      """
    )
    assert(errors.nonEmpty)
    assert(errors.head.message.contains("String"))
    assert(errors.head.message.contains("Double"))
  }
