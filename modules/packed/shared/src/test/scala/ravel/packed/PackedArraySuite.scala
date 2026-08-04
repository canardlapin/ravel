package ravel.packed

import munit.FunSuite
import ravel.*

final class PackedArraySuite extends FunSuite:
  private val widths = Vector(PackedBits.B1, PackedBits.B2, PackedBits.B4)

  test("codes round-trip at sizes that do not fill the final word"):
    widths.foreach { bits =>
      val shape = Shape(5, 7)
      val codes = Vector.tabulate(35)(index => (index * 3 + 1) % (bits.maxCode + 1))
      val packed = packedRight(PackedArray.fromCodes(shape, bits, codes))

      assertEquals(packed.codeVector, codes, s"width $bits")
      assertEquals(packed.shape, shape)
      assert(packed.isCanonical)
      var linear = 0
      while linear < 35 do
        assertEquals(packed.codeAt(linear), codes(linear))
        assertEquals(packed(linear / 7, linear % 7), codes(linear))
        linear += 1
    }

  test("construction handles scalars and empty axes"):
    widths.foreach { bits =>
      val scalar = packedRight(PackedArray.scalar(bits.maxCode, bits))
      assertEquals(scalar.shape, Shape.scalar)
      assertEquals(scalar.item, bits.maxCode)

      val selectedScalar =
        packedRight(PackedArray.fromCodes(Shape(1), bits, Vector(bits.maxCode))).select(-1, -1)
      assertEquals(selectedScalar.shape, Shape.scalar)
      assertEquals(selectedScalar.item, bits.maxCode)

      val empty = PackedArray.zeros(Shape(2, 0, 3), bits)
      assertEquals(empty.size, 0)
      assertEquals(empty.codeVector, Vector.empty)
      assertEquals(empty.sumCodes, 0L)
      assertEquals(empty.reverse(-2).shape, Shape(2, 0, 3))
      assertEquals(empty.slice(-1, Slice.reverse).shape, Shape(2, 0, 3))
      assertEquals(empty.narrow(-2, 0, 0).shape, Shape(2, 0, 3))

      val selected = empty.select(-1, -1)
      assertEquals(selected.shape, Shape(2, 0))
      assertEquals(selected.size, 0)
    }

  test("construction rejects out-of-range codes and wrong code counts"):
    PackedArray.fromCodes(Shape(4), PackedBits.B2, Vector(0, 1, 4, 0)) match
      case Left(PackedError.InvalidCode(index, code, maxCode)) =>
        assertEquals(index, 2)
        assertEquals(code, 4)
        assertEquals(maxCode, 3)
      case other => fail(s"expected InvalidCode, got $other")
    assert(PackedArray.fromCodes(Shape(4), PackedBits.B1, Vector(1, 0, 1)).isLeft)
    assert(PackedArray.fromCodes(Shape(2), PackedBits.B1, Vector(1, 0, 1)).isLeft)

  test("word layout is least-significant-slot first and fixed across platforms"):
    val oneBit = packedRight(
      PackedArray.fromCodes(Shape(8), PackedBits.B1, Vector(1, 0, 1, 0, 1, 0, 1, 0))
    )
    assertEquals(oneBit.wordVector, Vector(0x55))

    val fourBit = packedRight(
      PackedArray.fromCodes(Shape(6), PackedBits.B4, Vector(1, 2, 3, 4, 5, 6))
    )
    assertEquals(fourBit.wordVector, Vector(0x654321))

    val twoBit =
      packedRight(PackedArray.fromCodes(Shape(4), PackedBits.B2, Vector(3, 0, 2, 1)))
    assertEquals(twoBit.wordVector, Vector((1 << 6) | (2 << 4) | 3))

  test("serialized words round-trip and reject non-canonical input"):
    val shape = Shape(3, 11)
    val codes = Vector.tabulate(33)(index => index % 4)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B2, codes))
    val restored = packedRight(PackedArray.fromWords(shape, PackedBits.B2, packed.wordVector))

    assertEquals(restored.codeVector, codes)
    PackedArray.fromWords(shape, PackedBits.B2, packed.wordVector.dropRight(1)) match
      case Left(PackedError.WordLengthMismatch(expected, actual)) =>
        assertEquals(expected, 3)
        assertEquals(actual, 2)
      case other => fail(s"expected WordLengthMismatch, got $other")
    val corrupted = packed.wordVector.updated(2, packed.wordVector(2) | (1 << 5))
    PackedArray.fromWords(shape, PackedBits.B2, corrupted) match
      case Left(PackedError.NonCanonicalTail(_)) => ()
      case other => fail(s"expected NonCanonicalTail, got $other")

  test("select, slice, narrow, reverse, and permutation compose against a Vector model"):
    val shape = Shape(2, 3, 4)
    val codes = Vector.tabulate(shape.size)(index => index % 16)
    val base = packedRight(PackedArray.fromCodes(shape, PackedBits.B4, codes))
    val baseModel = Model(Vector(2, 3, 4), codes)
    val permutations = Vector(
      Vector(0, 1, 2),
      Vector(0, 2, 1),
      Vector(1, 0, 2),
      Vector(1, 2, 0),
      Vector(2, 0, 1),
      Vector(2, 1, 0)
    )

    permutations.foreach { order =>
      var packed = base.permuteAxes(order*)
      var model = baseModel.permute(order)
      var axis = 0
      while axis < 3 do
        packed = packed.reverse(axis - 3)
        model = model.reverse(axis)
        axis += 1
      axis = 0
      while axis < 3 do
        packed = packed.slice(axis, Slice.every(2))
        model = model.slice(axis, Vector.range(0, model.shape(axis), 2))
        axis += 1

      assertPackedMatches(packed, model, s"order=$order")
      axis = 0
      while axis < 3 do
        val selected = packed.select(axis - 3, -1)
        val selectedModel = model.select(axis, model.shape(axis) - 1)
        assertPackedMatches(selected, selectedModel, s"order=$order selected=$axis")
        axis += 1
    }

    val negativePermutation = base.permuteAxes(-1, 0, 1)
    assertPackedMatches(negativePermutation, baseModel.permute(Vector(2, 0, 1)), "negative axes")

  test("narrow follows core exact checked and throwing semantics"):
    val packed = packedRight(PackedArray.fromCodes(Shape(5), PackedBits.B4, 0 until 5))
    assertEquals(packed.narrow(0, -1, 1).codeVector, Vector(4))
    assertEquals(packed.narrow(-1, -5, 5).codeVector, Vector(0, 1, 2, 3, 4))
    assertEquals(packed.narrow(0, 5, 0).codeVector, Vector.empty)
    assert(packed.narrowChecked(0, 5, 1).isLeft)
    assert(packed.narrowChecked(0, -6, 1).isLeft)
    assert(packed.narrowChecked(0, 0, -1).isLeft)
    intercept[InvalidNarrowException](packed.narrow(0, 5, 1))

  test("negative coordinate indexing and view copying preserve logical order"):
    val shape = Shape(4, 6)
    val codes = Vector.tabulate(shape.size)(index => index % 16)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B4, codes))

    assertEquals(packed(-1, -1), codes.last)
    val row = packed.select(0, -2)
    assertEquals(row.codeVector, Vector.tabulate(6)(column => codes(2 * 6 + column)))

    val block = packed.narrow(0, 1, 2).slice(1, Slice.between(2, 5))
    assertEquals(
      block.codeVector,
      Vector.tabulate(6)(linear => codes((1 + linear / 3) * 6 + 2 + linear % 3))
    )
    val copied = block.copy
    assert(copied.isCanonical)
    assertEquals(copied.codeVector, block.codeVector)

  test("fixed-rank packed indexing rejects wrong arity at compile time"):
    val errors = compileErrors("""
      import ravel.*
      import ravel.packed.*
      val packed: PackedArray1 = PackedArray.zeros(Shape(4), PackedBits.B1)
      packed(0, 1)
    """)
    assert(errors.nonEmpty)

  test("sumCodes agrees between canonical fast path and view fallback"):
    val shape = Shape(9, 5)
    val codes = Vector.tabulate(shape.size)(index => (index * 7) % 4)
    val packed = packedRight(PackedArray.fromCodes(shape, PackedBits.B2, codes))
    val view = packed.narrow(0, 1, 7)

    assertEquals(packed.sumCodes, codes.map(_.toLong).sum)
    assertEquals(view.sumCodes, view.codeVector.map(_.toLong).sum)

  test("mutable workspace freezes by ownership transfer and by copy"):
    val workspace = MutablePackedArray.allocate(Shape(10), PackedBits.B2)
    var linear = 0
    while linear < 10 do
      workspace.setCode(linear, linear % 4)
      linear += 1
    val copied = workspace.freezeCopy
    workspace.setCode(0, 3)
    val frozen = workspace.freeze

    assertEquals(copied.codeAt(0), 0)
    assertEquals(frozen.codeAt(0), 3)
    assertEquals(copied.codeVector.drop(1), frozen.codeVector.drop(1))

  private def assertPackedMatches(
      packed: PackedArray[?],
      model: Model,
      clue: String
  ): Unit =
    assertEquals(packed.shape.toIArray.toVector, model.shape, clue)
    assertEquals(packed.codeVector, model.values, clue)

  private def packedRight[A](value: Either[PackedError, A]): A =
    value match
      case Right(result) => result
      case Left(error) => fail(error.message)

  private final case class Model(shape: Vector[Int], values: Vector[Int]):
    def select(axis: Int, index: Int): Model =
      val targetShape = shape.patch(axis, Nil, 1)
      transform(targetShape) { target =>
        target.patch(axis, Vector(index), 0)
      }

    def slice(axis: Int, indices: Vector[Int]): Model =
      transform(shape.updated(axis, indices.length)) { target =>
        target.updated(axis, indices(target(axis)))
      }

    def reverse(axis: Int): Model =
      slice(axis, (0 until shape(axis)).reverse.toVector)

    def permute(order: Vector[Int]): Model =
      val targetShape = order.map(shape)
      transform(targetShape) { target =>
        val source = Array.fill(shape.length)(0)
        var axis = 0
        while axis < order.length do
          source(order(axis)) = target(axis)
          axis += 1
        source.toVector
      }

    private def transform(
        targetShape: Vector[Int]
    )(sourceCoordinates: Vector[Int] => Vector[Int]): Model =
      val targetSize = if targetShape.isEmpty then 1 else targetShape.product
      Model(
        targetShape,
        Vector.tabulate(targetSize) { linear =>
          values(linearIndex(sourceCoordinates(coordinates(linear, targetShape)), shape))
        }
      )

  private def coordinates(linear: Int, shape: Vector[Int]): Vector[Int] =
    if shape.isEmpty then Vector.empty
    else
      val result = new Array[Int](shape.length)
      var remaining = linear
      var axis = shape.length - 1
      while axis >= 0 do
        result(axis) = remaining % shape(axis)
        remaining /= shape(axis)
        axis -= 1
      result.toVector

  private def linearIndex(coordinates: Vector[Int], shape: Vector[Int]): Int =
    var linear = 0
    var axis = 0
    while axis < shape.length do
      linear = linear * shape(axis) + coordinates(axis)
      axis += 1
    linear
