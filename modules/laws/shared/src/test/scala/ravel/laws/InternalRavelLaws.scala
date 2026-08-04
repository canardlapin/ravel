package ravel.laws

import munit.Assertions
import org.scalacheck.Gen
import org.scalacheck.Prop
import org.scalacheck.Prop.forAll
import org.typelevel.discipline.Laws
import ravel.*
import ravel.DType.given

private[laws] object ShapeLaws extends Assertions:
  def valid(shape: Shape[?]): Unit =
    assertEquals(shape.rank, shape.toIArray.length)
    var product = 1L
    var axis = 0
    while axis < shape.rank do
      assert(shape(axis) >= 0, s"negative dimension at axis $axis")
      product *= shape(axis).toLong
      axis += 1
    assertEquals(shape.size.toLong, product)

private[laws] object LayoutLaws extends Assertions:
  def materialization[A](array: NDArray[A, ?]): Unit =
    assert(array.contiguous.sameElements(array))

private[laws] object ViewLaws extends Assertions:
  def reverseInvolution[A](array: NDArray[A, ?], axis: Int): Unit =
    assert(array.reverse(axis).reverse(axis).sameElements(array))

  def swapInvolution[A](array: NDArray[A, ?], first: Int, second: Int): Unit =
    assert(array.swapAxes(first, second).swapAxes(first, second).sameElements(array))

  def pureMapCommutesWithSlice[A, B](
      array: NDArray[A, ?],
      axis: Int,
      slice: Slice,
      function: A => B
  )(using DType[B]): Unit =
    assert(
      array
        .slice(axis, slice)
        .map(function)
        .sameElements(array.map(function).slice(axis, slice))
    )

private[laws] object BroadcastLaws extends Assertions:
  def commutativeShape[A: ArithmeticDType](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Unit =
    val summed: Shape[?] = (left + right).shape
    val swapped: Shape[?] = (right + left).shape
    assertEquals(summed, swapped)

private[laws] object DTypeLaws extends Assertions:
  def named[A](dtype: DType[A], expected: String): Unit =
    assertEquals(dtype.name, expected)

private[laws] object CastLaws extends Assertions:
  def exactRoundTrip[A: NumericDType, B: NumericDType](values: NDArray[A, ?]): Unit =
    assert(values.cast[B].cast[A].sameElements(values))

private[laws] object KernelLaws extends Assertions:
  def additionMatchesCallback[A: ArithmeticDType](
      left: NDArray[A, ?],
      right: NDArray[A, ?],
      add: (A, A) => A
  ): Unit =
    assert((left + right).sameElements(left.zipMap(right)(add)))

private[laws] object MutableLaws extends Assertions:
  def freezeIsolation[A, R <: AnyRank](
      mutable: MutableNDArray[A, R],
      mutate: MutableNDArray[A, R] => Unit
  ): Unit =
    val frozen = mutable.freezeCopy()
    val snapshot = frozen.copy
    mutate(mutable)
    assert(frozen.sameElements(snapshot))

private[laws] object InteropLaws extends Assertions:
  def copyIsolation[A](
      readCopied: () => A,
      mutateOriginal: () => Unit,
      expected: A
  ): Unit =
    mutateOriginal()
    assertEquals(readCopied(), expected)

private[laws] object BorrowedInteropLaws extends Assertions:
  def externalMutationObserved[A](
      borrowedRead: () => A,
      mutateExternal: () => Unit,
      expectedAfter: A
  ): Unit =
    mutateExternal()
    assertEquals(borrowedRead(), expectedAfter)

private[laws] object InternalGenerators:
  val dimensions: Gen[Vector[Int]] =
    Gen
      .choose(0, 6)
      .flatMap(rank =>
        Gen
          .listOfN(
            rank,
            Gen.frequency(2 -> Gen.const(0), 2 -> Gen.const(1), 6 -> Gen.choose(2, 4))
          )
          .map(_.toVector)
      )

  val viewScenario: Gen[(Vector[Int], Int)] =
    Gen
      .choose(1, 6)
      .flatMap(rank =>
        Gen
          .listOfN(rank, Gen.choose(0, 4))
          .flatMap(dimensions =>
            Gen.chooseNum(Int.MinValue, Int.MaxValue).map(seed => (dimensions.toVector, seed))
          )
      )

  val broadcastScenario: Gen[(Int, Int)] =
    for
      rows <- Gen.choose(0, 4)
      columns <- Gen.choose(0, 4)
    yield (rows, columns)

  val exactInts: Gen[Vector[Int]] =
    Gen
      .choose(0, 24)
      .flatMap(size => Gen.listOfN(size, Gen.choose(-1000000, 1000000)).map(_.toVector))

  def shape(dimensions: Vector[Int]): Shape[AnyRank] =
    Shape.from(dimensions).fold(error => throw AssertionError(error.reason), identity)

  def intArray(dimensions: Vector[Int]): NDArray[Int, AnyRank] =
    val resultShape = shape(dimensions)
    NDArray.fromSeq(resultShape, Vector.tabulate(resultShape.size)(identity))

private[laws] object InternalRavelDiscipline extends Laws:
  import InternalGenerators.*

  def core: RuleSet =
    new DefaultRuleSet(
      name = "ravel-internal-generated",
      parent = None,
      "shape products rank zero through six" -> forAll(dimensions) { dimensions =>
        Prop.secure {
          ShapeLaws.valid(shape(dimensions))
          true
        }
      },
      "reverse and copy preserve generated logical traversal" -> forAll(viewScenario) {
        case (dimensions, seed) =>
          Prop.secure {
            val array = intArray(dimensions)
            val axis = Math.floorMod(seed, dimensions.length)
            val reversed = array.reverse(axis)
            ViewLaws.reverseInvolution(array, axis)
            assert(reversed.copy.sameElements(reversed))
            true
          }
      },
      "generated trailing broadcast shape is symmetric" -> forAll(broadcastScenario) {
        case (rows, columns) =>
          Prop.secure {
            val matrix = NDArray.zeros[Int](rows, columns)
            val trailing = NDArray.zeros[Int](columns)
            BroadcastLaws.commutativeShape(matrix, trailing)
            true
          }
      },
      "generated exact Int Double cast round trips" -> forAll(exactInts) { values =>
        Prop.secure {
          CastLaws.exactRoundTrip[Int, Double](NDArray.fromSeq(Shape(values.size), values))
          true
        }
      },
      "generated mutable writes do not alter frozen snapshots" -> forAll(exactInts) { values =>
        Prop.secure {
          val mutable = NDArray.fromSeq(Shape(values.size), values).mutableCopy
          MutableLaws.freezeIsolation(
            mutable,
            array => if array.size > 0 then array.updateAt(IArray(0), Int.MinValue)
          )
          true
        }
      },
      "invalid negative dimensions remain pure errors" -> forAll(
        Gen.choose(0, 5),
        Gen.choose(1, Int.MaxValue)
      ) { (prefix, magnitude) =>
        val invalid = Vector.fill(prefix)(1) :+ -magnitude
        Prop(Shape.from(invalid).isLeft)
      }
    )
