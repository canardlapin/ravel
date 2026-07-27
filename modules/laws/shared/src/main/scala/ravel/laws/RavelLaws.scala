package ravel.laws

import munit.Assertions
import org.scalacheck.Prop
import org.typelevel.discipline.Laws
import ravel.*

/** Shape invariants expressed only through the public API. */
object ShapeLaws extends Assertions:
  def valid(shape: Shape[?]): Unit =
    assertEquals(shape.rank, shape.toIArray.length)
    var product = 1L
    var axis = 0
    while axis < shape.rank do
      assert(shape(axis) >= 0, s"negative dimension at axis $axis")
      product *= shape(axis).toLong
      axis += 1
    assertEquals(shape.size.toLong, product)

/** Layout laws observable through public indexing and materialization. */
object LayoutLaws extends Assertions:
  def materialization[A](array: NDArray[A, ?]): Unit =
    assert(array.contiguous.sameElements(array))

/** Reindexing laws for the structural view algebra. */
object ViewLaws extends Assertions:
  def reverseInvolution[A](array: NDArray[A, ?], axis: Int): Unit =
    assert(array.reverse(axis).reverse(axis).sameElements(array))

  def swapInvolution[A](
      array: NDArray[A, ?],
      first: Int,
      second: Int
  ): Unit =
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

/** Trailing-axis broadcast laws. */
object BroadcastLaws extends Assertions:
  def commutativeShape[A: ArithmeticDType](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Unit =
    assertEquals((left + right).shape.toString, (right + left).shape.toString)

/** Closed dtype capability assertions. */
object DTypeLaws extends Assertions:
  def named[A](dtype: DType[A], expected: String): Unit =
    assertEquals(dtype.name, expected)

/** Explicit cast round-trip laws where the caller knows the conversion is exact. */
object CastLaws extends Assertions:
  def exactRoundTrip[A: NumericDType, B: NumericDType](
      values: NDArray[A, ?]
  ): Unit =
    assert(values.cast[B].cast[A].sameElements(values))

/** Specialized kernel/reference agreement. */
object KernelLaws extends Assertions:
  def additionMatchesCallback[A: ArithmeticDType](
      left: NDArray[A, ?],
      right: NDArray[A, ?],
      add: (A, A) => A
  ): Unit =
    assert((left + right).sameElements(left.zipMap(right)(add)))

/** Mutation locality and freeze isolation. */
object MutableLaws extends Assertions:
  def freezeIsolation[A, R <: AnyRank](
      mutable: MutableNDArray[A, R],
      mutate: MutableNDArray[A, R] => Unit
  ): Unit =
    val frozen = mutable.freezeCopy()
    val snapshot = frozen.copy
    mutate(mutable)
    assert(frozen.sameElements(snapshot))

/** Copying interop must isolate later external changes. */
object InteropLaws extends Assertions:
  def copyIsolation[A](
      readCopied: () => A,
      mutateOriginal: () => Unit,
      expected: A
  ): Unit =
    mutateOriginal()
    assertEquals(readCopied(), expected)

/** Borrowing interop must make external aliasing observable without losing type provenance. */
object BorrowedInteropLaws extends Assertions:
  def externalMutationObserved[A](
      borrowedRead: () => A,
      mutateExternal: () => Unit,
      expectedAfter: A
  ): Unit =
    mutateExternal()
    assertEquals(borrowedRead(), expectedAfter)

/** A small Discipline entry point for downstream property suites. */
object RavelDiscipline extends Laws:
  def core: RuleSet =
    new DefaultRuleSet(
      name = "ravel-core",
      parent = None,
      "scalar shape product" -> Prop(Shape.scalar.size == 1),
      "zero with one broadcasts to zero" -> Prop(
        (
          NDArray.zeros[Int](0)(using DType.intDType) +
            NDArray.fromSeq(Shape(1), Seq(7))(using DType.intDType)
        ).size == 0
      )
    )
