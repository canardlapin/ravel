package ravel

import munit.ScalaCheckSuite
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll
import ravel.DType.given
import ravel.internal.Layout
import ravel.internal.ProbeApi

final class ModelBasedLayoutCompositionSuite extends ScalaCheckSuite:
  override def scalaCheckTestParameters =
    super.scalaCheckTestParameters
      .withMinSuccessfulTests(300)
      .withMaxDiscardRatio(1)
      .withWorkers(1)

  private final case class Model(
      shape: Vector[Int],
      values: Vector[Int],
      addresses: Vector[Int]
  ):
    require(values.size == logicalSize(shape))
    require(addresses.size == values.size)

    def applyStep(step: Step): Model =
      step match
        case Step.Select(_, axis, _, index) =>
          val targetShape = shape.patch(axis, Nil, 1)
          transform(targetShape) { targetCoordinate =>
            targetCoordinate.patch(axis, Vector(index), 0)
          }
        case Step.SliceAxis(_, axis, _, _, _, indices) =>
          val targetShape = shape.updated(axis, indices.size)
          transform(targetShape) { targetCoordinate =>
            targetCoordinate.updated(axis, indices(targetCoordinate(axis)))
          }
        case Step.Narrow(_, axis, _, start, length) =>
          val targetShape = shape.updated(axis, length)
          transform(targetShape) { targetCoordinate =>
            targetCoordinate.updated(axis, start + targetCoordinate(axis))
          }
        case Step.Reverse(_, axis) =>
          transform(shape) { targetCoordinate =>
            targetCoordinate.updated(axis, shape(axis) - 1 - targetCoordinate(axis))
          }
        case Step.NewAxis(_, axis) =>
          val targetShape = shape.patch(axis, Vector(1), 0)
          transform(targetShape)(_.patch(axis, Nil, 1))
        case Step.Squeeze(_, axis) =>
          val targetShape = shape.patch(axis, Nil, 1)
          transform(targetShape) { targetCoordinate =>
            targetCoordinate.patch(axis, Vector(0), 0)
          }
        case Step.Permute(_, order) =>
          val targetShape = order.map(shape)
          transform(targetShape) { targetCoordinate =>
            val source = Array.fill(shape.size)(0)
            var targetAxis = 0
            while targetAxis < order.size do
              source(order(targetAxis)) = targetCoordinate(targetAxis)
              targetAxis += 1
            source.toVector
          }
        case Step.Broadcast(targetShape) =>
          val leading = targetShape.size - shape.size
          transform(targetShape) { targetCoordinate =>
            Vector.tabulate(shape.size) { sourceAxis =>
              if shape(sourceAxis) == 1 then 0 else targetCoordinate(leading + sourceAxis)
            }
          }
        case Step.Reshape(targetShape) =>
          copy(shape = targetShape)

    private def transform(
        targetShape: Vector[Int]
    )(sourceCoordinate: Vector[Int] => Vector[Int]): Model =
      val targetSize = logicalSize(targetShape)
      val targetValues = Vector.tabulate(targetSize) { linear =>
        val sourceLinear = linearIndex(shape, sourceCoordinate(coordinates(targetShape, linear)))
        values(sourceLinear)
      }
      val targetAddresses = Vector.tabulate(targetSize) { linear =>
        val sourceLinear = linearIndex(shape, sourceCoordinate(coordinates(targetShape, linear)))
        addresses(sourceLinear)
      }
      Model(targetShape, targetValues, targetAddresses)

  private enum Step:
    case Select(rawAxis: Int, axis: Int, rawIndex: Int, index: Int)
    case SliceAxis(
        rawAxis: Int,
        axis: Int,
        start: Int,
        stop: Int,
        stride: Int,
        indices: Vector[Int]
    )
    case Narrow(rawAxis: Int, axis: Int, rawStart: Int, start: Int, length: Int)
    case Reverse(rawAxis: Int, axis: Int)
    case NewAxis(rawAxis: Int, axis: Int)
    case Squeeze(rawAxis: Int, axis: Int)
    case Permute(rawOrder: Vector[Int], order: Vector[Int])
    case Broadcast(targetShape: Vector[Int])
    case Reshape(targetShape: Vector[Int])

    def family: String =
      this match
        case _: Select => "select"
        case _: SliceAxis => "slice"
        case _: Narrow => "narrow"
        case _: Reverse => "reverse"
        case _: NewAxis => "newAxis"
        case _: Squeeze => "squeeze"
        case _: Permute => "permute"
        case _: Broadcast => "broadcastTo"
        case _: Reshape => "reshapeView"

  private final case class Scenario(seed: Int, initial: Model, steps: Vector[Step])

  private final class DeterministicRandom(seed: Int):
    private var state = seed ^ 0x6d2b79f5

    def nextInt(bound: Int): Int =
      require(bound > 0)
      state = state * 1664525 + 1013904223
      (state & Int.MaxValue) % bound

    def nextBoolean(): Boolean = nextInt(2) == 0

    def choose[A](values: Vector[A]): A = values(nextInt(values.size))

    def permutation(size: Int): Vector[Int] =
      val values = Array.tabulate(size)(identity)
      var index = size - 1
      while index > 0 do
        val selected = nextInt(index + 1)
        val temporary = values(index)
        values(index) = values(selected)
        values(selected) = temporary
        index -= 1
      values.toVector

  property("owned and borrowed composed views match the independent logical model") {
    forAll(Gen.chooseNum(Int.MinValue, Int.MaxValue)) { seed =>
      val scenario = generateScenario(seed, allowBroadcast = true)
      runOwnedAndBorrowed(scenario)
      true
    }
  }

  property("mutable composed views match the model and writes remain local") {
    forAll(Gen.chooseNum(Int.MinValue, Int.MaxValue)) { seed =>
      val scenario = generateScenario(seed, allowBroadcast = false)
      runMutable(scenario)
      true
    }
  }

  test("the deterministic corpus covers every view family and boundary form") {
    val steps =
      (0 until 512).flatMap { seed =>
        generateScenario(seed, allowBroadcast = true).steps
      }
    assertEquals(
      steps.map(_.family).toSet,
      Set(
        "select",
        "slice",
        "narrow",
        "reverse",
        "newAxis",
        "squeeze",
        "permute",
        "broadcastTo",
        "reshapeView"
      )
    )
    assert(steps.exists(hasNegativeAxis), "expected negative-axis operations")
    assert(
      steps.exists {
        case Step.SliceAxis(_, _, _, _, stride, _) => stride < 0
        case _ => false
      },
      "expected negative-step slices"
    )
    assert(
      steps.exists {
        case Step.SliceAxis(_, _, _, _, _, indices) => indices.isEmpty
        case _ => false
      },
      "expected empty slices"
    )
  }

  test("select canonicalizes the unobservable offset of an empty composed view") {
    val source = NDArray.zeros[Int](0).reshapeView(Shape(0, 3, 2))
    val owned = source.select(1, 1)
    val borrowed = new BorrowedNDArray(source).select(1, 1)
    val mutable = source.mutableCopy.select(1, 1)

    assertEquals(owned.shape, Shape(0, 2))
    assertEquals(owned.elementsIterator.toVector, Vector.empty)
    assertEquals(owned.layout.offset, 0)
    assertEquals(borrowed.layout.offset, 0)
    assertEquals(mutable.layout.offset, 0)
  }

  test("invalid boundaries fail without changing the source") {
    val source = NDArray
      .fromSeq[Int, AnyRank](shape(Vector(2, 1, 3)), 0 until 6)
      .reverse(-1)
    val beforeValues = source.elementsIterator.toVector
    val beforeAddresses = physicalAddresses(source.layout)

    assert(source.permuteAxesChecked(0, 0, 2).isLeft)
    assert(source.permuteAxesChecked(0, 1).isLeft)
    assert(source.narrowChecked(0, Int.MinValue, 1).isLeft)
    assert(source.narrowChecked(0, Int.MaxValue, 0).isLeft)
    intercept[InvalidAxis](source.select(-4, 0))
    intercept[InvalidIndex](source.select(0, 2))
    intercept[InvalidSlice](source.slice(0, Slice(Int.MaxValue, Int.MaxValue)))
    intercept[InvalidAxis](source.newAxis(5))
    intercept[InvalidShape](source.squeeze(0))
    assert(Shape.from(Seq(Int.MaxValue, 2)).isLeft)

    assertEquals(source.elementsIterator.toVector, beforeValues)
    assertEquals(physicalAddresses(source.layout), beforeAddresses)
  }

  private def runOwnedAndBorrowed(scenario: Scenario): Unit =
    val base = NDArray.fromSeq[Int, AnyRank](shape(scenario.initial.shape), scenario.initial.values)
    var owned = base
    var borrowed = new BorrowedNDArray(base)
    var model = scenario.initial
    assertState(owned, model, s"seed ${scenario.seed}, initial owned")
    assertState(borrowed, model, s"seed ${scenario.seed}, initial borrowed")

    scenario.steps.zipWithIndex.foreach { case (step, index) =>
      owned = applyOwned(owned, step)
      borrowed = applyBorrowed(borrowed, step)
      model = model.applyStep(step)
      val clue = s"seed ${scenario.seed}, step $index: $step"
      assertState(owned, model, clue)
      assertState(borrowed, model, clue)
      assert(owned.storage eq base.storage, s"owned view copied at $clue")
      assert(borrowed.storage eq base.storage, s"borrowed view copied at $clue")
    }

    val ownedCopy = owned.copy
    val borrowedCopy = borrowed.copy
    assert(!(ownedCopy.storage eq base.storage))
    assert(!(borrowedCopy.storage eq base.storage))
    assertEquals(ownedCopy.elementsIterator.toVector, model.values)
    assertEquals(borrowedCopy.elementsIterator.toVector, model.values)

    if model.addresses.nonEmpty then
      val changedAddress = model.addresses(model.addresses.size / 2)
      ProbeApi.set(base.storage, changedAddress, -1)
      val expected =
        model.addresses.map(address => if address == changedAddress then -1 else address)
      assertEquals(borrowed.elementsIterator.toVector, expected)
      assertEquals(ownedCopy.elementsIterator.toVector, model.values)
      assertEquals(borrowedCopy.elementsIterator.toVector, model.values)

  private def runMutable(scenario: Scenario): Unit =
    val base = NDArray
      .fromSeq[Int, AnyRank](shape(scenario.initial.shape), scenario.initial.values)
      .mutableCopy
    var view = base
    var model = scenario.initial
    assertState(view, model, s"seed ${scenario.seed}, initial mutable")

    scenario.steps.zipWithIndex.foreach { case (step, index) =>
      view = applyMutable(view, step)
      model = model.applyStep(step)
      assertState(view, model, s"seed ${scenario.seed}, mutable step $index: $step")
    }

    assertEquals(model.addresses.distinct.size, model.addresses.size)
    if model.addresses.nonEmpty then
      val logical = model.addresses.size / 2
      val coordinate = coordinates(model.shape, logical)
      val changedAddress = model.addresses(logical)
      view.updateAt(IArray.unsafeFromArray(coordinate.toArray), -1)
      val expectedBase = scenario.initial.values.updated(changedAddress, -1)
      val expectedView = model.values.updated(logical, -1)
      assertEquals(base.freezeCopy().elementsIterator.toVector, expectedBase)
      assertEquals(view.freezeCopy().elementsIterator.toVector, expectedView)

  private def applyOwned(
      array: NDArray[Int, AnyRank],
      step: Step
  ): NDArray[Int, AnyRank] =
    step match
      case Step.Select(rawAxis, _, rawIndex, _) => array.select(rawAxis, rawIndex)
      case Step.SliceAxis(rawAxis, _, start, stop, stride, _) =>
        array.slice(rawAxis, Slice(start, stop, stride))
      case Step.Narrow(rawAxis, _, rawStart, _, length) =>
        array.narrow(rawAxis, rawStart, length)
      case Step.Reverse(rawAxis, _) => array.reverse(rawAxis)
      case Step.NewAxis(rawAxis, _) => array.newAxis(rawAxis)
      case Step.Squeeze(rawAxis, _) => array.squeeze(rawAxis)
      case Step.Permute(rawOrder, _) => array.permuteAxes(rawOrder*)
      case Step.Broadcast(targetShape) => array.broadcastTo(shape(targetShape))
      case Step.Reshape(targetShape) => array.reshapeView(shape(targetShape))

  private def applyBorrowed(
      array: BorrowedNDArray[Int, AnyRank],
      step: Step
  ): BorrowedNDArray[Int, AnyRank] =
    step match
      case Step.Select(rawAxis, _, rawIndex, _) =>
        array.select(rawAxis, rawIndex).asInstanceOf[BorrowedNDArray[Int, AnyRank]]
      case Step.SliceAxis(rawAxis, _, start, stop, stride, _) =>
        array.slice(rawAxis, Slice(start, stop, stride))
      case Step.Narrow(rawAxis, _, rawStart, _, length) =>
        array.narrow(rawAxis, rawStart, length)
      case Step.Reverse(rawAxis, _) => array.reverse(rawAxis)
      case Step.NewAxis(rawAxis, _) =>
        array.newAxis(rawAxis).asInstanceOf[BorrowedNDArray[Int, AnyRank]]
      case Step.Squeeze(rawAxis, _) =>
        array.squeeze(rawAxis).asInstanceOf[BorrowedNDArray[Int, AnyRank]]
      case Step.Permute(rawOrder, _) => array.permuteAxes(rawOrder*)
      case Step.Broadcast(targetShape) => array.broadcastTo(shape(targetShape))
      case Step.Reshape(targetShape) => array.reshapeView(shape(targetShape))

  private def applyMutable(
      array: MutableNDArray[Int, AnyRank],
      step: Step
  ): MutableNDArray[Int, AnyRank] =
    step match
      case Step.Select(rawAxis, _, rawIndex, _) =>
        array.select(rawAxis, rawIndex).asInstanceOf[MutableNDArray[Int, AnyRank]]
      case Step.SliceAxis(rawAxis, _, start, stop, stride, _) =>
        array.slice(rawAxis, Slice(start, stop, stride))
      case Step.Narrow(rawAxis, _, rawStart, _, length) =>
        array.narrow(rawAxis, rawStart, length)
      case Step.Reverse(rawAxis, _) => array.reverse(rawAxis)
      case Step.NewAxis(rawAxis, _) =>
        array.newAxis(rawAxis).asInstanceOf[MutableNDArray[Int, AnyRank]]
      case Step.Squeeze(rawAxis, _) =>
        array.squeeze(rawAxis).asInstanceOf[MutableNDArray[Int, AnyRank]]
      case Step.Permute(rawOrder, _) => array.permuteAxes(rawOrder*)
      case Step.Reshape(targetShape) => array.reshapeView(shape(targetShape))
      case _: Step.Broadcast => fail("mutable scenarios must not contain broadcast views")

  private def assertState(
      array: NDArray[Int, ?],
      model: Model,
      clue: String
  ): Unit =
    assertEquals(shapeVector(array.shape), model.shape, clue)
    assertEquals(array.elementsIterator.toVector, model.values, clue)
    assertEquals(physicalAddresses(array.layout), model.addresses, clue)

  private def assertState(
      array: BorrowedNDArray[Int, ?],
      model: Model,
      clue: String
  ): Unit =
    assertEquals(shapeVector(array.shape), model.shape, clue)
    assertEquals(array.elementsIterator.toVector, model.values, clue)
    assertEquals(physicalAddresses(array.layout), model.addresses, clue)

  private def assertState(
      array: MutableNDArray[Int, ?],
      model: Model,
      clue: String
  ): Unit =
    assertEquals(shapeVector(array.shape), model.shape, clue)
    assertEquals(array.freezeCopy().elementsIterator.toVector, model.values, clue)
    assertEquals(physicalAddresses(array.layout), model.addresses, clue)

  private def generateScenario(seed: Int, allowBroadcast: Boolean): Scenario =
    val random = new DeterministicRandom(seed)
    val rank = random.nextInt(7)
    val dimensions = Vector.fill(rank)(random.nextInt(6))
    val size = logicalSize(dimensions)
    val initial =
      Model(dimensions, Vector.tabulate(size)(identity), Vector.tabulate(size)(identity))
    val steps = Vector.newBuilder[Step]
    var current = initial
    var remaining = 12 + random.nextInt(9)
    while remaining > 0 do
      val step = generateStep(current, random, allowBroadcast)
      steps += step
      current = current.applyStep(step)
      remaining -= 1
    Scenario(seed, initial, steps.result())

  private def generateStep(
      model: Model,
      random: DeterministicRandom,
      allowBroadcast: Boolean
  ): Step =
    val nonemptyAxes = model.shape.indices.filter(model.shape(_) > 0).toVector
    val singletonAxes = model.shape.indices.filter(model.shape(_) == 1).toVector
    val families = Vector.newBuilder[String]
    if nonemptyAxes.nonEmpty then families += "select"
    if model.shape.nonEmpty then
      families += "slice"
      families += "narrow"
      families += "reverse"
    if model.shape.size < 6 then families += "newAxis"
    if singletonAxes.nonEmpty then families += "squeeze"
    if model.shape.size > 1 then families += "permute"
    if allowBroadcast && (model.shape.size < 6 || singletonAxes.nonEmpty) then
      families += "broadcast"
    if reshapeEligible(model.addresses) then families += "reshape"

    random.choose(families.result()) match
      case "select" =>
        val axis = random.choose(nonemptyAxes)
        val index = random.nextInt(model.shape(axis))
        Step.Select(
          encodedAxis(axis, model.shape.size, random),
          axis,
          encodedIndex(index, model.shape(axis), random),
          index
        )
      case "slice" => generateSlice(model, random)
      case "narrow" =>
        val axis = random.nextInt(model.shape.size)
        val extent = model.shape(axis)
        val start = random.nextInt(extent + 1)
        val length = random.nextInt(extent - start + 1)
        Step.Narrow(
          encodedAxis(axis, model.shape.size, random),
          axis,
          encodedBoundary(start, extent, random),
          start,
          length
        )
      case "reverse" =>
        val axis = random.nextInt(model.shape.size)
        Step.Reverse(encodedAxis(axis, model.shape.size, random), axis)
      case "newAxis" =>
        val axis = random.nextInt(model.shape.size + 1)
        val rawAxis =
          if random.nextBoolean() then axis - model.shape.size - 1 else axis
        Step.NewAxis(rawAxis, axis)
      case "squeeze" =>
        val axis = random.choose(singletonAxes)
        Step.Squeeze(encodedAxis(axis, model.shape.size, random), axis)
      case "permute" =>
        val order = random.permutation(model.shape.size)
        val raw = order.map(axis => encodedAxis(axis, model.shape.size, random))
        Step.Permute(raw, order)
      case "broadcast" =>
        Step.Broadcast(broadcastShape(model.shape, random))
      case "reshape" =>
        Step.Reshape(reshapeTarget(model.values.size, random))
      case unexpected => fail(s"unknown generated family $unexpected")

  private def generateSlice(model: Model, random: DeterministicRandom): Step =
    val axis = random.nextInt(model.shape.size)
    val extent = model.shape(axis)
    val rawAxis = encodedAxis(axis, model.shape.size, random)
    if extent > 0 && random.nextBoolean() then
      val normalizedStart = random.nextInt(extent)
      val normalizedStop =
        if random.nextBoolean() then -1 else random.nextInt(normalizedStart + 1)
      val stride = -(1 + random.nextInt(3))
      val indices =
        Iterator
          .iterate(normalizedStart)(_ + stride)
          .takeWhile(_ > normalizedStop)
          .toVector
      val start = encodedIndex(normalizedStart, extent, random)
      val stop =
        if normalizedStop == -1 then -1
        else encodedBoundary(normalizedStop, extent, random)
      Step.SliceAxis(rawAxis, axis, start, stop, stride, indices)
    else
      val normalizedStart = random.nextInt(extent + 1)
      val normalizedStop =
        normalizedStart + random.nextInt(extent - normalizedStart + 1)
      val stride = 1 + random.nextInt(3)
      val indices = (normalizedStart until normalizedStop by stride).toVector
      Step.SliceAxis(
        rawAxis,
        axis,
        encodedBoundary(normalizedStart, extent, random),
        encodedBoundary(normalizedStop, extent, random),
        stride,
        indices
      )

  private def broadcastShape(
      source: Vector[Int],
      random: DeterministicRandom
  ): Vector[Int] =
    val added =
      if source.size < 6 then 1 + random.nextInt(6 - source.size)
      else 0
    val leading = Vector.fill(added)(random.nextInt(4))
    val aligned = source.map { dimension =>
      if dimension == 1 then random.nextInt(6) else dimension
    }
    leading ++ aligned

  private def reshapeTarget(size: Int, random: DeterministicRandom): Vector[Int] =
    if size == 0 then
      val rank = 1 + random.nextInt(6)
      val zeroAxis = random.nextInt(rank)
      Vector.tabulate(rank)(axis => if axis == zeroAxis then 0 else 1 + random.nextInt(3))
    else if size == 1 then Vector.fill(random.nextInt(7))(1)
    else
      val trailingOnes = random.nextInt(6)
      Vector(size) ++ Vector.fill(trailingOnes)(1)

  private def reshapeEligible(addresses: Vector[Int]): Boolean =
    addresses.size <= 1 ||
      {
        val stride = addresses(1) - addresses(0)
        stride != 0 && addresses.indices.drop(1).forall { index =>
          addresses(index) - addresses(index - 1) == stride
        }
      }

  private def encodedAxis(
      axis: Int,
      rank: Int,
      random: DeterministicRandom
  ): Int =
    if random.nextBoolean() then axis - rank else axis

  private def encodedIndex(
      index: Int,
      extent: Int,
      random: DeterministicRandom
  ): Int =
    if random.nextBoolean() then index - extent else index

  private def encodedBoundary(
      boundary: Int,
      extent: Int,
      random: DeterministicRandom
  ): Int =
    if boundary < extent && random.nextBoolean() then boundary - extent else boundary

  private def hasNegativeAxis(step: Step): Boolean =
    step match
      case Step.Select(rawAxis, _, _, _) => rawAxis < 0
      case Step.SliceAxis(rawAxis, _, _, _, _, _) => rawAxis < 0
      case Step.Narrow(rawAxis, _, _, _, _) => rawAxis < 0
      case Step.Reverse(rawAxis, _) => rawAxis < 0
      case Step.NewAxis(rawAxis, _) => rawAxis < 0
      case Step.Squeeze(rawAxis, _) => rawAxis < 0
      case Step.Permute(rawOrder, _) => rawOrder.exists(_ < 0)
      case _ => false

  private def physicalAddresses(layout: Layout): Vector[Int] =
    val addresses = Vector.newBuilder[Int]
    layout.foreachPhysicalIndex(addresses += _)
    addresses.result()

  private def shape(dimensions: Vector[Int]): Shape[AnyRank] =
    Shape.from(dimensions).fold(error => fail(error.reason), identity)

  private def shapeVector(shape: Shape[?]): Vector[Int] =
    IArray.genericWrapArray(shape.toIArray).toVector

  private def logicalSize(shape: Vector[Int]): Int =
    shape.foldLeft(1)(_ * _)

  private def coordinates(shape: Vector[Int], linear: Int): Vector[Int] =
    val coordinate = Array.fill(shape.size)(0)
    var remainder = linear
    var axis = shape.size - 1
    while axis >= 0 do
      coordinate(axis) = remainder % shape(axis)
      remainder /= shape(axis)
      axis -= 1
    coordinate.toVector

  private def linearIndex(shape: Vector[Int], coordinate: Vector[Int]): Int =
    var linear = 0
    var axis = 0
    while axis < shape.size do
      linear = linear * shape(axis) + coordinate(axis)
      axis += 1
    linear
