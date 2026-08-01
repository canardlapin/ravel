package ravel.stencil

import munit.FunSuite
import ravel.DType.given
import ravel.MutableNDArray
import ravel.NDArray
import ravel.Rank
import ravel.Shape

final class ReferenceNeighborhoodExecutorSuite extends FunSuite:
  test("identity neighborhood copies the source under Same extent"):
    val source =
      NDArray.tabulate[Int](3, 2)((i, j) => 10 * i + j)
    val destination = MutableNDArray.zeros[Int, Rank[2]](Shape(3, 2))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(0, 0)),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(3, 2)
      )
    val reducer =
      new NeighborhoodReducer[Int, Int, Int]:
        def zero: Int = 0
        def accumulate(acc: Int, value: Int, offsetIndex: Int): Int = value
        def finish(acc: Int): Int = acc

    ReferenceNeighborhoodExecutor.run(
      source,
      destination,
      spec,
      reducer,
      constant = -1
    )

    val frozen = destination.freezeCopy()
    assertEquals(frozen(0, 0), 0)
    assertEquals(frozen(2, 1), 21)

  test("Constant border feeds the fill value outside the domain"):
    val source = NDArray.tabulate[Double](2, 2)((i, j) => (i + j).toDouble)
    val destination = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 2))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(2, 2)
      )
    val reducer =
      new NeighborhoodReducer[Double, Double, Double]:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value
        def finish(acc: Double): Double = acc

    ReferenceNeighborhoodExecutor.run(
      source,
      destination,
      spec,
      reducer,
      constant = 7.0
    )

    val result = destination.freezeCopy()(0, 0)
    assertEquals(result, 8.0)

  test("trailing batch axis is held fixed"):
    val source =
      NDArray.tabulate[Int](2, 2, 2)((i, j, t) => 100 * t + 10 * i + j)
    val destination = MutableNDArray.zeros[Int, Rank[3]](Shape(2, 2, 2))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(0, 0)),
        border = BorderMode.Replicate,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(2, 2)
      )
    val reducer =
      new NeighborhoodReducer[Int, Int, Int]:
        def zero: Int = 0
        def accumulate(acc: Int, value: Int, offsetIndex: Int): Int = value
        def finish(acc: Int): Int = acc

    ReferenceNeighborhoodExecutor.run(
      source,
      destination,
      spec,
      reducer,
      constant = 0
    )

    val frozen = destination.freezeCopy()
    assertEquals(frozen(1, 1, 0), 11)
    assertEquals(frozen(1, 1, 1), 111)

  test("direct executor agrees with reference on strided source views"):
    val base =
      NDArray.tabulate[Int](5, 4)((i, j) => 100 * i + j)
    val source = base.reverse(0).swapAxes(0, 1)
    val reference = MutableNDArray.zeros[Int, Rank[2]](source.shape)
    val direct = MutableNDArray.zeros[Int, Rank[2]](source.shape)
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(
          Vector(-1, 0),
          Vector(0, -1),
          Vector(0, 0),
          Vector(0, 1),
          Vector(1, 0)
        ),
        border = BorderMode.ReflectWithoutEdge,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(source.shape(0), source.shape(1))
      )
    val reducer =
      new NeighborhoodReducer[Int, Int, Int]:
        def zero: Int = 0
        def accumulate(acc: Int, value: Int, offsetIndex: Int): Int =
          acc + value * (offsetIndex + 1)
        def finish(acc: Int): Int = acc

    ReferenceNeighborhoodExecutor.run(
      source,
      reference,
      spec,
      reducer,
      constant = -7
    )
    DirectNeighborhoodExecutor.run(
      source,
      direct,
      spec,
      reducer,
      constant = -7
    )

    assert(direct.freezeCopy().sameElements(reference.freezeCopy()))

  test("direct executor agrees with reference for every border mode"):
    val source =
      NDArray.tabulate[Int](3, 2)((i, j) => 10 * i + j)
    val specFor = (border: BorderMode) =>
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
        border = border,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(3, 2)
      )
    val reducer =
      new NeighborhoodReducer[Int, Int, Int]:
        def zero: Int = 0
        def accumulate(acc: Int, value: Int, offsetIndex: Int): Int =
          acc + value
        def finish(acc: Int): Int = acc

    BorderMode.values.foreach { border =>
      val reference = MutableNDArray.zeros[Int, Rank[2]](Shape(3, 2))
      val direct = MutableNDArray.zeros[Int, Rank[2]](Shape(3, 2))
      val spec = specFor(border)
      ReferenceNeighborhoodExecutor.run(
        source,
        reference,
        spec,
        reducer,
        constant = -4
      )
      DirectNeighborhoodExecutor.run(
        source,
        direct,
        spec,
        reducer,
        constant = -4
      )
      assert(
        direct.freezeCopy().sameElements(reference.freezeCopy()),
        s"direct executor diverged for border mode $border"
      )
    }

  test("primitive Double direct path agrees with generic reference"):
    val source =
      NDArray.tabulate[Double](4, 3, 2)((i, j, t) => 100.0 * t + 10.0 * i + j)
    val reference = MutableNDArray.zeros[Double, Rank[3]](Shape(4, 3, 2))
    val direct = MutableNDArray.zeros[Double, Rank[3]](Shape(4, 3, 2))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(
          Vector(-1, 0),
          Vector(0, -1),
          Vector(0, 0),
          Vector(0, 1),
          Vector(1, 0)
        ),
        border = BorderMode.Wrap,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(4, 3)
      )
    val generic =
      new NeighborhoodReducer[Double, Double, Double]:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value
        def finish(acc: Double): Double = acc
    val primitive =
      new DoubleNeighborhoodReducer:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value
        def finish(acc: Double): Double = acc

    ReferenceNeighborhoodExecutor.run(
      source,
      reference,
      spec,
      generic,
      constant = 0.0
    )
    DirectNeighborhoodExecutor.runDouble(
      source,
      direct,
      spec,
      primitive,
      constant = 0.0
    )

    assert(direct.freezeCopy().sameElements(reference.freezeCopy()))

  test("direct executor matches reference for cropped and broadcast source strides"):
    val base =
      NDArray.tabulate[Int](5, 4)((i, j) => 10 * i + j)
    val cropped =
      base
        .narrow(axis = 0, from = 1, length = 3)
        .narrow(
          axis = 1,
          from = 1,
          length = 2
        )
    val broadcast =
      NDArray
        .tabulate[Int](1, 4)((_, column) => 10 + column)
        .broadcastTo(Shape(3, 4))
    val reducer =
      new NeighborhoodReducer[Int, Int, Int]:
        def zero: Int = 0
        def accumulate(acc: Int, value: Int, offsetIndex: Int): Int =
          acc + value
        def finish(acc: Int): Int = acc

    Vector(cropped, broadcast).foreach { source =>
      val reference =
        MutableNDArray.zeros[Int, Rank[2]](source.shape)
      val direct =
        MutableNDArray.zeros[Int, Rank[2]](source.shape)
      val spec =
        NeighborhoodSpec(
          spatialAxes = 2,
          offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
          border = BorderMode.Replicate,
          outputOrigin = Vector(0, 0),
          outputSpatialShape = Vector(source.shape(0), source.shape(1))
        )

      ReferenceNeighborhoodExecutor.run(
        source,
        reference,
        spec,
        reducer,
        constant = 0
      )
      DirectNeighborhoodExecutor.run(
        source,
        direct,
        spec,
        reducer,
        constant = 0
      )

      assert(
        direct.freezeCopy().sameElements(reference.freezeCopy()),
        s"direct executor diverged for source layout ${source.shape}"
      )
    }
