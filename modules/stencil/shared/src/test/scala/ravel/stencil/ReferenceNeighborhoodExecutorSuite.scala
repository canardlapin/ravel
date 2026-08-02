package ravel.stencil

import munit.FunSuite
import ravel.DType.given
import ravel.MutableNDArray
import ravel.NDArray
import ravel.Rank
import ravel.Shape
import ravel.mutableCopy

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

  test("prepared primitive Float, Byte, and Short paths agree with reference"):
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
        border = BorderMode.Replicate,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(2, 2)
      )

    val floatSource = NDArray.tabulate[Float](2, 2)((i, j) => (10 * i + j).toFloat)
    val floatReference = MutableNDArray.zeros[Float, Rank[2]](Shape(2, 2))
    val floatDirect = MutableNDArray.zeros[Float, Rank[2]](Shape(2, 2))
    val floatGeneric =
      new NeighborhoodReducer[Float, Float, Float]:
        def zero: Float = 0.0f
        def accumulate(acc: Float, value: Float, offsetIndex: Int): Float = acc + value
        def finish(acc: Float): Float = acc
    val floatPrimitive =
      new FloatNeighborhoodReducer:
        def zero: Float = 0.0f
        def accumulate(acc: Float, value: Float, offsetIndex: Int): Float = acc + value
        def finish(acc: Float): Float = acc
    ReferenceNeighborhoodExecutor.run(
      floatSource,
      floatReference,
      spec,
      floatGeneric,
      constant = 0.0f
    )
    DirectNeighborhoodExecutor
      .prepare(floatSource, floatDirect, spec)
      .runFloat(floatSource, floatDirect, floatPrimitive, constant = 0.0f)
    assert(floatDirect.freezeCopy().sameElements(floatReference.freezeCopy()))

    val byteSource = NDArray.tabulate[Byte](2, 2)((i, j) => (10 * i + j).toByte)
    val byteReference = MutableNDArray.zeros[Byte, Rank[2]](Shape(2, 2))
    val byteDirect = MutableNDArray.zeros[Byte, Rank[2]](Shape(2, 2))
    val byteGeneric =
      new NeighborhoodReducer[Byte, Byte, Byte]:
        def zero: Byte = 0
        def accumulate(acc: Byte, value: Byte, offsetIndex: Int): Byte =
          (acc + value).toByte
        def finish(acc: Byte): Byte = acc
    val bytePrimitive =
      new ByteNeighborhoodReducer:
        def zero: Byte = 0
        def accumulate(acc: Byte, value: Byte, offsetIndex: Int): Byte =
          (acc + value).toByte
        def finish(acc: Byte): Byte = acc
    ReferenceNeighborhoodExecutor.run(
      byteSource,
      byteReference,
      spec,
      byteGeneric,
      constant = 0.toByte
    )
    DirectNeighborhoodExecutor
      .prepare(byteSource, byteDirect, spec)
      .runByte(byteSource, byteDirect, bytePrimitive, constant = 0.toByte)
    assert(byteDirect.freezeCopy().sameElements(byteReference.freezeCopy()))

    val shortSource =
      NDArray.tabulate[Short](2, 2)((i, j) => (100 * i + j).toShort)
    val shortReference = MutableNDArray.zeros[Short, Rank[2]](Shape(2, 2))
    val shortDirect = MutableNDArray.zeros[Short, Rank[2]](Shape(2, 2))
    val shortGeneric =
      new NeighborhoodReducer[Short, Short, Short]:
        def zero: Short = 0
        def accumulate(acc: Short, value: Short, offsetIndex: Int): Short =
          (acc + value).toShort
        def finish(acc: Short): Short = acc
    val shortPrimitive =
      new ShortNeighborhoodReducer:
        def zero: Short = 0
        def accumulate(acc: Short, value: Short, offsetIndex: Int): Short =
          (acc + value).toShort
        def finish(acc: Short): Short = acc
    ReferenceNeighborhoodExecutor.run(
      shortSource,
      shortReference,
      spec,
      shortGeneric,
      constant = 0.toShort
    )
    DirectNeighborhoodExecutor
      .prepare(shortSource, shortDirect, spec)
      .runShort(shortSource, shortDirect, shortPrimitive, constant = 0.toShort)
    assert(shortDirect.freezeCopy().sameElements(shortReference.freezeCopy()))

  test("prepared mutable Double source agrees with immutable source without freezeCopy"):
    val immutable =
      NDArray.tabulate[Double](4, 3)((i, j) => 10.0 * i + j)
    val mutableSource = immutable.mutableCopy
    val fromImmutable = MutableNDArray.zeros[Double, Rank[2]](Shape(4, 3))
    val fromMutable = MutableNDArray.zeros[Double, Rank[2]](Shape(4, 3))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
        border = BorderMode.Replicate,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(4, 3)
      )
    val reducer =
      new DoubleNeighborhoodReducer:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          acc + value * (offsetIndex + 1).toDouble
        def finish(acc: Double): Double = acc

    val prepared =
      DirectNeighborhoodExecutor.prepare(immutable, fromImmutable, spec)
    prepared.runDouble(immutable, fromImmutable, reducer, constant = 0.0)
    prepared.runDouble(mutableSource, fromMutable, reducer, constant = 0.0)

    assert(fromMutable.freezeCopy().sameElements(fromImmutable.freezeCopy()))

  test("prepared Float ping-pong across two workspaces matches freezeCopy chaining"):
    val source =
      NDArray.tabulate[Float](3, 3)((i, j) => (10 * i + j).toFloat)
    val axis0 =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(3, 3)
      )
    val axis1 =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(0, -1), Vector(0, 0), Vector(0, 1)),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(3, 3)
      )
    val reducer =
      new FloatNeighborhoodReducer:
        def zero: Float = 0.0f
        def accumulate(acc: Float, value: Float, offsetIndex: Int): Float =
          acc + value
        def finish(acc: Float): Float = acc / 3.0f

    val freezeA = MutableNDArray.zeros[Float, Rank[2]](Shape(3, 3))
    val freezeB = MutableNDArray.zeros[Float, Rank[2]](Shape(3, 3))
    val pass0 = DirectNeighborhoodExecutor.prepare(source, freezeA, axis0)
    pass0.runFloat(source, freezeA, reducer, constant = 0.0f)
    val mid = freezeA.freezeCopy()
    val pass1 = DirectNeighborhoodExecutor.prepare(mid, freezeB, axis1)
    pass1.runFloat(mid, freezeB, reducer, constant = 0.0f)

    val workspaceA = MutableNDArray.zeros[Float, Rank[2]](Shape(3, 3))
    val workspaceB = MutableNDArray.zeros[Float, Rank[2]](Shape(3, 3))
    val mutablePass0 =
      DirectNeighborhoodExecutor.prepare(source, workspaceA, axis0)
    mutablePass0.runFloat(source, workspaceA, reducer, constant = 0.0f)
    val mutablePass1 =
      DirectNeighborhoodExecutor.prepare(workspaceA, workspaceB, axis1)
    mutablePass1.runFloat(workspaceA, workspaceB, reducer, constant = 0.0f)

    assert(workspaceB.freezeCopy().sameElements(freezeB.freezeCopy()))

  test("prepared mutable source rejects aliased destination storage"):
    val workspace = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 2))
    val spec =
      NeighborhoodSpec(
        spatialAxes = 2,
        offsets = Vector(Vector(0, 0)),
        border = BorderMode.Constant,
        outputOrigin = Vector(0, 0),
        outputSpatialShape = Vector(2, 2)
      )
    intercept[IllegalArgumentException]:
      DirectNeighborhoodExecutor.prepare(workspace, workspace, spec)

    val prepared =
      DirectNeighborhoodExecutor.prepare(
        NDArray.zeros[Double](2, 2),
        MutableNDArray.zeros[Double, Rank[2]](Shape(2, 2)),
        spec
      )
    val reducer =
      new DoubleNeighborhoodReducer:
        def zero: Double = 0.0
        def accumulate(acc: Double, value: Double, offsetIndex: Int): Double =
          value
        def finish(acc: Double): Double = acc
    intercept[IllegalArgumentException]:
      prepared.runDouble(workspace, workspace, reducer, constant = 0.0)

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
