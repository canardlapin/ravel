package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import ravel.SumAs.given
import scala.compiletime.uninitialized

/** Broad public-operation matrix used to find the next performance bottleneck.
  *
  * The benchmark is parameterized by case so the semantic registry below is also used by the parity
  * receipt. The single case lookup and closure call are constant per invocation and are
  * intentionally mirrored by the Python harness. Large-array timings remain diagnostic; focused
  * follow-up benchmarks must confirm any optimization target.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 300, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 300, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class OperationMatrixBenchmarks:
  @Param(Array("256", "1024"))
  var side: Int = 0

  @Param(
    Array(
      "contiguous_subtract_double",
      "inner_stride_multiply_double",
      "outer_stride_divide_double",
      "reverse_minimum_double",
      "transpose_maximum_double",
      "broadcast_subtract_double",
      "scalar_add_double",
      "scalar_subtract_double",
      "scalar_multiply_double",
      "scalar_divide_double",
      "scalar_minimum_double",
      "scalar_maximum_double",
      "clip_double",
      "negate_double",
      "abs_double",
      "sqrt_double",
      "exp_double",
      "log_double",
      "sin_double",
      "cos_double",
      "tan_double",
      "floor_double",
      "ceil_double",
      "equal_double",
      "not_equal_double",
      "less_inner_stride_double",
      "less_equal_outer_stride_double",
      "greater_broadcast_double",
      "greater_equal_scalar_double",
      "is_nan_double",
      "is_finite_double",
      "cast_double_int",
      "cast_float_double",
      "cast_int_long",
      "cast_long_float",
      "contiguous_add_float",
      "contiguous_multiply_float",
      "scalar_divide_float",
      "contiguous_add_int",
      "contiguous_multiply_int",
      "scalar_quot_int",
      "contiguous_add_long",
      "contiguous_multiply_long",
      "scalar_quot_long",
      "full_product_double",
      "full_min_double",
      "full_max_double",
      "full_arg_min_double",
      "full_arg_max_double",
      "full_mean_double",
      "axis0_product_double",
      "axis1_product_double",
      "axis0_min_double",
      "axis1_max_double",
      "axis0_arg_min_double",
      "axis1_arg_max_double",
      "axis0_mean_double",
      "axis1_mean_double",
      "inner_stride_product_double",
      "reverse_min_double",
      "transpose_max_double",
      "full_sum_float",
      "full_product_float",
      "full_mean_float",
      "full_sum_int",
      "full_product_int",
      "full_sum_long",
      "full_product_long",
      "sum_as_long_int",
      "sum_as_double_float",
      "inplace_add_double",
      "inplace_subtract_double",
      "inplace_multiply_double",
      "inplace_divide_double",
      "inplace_add_float",
      "inplace_add_int",
      "inplace_add_long",
      "inplace_reverse_add_double",
      "inplace_inner_stride_multiply_double"
    )
  )
  var caseName: String = ""

  private var operation: OperationMatrixCase = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val cases = OperationMatrixCases(new OperationMatrixFixture(side))
    operation = cases.find(_.name == caseName).getOrElse {
      throw new IllegalArgumentException(s"unknown operation-matrix case: $caseName")
    }

  @Benchmark
  def operation_matrix(): Any =
    operation.run()

private[bench] final case class OperationMatrixCase(
    name: String,
    family: String,
    inputDType: String,
    resultDType: String,
    inputLayout: String,
    logicalWorkUnits: Long,
    workUnit: String,
    comparison: String,
    run: () => Any,
    signature: () => Any
)

private[bench] final class OperationMatrixFixture(val side: Int):
  require(side > 0 && side % 2 == 0, s"side must be a positive even integer, got $side")
  Math.multiplyExact(side, side)

  val fullSize: Long = side.toLong * side.toLong
  val halfSize: Long = fullSize / 2L

  val doubleLeft: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0
    }
  val doubleRight: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      ((row * 43 + column * 19) % 257 - 128).toDouble / 32.0
    }
  val doublePositive: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      ((row * 23 + column * 11) % 97 + 1).toDouble / 17.0
    }
  val doubleProduct: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      1.0 + (((row * 13 + column * 7) % 17) - 8).toDouble * 1.0e-7
    }
  val doubleNonFinite: Array2[Double] =
    NDArray.tabulate(side, side) { (row, column) =>
      val index = row * side + column
      if index % 257 == 0 then Double.NaN
      else if index % 263 == 0 then Double.PositiveInfinity
      else doubleFixtureValue(row, column)
    }
  val doubleRow: Array1[Double] =
    NDArray.tabulate(side) { column =>
      ((column * 29) % 127 - 63).toDouble / 8.0
    }

  val floatLeft: Array2[Float] =
    NDArray.tabulate(side, side) { (row, column) =>
      doubleFixtureValue(row, column).toFloat
    }
  val floatRight: Array2[Float] =
    NDArray.tabulate(side, side) { (row, column) =>
      (((row * 43 + column * 19) % 257 - 128).toDouble / 32.0).toFloat
    }
  val floatProduct: Array2[Float] =
    NDArray.tabulate(side, side) { (row, column) =>
      (1.0 + (((row * 13 + column * 7) % 17) - 8).toDouble * 1.0e-7).toFloat
    }

  val intLeft: Array2[Int] =
    NDArray.tabulate(side, side)((row, column) => (row * 17 + column * 5) % 23 - 11)
  val intRight: Array2[Int] =
    NDArray.tabulate(side, side)((row, column) => (row * 7 + column * 3) % 11 - 5)
  val intPositive: Array2[Int] =
    NDArray.tabulate(side, side)((row, column) => (row * 3 + column * 5) % 7 + 1)
  val intProduct: Array2[Int] =
    NDArray.tabulate(side, side)((row, column) => if (row + column) % 3 == 0 then -1 else 1)

  val longLeft: Array2[Long] =
    NDArray.tabulate(side, side)((row, column) => ((row * 31 + column * 7) % 41 - 20).toLong)
  val longRight: Array2[Long] =
    NDArray.tabulate(side, side)((row, column) => ((row * 5 + column * 11) % 19 - 9).toLong)
  val longPositive: Array2[Long] =
    NDArray.tabulate(side, side)((row, column) => ((row * 7 + column * 3) % 9 + 1).toLong)
  val longProduct: Array2[Long] =
    NDArray.tabulate(side, side)((row, column) => if (row + column) % 5 == 0 then -1L else 1L)

  val innerDoubleLeft: Array2[Double] =
    doubleLeft.slice(axis = 1, Slice(0, side, 2))
  val innerDoubleRight: Array2[Double] =
    doubleRight.slice(axis = 1, Slice(1, side, 2))
  val outerDoubleLeft: Array2[Double] =
    doubleLeft.slice(axis = 0, Slice(0, side, 2))
  val outerDoublePositive: Array2[Double] =
    doublePositive.slice(axis = 0, Slice(1, side, 2))
  val reversedDouble: Array2[Double] = doubleLeft.reverse(axis = 1)
  val transposedDouble: Array2[Double] = doubleLeft.transpose
  val transposedRight: Array2[Double] = doubleRight.transpose

  val mutableAddDouble: MutableNDArray[Double, Rank[2]] = doubleLeft.mutableCopy
  val mutableSubtractDouble: MutableNDArray[Double, Rank[2]] = doubleLeft.mutableCopy
  val mutableMultiplyDouble: MutableNDArray[Double, Rank[2]] = doubleLeft.mutableCopy
  val mutableDivideDouble: MutableNDArray[Double, Rank[2]] = doublePositive.mutableCopy
  val mutableAddFloat: MutableNDArray[Float, Rank[2]] = floatLeft.mutableCopy
  val mutableAddInt: MutableNDArray[Int, Rank[2]] = intLeft.mutableCopy
  val mutableAddLong: MutableNDArray[Long, Rank[2]] = longLeft.mutableCopy
  val mutableReverseDouble: MutableNDArray[Double, Rank[2]] =
    doubleLeft.mutableCopy.reverse(axis = 1)
  val mutableInnerDouble: MutableNDArray[Double, Rank[2]] =
    doubleProduct.mutableCopy.slice(axis = 1, Slice(0, side, 2))

  private def doubleFixtureValue(row: Int, column: Int): Double =
    ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0

private[bench] object OperationMatrixCases:
  def apply(fixture: OperationMatrixFixture): Vector[OperationMatrixCase] =
    import fixture.*

    Vector(
      valueCase(
        "contiguous_subtract_double",
        "binary",
        "float64",
        "float64",
        "contiguous",
        fullSize
      )(
        doubleLeft - doubleRight
      ),
      valueCase(
        "inner_stride_multiply_double",
        "binary",
        "float64",
        "float64",
        "inner_stride",
        halfSize
      )(innerDoubleLeft * innerDoubleRight),
      valueCase(
        "outer_stride_divide_double",
        "binary",
        "float64",
        "float64",
        "outer_stride",
        halfSize
      )(outerDoubleLeft / outerDoublePositive),
      valueCase("reverse_minimum_double", "binary", "float64", "float64", "reversed", fullSize)(
        reversedDouble.minimum(doubleRight)
      ),
      valueCase(
        "transpose_maximum_double",
        "binary",
        "float64",
        "float64",
        "transposed",
        fullSize
      )(transposedDouble.maximum(transposedRight)),
      valueCase(
        "broadcast_subtract_double",
        "binary",
        "float64",
        "float64",
        "broadcast",
        fullSize
      )(doubleLeft - doubleRow),
      valueCase("scalar_add_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft + 1.25
      ),
      valueCase("scalar_subtract_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft - 0.75
      ),
      valueCase("scalar_multiply_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft * -1.5
      ),
      valueCase("scalar_divide_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft / 3.0
      ),
      valueCase("scalar_minimum_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.minimum(0.5)
      ),
      valueCase("scalar_maximum_double", "scalar", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.maximum(-0.5)
      ),
      valueCase("clip_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.clip(-2.0, 3.0)
      ),
      valueCase("negate_double", "unary", "float64", "float64", "contiguous", fullSize)(
        -doubleLeft
      ),
      valueCase("abs_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.abs
      ),
      valueCase("sqrt_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doublePositive.sqrt
      ),
      valueCase("exp_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.exp
      ),
      valueCase("log_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doublePositive.log
      ),
      valueCase("sin_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.sin
      ),
      valueCase("cos_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.cos
      ),
      valueCase("tan_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.tan
      ),
      valueCase("floor_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.floor
      ),
      valueCase("ceil_double", "unary", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.ceil
      ),
      exactCase("equal_double", "comparison", "float64", "bool", "contiguous", fullSize)(
        doubleLeft === doubleRight
      ),
      exactCase("not_equal_double", "comparison", "float64", "bool", "contiguous", fullSize)(
        doubleLeft =!= doubleRight
      ),
      exactCase(
        "less_inner_stride_double",
        "comparison",
        "float64",
        "bool",
        "inner_stride",
        halfSize
      )(innerDoubleLeft < innerDoubleRight),
      exactCase(
        "less_equal_outer_stride_double",
        "comparison",
        "float64",
        "bool",
        "outer_stride",
        halfSize
      )(outerDoubleLeft <= outerDoublePositive),
      exactCase("greater_broadcast_double", "comparison", "float64", "bool", "broadcast", fullSize)(
        doubleLeft > doubleRow
      ),
      exactCase(
        "greater_equal_scalar_double",
        "comparison",
        "float64",
        "bool",
        "contiguous",
        fullSize
      )(doubleLeft >= 0.0),
      exactCase("is_nan_double", "predicate", "float64", "bool", "contiguous", fullSize)(
        doubleNonFinite.isNaN
      ),
      exactCase("is_finite_double", "predicate", "float64", "bool", "contiguous", fullSize)(
        doubleNonFinite.isFinite
      ),
      exactCase("cast_double_int", "cast", "float64", "int32", "contiguous", fullSize)(
        doubleLeft.cast[Int]
      ),
      valueCase("cast_float_double", "cast", "float32", "float64", "contiguous", fullSize)(
        floatLeft.cast[Double]
      ),
      exactCase("cast_int_long", "cast", "int32", "int64", "contiguous", fullSize)(
        intLeft.cast[Long]
      ),
      valueCase("cast_long_float", "cast", "int64", "float32", "contiguous", fullSize)(
        longLeft.cast[Float]
      ),
      valueCase("contiguous_add_float", "binary", "float32", "float32", "contiguous", fullSize)(
        floatLeft + floatRight
      ),
      valueCase(
        "contiguous_multiply_float",
        "binary",
        "float32",
        "float32",
        "contiguous",
        fullSize
      )(floatLeft * floatRight),
      valueCase("scalar_divide_float", "scalar", "float32", "float32", "contiguous", fullSize)(
        floatLeft / 3.0f
      ),
      exactCase("contiguous_add_int", "binary", "int32", "int32", "contiguous", fullSize)(
        intLeft + intRight
      ),
      exactCase(
        "contiguous_multiply_int",
        "binary",
        "int32",
        "int32",
        "contiguous",
        fullSize
      )(intLeft * intRight),
      exactCase("scalar_quot_int", "scalar", "int32", "int32", "contiguous", fullSize)(
        intPositive.quot(3)
      ),
      exactCase("contiguous_add_long", "binary", "int64", "int64", "contiguous", fullSize)(
        longLeft + longRight
      ),
      exactCase(
        "contiguous_multiply_long",
        "binary",
        "int64",
        "int64",
        "contiguous",
        fullSize
      )(longLeft * longRight),
      exactCase("scalar_quot_long", "scalar", "int64", "int64", "contiguous", fullSize)(
        longPositive.quot(3L)
      ),
      valueCase("full_product_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleProduct.product
      ),
      valueCase("full_min_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.min
      ),
      valueCase("full_max_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.max
      ),
      exactCase("full_arg_min_double", "reduction", "float64", "int32", "contiguous", fullSize)(
        doubleLeft.argMin
      ),
      exactCase("full_arg_max_double", "reduction", "float64", "int32", "contiguous", fullSize)(
        doubleLeft.argMax
      ),
      valueCase("full_mean_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.mean
      ),
      valueCase("axis0_product_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleProduct.product(axis = 0)
      ),
      valueCase("axis1_product_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleProduct.product(axis = 1)
      ),
      valueCase("axis0_min_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.min(axis = 0)
      ),
      valueCase("axis1_max_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.max(axis = 1)
      ),
      exactCase("axis0_arg_min_double", "reduction", "float64", "int32", "contiguous", fullSize)(
        doubleLeft.argMin(axis = 0)
      ),
      exactCase("axis1_arg_max_double", "reduction", "float64", "int32", "contiguous", fullSize)(
        doubleLeft.argMax(axis = 1)
      ),
      valueCase("axis0_mean_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.mean(axis = 0)
      ),
      valueCase("axis1_mean_double", "reduction", "float64", "float64", "contiguous", fullSize)(
        doubleLeft.mean(axis = 1)
      ),
      valueCase(
        "inner_stride_product_double",
        "reduction",
        "float64",
        "float64",
        "inner_stride",
        halfSize
      )(doubleProduct.slice(axis = 1, Slice(0, side, 2)).product),
      valueCase("reverse_min_double", "reduction", "float64", "float64", "reversed", fullSize)(
        reversedDouble.min
      ),
      valueCase(
        "transpose_max_double",
        "reduction",
        "float64",
        "float64",
        "transposed",
        fullSize
      )(transposedDouble.max),
      valueCase("full_sum_float", "reduction", "float32", "float32", "contiguous", fullSize)(
        floatLeft.sum
      ),
      valueCase(
        "full_product_float",
        "reduction",
        "float32",
        "float32",
        "contiguous",
        fullSize
      )(floatProduct.product),
      valueCase("full_mean_float", "reduction", "float32", "float32", "contiguous", fullSize)(
        floatLeft.mean
      ),
      exactCase("full_sum_int", "reduction", "int32", "int32", "contiguous", fullSize)(
        intLeft.sum
      ),
      exactCase("full_product_int", "reduction", "int32", "int32", "contiguous", fullSize)(
        intProduct.product
      ),
      exactCase("full_sum_long", "reduction", "int64", "int64", "contiguous", fullSize)(
        longLeft.sum
      ),
      exactCase("full_product_long", "reduction", "int64", "int64", "contiguous", fullSize)(
        longProduct.product
      ),
      exactCase("sum_as_long_int", "reduction", "int32", "int64", "contiguous", fullSize)(
        intLeft.sumAs[Long]
      ),
      valueCase("sum_as_double_float", "reduction", "float32", "float64", "contiguous", fullSize)(
        floatLeft.sumAs[Double]
      ),
      inPlaceCase("inplace_add_double", "float64", "contiguous", fullSize)(
        {
          mutableAddDouble.addInPlace(0.25)
          mutableAddDouble(side - 1, side - 1)
        }
      ) {
        mutableAddDouble.addInPlace(0.25)
        mutableAddDouble.freezeCopy()
      },
      inPlaceCase("inplace_subtract_double", "float64", "contiguous", fullSize)(
        {
          mutableSubtractDouble.subtractInPlace(0.125)
          mutableSubtractDouble(side - 1, side - 1)
        }
      ) {
        mutableSubtractDouble.subtractInPlace(0.125)
        mutableSubtractDouble.freezeCopy()
      },
      inPlaceCase("inplace_multiply_double", "float64", "contiguous", fullSize)(
        {
          mutableMultiplyDouble.multiplyInPlace(-1.0)
          mutableMultiplyDouble(side - 1, side - 1)
        }
      ) {
        mutableMultiplyDouble.multiplyInPlace(-1.0)
        mutableMultiplyDouble.freezeCopy()
      },
      inPlaceCase("inplace_divide_double", "float64", "contiguous", fullSize)(
        {
          mutableDivideDouble.divideInPlace(-1.0)
          mutableDivideDouble(side - 1, side - 1)
        }
      ) {
        mutableDivideDouble.divideInPlace(-1.0)
        mutableDivideDouble.freezeCopy()
      },
      inPlaceCase("inplace_add_float", "float32", "contiguous", fullSize)(
        {
          mutableAddFloat.addInPlace(0.25f)
          mutableAddFloat(side - 1, side - 1)
        }
      ) {
        mutableAddFloat.addInPlace(0.25f)
        mutableAddFloat.freezeCopy()
      },
      inPlaceExactCase("inplace_add_int", "int32", "contiguous", fullSize)(
        {
          mutableAddInt.addInPlace(1)
          mutableAddInt(side - 1, side - 1)
        }
      ) {
        mutableAddInt.addInPlace(1)
        mutableAddInt.freezeCopy()
      },
      inPlaceExactCase("inplace_add_long", "int64", "contiguous", fullSize)(
        {
          mutableAddLong.addInPlace(1L)
          mutableAddLong(side - 1, side - 1)
        }
      ) {
        mutableAddLong.addInPlace(1L)
        mutableAddLong.freezeCopy()
      },
      inPlaceCase("inplace_reverse_add_double", "float64", "reversed", fullSize)(
        {
          mutableReverseDouble.addInPlace(0.25)
          mutableReverseDouble(0, 0)
        }
      ) {
        mutableReverseDouble.addInPlace(0.25)
        mutableReverseDouble.freezeCopy()
      },
      inPlaceCase(
        "inplace_inner_stride_multiply_double",
        "float64",
        "inner_stride",
        halfSize
      )(
        {
          mutableInnerDouble.multiplyInPlace(-1.0)
          mutableInnerDouble(side - 1, side / 2 - 1)
        }
      ) {
        mutableInnerDouble.multiplyInPlace(-1.0)
        mutableInnerDouble.freezeCopy()
      }
    )

  private def valueCase(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      work: Long
  )(body: => Any): OperationMatrixCase =
    val thunk = () => body
    OperationMatrixCase(
      name,
      family,
      inputDType,
      resultDType,
      inputLayout,
      work,
      "element",
      "floating",
      thunk,
      thunk
    )

  private def exactCase(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      work: Long
  )(body: => Any): OperationMatrixCase =
    val thunk = () => body
    OperationMatrixCase(
      name,
      family,
      inputDType,
      resultDType,
      inputLayout,
      work,
      "element",
      "exact",
      thunk,
      thunk
    )

  private def inPlaceCase(
      name: String,
      dtype: String,
      inputLayout: String,
      work: Long
  )(body: => Any)(signatureBody: => Any): OperationMatrixCase =
    OperationMatrixCase(
      name,
      "in-place",
      dtype,
      dtype,
      inputLayout,
      work,
      "element",
      "floating",
      () => body,
      () => signatureBody
    )

  private def inPlaceExactCase(
      name: String,
      dtype: String,
      inputLayout: String,
      work: Long
  )(body: => Any)(signatureBody: => Any): OperationMatrixCase =
    OperationMatrixCase(
      name,
      "in-place",
      dtype,
      dtype,
      inputLayout,
      work,
      "element",
      "exact",
      () => body,
      () => signatureBody
    )
