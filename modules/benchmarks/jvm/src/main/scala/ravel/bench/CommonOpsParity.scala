package ravel.bench

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import ravel.*
import ravel.DType.given
import ravel.SumAs.given

/** Elementwise NumPy oracle for common public 0.1 operations.
  *
  * Fixtures stay small so CI can compare every logical value, not just checksums. The existing
  * operation-matrix signatures remain the performance locator.
  */
object CommonOpsParity:
  private final case class Config(
      output: Path = Paths.get("target/common-ops/ravel-signatures.json")
  )

  final case class Case(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      comparison: String,
      run: () => Any
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args)
    val rows = cases.map(row)
    val names = rows.map(_("case"))
    require(names.distinct.size == names.size, "duplicate common-ops case name")
    val parent = config.output.toAbsolutePath.getParent
    if parent != null then
      val _ = Files.createDirectories(parent)
    val _ = Files.writeString(config.output, render(rows), StandardCharsets.UTF_8)
    println(s"Wrote ${rows.size} common-ops signatures to ${config.output}")

  def cases: Vector[Case] =
    val f = Fixture()
    Vector(
      exact(
        "empty_int_sum",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.sum
      ),
      exact(
        "empty_bool_all",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.emptyBool.all
      ),
      exact(
        "empty_bool_any",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.emptyBool.any
      ),
      exact(
        "rank1_int_add",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.int1Left + f.int1Right
      ),
      exact(
        "rank2_int_add",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left + f.int2Right
      ),
      exact(
        "rank2_int_multiply",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left * f.int2Right
      ),
      exact(
        "rank2_int_quot",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.quot(f.int2Positive)
      ),
      exact(
        "broadcast_int_add",
        "binary",
        "int32",
        "int32",
        "broadcast",
        () => f.int2Left + f.intRow
      ),
      exact(
        "rank2_int_sum",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sum
      ),
      exact(
        "rank2_int_sum_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sum(0)
      ),
      exact(
        "rank2_int_sum_keepdims",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sumKeepDims(1)
      ),
      exact(
        "rank3_int_sum_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int3.sum(0)
      ),
      exact(
        "rank2_int_min",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.minimum(f.int2Right)
      ),
      exact(
        "rank2_int_less",
        "comparison",
        "int32",
        "bool",
        "contiguous",
        () => f.int2Left < f.int2Right
      ),
      exact(
        "inner_stride_int_add",
        "binary",
        "int32",
        "int32",
        "inner_stride",
        () => f.intInnerLeft + f.intInnerRight
      ),
      exact(
        "reverse_int_maximum",
        "binary",
        "int32",
        "int32",
        "reversed",
        () => f.intReversed.maximum(f.int2Right)
      ),
      exact(
        "transpose_int_equal",
        "comparison",
        "int32",
        "bool",
        "transposed",
        () => f.intTransposed === f.int2Right.transpose
      ),
      exact(
        "bool_all",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.bool2.all
      ),
      exact(
        "bool_any",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.bool2.any
      ),
      exact(
        "bool_all_axis0",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.bool2.all(0)
      ),
      exact(
        "bool_any_keepdims",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.bool2.anyKeepDims(1)
      ),
      exact(
        "bool_count_true",
        "reduction",
        "bool",
        "int32",
        "contiguous",
        () => f.bool2.countTrue
      ),
      exact(
        "rank3_bool_all_axes",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.bool3.all(Axes.require(3, 0, 1))
      ),
      exact(
        "uint8_minimum",
        "binary",
        "uint8",
        "uint8",
        "contiguous",
        () => f.uint8Left.minimum(f.uint8Right)
      ),
      exact(
        "uint8_clip",
        "unary",
        "uint8",
        "uint8",
        "contiguous",
        () => f.uint8Left.clip(UInt8.unsafe(40), UInt8.unsafe(80))
      ),
      exact(
        "uint8_greater_equal",
        "comparison",
        "uint8",
        "bool",
        "contiguous",
        () => f.uint8Left >= UInt8.unsafe(50)
      ),
      exact(
        "uint8_max_axis1",
        "reduction",
        "uint8",
        "uint8",
        "contiguous",
        () => f.uint8Left.max(1)
      ),
      exact(
        "inner_stride_uint8_min",
        "binary",
        "uint8",
        "uint8",
        "inner_stride",
        () => f.uint8InnerLeft.minimum(f.uint8InnerRight)
      ),
      exact(
        "uint8_cast_float32",
        "cast",
        "uint8",
        "float32",
        "contiguous",
        () => f.uint8Left.cast[Float]
      ),
      exact(
        "uint16_maximum",
        "binary",
        "uint16",
        "uint16",
        "contiguous",
        () => f.uint16Left.maximum(f.uint16Right)
      ),
      exact(
        "uint16_less",
        "comparison",
        "uint16",
        "bool",
        "contiguous",
        () => f.uint16Left < f.uint16Right
      ),
      exact(
        "uint16_cast_int32",
        "cast",
        "uint16",
        "int32",
        "contiguous",
        () => f.uint16Left.cast[Int]
      ),
      exact(
        "cast_float64_int32",
        "cast",
        "float64",
        "int32",
        "contiguous",
        () => f.doubleTrunc.cast[Int]
      ),
      floating(
        "rank2_double_add",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft + f.doubleRight
      ),
      floating(
        "broadcast_double_subtract",
        "binary",
        "float64",
        "float64",
        "broadcast",
        () => f.doubleLeft - f.doubleRow
      ),
      floating(
        "inner_stride_double_multiply",
        "binary",
        "float64",
        "float64",
        "inner_stride",
        () => f.doubleInnerLeft * f.doubleInnerRight
      ),
      floating(
        "reverse_double_divide",
        "binary",
        "float64",
        "float64",
        "reversed",
        () => f.doubleReversed / f.doublePositive
      ),
      floating(
        "transpose_double_minimum",
        "binary",
        "float64",
        "float64",
        "transposed",
        () => f.doubleTransposed.minimum(f.doubleRight.transpose)
      ),
      floating(
        "double_abs",
        "unary",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft.abs
      ),
      floating(
        "double_clip",
        "unary",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft.clip(-2.0, 3.0)
      ),
      exact(
        "double_less_equal",
        "comparison",
        "float64",
        "bool",
        "contiguous",
        () => f.doubleLeft <= f.doubleRight
      ),
      exact(
        "double_is_nan",
        "predicate",
        "float64",
        "bool",
        "contiguous",
        () => f.doubleNonFinite.isNaN
      ),
      exact(
        "double_is_finite",
        "predicate",
        "float64",
        "bool",
        "contiguous",
        () => f.doubleNonFinite.isFinite
      ),
      floating(
        "double_mean",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft.mean
      ),
      floating(
        "double_mean_axis0",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft.mean(0)
      ),
      floating(
        "empty_float_mean",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.emptyDouble.mean
      ),
      floating(
        "rank3_double_mean_axis2",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.double3.mean(2)
      ),
      floating(
        "rank2_float32_add",
        "binary",
        "float32",
        "float32",
        "contiguous",
        () => f.floatLeft + f.floatRight
      ),
      exact(
        "scalar_add_int",
        "scalar",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left + 1
      ),
      exact(
        "scalar_subtract_int",
        "scalar",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left - 2
      ),
      exact(
        "empty_int_product",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.product
      ),
      exact(
        "rank2_int_product",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Sign.product
      ),
      exact(
        "rank2_int_arg_min",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.argMin
      ),
      exact(
        "rank2_int_arg_max",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.argMax
      ),
      exact(
        "rank2_int_arg_min_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.argMin(0)
      ),
      exact(
        "rank2_int_sum_as_long",
        "reduction",
        "int32",
        "int64",
        "contiguous",
        () => f.int2Left.sumAs[Long]
      ),
      exact(
        "byte_minimum",
        "binary",
        "int8",
        "int8",
        "contiguous",
        () => f.byteLeft.minimum(f.byteRight)
      ),
      exact(
        "byte_clip",
        "unary",
        "int8",
        "int8",
        "contiguous",
        () => f.byteLeft.clip(-4.toByte, 6.toByte)
      ),
      exact(
        "byte_less",
        "comparison",
        "int8",
        "bool",
        "contiguous",
        () => f.byteLeft < f.byteRight
      ),
      exact(
        "byte_min",
        "reduction",
        "int8",
        "int8",
        "contiguous",
        () => f.byteLeft.min
      ),
      exact(
        "byte_arg_max",
        "reduction",
        "int8",
        "int32",
        "contiguous",
        () => f.byteLeft.argMax
      ),
      exact(
        "byte_cast_int32",
        "cast",
        "int8",
        "int32",
        "contiguous",
        () => f.byteLeft.cast[Int]
      ),
      exact(
        "short_maximum",
        "binary",
        "int16",
        "int16",
        "contiguous",
        () => f.shortLeft.maximum(f.shortRight)
      ),
      exact(
        "short_greater_equal",
        "comparison",
        "int16",
        "bool",
        "contiguous",
        () => f.shortLeft >= 0.toShort
      ),
      exact(
        "short_arg_min",
        "reduction",
        "int16",
        "int32",
        "contiguous",
        () => f.shortLeft.argMin
      ),
      exact(
        "short_max_axis0",
        "reduction",
        "int16",
        "int16",
        "contiguous",
        () => f.shortLeft.max(0)
      ),
      exact(
        "short_cast_float64",
        "cast",
        "int16",
        "float64",
        "contiguous",
        () => f.shortLeft.cast[Double]
      ),
      exact(
        "rank2_long_add",
        "binary",
        "int64",
        "int64",
        "contiguous",
        () => f.longLeft + f.longRight
      ),
      exact(
        "rank2_long_multiply",
        "binary",
        "int64",
        "int64",
        "contiguous",
        () => f.longLeft * f.longRight
      ),
      exact(
        "scalar_add_long",
        "scalar",
        "int64",
        "int64",
        "contiguous",
        () => f.longLeft + 1L
      ),
      exact(
        "rank2_long_quot",
        "binary",
        "int64",
        "int64",
        "contiguous",
        () => f.longLeft.quot(f.longPositive)
      ),
      exact(
        "rank2_long_sum",
        "reduction",
        "int64",
        "int64",
        "contiguous",
        () => f.longLeft.sum
      ),
      exact(
        "rank2_long_product",
        "reduction",
        "int64",
        "int64",
        "contiguous",
        () => f.longSign.product
      ),
      exact(
        "long_arg_min_axis1",
        "reduction",
        "int64",
        "int32",
        "contiguous",
        () => f.longLeft.argMin(1)
      ),
      exact(
        "scalar_add_double",
        "scalar",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft + 0.25
      ),
      exact(
        "rank2_double_product",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleSign.product
      ),
      floating(
        "scalar_divide_float",
        "scalar",
        "float32",
        "float32",
        "contiguous",
        () => f.floatLeft / 2.0f
      ),
      floating(
        "float_sum_as_double",
        "reduction",
        "float32",
        "float64",
        "contiguous",
        () => f.floatLeft.sumAs[Double]
      ),
      exact(
        "byte_convert_int32",
        "convert",
        "int8",
        "int32",
        "contiguous",
        () => f.byteLeft.convert[Int]()
      ),
      exact(
        "byte_convert_int64",
        "convert",
        "int8",
        "int64",
        "contiguous",
        () => f.byteLeft.convert[Long]()
      ),
      exact(
        "uint8_convert_int32",
        "convert",
        "uint8",
        "int32",
        "contiguous",
        () => f.uint8Left.convert[Int]()
      ),
      exact(
        "uint16_convert_int64",
        "convert",
        "uint16",
        "int64",
        "contiguous",
        () => f.uint16Left.convert[Long]()
      ),
      exact(
        "int32_convert_int64",
        "convert",
        "int32",
        "int64",
        "contiguous",
        () => f.int2Left.convert[Long]()
      ),
      exact(
        "int32_convert_float64",
        "convert",
        "int32",
        "float64",
        "contiguous",
        () => f.int2Left.convert[Double]()
      ),
      exact(
        "uint8_convert_float32",
        "convert",
        "uint8",
        "float32",
        "contiguous",
        () => f.uint8Left.convert[Float]()
      ),
      exact(
        "short_convert_int8",
        "convert",
        "int16",
        "int8",
        "contiguous",
        () => f.shortLeft.convert[Byte]()
      ),
      exact(
        "uint8_convert_int8",
        "convert",
        "uint8",
        "int8",
        "contiguous",
        () => f.uint8Left.convert[Byte]()
      ),
      exact(
        "uint16_convert_uint8",
        "convert",
        "uint16",
        "uint8",
        "contiguous",
        () => f.uint16Left.convert[UInt8]()
      ),
      exact(
        "int32_convert_uint8_inrange",
        "convert",
        "int32",
        "uint8",
        "contiguous",
        () => f.convertInRangeUInt8.convert[UInt8]()
      ),
      exact(
        "long_convert_int32",
        "convert",
        "int64",
        "int32",
        "contiguous",
        () => f.longLeft.convert[Int]()
      ),
      exact(
        "float32_convert_float64",
        "convert",
        "float32",
        "float64",
        "contiguous",
        () => f.floatLeft.convert[Double]()
      ),
      exact(
        "float64_convert_float32_inrange",
        "convert",
        "float64",
        "float32",
        "contiguous",
        () => f.doubleLeft.convert[Float]()
      ),
      exact(
        "reversed_int32_convert_int8",
        "convert",
        "int32",
        "int8",
        "reversed",
        () => f.intReversed.convert[Byte]()
      ),
      exact(
        "double_convert_int32_nearest_even",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () => f.convertRound.convert[Int]()
      ),
      exact(
        "double_convert_int32_toward_zero",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () =>
          f.convertToward.convert[Int](
            ConversionPolicy(Rounding.TowardZero, Overflow.Reject)
          )
      ),
      exact(
        "double_convert_int32_floor",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () =>
          f.convertToward.convert[Int](
            ConversionPolicy(Rounding.Floor, Overflow.Reject)
          )
      ),
      exact(
        "double_convert_int32_ceiling",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () =>
          f.convertToward.convert[Int](
            ConversionPolicy(Rounding.Ceiling, Overflow.Reject)
          )
      ),
      exact(
        "double_convert_uint8_inrange",
        "convert",
        "float64",
        "uint8",
        "contiguous",
        () => f.convertInRangeUInt8Float.convert[UInt8]()
      ),
      exact(
        "int32_convert_int8_reject",
        "convert",
        "int32",
        "int8",
        "contiguous",
        () => f.convertWideInt.convert[Byte]()
      ),
      exact(
        "int32_convert_uint8_reject",
        "convert",
        "int32",
        "uint8",
        "contiguous",
        () => f.convertNegInt.convert[UInt8]()
      ),
      exact(
        "uint16_convert_uint8_reject",
        "convert",
        "uint16",
        "uint8",
        "contiguous",
        () => f.convertWideUInt16.convert[UInt8]()
      ),
      exact(
        "long_convert_int32_reject",
        "convert",
        "int64",
        "int32",
        "contiguous",
        () => f.convertWideLong.convert[Int]()
      ),
      exact(
        "double_convert_int8_reject",
        "convert",
        "float64",
        "int8",
        "contiguous",
        () => f.convertOverflowByte.convert[Byte]()
      ),
      exact(
        "double_convert_int32_nan_reject",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () => f.convertNan.convert[Int]()
      ),
      exact(
        "double_convert_int32_inf_reject",
        "convert",
        "float64",
        "int32",
        "contiguous",
        () => f.convertInf.convert[Int]()
      ),
      exact(
        "double_convert_float32_reject",
        "convert",
        "float64",
        "float32",
        "contiguous",
        () => f.convertHugeDouble.convert[Float]()
      ),
      exact(
        "float32_convert_int8_reject",
        "convert",
        "float32",
        "int8",
        "contiguous",
        () => f.convertOverflowFloat.convert[Byte]()
      ),
      exact(
        "reversed_int32_convert_int8_reject",
        "convert",
        "int32",
        "int8",
        "reversed",
        () => f.convertIntOverflowReversed.convert[Byte]()
      ),
      exact(
        "empty_int_min_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.min
      ),
      exact(
        "empty_int_max_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.max
      ),
      exact(
        "empty_int_arg_min_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.argMin
      ),
      exact(
        "empty_int_arg_max_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyInt.argMax
      ),
      exact(
        "empty_fiber_int_sum",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberInt.sum(1)
      ),
      exact(
        "empty_fiber_int_product",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberInt.product(1)
      ),
      exact(
        "empty_fiber_int_min_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberInt.min(1)
      ),
      exact(
        "empty_fiber_int_arg_max_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberInt.argMax(1)
      ),
      floating(
        "empty_fiber_double_mean",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.emptyFiberDouble.mean(1)
      ),
      exact(
        "empty_fiber_rank3_sum",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberRank3.sum(1)
      ),
      exact(
        "empty_fiber_rank3_min_reject",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberRank3.min(1)
      ),
      exact(
        "empty_output_int_sum_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyRowsInt.sum(0)
      ),
      exact(
        "empty_output_int_sum_axis1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyRowsInt.sum(1)
      ),
      exact(
        "empty_output_int_min_axis1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyRowsInt.min(1)
      ),
      exact(
        "empty_output_rank3_min_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberRank3.min(0)
      ),
      exact(
        "empty_fiber_bool_all",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.emptyFiberBool.all(1)
      ),
      exact(
        "empty_fiber_bool_any",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.emptyFiberBool.any(1)
      ),
      exact(
        "rank2_int_sum_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sum(-1)
      ),
      exact(
        "rank2_int_sum_axis_neg2",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sum(-2)
      ),
      exact(
        "rank2_int_product_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Sign.product(-1)
      ),
      exact(
        "rank2_int_min_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.min(-1)
      ),
      exact(
        "rank2_int_max_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.max(-1)
      ),
      exact(
        "rank2_int_arg_min_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.argMin(-1)
      ),
      exact(
        "rank2_int_arg_max_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.argMax(-1)
      ),
      exact(
        "rank2_int_sum_keepdims_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int2Left.sumKeepDims(-1)
      ),
      exact(
        "rank3_int_sum_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.int3.sum(-1)
      ),
      floating(
        "rank2_double_mean_axis_neg1",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.doubleLeft.mean(-1)
      ),
      exact(
        "uint8_max_axis_neg1",
        "reduction",
        "uint8",
        "uint8",
        "contiguous",
        () => f.uint8Left.max(-1)
      ),
      exact(
        "long_arg_min_axis_neg1",
        "reduction",
        "int64",
        "int32",
        "contiguous",
        () => f.longLeft.argMin(-1)
      ),
      exact(
        "empty_fiber_int_sum_axis_neg1",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.emptyFiberInt.sum(-1)
      ),
      exact(
        "empty_fiber_bool_all_axis_neg2",
        "reduction",
        "bool",
        "bool",
        "contiguous",
        () => f.emptyFiberBool.all(-2)
      ),
      floating(
        "nan_add",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.nanLeft + f.nanRight
      ),
      floating(
        "nan_multiply",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.nanLeft * f.nanRight
      ),
      floating(
        "nan_minimum",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.nanLeft.minimum(f.nanRight)
      ),
      floating(
        "nan_min",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.nanLeft.min
      ),
      floating(
        "nan_max",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.nanLeft.max
      ),
      exact(
        "nan_arg_min",
        "reduction",
        "float64",
        "int32",
        "contiguous",
        () => f.nanTies.argMin
      ),
      exact(
        "nan_arg_max",
        "reduction",
        "float64",
        "int32",
        "contiguous",
        () => f.nanTies.argMax
      ),
      exact(
        "nan_less",
        "comparison",
        "float64",
        "bool",
        "contiguous",
        () => f.nanLeft < f.nanRight
      ),
      exact(
        "nan_equal",
        "comparison",
        "float64",
        "bool",
        "contiguous",
        () => f.nanLeft === f.nanRight
      ),
      floating(
        "nan_min_axis1",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.nanMatrix.min(1)
      ),
      floating(
        "inf_minus_inf",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.infLeft - f.infRight
      ),
      bits(
        "one_div_zero",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.oneDiv / f.zeroDiv
      ),
      bits(
        "signed_zero_min",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZero.min
      ),
      bits(
        "signed_zero_max",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZero.max
      ),
      exact(
        "signed_zero_arg_min",
        "reduction",
        "float64",
        "int32",
        "contiguous",
        () => f.signedZero.argMin
      ),
      exact(
        "signed_zero_arg_max",
        "reduction",
        "float64",
        "int32",
        "contiguous",
        () => f.signedZero.argMax
      ),
      bits(
        "signed_zero_minimum",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZeroLeft.minimum(f.signedZeroRight)
      ),
      bits(
        "signed_zero_add",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZeroLeft + f.signedZeroRight
      ),
      bits(
        "signed_zero_multiply",
        "binary",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZeroMulLeft * f.signedZeroMulRight
      ),
      bits(
        "signed_zero_min_axis0",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        () => f.signedZeroMatrix.min(0)
      ),
      exact(
        "int_sum_overflow",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.intOverflow.sum
      ),
      exact(
        "int_sum_overflow_axis0",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.intOverflowMatrix.sum(0)
      ),
      exact(
        "int_product_overflow",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        () => f.intOverflowProduct.product
      ),
      exact(
        "int_add_overflow",
        "binary",
        "int32",
        "int32",
        "contiguous",
        () => f.intOverflowLeft + f.intOverflowRight
      ),
      exact(
        "long_sum_overflow",
        "reduction",
        "int64",
        "int64",
        "contiguous",
        () => f.longOverflow.sum
      ),
      exact(
        "long_add_overflow",
        "binary",
        "int64",
        "int64",
        "contiguous",
        () => f.longOverflowLeft + f.longOverflowRight
      ),
      floating(
        "clip_nan",
        "unary",
        "float64",
        "float64",
        "contiguous",
        () => f.clipAdversarial.clip(-2.0, 3.0)
      ),
      bits(
        "clip_inf",
        "unary",
        "float64",
        "float64",
        "contiguous",
        () => f.clipInf.clip(-2.0, 3.0)
      ),
      exact(
        "int_clip_edges",
        "unary",
        "int32",
        "int32",
        "contiguous",
        () => f.intClipEdges.clip(-10, 10)
      )
    )

  private def exact(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      run: () => Any
  ): Case =
    Case(name, family, inputDType, resultDType, inputLayout, "exact", run)

  private def floating(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      run: () => Any
  ): Case =
    Case(name, family, inputDType, resultDType, inputLayout, "floating", run)

  private def bits(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      run: () => Any
  ): Case =
    Case(name, family, inputDType, resultDType, inputLayout, "bits", run)

  private final class Fixture:
    val emptyInt: Array1[Int] = NDArray.zeros[Int](0)
    val emptyBool: Array1[Boolean] = NDArray.zeros[Boolean](0)
    val emptyDouble: Array1[Double] = NDArray.zeros[Double](0)
    val emptyFiberInt: Array2[Int] = NDArray.zeros[Int](3, 0)
    val emptyFiberDouble: Array2[Double] = NDArray.zeros[Double](3, 0)
    val emptyFiberRank3: Array3[Int] = NDArray.zeros[Int](2, 0, 3)
    val emptyRowsInt: Array2[Int] = NDArray.zeros[Int](0, 5)
    val emptyFiberBool: Array3[Boolean] = NDArray.zeros[Boolean](2, 0, 3)

    val int1Left: Array1[Int] =
      NDArray.tabulate[Int](7)(i => (i * 17) % 23 - 11)
    val int1Right: Array1[Int] =
      NDArray.tabulate[Int](7)(i => (i * 7) % 11 - 5)

    val int2Left: Array2[Int] =
      NDArray.tabulate[Int](4, 3)((i, j) => (i * 17 + j * 5) % 23 - 11)
    val int2Right: Array2[Int] =
      NDArray.tabulate[Int](4, 3)((i, j) => (i * 7 + j * 3) % 11 - 5)
    val int2Positive: Array2[Int] =
      NDArray.tabulate[Int](4, 3)((i, j) => (i * 3 + j * 5) % 7 + 1)
    val intRow: Array1[Int] =
      NDArray.tabulate[Int](3)(j => (j * 5) % 7 - 3)
    val int3: Array3[Int] =
      NDArray.tabulate[Int](2, 3, 2)((i, j, k) => i * 10 + j * 3 + k)
    val intWide: Array2[Int] =
      NDArray.tabulate[Int](4, 6)((i, j) => (i * 17 + j * 5) % 23 - 11)
    val intInnerLeft: Array2[Int] =
      intWide.slice(1, Slice.every(2))
    val intInnerRight: Array2[Int] =
      NDArray
        .tabulate[Int](4, 6)((i, j) => (i * 7 + j * 3) % 11 - 5)
        .slice(1, Slice(1, 6, 2))
    val intReversed: Array2[Int] =
      int2Left.reverse(1)
    val intTransposed: Array2[Int] =
      int2Left.transpose
    val int2Sign: Array2[Int] =
      NDArray.tabulate[Int](4, 3)((i, j) => if (i + j) % 3 == 0 then -1 else 1)

    val byteLeft: Array2[Byte] =
      NDArray.tabulate[Byte](4, 3)((i, j) => ((i * 17 + j * 5) % 23 - 11).toByte)
    val byteRight: Array2[Byte] =
      NDArray.tabulate[Byte](4, 3)((i, j) => ((i * 7 + j * 3) % 11 - 5).toByte)

    val shortLeft: Array2[Short] =
      NDArray.tabulate[Short](4, 3)((i, j) => ((i * 31 + j * 7) % 41 - 20).toShort)
    val shortRight: Array2[Short] =
      NDArray.tabulate[Short](4, 3)((i, j) => ((i * 5 + j * 11) % 19 - 9).toShort)

    val longLeft: Array2[Long] =
      NDArray.tabulate[Long](4, 3)((i, j) => (i * 17 + j * 5) % 23 - 11)
    val longRight: Array2[Long] =
      NDArray.tabulate[Long](4, 3)((i, j) => (i * 7 + j * 3) % 11 - 5)
    val longPositive: Array2[Long] =
      NDArray.tabulate[Long](4, 3)((i, j) => (i * 3 + j * 5) % 7 + 1)
    val longSign: Array2[Long] =
      NDArray.tabulate[Long](4, 3)((i, j) => if (i + j) % 3 == 0 then -1L else 1L)

    val doubleSign: Array2[Double] =
      NDArray.tabulate[Double](4, 3)((i, j) => if (i + j) % 3 == 0 then -1.0 else 1.0)

    val bool2: Array2[Boolean] =
      NDArray.tabulate[Boolean](4, 3)((i, j) => (i + j) % 2 == 0)
    val bool3: Array3[Boolean] =
      NDArray.tabulate[Boolean](2, 2, 2)((i, j, k) => i == 0 || (j == 1 && k == 1))

    val uint8Left: Array2[UInt8] =
      NDArray.tabulate[UInt8](4, 3)((i, j) => UInt8.unsafe((i * 17 + j * 5) % 23 + 40))
    val uint8Right: Array2[UInt8] =
      NDArray.tabulate[UInt8](4, 3)((i, j) => UInt8.unsafe((i * 7 + j * 3) % 11 + 30))
    val uint8Wide: Array2[UInt8] =
      NDArray.tabulate[UInt8](4, 6)((i, j) => UInt8.unsafe((i * 17 + j * 5) % 23 + 40))
    val uint8InnerLeft: Array2[UInt8] =
      uint8Wide.slice(1, Slice.every(2))
    val uint8InnerRight: Array2[UInt8] =
      NDArray
        .tabulate[UInt8](4, 6)((i, j) => UInt8.unsafe((i * 7 + j * 3) % 11 + 30))
        .slice(1, Slice(1, 6, 2))

    val uint16Left: Array2[UInt16] =
      NDArray.tabulate[UInt16](4, 3)((i, j) => UInt16.unsafe((i * 31 + j * 7) % 41 + 200))
    val uint16Right: Array2[UInt16] =
      NDArray.tabulate[UInt16](4, 3)((i, j) => UInt16.unsafe((i * 5 + j * 11) % 19 + 180))

    val doubleLeft: Array2[Double] =
      NDArray.tabulate[Double](4, 3)((i, j) => ((i * 131 + j * 17) % 251 - 125).toDouble / 16.0)
    val doubleRight: Array2[Double] =
      NDArray.tabulate[Double](4, 3)((i, j) => ((i * 43 + j * 19) % 257 - 128).toDouble / 32.0)
    val doublePositive: Array2[Double] =
      NDArray.tabulate[Double](4, 3)((i, j) => ((i * 23 + j * 11) % 97 + 1).toDouble / 17.0)
    val doubleRow: Array1[Double] =
      NDArray.tabulate[Double](3)(j => ((j * 29) % 127 - 63).toDouble / 8.0)
    val doubleWide: Array2[Double] =
      NDArray.tabulate[Double](4, 6)((i, j) => ((i * 131 + j * 17) % 251 - 125).toDouble / 16.0)
    val doubleInnerLeft: Array2[Double] =
      doubleWide.slice(1, Slice.every(2))
    val doubleInnerRight: Array2[Double] =
      NDArray
        .tabulate[Double](4, 6)((i, j) => ((i * 43 + j * 19) % 257 - 128).toDouble / 32.0)
        .slice(1, Slice(1, 6, 2))
    val doubleReversed: Array2[Double] =
      doubleLeft.reverse(1)
    val doubleTransposed: Array2[Double] =
      doubleLeft.transpose
    val double3: Array3[Double] =
      NDArray.tabulate[Double](2, 3, 2)((i, j, k) => (i * 10 + j + k).toDouble / 4.0)
    val doubleTrunc: Array2[Double] =
      NDArray.tabulate[Double](4, 3)((i, j) => i.toDouble + j.toDouble / 2.0)
    val doubleNonFinite: Array2[Double] =
      val values = doubleLeft.mutableCopy
      values.update(0, 0, Double.NaN)
      values.update(1, 1, Double.PositiveInfinity)
      values.freezeCopy()
    val floatLeft: Array2[Float] =
      doubleLeft.cast[Float]
    val floatRight: Array2[Float] =
      doubleRight.cast[Float]

    val convertRound: Array1[Double] =
      NDArray.fromSeq(Shape(4), Seq(2.5, 3.5, -2.5, -3.5))
    val convertToward: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(1.9, -1.9))
    val convertInRangeUInt8: Array1[Int] =
      NDArray.fromSeq(Shape(3), Seq(0, 128, 255))
    val convertInRangeUInt8Float: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(0.4, 200.4, 255.0))
    val convertWideInt: Array1[Int] =
      NDArray.fromSeq(Shape(3), Seq(127, 128, 0))
    val convertNegInt: Array1[Int] =
      NDArray.fromSeq(Shape(3), Seq(1, -1, 2))
    val convertWideUInt16: Array1[UInt16] =
      NDArray.fromSeq(Shape(2), Seq(UInt16.unsafe(200), UInt16.unsafe(300)))
    val convertWideLong: Array1[Long] =
      NDArray.fromSeq(Shape(2), Seq(1L, Int.MaxValue.toLong + 1L))
    val convertOverflowByte: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(1.0, 128.0, 2.0))
    val convertNan: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(1.0, Double.NaN, 3.0))
    val convertInf: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(1.0, Double.PositiveInfinity))
    val convertHugeDouble: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(1.0, 1.0e40))
    val convertOverflowFloat: Array1[Float] =
      NDArray.fromSeq(Shape(2), Seq(1.0f, 200.0f))
    val convertIntOverflow: Array2[Int] =
      NDArray.fromSeq(Shape(2, 3), Seq(1, 2, 200, 3, 4, 5))
    val convertIntOverflowReversed: Array2[Int] =
      convertIntOverflow.reverse(1)

    val nanLeft: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(1.0, Double.NaN, 2.0))
    val nanRight: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(3.0, 4.0, Double.NaN))
    val nanTies: Array1[Double] =
      NDArray.fromSeq(Shape(5), Seq(3.0, Double.NaN, 1.0, Double.NaN, 1.0))
    val nanMatrix: Array2[Double] =
      NDArray.fromSeq(Shape(2, 4), Seq(3.0, Double.NaN, 1.0, 2.0, 4.0, 5.0, 5.0, -1.0))
    val infLeft: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(Double.PositiveInfinity, 1.0))
    val infRight: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(Double.PositiveInfinity, 2.0))
    val oneDiv: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(1.0, -1.0, 0.0))
    val zeroDiv: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(0.0, 0.0, 0.0))
    val signedZero: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(0.0, -0.0))
    val signedZeroLeft: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(0.0, -0.0, -0.0))
    val signedZeroRight: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(0.0, 0.0, -0.0))
    val signedZeroMulLeft: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(1.0, -1.0, 1.0))
    val signedZeroMulRight: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(0.0, 0.0, -0.0))
    val signedZeroMatrix: Array2[Double] =
      NDArray.fromSeq(Shape(2, 2), Seq(0.0, 2.0, -0.0, 1.0))
    val intOverflow: Array1[Int] =
      NDArray.fromSeq(Shape(2), Seq(Int.MaxValue, 1))
    val intOverflowMatrix: Array2[Int] =
      NDArray.fromSeq(Shape(2, 2), Seq(Int.MaxValue, 2, 1, Int.MaxValue))
    val intOverflowProduct: Array1[Int] =
      NDArray.fromSeq(Shape(2), Seq(Int.MaxValue, 2))
    val intOverflowLeft: Array1[Int] =
      NDArray.fromSeq(Shape(2), Seq(Int.MaxValue, -1))
    val intOverflowRight: Array1[Int] =
      NDArray.fromSeq(Shape(2), Seq(1, Int.MinValue))
    val longOverflow: Array1[Long] =
      NDArray.fromSeq(Shape(2), Seq(Long.MaxValue, 1L))
    val longOverflowLeft: Array1[Long] =
      NDArray.fromSeq(Shape(2), Seq(Long.MaxValue, -1L))
    val longOverflowRight: Array1[Long] =
      NDArray.fromSeq(Shape(2), Seq(1L, Long.MinValue))
    val clipAdversarial: Array1[Double] =
      NDArray.fromSeq(Shape(3), Seq(Double.NaN, 1.0, 4.0))
    val clipInf: Array1[Double] =
      NDArray.fromSeq(Shape(2), Seq(Double.PositiveInfinity, Double.NegativeInfinity))
    val intClipEdges: Array1[Int] =
      NDArray.fromSeq(Shape(4), Seq(Int.MinValue, -10, 10, Int.MaxValue))

  private def parseArgs(args: Array[String]): Config =
    var config = Config()
    var index = 0
    while index < args.length do
      args(index) match
        case "--out" =>
          index += 1
          require(index < args.length, "--out requires a path")
          config = config.copy(output = Paths.get(args(index)))
        case argument =>
          throw new IllegalArgumentException(s"unknown argument: $argument")
      index += 1
    config

  private def row(operation: Case): Map[String, String] =
    val encoded =
      try encode(operation.run(), operation.comparison)
      catch
        case error: EmptyReduction =>
          Encoded(
            "error",
            "none",
            "[]",
            s"""["empty_reduction", ${jsonString(error.operation)}]"""
          )
    Map(
      "case" -> jsonString(operation.name),
      "family" -> jsonString(operation.family),
      "input_dtype" -> jsonString(operation.inputDType),
      "result_dtype" -> jsonString(operation.resultDType),
      "input_layout" -> jsonString(operation.inputLayout),
      "comparison" -> jsonString(operation.comparison),
      "result_kind" -> jsonString(encoded.kind),
      "result_layout" -> jsonString(encoded.layout),
      "shape" -> encoded.shape,
      "values" -> encoded.values
    )

  private final case class Encoded(kind: String, layout: String, shape: String, values: String)

  private def encode(value: Any, comparison: String): Encoded =
    val bits = comparison == "bits"
    value match
      case Left(error: ConversionError) =>
        Encoded("error", "none", "[]", conversionErrorValues(error))
      case Right(inner) =>
        encode(inner, comparison)
      case array: NDArray[?, ?] =>
        Encoded(
          "array",
          if array.isContiguous then "c_contiguous" else "strided_view",
          shapeJson(array.shape),
          valuesJson(array.asInstanceOf[NDArray[Any, ?]], bits)
        )
      case boolean: Boolean =>
        Encoded("scalar", "scalar", "[]", s"[${if boolean then "true" else "false"}]")
      case byte: Byte =>
        Encoded("scalar", "scalar", "[]", s"[$byte]")
      case short: Short =>
        Encoded("scalar", "scalar", "[]", s"[$short]")
      case int: Int =>
        Encoded("scalar", "scalar", "[]", s"[$int]")
      case long: Long =>
        Encoded("scalar", "scalar", "[]", s"[$long]")
      case float: Float =>
        Encoded(
          "scalar",
          "scalar",
          "[]",
          s"[${if bits then bitsJson(float) else numberJson(float.toDouble)}]"
        )
      case double: Double =>
        Encoded(
          "scalar",
          "scalar",
          "[]",
          s"[${if bits then bitsJson(double) else numberJson(double)}]"
        )
      case other =>
        throw new IllegalArgumentException(
          s"unsupported common-ops result: ${other.getClass.getName}"
        )

  private def shapeJson(shape: Shape[?]): String =
    (0 until shape.rank).map(axis => shape(axis).toString).mkString("[", ", ", "]")

  private def valuesJson(array: NDArray[Any, ?], bits: Boolean): String =
    val iterator = array.elementsIterator
    val rendered = new StringBuilder
    rendered.append('[')
    var first = true
    while iterator.hasNext do
      if !first then rendered.append(", ")
      rendered.append(elementJson(array.dtype.name, iterator.next(), bits))
      first = false
    rendered.append(']')
    rendered.result()

  private def elementJson(dtypeName: String, value: Any, bits: Boolean): String =
    dtypeName match
      case "UInt8" =>
        value.asInstanceOf[UInt8].toInt.toString
      case "UInt16" =>
        value.asInstanceOf[UInt16].toInt.toString
      case _ =>
        value match
          case boolean: Boolean => if boolean then "true" else "false"
          case byte: Byte => byte.toString
          case short: Short => short.toString
          case int: Int => int.toString
          case long: Long => long.toString
          case float: Float =>
            if bits then bitsJson(float) else numberJson(float.toDouble)
          case double: Double =>
            if bits then bitsJson(double) else numberJson(double)
          case other =>
            throw new IllegalArgumentException(
              s"unsupported common-ops element: ${other.getClass.getName}"
            )

  private def bitsJson(value: Double): String =
    jsonString(f"${java.lang.Double.doubleToRawLongBits(value)}%016x")

  private def bitsJson(value: Float): String =
    jsonString(f"${java.lang.Float.floatToRawIntBits(value)}%08x")

  private def conversionErrorValues(error: ConversionError): String =
    error match
      case ConversionError.OutOfRange(index, source, target) =>
        s"""["out_of_range", $index, ${jsonString(source)}, ${jsonString(target)}]"""
      case ConversionError.NonFiniteToIntegral(index, source, target) =>
        s"""["non_finite", $index, ${jsonString(source)}, ${jsonString(target)}]"""

  private def numberJson(value: Double): String =
    if value.isNaN then "\"nan\""
    else if value == Double.PositiveInfinity then "\"inf\""
    else if value == Double.NegativeInfinity then "\"-inf\""
    else value.toString

  private def render(rows: Vector[Map[String, String]]): String =
    val body = rows
      .map { row =>
        s"""    {
           |      "case": ${row("case")},
           |      "family": ${row("family")},
           |      "input_dtype": ${row("input_dtype")},
           |      "result_dtype": ${row("result_dtype")},
           |      "input_layout": ${row("input_layout")},
           |      "comparison": ${row("comparison")},
           |      "result_kind": ${row("result_kind")},
           |      "result_layout": ${row("result_layout")},
           |      "shape": ${row("shape")},
           |      "values": ${row("values")}
           |    }""".stripMargin
      }
      .mkString(",\n")
    s"""{
       |  "schema": "ravel.common-ops.signatures.v1",
       |  "metadata": {
       |    "library": "ravel",
       |    "oracle": "elementwise",
       |    "release_target": "0.1"
       |  },
       |  "results": [
       |$body
       |  ]
       |}
       |""".stripMargin

  private def jsonString(value: String): String =
    val builder = new StringBuilder
    builder.append('"')
    value.foreach {
      case '"' => builder.append("\\\"")
      case '\\' => builder.append("\\\\")
      case '\n' => builder.append("\\n")
      case char => builder.append(char)
    }
    builder.append('"')
    builder.result()
