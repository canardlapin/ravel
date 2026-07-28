# Ravel JVM vs NumPy public-operation matrix

Semantic parity passed for every reported row before timing comparison. This broad matrix locates follow-up targets; focused JMH controls remain the authority for optimization claims.

Ravel: Java 25.0.1 (Homebrew), Mac OS X aarch64. NumPy 2.4.3 on Python 3.14.3, macOS-14.3-arm64-arm-64bit-Mach-O.

## Family summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `binary` | 24 | 0.669x | 0.295x | 1.540x |
| `cast` | 8 | 0.425x | 0.111x | 1.254x |
| `comparison` | 12 | 0.690x | 0.345x | 1.178x |
| `in-place` | 18 | 0.027x | 0.011x | 0.041x |
| `predicate` | 4 | 0.278x | 0.176x | 0.358x |
| `reduction` | 52 | 0.440x | 0.068x | 2.425x |
| `scalar` | 18 | 0.646x | 0.194x | 2.759x |
| `unary` | 22 | 0.773x | 0.300x | 3.159x |

## Input dtype summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `float32` | 18 | 0.351x | 0.016x | 1.074x |
| `float64` | 110 | 0.434x | 0.011x | 3.159x |
| `int32` | 16 | 0.231x | 0.020x | 1.520x |
| `int64` | 14 | 0.386x | 0.037x | 1.312x |

## Input layout summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `broadcast` | 4 | 0.649x | 0.522x | 0.743x |
| `contiguous` | 132 | 0.390x | 0.016x | 3.159x |
| `inner_stride` | 8 | 0.372x | 0.031x | 1.271x |
| `outer_stride` | 4 | 1.204x | 1.001x | 1.540x |
| `reversed` | 6 | 0.184x | 0.011x | 0.860x |
| `transposed` | 4 | 0.372x | 0.091x | 1.299x |

## Detailed results

| side | family | dtype | layout | case | work | Ravel ns/op | NumPy ns/op | Ravel speed |
|---:|---|---|---|---|---:|---:|---:|---:|
| 256 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 65,536 element | 34,939.0 | 11,431.8 | 0.327x |
| 256 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 32,768 element | 17,905.7 | 10,695.5 | 0.597x |
| 256 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 32,768 element | 20,052.4 | 20,070.2 | 1.001x |
| 256 | binary | `float64` | `reversed` | `reverse_minimum_double` | 65,536 element | 41,973.2 | 36,101.6 | 0.860x |
| 256 | binary | `float64` | `transposed` | `transpose_maximum_double` | 65,536 element | 82,049.8 | 106,604.8 | 1.299x |
| 256 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 65,536 element | 39,666.9 | 25,222.8 | 0.636x |
| 256 | scalar | `float64` | `contiguous` | `scalar_add_double` | 65,536 element | 16,927.7 | 11,383.4 | 0.672x |
| 256 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 65,536 element | 17,174.5 | 11,374.9 | 0.662x |
| 256 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 65,536 element | 17,925.1 | 10,592.7 | 0.591x |
| 256 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 65,536 element | 21,544.4 | 10,722.3 | 0.498x |
| 256 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 65,536 element | 18,026.4 | 20,226.4 | 1.122x |
| 256 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 65,536 element | 16,994.0 | 20,111.0 | 1.183x |
| 256 | unary | `float64` | `contiguous` | `clip_double` | 65,536 element | 16,008.3 | 20,195.3 | 1.262x |
| 256 | unary | `float64` | `contiguous` | `negate_double` | 65,536 element | 17,180.5 | 7,360.4 | 0.428x |
| 256 | unary | `float64` | `contiguous` | `abs_double` | 65,536 element | 17,208.3 | 10,458.8 | 0.608x |
| 256 | unary | `float64` | `contiguous` | `sqrt_double` | 65,536 element | 39,500.9 | 18,575.2 | 0.470x |
| 256 | unary | `float64` | `contiguous` | `exp_double` | 65,536 element | 292,724.9 | 159,201.0 | 0.544x |
| 256 | unary | `float64` | `contiguous` | `log_double` | 65,536 element | 274,668.8 | 175,904.1 | 0.640x |
| 256 | unary | `float64` | `contiguous` | `sin_double` | 65,536 element | 283,980.8 | 209,899.4 | 0.739x |
| 256 | unary | `float64` | `contiguous` | `cos_double` | 65,536 element | 286,391.6 | 219,043.6 | 0.765x |
| 256 | unary | `float64` | `contiguous` | `tan_double` | 65,536 element | 752,541.4 | 270,430.5 | 0.359x |
| 256 | unary | `float64` | `contiguous` | `floor_double` | 65,536 element | 16,843.0 | 20,301.8 | 1.205x |
| 256 | unary | `float64` | `contiguous` | `ceil_double` | 65,536 element | 16,954.8 | 13,196.8 | 0.778x |
| 256 | comparison | `float64` | `contiguous` | `equal_double` | 65,536 element | 20,787.1 | 17,737.0 | 0.853x |
| 256 | comparison | `float64` | `contiguous` | `not_equal_double` | 65,536 element | 20,871.1 | 16,130.3 | 0.773x |
| 256 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 32,768 element | 18,290.5 | 11,131.9 | 0.609x |
| 256 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 32,768 element | 19,197.6 | 22,207.8 | 1.157x |
| 256 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 65,536 element | 36,743.3 | 27,305.7 | 0.743x |
| 256 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 65,536 element | 37,592.2 | 12,974.9 | 0.345x |
| 256 | predicate | `float64` | `contiguous` | `is_nan_double` | 65,536 element | 17,001.9 | 6,094.2 | 0.358x |
| 256 | predicate | `float64` | `contiguous` | `is_finite_double` | 65,536 element | 20,520.8 | 5,973.9 | 0.291x |
| 256 | cast | `float64` | `contiguous` | `cast_double_int` | 65,536 element | 62,444.5 | 20,730.0 | 0.332x |
| 256 | cast | `float32` | `contiguous` | `cast_float_double` | 65,536 element | 13,323.8 | 7,068.8 | 0.531x |
| 256 | cast | `int32` | `contiguous` | `cast_int_long` | 65,536 element | 13,391.4 | 8,227.2 | 0.614x |
| 256 | cast | `int64` | `contiguous` | `cast_long_float` | 65,536 element | 20,118.4 | 19,611.5 | 0.975x |
| 256 | binary | `float32` | `contiguous` | `contiguous_add_float` | 65,536 element | 10,059.5 | 8,216.7 | 0.817x |
| 256 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 65,536 element | 24,895.8 | 7,878.4 | 0.316x |
| 256 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 65,536 element | 20,291.0 | 5,981.8 | 0.295x |
| 256 | binary | `int32` | `contiguous` | `contiguous_add_int` | 65,536 element | 9,841.2 | 8,157.4 | 0.829x |
| 256 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 65,536 element | 19,728.2 | 8,439.8 | 0.428x |
| 256 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 65,536 element | 51,396.0 | 18,119.0 | 0.353x |
| 256 | binary | `int64` | `contiguous` | `contiguous_add_long` | 65,536 element | 19,819.8 | 16,513.9 | 0.833x |
| 256 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 65,536 element | 25,459.0 | 21,656.7 | 0.851x |
| 256 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 65,536 element | 42,370.7 | 55,610.8 | 1.312x |
| 256 | reduction | `float64` | `contiguous` | `full_product_double` | 65,536 element | 72,440.7 | 79,260.8 | 1.094x |
| 256 | reduction | `float64` | `contiguous` | `full_min_double` | 65,536 element | 36,214.8 | 7,268.5 | 0.201x |
| 256 | reduction | `float64` | `contiguous` | `full_max_double` | 65,536 element | 36,276.2 | 7,248.7 | 0.200x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 65,536 element | 18,388.0 | 44,587.4 | 2.425x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 65,536 element | 18,560.9 | 44,291.4 | 2.386x |
| 256 | reduction | `float64` | `contiguous` | `full_mean_double` | 65,536 element | 32,736.3 | 11,166.8 | 0.341x |
| 256 | reduction | `float64` | `contiguous` | `axis0_product_double` | 65,536 element | 64,153.9 | 12,024.8 | 0.187x |
| 256 | reduction | `float64` | `contiguous` | `axis1_product_double` | 65,536 element | 62,433.8 | 59,900.0 | 0.959x |
| 256 | reduction | `float64` | `contiguous` | `axis0_min_double` | 65,536 element | 56,850.9 | 12,263.2 | 0.216x |
| 256 | reduction | `float64` | `contiguous` | `axis1_max_double` | 65,536 element | 34,233.7 | 9,895.7 | 0.289x |
| 256 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 65,536 element | 42,559.3 | 88,105.0 | 2.070x |
| 256 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 65,536 element | 25,546.3 | 38,862.5 | 1.521x |
| 256 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 65,536 element | 44,925.1 | 13,069.9 | 0.291x |
| 256 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 65,536 element | 29,745.9 | 13,230.5 | 0.445x |
| 256 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 32,768 element | 36,607.2 | 40,285.2 | 1.100x |
| 256 | reduction | `float64` | `reversed` | `reverse_min_double` | 65,536 element | 36,385.5 | 30,952.2 | 0.851x |
| 256 | reduction | `float64` | `transposed` | `transpose_max_double` | 65,536 element | 39,579.5 | 7,251.7 | 0.183x |
| 256 | reduction | `float32` | `contiguous` | `full_sum_float` | 65,536 element | 20,280.7 | 10,817.3 | 0.533x |
| 256 | reduction | `float32` | `contiguous` | `full_product_float` | 65,536 element | 72,445.0 | 77,722.4 | 1.073x |
| 256 | reduction | `float32` | `contiguous` | `full_mean_float` | 65,536 element | 28,691.7 | 12,512.2 | 0.436x |
| 256 | reduction | `int32` | `contiguous` | `full_sum_int` | 65,536 element | 18,310.0 | 4,352.7 | 0.238x |
| 256 | reduction | `int32` | `contiguous` | `full_product_int` | 65,536 element | 54,792.2 | 5,106.2 | 0.093x |
| 256 | reduction | `int64` | `contiguous` | `full_sum_long` | 65,536 element | 18,196.4 | 7,183.9 | 0.395x |
| 256 | reduction | `int64` | `contiguous` | `full_product_long` | 65,536 element | 54,921.6 | 15,953.5 | 0.290x |
| 256 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 65,536 element | 36,382.1 | 14,481.7 | 0.398x |
| 256 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 65,536 element | 28,740.0 | 18,230.5 | 0.634x |
| 256 | in-place | `float64` | `contiguous` | `inplace_add_double` | 65,536 element | 272,389.0 | 9,572.4 | 0.035x |
| 256 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 65,536 element | 266,308.0 | 9,511.1 | 0.036x |
| 256 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 65,536 element | 267,232.0 | 9,626.0 | 0.036x |
| 256 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 65,536 element | 273,650.4 | 10,188.6 | 0.037x |
| 256 | in-place | `float32` | `contiguous` | `inplace_add_float` | 65,536 element | 254,055.5 | 5,175.5 | 0.020x |
| 256 | in-place | `int32` | `contiguous` | `inplace_add_int` | 65,536 element | 230,395.8 | 5,140.7 | 0.022x |
| 256 | in-place | `int64` | `contiguous` | `inplace_add_long` | 65,536 element | 236,255.0 | 9,653.6 | 0.041x |
| 256 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 65,536 element | 833,578.1 | 10,011.5 | 0.012x |
| 256 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 32,768 element | 406,266.5 | 12,595.3 | 0.031x |
| 1024 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 1,048,576 element | 1,612,144.6 | 744,142.9 | 0.462x |
| 1024 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 524,288 element | 278,671.1 | 354,289.4 | 1.271x |
| 1024 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 524,288 element | 360,190.3 | 554,795.0 | 1.540x |
| 1024 | binary | `float64` | `reversed` | `reverse_minimum_double` | 1,048,576 element | 2,315,184.3 | 1,284,757.6 | 0.555x |
| 1024 | binary | `float64` | `transposed` | `transpose_maximum_double` | 1,048,576 element | 4,509,362.2 | 3,964,597.6 | 0.879x |
| 1024 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 1,048,576 element | 2,056,556.1 | 1,072,842.4 | 0.522x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_add_double` | 1,048,576 element | 1,532,488.9 | 507,637.3 | 0.331x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 1,048,576 element | 1,861,420.4 | 482,673.5 | 0.259x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 1,048,576 element | 2,676,288.7 | 519,519.8 | 0.194x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 1,048,576 element | 895,978.2 | 566,474.4 | 0.632x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 1,048,576 element | 273,127.8 | 681,657.5 | 2.496x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 1,048,576 element | 271,883.1 | 750,200.2 | 2.759x |
| 1024 | unary | `float64` | `contiguous` | `clip_double` | 1,048,576 element | 260,236.0 | 822,003.7 | 3.159x |
| 1024 | unary | `float64` | `contiguous` | `negate_double` | 1,048,576 element | 440,595.0 | 461,543.4 | 1.048x |
| 1024 | unary | `float64` | `contiguous` | `abs_double` | 1,048,576 element | 297,785.3 | 454,060.6 | 1.525x |
| 1024 | unary | `float64` | `contiguous` | `sqrt_double` | 1,048,576 element | 2,381,582.6 | 714,905.0 | 0.300x |
| 1024 | unary | `float64` | `contiguous` | `exp_double` | 1,048,576 element | 5,248,369.0 | 3,071,773.6 | 0.585x |
| 1024 | unary | `float64` | `contiguous` | `log_double` | 1,048,576 element | 4,989,208.1 | 3,051,696.0 | 0.612x |
| 1024 | unary | `float64` | `contiguous` | `sin_double` | 1,048,576 element | 5,030,592.8 | 3,779,750.0 | 0.751x |
| 1024 | unary | `float64` | `contiguous` | `cos_double` | 1,048,576 element | 5,036,249.3 | 3,666,753.9 | 0.728x |
| 1024 | unary | `float64` | `contiguous` | `tan_double` | 1,048,576 element | 12,588,840.0 | 4,489,137.3 | 0.357x |
| 1024 | unary | `float64` | `contiguous` | `floor_double` | 1,048,576 element | 384,380.9 | 558,896.4 | 1.454x |
| 1024 | unary | `float64` | `contiguous` | `ceil_double` | 1,048,576 element | 269,141.8 | 595,934.0 | 2.214x |
| 1024 | comparison | `float64` | `contiguous` | `equal_double` | 1,048,576 element | 344,315.7 | 255,358.6 | 0.742x |
| 1024 | comparison | `float64` | `contiguous` | `not_equal_double` | 1,048,576 element | 377,715.5 | 237,461.4 | 0.629x |
| 1024 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 524,288 element | 280,723.5 | 184,010.6 | 0.655x |
| 1024 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 524,288 element | 311,645.3 | 367,272.3 | 1.178x |
| 1024 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 1,048,576 element | 608,342.3 | 436,546.3 | 0.718x |
| 1024 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 1,048,576 element | 520,537.5 | 195,932.9 | 0.376x |
| 1024 | predicate | `float64` | `contiguous` | `is_nan_double` | 1,048,576 element | 282,732.8 | 92,261.0 | 0.326x |
| 1024 | predicate | `float64` | `contiguous` | `is_finite_double` | 1,048,576 element | 513,484.3 | 90,235.8 | 0.176x |
| 1024 | cast | `float64` | `contiguous` | `cast_double_int` | 1,048,576 element | 1,242,277.5 | 414,165.6 | 0.333x |
| 1024 | cast | `float32` | `contiguous` | `cast_float_double` | 1,048,576 element | 1,665,974.0 | 361,919.2 | 0.217x |
| 1024 | cast | `int32` | `contiguous` | `cast_int_long` | 1,048,576 element | 2,202,088.5 | 244,972.0 | 0.111x |
| 1024 | cast | `int64` | `contiguous` | `cast_long_float` | 1,048,576 element | 310,199.9 | 389,113.6 | 1.254x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_add_float` | 1,048,576 element | 212,855.2 | 196,633.8 | 0.924x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 1,048,576 element | 748,246.3 | 227,845.5 | 0.305x |
| 1024 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 1,048,576 element | 318,308.6 | 173,298.8 | 0.544x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_add_int` | 1,048,576 element | 149,832.8 | 227,770.9 | 1.520x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 1,048,576 element | 341,201.0 | 232,046.7 | 0.680x |
| 1024 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 1,048,576 element | 959,903.0 | 456,594.3 | 0.476x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_add_long` | 1,048,576 element | 1,866,601.3 | 684,612.1 | 0.367x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 1,048,576 element | 2,258,645.2 | 666,118.6 | 0.295x |
| 1024 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 1,048,576 element | 1,500,968.5 | 1,304,140.2 | 0.869x |
| 1024 | reduction | `float64` | `contiguous` | `full_product_double` | 1,048,576 element | 1,169,177.8 | 1,189,671.2 | 1.018x |
| 1024 | reduction | `float64` | `contiguous` | `full_min_double` | 1,048,576 element | 584,536.5 | 100,296.0 | 0.172x |
| 1024 | reduction | `float64` | `contiguous` | `full_max_double` | 1,048,576 element | 585,055.5 | 99,091.0 | 0.169x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 1,048,576 element | 300,575.7 | 675,240.3 | 2.246x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 1,048,576 element | 308,020.1 | 675,421.5 | 2.193x |
| 1024 | reduction | `float64` | `contiguous` | `full_mean_double` | 1,048,576 element | 550,763.5 | 142,297.8 | 0.258x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_product_double` | 1,048,576 element | 1,335,146.2 | 160,598.6 | 0.120x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_product_double` | 1,048,576 element | 1,145,779.9 | 1,128,822.8 | 0.985x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_min_double` | 1,048,576 element | 1,953,606.6 | 162,153.8 | 0.083x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_max_double` | 1,048,576 element | 551,885.9 | 107,578.1 | 0.195x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 1,048,576 element | 1,189,013.9 | 2,178,951.1 | 1.833x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 1,048,576 element | 385,925.4 | 669,172.8 | 1.734x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 1,048,576 element | 1,113,375.4 | 165,026.3 | 0.148x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 1,048,576 element | 482,228.7 | 150,361.0 | 0.312x |
| 1024 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 524,288 element | 585,643.4 | 624,099.3 | 1.066x |
| 1024 | reduction | `float64` | `reversed` | `reverse_min_double` | 1,048,576 element | 604,087.1 | 427,618.5 | 0.708x |
| 1024 | reduction | `float64` | `transposed` | `transpose_max_double` | 1,048,576 element | 1,067,844.8 | 97,467.6 | 0.091x |
| 1024 | reduction | `float32` | `contiguous` | `full_sum_float` | 1,048,576 element | 327,265.2 | 145,825.4 | 0.446x |
| 1024 | reduction | `float32` | `contiguous` | `full_product_float` | 1,048,576 element | 1,171,728.9 | 1,257,934.0 | 1.074x |
| 1024 | reduction | `float32` | `contiguous` | `full_mean_float` | 1,048,576 element | 463,453.2 | 150,714.3 | 0.325x |
| 1024 | reduction | `int32` | `contiguous` | `full_sum_int` | 1,048,576 element | 292,042.6 | 46,231.0 | 0.158x |
| 1024 | reduction | `int32` | `contiguous` | `full_product_int` | 1,048,576 element | 876,934.7 | 60,030.1 | 0.068x |
| 1024 | reduction | `int64` | `contiguous` | `full_sum_long` | 1,048,576 element | 293,912.4 | 93,560.5 | 0.318x |
| 1024 | reduction | `int64` | `contiguous` | `full_product_long` | 1,048,576 element | 882,688.2 | 248,707.1 | 0.282x |
| 1024 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 1,048,576 element | 582,924.0 | 220,208.4 | 0.378x |
| 1024 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 1,048,576 element | 462,358.7 | 278,691.4 | 0.603x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_add_double` | 1,048,576 element | 4,787,766.9 | 149,761.9 | 0.031x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 1,048,576 element | 4,797,817.1 | 149,814.1 | 0.031x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 1,048,576 element | 4,789,798.8 | 151,355.2 | 0.032x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 1,048,576 element | 4,661,554.0 | 162,900.2 | 0.035x |
| 1024 | in-place | `float32` | `contiguous` | `inplace_add_float` | 1,048,576 element | 4,562,873.9 | 74,049.2 | 0.016x |
| 1024 | in-place | `int32` | `contiguous` | `inplace_add_int` | 1,048,576 element | 3,833,218.2 | 75,414.0 | 0.020x |
| 1024 | in-place | `int64` | `contiguous` | `inplace_add_long` | 1,048,576 element | 4,173,379.1 | 153,177.2 | 0.037x |
| 1024 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 1,048,576 element | 13,583,048.9 | 152,352.3 | 0.011x |
| 1024 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 524,288 element | 6,354,125.4 | 210,085.0 | 0.033x |

A speed value above 1.0x favors Ravel. In-place rows reuse their destination; allocating array operations include result allocation. All other interpretation cautions from the access-pattern suite apply.
