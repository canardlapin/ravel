# Ravel JVM vs NumPy public-operation matrix

Semantic parity passed for every reported row before timing comparison. This broad matrix locates follow-up targets; focused JMH controls remain the authority for optimization claims.

Ravel: Java 21.0.11 (Eclipse Adoptium), Mac OS X aarch64. NumPy 2.4.3 on Python 3.14.3, macOS-14.3-arm64-arm-64bit-Mach-O.

## Family summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `binary` | 24 | 0.804x | 0.349x | 2.577x |
| `cast` | 8 | 1.034x | 0.329x | 3.149x |
| `comparison` | 12 | 0.683x | 0.376x | 1.118x |
| `in-place` | 18 | 0.026x | 0.011x | 0.039x |
| `predicate` | 4 | 0.312x | 0.264x | 0.360x |
| `reduction` | 52 | 0.429x | 0.065x | 2.273x |
| `scalar` | 18 | 0.783x | 0.286x | 3.008x |
| `unary` | 22 | 0.731x | 0.291x | 2.647x |

## Input dtype summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `float32` | 18 | 0.427x | 0.017x | 2.723x |
| `float64` | 110 | 0.437x | 0.011x | 3.008x |
| `int32` | 16 | 0.284x | 0.020x | 3.149x |
| `int64` | 14 | 0.541x | 0.036x | 2.577x |

## Input layout summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `broadcast` | 4 | 0.574x | 0.418x | 0.705x |
| `contiguous` | 132 | 0.433x | 0.017x | 3.149x |
| `inner_stride` | 8 | 0.350x | 0.028x | 1.145x |
| `outer_stride` | 4 | 1.144x | 0.925x | 1.513x |
| `reversed` | 6 | 0.177x | 0.011x | 0.862x |
| `transposed` | 4 | 0.347x | 0.084x | 1.193x |

## Detailed results

| side | family | dtype | layout | case | work | Ravel ns/op | NumPy ns/op | Ravel speed |
|---:|---|---|---|---|---:|---:|---:|---:|
| 256 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 65,536 element | 19,492.1 | 10,756.9 | 0.552x |
| 256 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 32,768 element | 17,790.2 | 10,384.0 | 0.584x |
| 256 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 32,768 element | 20,494.8 | 18,962.2 | 0.925x |
| 256 | binary | `float64` | `reversed` | `reverse_minimum_double` | 65,536 element | 39,859.5 | 34,344.4 | 0.862x |
| 256 | binary | `float64` | `transposed` | `transpose_maximum_double` | 65,536 element | 77,131.5 | 92,003.6 | 1.193x |
| 256 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 65,536 element | 40,292.1 | 21,086.2 | 0.523x |
| 256 | scalar | `float64` | `contiguous` | `scalar_add_double` | 65,536 element | 17,281.0 | 9,881.5 | 0.572x |
| 256 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 65,536 element | 17,305.8 | 10,259.3 | 0.593x |
| 256 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 65,536 element | 17,221.6 | 10,186.1 | 0.591x |
| 256 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 65,536 element | 21,835.7 | 10,132.2 | 0.464x |
| 256 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 65,536 element | 17,110.3 | 20,304.5 | 1.187x |
| 256 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 65,536 element | 17,142.9 | 19,829.5 | 1.157x |
| 256 | unary | `float64` | `contiguous` | `clip_double` | 65,536 element | 16,300.3 | 19,452.4 | 1.193x |
| 256 | unary | `float64` | `contiguous` | `negate_double` | 65,536 element | 17,337.8 | 9,008.1 | 0.520x |
| 256 | unary | `float64` | `contiguous` | `abs_double` | 65,536 element | 17,291.0 | 10,308.2 | 0.596x |
| 256 | unary | `float64` | `contiguous` | `sqrt_double` | 65,536 element | 39,688.6 | 19,320.7 | 0.487x |
| 256 | unary | `float64` | `contiguous` | `exp_double` | 65,536 element | 350,892.3 | 154,551.7 | 0.440x |
| 256 | unary | `float64` | `contiguous` | `log_double` | 65,536 element | 317,659.3 | 156,010.1 | 0.491x |
| 256 | unary | `float64` | `contiguous` | `sin_double` | 65,536 element | 292,525.0 | 191,234.3 | 0.654x |
| 256 | unary | `float64` | `contiguous` | `cos_double` | 65,536 element | 288,865.7 | 197,704.6 | 0.684x |
| 256 | unary | `float64` | `contiguous` | `tan_double` | 65,536 element | 855,199.3 | 248,564.0 | 0.291x |
| 256 | unary | `float64` | `contiguous` | `floor_double` | 65,536 element | 17,123.2 | 10,228.4 | 0.597x |
| 256 | unary | `float64` | `contiguous` | `ceil_double` | 65,536 element | 16,982.8 | 10,334.9 | 0.609x |
| 256 | comparison | `float64` | `contiguous` | `equal_double` | 65,536 element | 20,968.8 | 15,616.7 | 0.745x |
| 256 | comparison | `float64` | `contiguous` | `not_equal_double` | 65,536 element | 20,806.5 | 15,272.2 | 0.734x |
| 256 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 32,768 element | 17,939.6 | 10,814.9 | 0.603x |
| 256 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 32,768 element | 19,539.1 | 21,839.7 | 1.118x |
| 256 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 65,536 element | 37,095.4 | 26,072.3 | 0.703x |
| 256 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 65,536 element | 32,710.5 | 12,406.8 | 0.379x |
| 256 | predicate | `float64` | `contiguous` | `is_nan_double` | 65,536 element | 16,443.0 | 5,911.5 | 0.360x |
| 256 | predicate | `float64` | `contiguous` | `is_finite_double` | 65,536 element | 19,703.6 | 5,756.3 | 0.292x |
| 256 | cast | `float64` | `contiguous` | `cast_double_int` | 65,536 element | 60,747.9 | 19,994.5 | 0.329x |
| 256 | cast | `float32` | `contiguous` | `cast_float_double` | 65,536 element | 9,395.5 | 7,959.0 | 0.847x |
| 256 | cast | `int32` | `contiguous` | `cast_int_long` | 65,536 element | 9,524.1 | 7,987.3 | 0.839x |
| 256 | cast | `int64` | `contiguous` | `cast_long_float` | 65,536 element | 19,128.5 | 19,169.4 | 1.002x |
| 256 | binary | `float32` | `contiguous` | `contiguous_add_float` | 65,536 element | 10,118.5 | 8,072.9 | 0.798x |
| 256 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 65,536 element | 17,387.4 | 6,060.4 | 0.349x |
| 256 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 65,536 element | 20,067.9 | 5,740.4 | 0.286x |
| 256 | binary | `int32` | `contiguous` | `contiguous_add_int` | 65,536 element | 10,023.6 | 7,954.7 | 0.794x |
| 256 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 65,536 element | 19,214.3 | 7,805.8 | 0.406x |
| 256 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 65,536 element | 39,798.2 | 17,224.8 | 0.433x |
| 256 | binary | `int64` | `contiguous` | `contiguous_add_long` | 65,536 element | 14,294.4 | 15,677.2 | 1.097x |
| 256 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 65,536 element | 23,240.8 | 20,324.1 | 0.875x |
| 256 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 65,536 element | 42,453.0 | 52,816.9 | 1.244x |
| 256 | reduction | `float64` | `contiguous` | `full_product_double` | 65,536 element | 72,563.2 | 74,860.1 | 1.032x |
| 256 | reduction | `float64` | `contiguous` | `full_min_double` | 65,536 element | 36,325.8 | 6,915.2 | 0.190x |
| 256 | reduction | `float64` | `contiguous` | `full_max_double` | 65,536 element | 36,294.1 | 6,940.7 | 0.191x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 65,536 element | 18,834.7 | 42,815.0 | 2.273x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 65,536 element | 18,795.3 | 42,641.5 | 2.269x |
| 256 | reduction | `float64` | `contiguous` | `full_mean_double` | 65,536 element | 32,658.3 | 10,564.8 | 0.323x |
| 256 | reduction | `float64` | `contiguous` | `axis0_product_double` | 65,536 element | 64,711.0 | 11,461.4 | 0.177x |
| 256 | reduction | `float64` | `contiguous` | `axis1_product_double` | 65,536 element | 62,360.4 | 57,471.7 | 0.922x |
| 256 | reduction | `float64` | `contiguous` | `axis0_min_double` | 65,536 element | 39,213.5 | 11,755.1 | 0.300x |
| 256 | reduction | `float64` | `contiguous` | `axis1_max_double` | 65,536 element | 27,724.8 | 9,441.8 | 0.341x |
| 256 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 65,536 element | 42,683.8 | 83,462.0 | 1.955x |
| 256 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 65,536 element | 25,354.6 | 37,113.3 | 1.464x |
| 256 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 65,536 element | 45,292.8 | 12,466.9 | 0.275x |
| 256 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 65,536 element | 29,421.8 | 12,967.7 | 0.441x |
| 256 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 32,768 element | 36,469.9 | 38,388.2 | 1.053x |
| 256 | reduction | `float64` | `reversed` | `reverse_min_double` | 65,536 element | 36,387.5 | 29,055.4 | 0.799x |
| 256 | reduction | `float64` | `transposed` | `transpose_max_double` | 65,536 element | 38,972.4 | 6,872.8 | 0.176x |
| 256 | reduction | `float32` | `contiguous` | `full_sum_float` | 65,536 element | 20,248.7 | 9,762.7 | 0.482x |
| 256 | reduction | `float32` | `contiguous` | `full_product_float` | 65,536 element | 72,292.5 | 75,830.5 | 1.049x |
| 256 | reduction | `float32` | `contiguous` | `full_mean_float` | 65,536 element | 28,723.6 | 11,998.4 | 0.418x |
| 256 | reduction | `int32` | `contiguous` | `full_sum_int` | 65,536 element | 18,188.1 | 4,121.9 | 0.227x |
| 256 | reduction | `int32` | `contiguous` | `full_product_int` | 65,536 element | 54,649.9 | 4,829.4 | 0.088x |
| 256 | reduction | `int64` | `contiguous` | `full_sum_long` | 65,536 element | 18,216.5 | 6,762.3 | 0.371x |
| 256 | reduction | `int64` | `contiguous` | `full_product_long` | 65,536 element | 54,576.7 | 15,348.1 | 0.281x |
| 256 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 65,536 element | 36,359.4 | 13,729.8 | 0.378x |
| 256 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 65,536 element | 28,067.3 | 17,430.8 | 0.621x |
| 256 | in-place | `float64` | `contiguous` | `inplace_add_double` | 65,536 element | 257,672.1 | 9,233.8 | 0.036x |
| 256 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 65,536 element | 260,539.3 | 9,188.1 | 0.035x |
| 256 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 65,536 element | 268,607.1 | 9,145.8 | 0.034x |
| 256 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 65,536 element | 270,876.2 | 9,774.9 | 0.036x |
| 256 | in-place | `float32` | `contiguous` | `inplace_add_float` | 65,536 element | 242,160.9 | 4,936.0 | 0.020x |
| 256 | in-place | `int32` | `contiguous` | `inplace_add_int` | 65,536 element | 229,473.2 | 4,920.0 | 0.021x |
| 256 | in-place | `int64` | `contiguous` | `inplace_add_long` | 65,536 element | 240,655.6 | 9,298.6 | 0.039x |
| 256 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 65,536 element | 816,362.9 | 9,477.8 | 0.012x |
| 256 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 32,768 element | 415,124.5 | 11,975.3 | 0.029x |
| 1024 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 1,048,576 element | 1,767,054.6 | 702,660.8 | 0.398x |
| 1024 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 524,288 element | 265,607.8 | 304,213.5 | 1.145x |
| 1024 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 524,288 element | 332,537.7 | 503,214.4 | 1.513x |
| 1024 | binary | `float64` | `reversed` | `reverse_minimum_double` | 1,048,576 element | 2,138,720.7 | 1,204,706.4 | 0.563x |
| 1024 | binary | `float64` | `transposed` | `transpose_maximum_double` | 1,048,576 element | 4,119,645.0 | 3,357,126.5 | 0.815x |
| 1024 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 1,048,576 element | 2,126,489.8 | 889,801.6 | 0.418x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_add_double` | 1,048,576 element | 810,226.0 | 486,428.6 | 0.600x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 1,048,576 element | 548,784.4 | 602,034.9 | 1.097x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 1,048,576 element | 933,899.0 | 588,872.1 | 0.631x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 1,048,576 element | 1,619,836.2 | 591,399.5 | 0.365x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 1,048,576 element | 267,730.5 | 707,032.3 | 2.641x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 1,048,576 element | 267,998.1 | 806,224.0 | 3.008x |
| 1024 | unary | `float64` | `contiguous` | `clip_double` | 1,048,576 element | 254,822.1 | 674,608.0 | 2.647x |
| 1024 | unary | `float64` | `contiguous` | `negate_double` | 1,048,576 element | 271,727.2 | 649,611.6 | 2.391x |
| 1024 | unary | `float64` | `contiguous` | `abs_double` | 1,048,576 element | 268,569.2 | 489,380.2 | 1.822x |
| 1024 | unary | `float64` | `contiguous` | `sqrt_double` | 1,048,576 element | 695,433.8 | 740,770.6 | 1.065x |
| 1024 | unary | `float64` | `contiguous` | `exp_double` | 1,048,576 element | 6,235,777.9 | 3,023,462.3 | 0.485x |
| 1024 | unary | `float64` | `contiguous` | `log_double` | 1,048,576 element | 5,723,157.1 | 3,011,025.0 | 0.526x |
| 1024 | unary | `float64` | `contiguous` | `sin_double` | 1,048,576 element | 5,162,194.5 | 3,641,970.4 | 0.706x |
| 1024 | unary | `float64` | `contiguous` | `cos_double` | 1,048,576 element | 5,141,761.8 | 3,752,851.7 | 0.730x |
| 1024 | unary | `float64` | `contiguous` | `tan_double` | 1,048,576 element | 14,205,570.1 | 4,560,322.3 | 0.321x |
| 1024 | unary | `float64` | `contiguous` | `floor_double` | 1,048,576 element | 1,105,370.2 | 642,455.5 | 0.581x |
| 1024 | unary | `float64` | `contiguous` | `ceil_double` | 1,048,576 element | 288,201.5 | 526,810.7 | 1.828x |
| 1024 | comparison | `float64` | `contiguous` | `equal_double` | 1,048,576 element | 343,095.9 | 252,920.9 | 0.737x |
| 1024 | comparison | `float64` | `contiguous` | `not_equal_double` | 1,048,576 element | 340,417.8 | 251,893.3 | 0.740x |
| 1024 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 524,288 element | 280,913.7 | 184,809.9 | 0.658x |
| 1024 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 524,288 element | 329,368.7 | 361,090.8 | 1.096x |
| 1024 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 1,048,576 element | 584,361.7 | 412,238.2 | 0.705x |
| 1024 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 1,048,576 element | 516,033.2 | 194,203.5 | 0.376x |
| 1024 | predicate | `float64` | `contiguous` | `is_nan_double` | 1,048,576 element | 264,718.2 | 90,027.7 | 0.340x |
| 1024 | predicate | `float64` | `contiguous` | `is_finite_double` | 1,048,576 element | 342,513.8 | 90,354.3 | 0.264x |
| 1024 | cast | `float64` | `contiguous` | `cast_double_int` | 1,048,576 element | 1,085,542.5 | 471,506.8 | 0.434x |
| 1024 | cast | `float32` | `contiguous` | `cast_float_double` | 1,048,576 element | 147,258.2 | 400,933.1 | 2.723x |
| 1024 | cast | `int32` | `contiguous` | `cast_int_long` | 1,048,576 element | 142,871.2 | 449,885.4 | 3.149x |
| 1024 | cast | `int64` | `contiguous` | `cast_long_float` | 1,048,576 element | 309,033.0 | 463,674.8 | 1.500x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_add_float` | 1,048,576 element | 148,799.7 | 190,842.2 | 1.283x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 1,048,576 element | 276,931.6 | 168,716.1 | 0.609x |
| 1024 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 1,048,576 element | 316,883.0 | 144,787.2 | 0.457x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_add_int` | 1,048,576 element | 148,536.8 | 185,011.3 | 1.246x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 1,048,576 element | 311,492.9 | 211,812.7 | 0.680x |
| 1024 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 1,048,576 element | 639,079.8 | 362,380.7 | 0.567x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_add_long` | 1,048,576 element | 263,748.1 | 679,672.0 | 2.577x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 1,048,576 element | 457,636.1 | 790,421.6 | 1.727x |
| 1024 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 1,048,576 element | 651,320.5 | 1,310,844.2 | 2.013x |
| 1024 | reduction | `float64` | `contiguous` | `full_product_double` | 1,048,576 element | 1,163,102.6 | 1,154,641.7 | 0.993x |
| 1024 | reduction | `float64` | `contiguous` | `full_min_double` | 1,048,576 element | 582,267.6 | 90,893.4 | 0.156x |
| 1024 | reduction | `float64` | `contiguous` | `full_max_double` | 1,048,576 element | 582,825.1 | 90,914.3 | 0.156x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 1,048,576 element | 307,504.3 | 655,590.3 | 2.132x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 1,048,576 element | 308,448.8 | 656,987.2 | 2.130x |
| 1024 | reduction | `float64` | `contiguous` | `full_mean_double` | 1,048,576 element | 529,264.2 | 136,920.2 | 0.259x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_product_double` | 1,048,576 element | 1,270,043.7 | 152,114.9 | 0.120x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_product_double` | 1,048,576 element | 1,129,967.5 | 1,106,118.8 | 0.979x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_min_double` | 1,048,576 element | 1,055,724.5 | 156,290.7 | 0.148x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_max_double` | 1,048,576 element | 552,446.4 | 104,213.5 | 0.189x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 1,048,576 element | 1,278,515.2 | 2,265,994.3 | 1.772x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 1,048,576 element | 383,408.6 | 664,594.3 | 1.733x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 1,048,576 element | 1,021,527.8 | 157,682.5 | 0.154x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 1,048,576 element | 473,115.3 | 146,217.9 | 0.309x |
| 1024 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 524,288 element | 584,209.5 | 586,239.0 | 1.003x |
| 1024 | reduction | `float64` | `reversed` | `reverse_min_double` | 1,048,576 element | 610,252.3 | 390,446.7 | 0.640x |
| 1024 | reduction | `float64` | `transposed` | `transpose_max_double` | 1,048,576 element | 1,105,876.6 | 93,378.2 | 0.084x |
| 1024 | reduction | `float32` | `contiguous` | `full_sum_float` | 1,048,576 element | 325,372.5 | 142,708.4 | 0.439x |
| 1024 | reduction | `float32` | `contiguous` | `full_product_float` | 1,048,576 element | 1,160,406.8 | 1,192,884.9 | 1.028x |
| 1024 | reduction | `float32` | `contiguous` | `full_mean_float` | 1,048,576 element | 461,414.9 | 144,547.0 | 0.313x |
| 1024 | reduction | `int32` | `contiguous` | `full_sum_int` | 1,048,576 element | 292,655.6 | 44,162.6 | 0.151x |
| 1024 | reduction | `int32` | `contiguous` | `full_product_int` | 1,048,576 element | 875,582.9 | 57,233.7 | 0.065x |
| 1024 | reduction | `int64` | `contiguous` | `full_sum_long` | 1,048,576 element | 293,899.3 | 89,424.8 | 0.304x |
| 1024 | reduction | `int64` | `contiguous` | `full_product_long` | 1,048,576 element | 878,416.2 | 222,723.5 | 0.254x |
| 1024 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 1,048,576 element | 582,172.0 | 189,821.5 | 0.326x |
| 1024 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 1,048,576 element | 450,844.5 | 249,281.9 | 0.553x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_add_double` | 1,048,576 element | 4,478,230.1 | 141,814.6 | 0.032x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 1,048,576 element | 4,488,477.9 | 140,608.4 | 0.031x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 1,048,576 element | 4,479,400.1 | 139,961.9 | 0.031x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 1,048,576 element | 4,519,471.3 | 151,783.3 | 0.034x |
| 1024 | in-place | `float32` | `contiguous` | `inplace_add_float` | 1,048,576 element | 4,264,424.4 | 70,433.2 | 0.017x |
| 1024 | in-place | `int32` | `contiguous` | `inplace_add_int` | 1,048,576 element | 3,595,339.7 | 70,504.4 | 0.020x |
| 1024 | in-place | `int64` | `contiguous` | `inplace_add_long` | 1,048,576 element | 3,887,110.4 | 141,711.9 | 0.036x |
| 1024 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 1,048,576 element | 13,430,404.9 | 141,060.4 | 0.011x |
| 1024 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 524,288 element | 6,468,848.1 | 181,020.8 | 0.028x |

A speed value above 1.0x favors Ravel. In-place rows reuse their destination; allocating array operations include result allocation. All other interpretation cautions from the access-pattern suite apply.
