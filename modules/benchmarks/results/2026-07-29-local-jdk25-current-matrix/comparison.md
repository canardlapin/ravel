# Ravel JVM vs NumPy public-operation matrix

Semantic parity passed for every reported row before timing comparison. This broad matrix locates follow-up targets; focused JMH controls remain the authority for optimization claims.

Ravel: Java 25.0.1 (Homebrew), Mac OS X aarch64. NumPy 2.4.3 on Python 3.14.3, macOS-14.3-arm64-arm-64bit-Mach-O.

## Family summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `binary` | 24 | 0.893x | 0.420x | 1.727x |
| `cast` | 8 | 0.912x | 0.406x | 1.739x |
| `comparison` | 12 | 0.530x | 0.242x | 0.874x |
| `in-place` | 18 | 0.812x | 0.146x | 1.116x |
| `predicate` | 4 | 0.277x | 0.246x | 0.324x |
| `reduction` | 52 | 0.701x | 0.264x | 1.741x |
| `scalar` | 18 | 0.909x | 0.272x | 2.855x |
| `unary` | 22 | 0.830x | 0.309x | 3.009x |

## Input dtype summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `float32` | 18 | 0.720x | 0.272x | 1.342x |
| `float64` | 110 | 0.756x | 0.146x | 3.009x |
| `int32` | 16 | 0.584x | 0.296x | 1.727x |
| `int64` | 14 | 1.071x | 0.547x | 1.880x |

## Input layout summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `broadcast` | 4 | 0.638x | 0.376x | 1.186x |
| `contiguous` | 132 | 0.756x | 0.242x | 3.009x |
| `inner_stride` | 8 | 0.825x | 0.550x | 1.346x |
| `outer_stride` | 4 | 1.007x | 0.827x | 1.459x |
| `reversed` | 6 | 0.693x | 0.146x | 1.553x |
| `transposed` | 4 | 0.627x | 0.417x | 1.145x |

## Detailed results

| side | family | dtype | layout | case | work | Ravel ns/op | NumPy ns/op | Ravel speed |
|---:|---|---|---|---|---:|---:|---:|---:|
| 256 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 65,536 element | 19,166.1 | 11,117.1 | 0.580x |
| 256 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 32,768 element | 17,600.6 | 10,485.3 | 0.596x |
| 256 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 32,768 element | 19,809.3 | 19,294.6 | 0.974x |
| 256 | binary | `float64` | `reversed` | `reverse_minimum_double` | 65,536 element | 47,835.1 | 35,174.8 | 0.735x |
| 256 | binary | `float64` | `transposed` | `transpose_maximum_double` | 65,536 element | 78,130.6 | 89,492.8 | 1.145x |
| 256 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 65,536 element | 39,553.8 | 22,686.5 | 0.574x |
| 256 | scalar | `float64` | `contiguous` | `scalar_add_double` | 65,536 element | 17,191.4 | 9,873.5 | 0.574x |
| 256 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 65,536 element | 16,949.4 | 7,684.1 | 0.453x |
| 256 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 65,536 element | 16,812.8 | 7,621.0 | 0.453x |
| 256 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 65,536 element | 21,534.0 | 9,452.9 | 0.439x |
| 256 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 65,536 element | 16,995.4 | 20,066.6 | 1.181x |
| 256 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 65,536 element | 17,654.0 | 19,100.1 | 1.082x |
| 256 | unary | `float64` | `contiguous` | `clip_double` | 65,536 element | 15,970.6 | 20,130.7 | 1.260x |
| 256 | unary | `float64` | `contiguous` | `negate_double` | 65,536 element | 17,002.5 | 9,718.0 | 0.572x |
| 256 | unary | `float64` | `contiguous` | `abs_double` | 65,536 element | 18,075.7 | 9,998.4 | 0.553x |
| 256 | unary | `float64` | `contiguous` | `sqrt_double` | 65,536 element | 40,559.0 | 17,902.9 | 0.441x |
| 256 | unary | `float64` | `contiguous` | `exp_double` | 65,536 element | 293,345.5 | 149,737.3 | 0.510x |
| 256 | unary | `float64` | `contiguous` | `log_double` | 65,536 element | 279,651.5 | 150,210.2 | 0.537x |
| 256 | unary | `float64` | `contiguous` | `sin_double` | 65,536 element | 288,219.3 | 195,614.9 | 0.679x |
| 256 | unary | `float64` | `contiguous` | `cos_double` | 65,536 element | 300,206.8 | 198,171.2 | 0.660x |
| 256 | unary | `float64` | `contiguous` | `tan_double` | 65,536 element | 806,429.8 | 249,018.7 | 0.309x |
| 256 | unary | `float64` | `contiguous` | `floor_double` | 65,536 element | 17,387.8 | 9,847.6 | 0.566x |
| 256 | unary | `float64` | `contiguous` | `ceil_double` | 65,536 element | 17,207.1 | 10,707.2 | 0.622x |
| 256 | comparison | `float64` | `contiguous` | `equal_double` | 65,536 element | 21,317.3 | 15,558.5 | 0.730x |
| 256 | comparison | `float64` | `contiguous` | `not_equal_double` | 65,536 element | 26,993.4 | 15,668.7 | 0.580x |
| 256 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 32,768 element | 18,211.2 | 10,015.5 | 0.550x |
| 256 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 32,768 element | 24,245.8 | 20,051.2 | 0.827x |
| 256 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 65,536 element | 38,747.3 | 25,104.2 | 0.648x |
| 256 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 65,536 element | 50,897.9 | 12,325.6 | 0.242x |
| 256 | predicate | `float64` | `contiguous` | `is_nan_double` | 65,536 element | 17,509.2 | 5,670.3 | 0.324x |
| 256 | predicate | `float64` | `contiguous` | `is_finite_double` | 65,536 element | 20,648.3 | 5,470.9 | 0.265x |
| 256 | cast | `float64` | `contiguous` | `cast_double_int` | 65,536 element | 13,725.9 | 19,295.6 | 1.406x |
| 256 | cast | `float32` | `contiguous` | `cast_float_double` | 65,536 element | 18,387.3 | 7,473.8 | 0.406x |
| 256 | cast | `int32` | `contiguous` | `cast_int_long` | 65,536 element | 14,320.8 | 7,349.1 | 0.513x |
| 256 | cast | `int64` | `contiguous` | `cast_long_float` | 65,536 element | 20,014.8 | 17,965.0 | 0.898x |
| 256 | binary | `float32` | `contiguous` | `contiguous_add_float` | 65,536 element | 9,706.2 | 7,697.3 | 0.793x |
| 256 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 65,536 element | 17,329.0 | 8,695.5 | 0.502x |
| 256 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 65,536 element | 20,229.2 | 5,508.4 | 0.272x |
| 256 | binary | `int32` | `contiguous` | `contiguous_add_int` | 65,536 element | 9,841.0 | 7,869.1 | 0.800x |
| 256 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 65,536 element | 19,217.5 | 8,080.8 | 0.420x |
| 256 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 65,536 element | 42,416.7 | 17,050.4 | 0.402x |
| 256 | binary | `int64` | `contiguous` | `contiguous_add_long` | 65,536 element | 20,891.6 | 14,542.8 | 0.696x |
| 256 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 65,536 element | 24,489.5 | 19,827.6 | 0.810x |
| 256 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 65,536 element | 43,945.5 | 50,915.4 | 1.159x |
| 256 | reduction | `float64` | `contiguous` | `full_product_double` | 65,536 element | 74,951.1 | 71,089.0 | 0.948x |
| 256 | reduction | `float64` | `contiguous` | `full_min_double` | 65,536 element | 16,512.5 | 6,897.5 | 0.418x |
| 256 | reduction | `float64` | `contiguous` | `full_max_double` | 65,536 element | 15,568.6 | 6,706.6 | 0.431x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 65,536 element | 35,685.7 | 41,808.4 | 1.172x |
| 256 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 65,536 element | 24,846.1 | 41,143.8 | 1.656x |
| 256 | reduction | `float64` | `contiguous` | `full_mean_double` | 65,536 element | 13,025.5 | 10,042.4 | 0.771x |
| 256 | reduction | `float64` | `contiguous` | `axis0_product_double` | 65,536 element | 19,163.1 | 11,042.4 | 0.576x |
| 256 | reduction | `float64` | `contiguous` | `axis1_product_double` | 65,536 element | 65,781.9 | 55,495.8 | 0.844x |
| 256 | reduction | `float64` | `contiguous` | `axis0_min_double` | 65,536 element | 19,480.8 | 11,054.3 | 0.567x |
| 256 | reduction | `float64` | `contiguous` | `axis1_max_double` | 65,536 element | 19,043.5 | 8,893.3 | 0.467x |
| 256 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 65,536 element | 76,341.9 | 78,716.0 | 1.031x |
| 256 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 65,536 element | 27,744.6 | 35,216.3 | 1.269x |
| 256 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 65,536 element | 19,340.9 | 11,921.6 | 0.616x |
| 256 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 65,536 element | 18,689.2 | 11,964.7 | 0.640x |
| 256 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 32,768 element | 45,912.3 | 37,094.4 | 0.808x |
| 256 | reduction | `float64` | `reversed` | `reverse_min_double` | 65,536 element | 17,953.6 | 27,889.3 | 1.553x |
| 256 | reduction | `float64` | `transposed` | `transpose_max_double` | 65,536 element | 12,068.2 | 6,572.8 | 0.545x |
| 256 | reduction | `float32` | `contiguous` | `full_sum_float` | 65,536 element | 10,755.2 | 9,788.8 | 0.910x |
| 256 | reduction | `float32` | `contiguous` | `full_product_float` | 65,536 element | 74,655.8 | 72,666.1 | 0.973x |
| 256 | reduction | `float32` | `contiguous` | `full_mean_float` | 65,536 element | 12,822.0 | 11,468.8 | 0.894x |
| 256 | reduction | `int32` | `contiguous` | `full_sum_int` | 65,536 element | 9,481.7 | 3,836.3 | 0.405x |
| 256 | reduction | `int32` | `contiguous` | `full_product_int` | 65,536 element | 10,585.0 | 4,776.1 | 0.451x |
| 256 | reduction | `int64` | `contiguous` | `full_sum_long` | 65,536 element | 8,916.6 | 6,714.7 | 0.753x |
| 256 | reduction | `int64` | `contiguous` | `full_product_long` | 65,536 element | 9,282.4 | 14,264.3 | 1.537x |
| 256 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 65,536 element | 36,707.8 | 13,341.2 | 0.363x |
| 256 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 65,536 element | 13,265.5 | 17,808.1 | 1.342x |
| 256 | in-place | `float64` | `contiguous` | `inplace_add_double` | 65,536 element | 8,593.6 | 9,587.4 | 1.116x |
| 256 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 65,536 element | 9,098.2 | 9,264.8 | 1.018x |
| 256 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 65,536 element | 12,843.8 | 9,387.0 | 0.731x |
| 256 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 65,536 element | 10,782.4 | 9,648.0 | 0.895x |
| 256 | in-place | `float32` | `contiguous` | `inplace_add_float` | 65,536 element | 4,467.7 | 4,904.0 | 1.098x |
| 256 | in-place | `int32` | `contiguous` | `inplace_add_int` | 65,536 element | 4,732.6 | 4,889.4 | 1.033x |
| 256 | in-place | `int64` | `contiguous` | `inplace_add_long` | 65,536 element | 8,969.2 | 9,142.2 | 1.019x |
| 256 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 65,536 element | 62,183.3 | 9,069.8 | 0.146x |
| 256 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 32,768 element | 10,434.1 | 11,269.5 | 1.080x |
| 1024 | binary | `float64` | `contiguous` | `contiguous_subtract_double` | 1,048,576 element | 313,789.4 | 375,998.3 | 1.198x |
| 1024 | binary | `float64` | `inner_stride` | `inner_stride_multiply_double` | 524,288 element | 274,336.4 | 369,324.2 | 1.346x |
| 1024 | binary | `float64` | `outer_stride` | `outer_stride_divide_double` | 524,288 element | 325,371.5 | 474,685.0 | 1.459x |
| 1024 | binary | `float64` | `reversed` | `reverse_minimum_double` | 1,048,576 element | 626,264.7 | 943,717.5 | 1.507x |
| 1024 | binary | `float64` | `transposed` | `transpose_maximum_double` | 1,048,576 element | 4,655,357.0 | 2,769,609.3 | 0.595x |
| 1024 | binary | `float64` | `broadcast` | `broadcast_subtract_double` | 1,048,576 element | 666,869.7 | 791,104.3 | 1.186x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_add_double` | 1,048,576 element | 262,286.5 | 544,079.2 | 2.074x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_subtract_double` | 1,048,576 element | 260,269.8 | 157,658.5 | 0.606x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_multiply_double` | 1,048,576 element | 269,734.3 | 492,426.6 | 1.826x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_divide_double` | 1,048,576 element | 342,758.4 | 518,968.6 | 1.514x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_minimum_double` | 1,048,576 element | 276,536.5 | 749,887.3 | 2.712x |
| 1024 | scalar | `float64` | `contiguous` | `scalar_maximum_double` | 1,048,576 element | 263,405.8 | 751,963.4 | 2.855x |
| 1024 | unary | `float64` | `contiguous` | `clip_double` | 1,048,576 element | 250,191.4 | 752,914.7 | 3.009x |
| 1024 | unary | `float64` | `contiguous` | `negate_double` | 1,048,576 element | 265,510.7 | 673,651.1 | 2.537x |
| 1024 | unary | `float64` | `contiguous` | `abs_double` | 1,048,576 element | 269,922.1 | 698,944.9 | 2.589x |
| 1024 | unary | `float64` | `contiguous` | `sqrt_double` | 1,048,576 element | 637,719.2 | 782,313.7 | 1.227x |
| 1024 | unary | `float64` | `contiguous` | `exp_double` | 1,048,576 element | 5,254,905.6 | 2,950,413.3 | 0.561x |
| 1024 | unary | `float64` | `contiguous` | `log_double` | 1,048,576 element | 5,065,456.2 | 3,267,841.3 | 0.645x |
| 1024 | unary | `float64` | `contiguous` | `sin_double` | 1,048,576 element | 5,187,973.1 | 4,051,779.6 | 0.781x |
| 1024 | unary | `float64` | `contiguous` | `cos_double` | 1,048,576 element | 5,340,537.4 | 3,948,371.0 | 0.739x |
| 1024 | unary | `float64` | `contiguous` | `tan_double` | 1,048,576 element | 13,950,495.3 | 4,541,376.2 | 0.326x |
| 1024 | unary | `float64` | `contiguous` | `floor_double` | 1,048,576 element | 282,330.5 | 632,960.7 | 2.242x |
| 1024 | unary | `float64` | `contiguous` | `ceil_double` | 1,048,576 element | 282,539.4 | 534,296.4 | 1.891x |
| 1024 | comparison | `float64` | `contiguous` | `equal_double` | 1,048,576 element | 375,710.7 | 236,889.2 | 0.631x |
| 1024 | comparison | `float64` | `contiguous` | `not_equal_double` | 1,048,576 element | 508,130.5 | 234,568.2 | 0.462x |
| 1024 | comparison | `float64` | `inner_stride` | `less_inner_stride_double` | 524,288 element | 298,553.8 | 164,925.3 | 0.552x |
| 1024 | comparison | `float64` | `outer_stride` | `less_equal_outer_stride_double` | 524,288 element | 363,033.4 | 317,172.8 | 0.874x |
| 1024 | comparison | `float64` | `broadcast` | `greater_broadcast_double` | 1,048,576 element | 1,009,811.0 | 379,384.4 | 0.376x |
| 1024 | comparison | `float64` | `contiguous` | `greater_equal_scalar_double` | 1,048,576 element | 613,180.4 | 187,226.0 | 0.305x |
| 1024 | predicate | `float64` | `contiguous` | `is_nan_double` | 1,048,576 element | 297,008.5 | 83,236.7 | 0.280x |
| 1024 | predicate | `float64` | `contiguous` | `is_finite_double` | 1,048,576 element | 341,782.4 | 83,974.2 | 0.246x |
| 1024 | cast | `float64` | `contiguous` | `cast_double_int` | 1,048,576 element | 264,596.4 | 460,195.7 | 1.739x |
| 1024 | cast | `float32` | `contiguous` | `cast_float_double` | 1,048,576 element | 241,458.0 | 117,644.4 | 0.487x |
| 1024 | cast | `int32` | `contiguous` | `cast_int_long` | 1,048,576 element | 258,681.7 | 433,259.0 | 1.675x |
| 1024 | cast | `int64` | `contiguous` | `cast_long_float` | 1,048,576 element | 364,021.0 | 464,953.1 | 1.277x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_add_float` | 1,048,576 element | 151,704.6 | 182,373.8 | 1.202x |
| 1024 | binary | `float32` | `contiguous` | `contiguous_multiply_float` | 1,048,576 element | 278,310.8 | 153,895.8 | 0.553x |
| 1024 | scalar | `float32` | `contiguous` | `scalar_divide_float` | 1,048,576 element | 314,238.6 | 222,604.0 | 0.708x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_add_int` | 1,048,576 element | 151,277.7 | 261,181.4 | 1.727x |
| 1024 | binary | `int32` | `contiguous` | `contiguous_multiply_int` | 1,048,576 element | 324,700.7 | 240,320.7 | 0.740x |
| 1024 | scalar | `int32` | `contiguous` | `scalar_quot_int` | 1,048,576 element | 656,541.5 | 392,075.4 | 0.597x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_add_long` | 1,048,576 element | 340,941.4 | 570,381.3 | 1.673x |
| 1024 | binary | `int64` | `contiguous` | `contiguous_multiply_long` | 1,048,576 element | 454,003.3 | 685,899.3 | 1.511x |
| 1024 | scalar | `int64` | `contiguous` | `scalar_quot_long` | 1,048,576 element | 691,219.6 | 1,299,816.8 | 1.880x |
| 1024 | reduction | `float64` | `contiguous` | `full_product_double` | 1,048,576 element | 1,637,908.9 | 1,145,634.8 | 0.699x |
| 1024 | reduction | `float64` | `contiguous` | `full_min_double` | 1,048,576 element | 257,912.6 | 87,711.3 | 0.340x |
| 1024 | reduction | `float64` | `contiguous` | `full_max_double` | 1,048,576 element | 277,761.0 | 87,490.0 | 0.315x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_min_double` | 1,048,576 element | 373,205.4 | 649,567.3 | 1.741x |
| 1024 | reduction | `float64` | `contiguous` | `full_arg_max_double` | 1,048,576 element | 454,719.0 | 655,245.2 | 1.441x |
| 1024 | reduction | `float64` | `contiguous` | `full_mean_double` | 1,048,576 element | 187,975.2 | 133,152.9 | 0.708x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_product_double` | 1,048,576 element | 315,196.4 | 152,929.0 | 0.485x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_product_double` | 1,048,576 element | 1,493,049.7 | 1,076,596.7 | 0.721x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_min_double` | 1,048,576 element | 311,283.2 | 152,177.1 | 0.489x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_max_double` | 1,048,576 element | 363,760.5 | 95,953.0 | 0.264x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_arg_min_double` | 1,048,576 element | 2,054,296.5 | 2,310,026.9 | 1.124x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_arg_max_double` | 1,048,576 element | 400,939.8 | 661,239.5 | 1.649x |
| 1024 | reduction | `float64` | `contiguous` | `axis0_mean_double` | 1,048,576 element | 309,146.0 | 158,515.6 | 0.513x |
| 1024 | reduction | `float64` | `contiguous` | `axis1_mean_double` | 1,048,576 element | 264,722.0 | 149,515.7 | 0.565x |
| 1024 | reduction | `float64` | `inner_stride` | `inner_stride_product_double` | 524,288 element | 589,035.6 | 609,329.4 | 1.034x |
| 1024 | reduction | `float64` | `reversed` | `reverse_min_double` | 1,048,576 element | 304,640.2 | 387,939.0 | 1.273x |
| 1024 | reduction | `float64` | `transposed` | `transpose_max_double` | 1,048,576 element | 220,271.0 | 91,884.8 | 0.417x |
| 1024 | reduction | `float32` | `contiguous` | `full_sum_float` | 1,048,576 element | 176,441.4 | 141,683.5 | 0.803x |
| 1024 | reduction | `float32` | `contiguous` | `full_product_float` | 1,048,576 element | 1,181,426.7 | 1,186,599.5 | 1.004x |
| 1024 | reduction | `float32` | `contiguous` | `full_mean_float` | 1,048,576 element | 235,338.7 | 142,048.7 | 0.604x |
| 1024 | reduction | `int32` | `contiguous` | `full_sum_int` | 1,048,576 element | 140,151.6 | 42,524.0 | 0.303x |
| 1024 | reduction | `int32` | `contiguous` | `full_product_int` | 1,048,576 element | 153,603.6 | 55,027.0 | 0.358x |
| 1024 | reduction | `int64` | `contiguous` | `full_sum_long` | 1,048,576 element | 152,656.7 | 83,429.7 | 0.547x |
| 1024 | reduction | `int64` | `contiguous` | `full_product_long` | 1,048,576 element | 159,402.4 | 211,225.4 | 1.325x |
| 1024 | reduction | `int32` | `contiguous` | `sum_as_long_int` | 1,048,576 element | 628,474.6 | 185,758.7 | 0.296x |
| 1024 | reduction | `float32` | `contiguous` | `sum_as_double_float` | 1,048,576 element | 508,112.6 | 249,095.5 | 0.490x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_add_double` | 1,048,576 element | 138,828.0 | 132,680.8 | 0.956x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 1,048,576 element | 139,326.8 | 134,197.5 | 0.963x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 1,048,576 element | 147,416.3 | 131,325.2 | 0.891x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 1,048,576 element | 178,702.9 | 145,641.2 | 0.815x |
| 1024 | in-place | `float32` | `contiguous` | `inplace_add_float` | 1,048,576 element | 70,016.4 | 68,068.1 | 0.972x |
| 1024 | in-place | `int32` | `contiguous` | `inplace_add_int` | 1,048,576 element | 72,963.2 | 68,192.3 | 0.935x |
| 1024 | in-place | `int64` | `contiguous` | `inplace_add_long` | 1,048,576 element | 153,008.1 | 131,583.9 | 0.860x |
| 1024 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 1,048,576 element | 393,687.3 | 135,966.5 | 0.345x |
| 1024 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 524,288 element | 175,538.6 | 171,015.5 | 0.974x |

A speed value above 1.0x favors Ravel. In-place rows reuse their destination; allocating array operations include result allocation. All other interpretation cautions from the access-pattern suite apply.
