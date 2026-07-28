# Ravel JVM vs NumPy public-operation matrix

Semantic parity passed for every reported row before timing comparison. This broad matrix locates follow-up targets; focused JMH controls remain the authority for optimization claims.

Ravel: Java 25.0.1 (Homebrew), Mac OS X aarch64. NumPy 2.4.3 on Python 3.14.3, macOS-14.3-arm64-arm-64bit-Mach-O.

## Family summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `in-place` | 18 | 0.990x | 0.499x | 1.360x |
| `reduction` | 2 | 0.904x | 0.819x | 0.998x |

## Input dtype summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `float32` | 4 | 1.003x | 0.819x | 1.179x |
| `float64` | 12 | 0.960x | 0.499x | 1.360x |
| `int32` | 2 | 1.044x | 1.027x | 1.062x |
| `int64` | 2 | 1.010x | 1.005x | 1.015x |

## Input layout summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `contiguous` | 16 | 1.025x | 0.819x | 1.179x |
| `inner_stride` | 2 | 1.336x | 1.312x | 1.360x |
| `reversed` | 2 | 0.508x | 0.499x | 0.518x |

## Detailed results

| side | family | dtype | layout | case | work | Ravel ns/op | NumPy ns/op | Ravel speed |
|---:|---|---|---|---|---:|---:|---:|---:|
| 256 | reduction | `float32` | `contiguous` | `full_sum_float` | 65,536 element | 10,834.0 | 10,817.3 | 0.998x |
| 256 | in-place | `float64` | `contiguous` | `inplace_add_double` | 65,536 element | 8,400.9 | 9,572.4 | 1.139x |
| 256 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 65,536 element | 9,045.6 | 9,511.1 | 1.051x |
| 256 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 65,536 element | 9,145.5 | 9,626.0 | 1.053x |
| 256 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 65,536 element | 10,350.6 | 10,188.6 | 0.984x |
| 256 | in-place | `float32` | `contiguous` | `inplace_add_float` | 65,536 element | 4,390.3 | 5,175.5 | 1.179x |
| 256 | in-place | `int32` | `contiguous` | `inplace_add_int` | 65,536 element | 4,841.8 | 5,140.7 | 1.062x |
| 256 | in-place | `int64` | `contiguous` | `inplace_add_long` | 65,536 element | 9,605.7 | 9,653.6 | 1.005x |
| 256 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 65,536 element | 20,071.4 | 10,011.5 | 0.499x |
| 256 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 32,768 element | 9,598.6 | 12,595.3 | 1.312x |
| 1024 | reduction | `float32` | `contiguous` | `full_sum_float` | 1,048,576 element | 178,037.9 | 145,825.4 | 0.819x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_add_double` | 1,048,576 element | 134,956.2 | 149,761.9 | 1.110x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 1,048,576 element | 143,230.5 | 149,814.1 | 1.046x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 1,048,576 element | 147,805.1 | 151,355.2 | 1.024x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 1,048,576 element | 181,685.0 | 162,900.2 | 0.897x |
| 1024 | in-place | `float32` | `contiguous` | `inplace_add_float` | 1,048,576 element | 70,535.4 | 74,049.2 | 1.050x |
| 1024 | in-place | `int32` | `contiguous` | `inplace_add_int` | 1,048,576 element | 73,429.2 | 75,414.0 | 1.027x |
| 1024 | in-place | `int64` | `contiguous` | `inplace_add_long` | 1,048,576 element | 150,886.7 | 153,177.2 | 1.015x |
| 1024 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 1,048,576 element | 293,982.9 | 152,352.3 | 0.518x |
| 1024 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 524,288 element | 154,428.6 | 210,085.0 | 1.360x |

A speed value above 1.0x favors Ravel. In-place rows reuse their destination; allocating array operations include result allocation. All other interpretation cautions from the access-pattern suite apply.
