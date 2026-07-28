# Ravel JVM vs NumPy public-operation matrix

Semantic parity passed for every reported row before timing comparison. This broad matrix locates follow-up targets; focused JMH controls remain the authority for optimization claims.

Ravel: Java 21.0.11 (Eclipse Adoptium), Mac OS X aarch64. NumPy 2.4.3 on Python 3.14.3, macOS-14.3-arm64-arm-64bit-Mach-O.

## Family summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `in-place` | 18 | 0.967x | 0.428x | 1.194x |
| `reduction` | 2 | 0.895x | 0.854x | 0.938x |

## Input dtype summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `float32` | 4 | 0.989x | 0.854x | 1.154x |
| `float64` | 12 | 0.919x | 0.428x | 1.194x |
| `int32` | 2 | 1.082x | 1.024x | 1.143x |
| `int64` | 2 | 1.046x | 1.001x | 1.092x |

## Input layout summary

| group | rows | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|
| `contiguous` | 16 | 1.034x | 0.854x | 1.154x |
| `inner_stride` | 2 | 1.154x | 1.115x | 1.194x |
| `reversed` | 2 | 0.441x | 0.428x | 0.455x |

## Detailed results

| side | family | dtype | layout | case | work | Ravel ns/op | NumPy ns/op | Ravel speed |
|---:|---|---|---|---|---:|---:|---:|---:|
| 256 | reduction | `float32` | `contiguous` | `full_sum_float` | 65,536 element | 10,403.8 | 9,762.7 | 0.938x |
| 256 | in-place | `float64` | `contiguous` | `inplace_add_double` | 65,536 element | 8,540.4 | 9,233.8 | 1.081x |
| 256 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 65,536 element | 8,494.3 | 9,188.1 | 1.082x |
| 256 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 65,536 element | 8,504.1 | 9,145.8 | 1.075x |
| 256 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 65,536 element | 9,167.8 | 9,774.9 | 1.066x |
| 256 | in-place | `float32` | `contiguous` | `inplace_add_float` | 65,536 element | 4,275.5 | 4,936.0 | 1.154x |
| 256 | in-place | `int32` | `contiguous` | `inplace_add_int` | 65,536 element | 4,302.8 | 4,920.0 | 1.143x |
| 256 | in-place | `int64` | `contiguous` | `inplace_add_long` | 65,536 element | 8,512.5 | 9,298.6 | 1.092x |
| 256 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 65,536 element | 20,849.8 | 9,477.8 | 0.455x |
| 256 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 32,768 element | 10,026.6 | 11,975.3 | 1.194x |
| 1024 | reduction | `float32` | `contiguous` | `full_sum_float` | 1,048,576 element | 167,010.5 | 142,708.4 | 0.854x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_add_double` | 1,048,576 element | 139,097.6 | 141,814.6 | 1.020x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_subtract_double` | 1,048,576 element | 139,632.9 | 140,608.4 | 1.007x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_multiply_double` | 1,048,576 element | 139,673.6 | 139,961.9 | 1.002x |
| 1024 | in-place | `float64` | `contiguous` | `inplace_divide_double` | 1,048,576 element | 150,484.0 | 151,783.3 | 1.009x |
| 1024 | in-place | `float32` | `contiguous` | `inplace_add_float` | 1,048,576 element | 68,250.7 | 70,433.2 | 1.032x |
| 1024 | in-place | `int32` | `contiguous` | `inplace_add_int` | 1,048,576 element | 68,819.2 | 70,504.4 | 1.024x |
| 1024 | in-place | `int64` | `contiguous` | `inplace_add_long` | 1,048,576 element | 141,587.9 | 141,711.9 | 1.001x |
| 1024 | in-place | `float64` | `reversed` | `inplace_reverse_add_double` | 1,048,576 element | 329,432.1 | 141,060.4 | 0.428x |
| 1024 | in-place | `float64` | `inner_stride` | `inplace_inner_stride_multiply_double` | 524,288 element | 162,409.2 | 181,020.8 | 1.115x |

A speed value above 1.0x favors Ravel. In-place rows reuse their destination; allocating array operations include result allocation. All other interpretation cautions from the access-pattern suite apply.
