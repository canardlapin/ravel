# Dtypes and ranks

## Dtype capabilities

| Dtype | Storage | Ordering | Same-dtype arithmetic | `/` | Explicit cast |
|---|---:|---:|---:|---:|---:|
| `Boolean` | yes | no | no | no | no |
| `Byte` | yes | yes | no | no | yes |
| `Short` | yes | yes | no | no | yes |
| `Int` | yes | yes | yes | `quot` / `truncDiv` | yes |
| `Long` | yes | yes | yes | `quot` / `truncDiv` | yes |
| `Float` | yes | yes | yes | yes | yes |
| `Double` | yes | yes | yes | yes | yes |

Integer operations use ordinary two's-complement overflow. Floating-to-integer
casts truncate toward zero, map NaN to zero, and clamp infinities and
out-of-range values before any final `Byte` or `Short` narrowing.

## Reduction behavior

- Empty sum is positive zero; empty product is one.
- Empty floating mean is NaN.
- Empty min, max, arg-min, and arg-max throw `EmptyReduction`.
- NaNs propagate through floating min, max, sum, and mean.
- Arg reductions choose the first logical NaN; ordinary ties choose the first
  logical occurrence.
- `Float.sumAs[Double]` and `Int.sumAs[Long]` are the supported widening sums.

## Rank aliases

| Alias | Expanded type |
|---|---|
| `Array0[A]` | `NDArray[A, Rank[0]]` |
| `Array1[A]` | `NDArray[A, Rank[1]]` |
| `Array2[A]` | `NDArray[A, Rank[2]]` |
| `Array3[A]` | `NDArray[A, Rank[3]]` |
| `Array4[A]` | `NDArray[A, Rank[4]]` |
| `AnyNDArray[A]` | `NDArray[A, AnyRank]` |

Dimension sizes are always runtime `Int` values. Static rank exists to make
axis-adding and axis-dropping APIs more precise, not to encode dimensions in
types.
