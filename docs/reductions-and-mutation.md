# Reductions and mutation

Floating sums use a fixed block-pairwise schedule:

1. Traverse values in logical row-major order, independent of physical layout.
2. Accumulate consecutive blocks of 128 values from left to right.
3. Merge adjacent block totals into the front of the block array.
4. Carry an unpaired total unchanged to the next level.
5. Repeat until one total remains.

`Float.sum` uses `Float` at every addition. `Float.sumAs[Double]` converts each
value to `Double` before the schedule. `Float.mean` uses that `Double` schedule
and rounds once at the end. `Double.sum` and `Double.mean` use `Double`
throughout.

Empty sum and product return their positive identities. Empty floating mean is
NaN. Min, max, arg-min, and arg-max reject an empty reduction fiber. NaNs
propagate; arg reductions choose the first logical NaN. Ordinary ties choose
the first logical occurrence. Min chooses negative zero and max chooses
positive zero.

## Mutable arrays

`MutableNDArray` is the only ordinary mutation surface. `mutableCopy` and
`freezeCopy` always copy. Mutable layouts have no constructor from arbitrary
strides: they begin canonical and can only pass through selection, nonzero-step
slicing, reversal, permutation, singleton insertion/removal, or legal
no-copy reshape. Each transformation preserves injectivity. Broadcasting is
intentionally unavailable because a non-singleton zero stride aliases cells.

The expert `ravel.kernel` API writes into caller-owned whole contiguous mutable
buffers. Built-in `addInto` and `multiplyInto` use specialized kernels;
callback `mapInto` and `zipMapInto` preserve once-per-element logical-order
callback semantics.
