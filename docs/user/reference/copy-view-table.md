# Copy and view behavior

| Operation | Result | Notes |
|---|---|---|
| `select`, `slice`, `narrow`, `reverse` | view | preserves ownership kind |
| `swapAxes`, `permuteAxes`, `transpose` | view | preserves ownership kind |
| `newAxis`, `squeeze`, `broadcastTo` | view | broadcasting is immutable/borrowed only |
| `reshapeView` | view or error | never copies |
| `reshape` | view or owned copy | copies only when view reshape is illegal |
| `reshapeCopy` | owned copy | always materializes |
| owned `contiguous` | same value or owned copy | returns `this` when already contiguous |
| borrowed `contiguous` | owned copy | removes external aliasing |
| `copy`, `flattenCopy` | owned copy | logical row-major order |
| `mutableCopy` | mutable copy | source is unchanged |
| `freezeCopy` | owned copy | mutable source remains separate |
| `map`, arithmetic, comparisons | owned allocation | eager |
| array-valued reductions | owned allocation | eager |
| `NDArray.build` | owned value | one destination, no final output-sized copy |
| `JvmInterop.copyToArray` / typed-array copy | platform copy | logical row-major order |
| `unsafeBorrow` | borrowed view | later external mutation is observable |

“View” means the result shares the same underlying storage and changes only
shape, strides, or offset. It does not mean the result is necessarily
contiguous or covers the whole buffer.
