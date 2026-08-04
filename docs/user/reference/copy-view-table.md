# Copy and view behavior

| Operation | Result | Notes |
|---|---|---|
| `select`, `slice`, `narrow`, `reverse` | view | preserves ownership kind |
| `swapAxes`, `permuteAxes`, `transpose` | view | preserves ownership kind |
| `newAxis`, `squeeze`, `broadcastTo` | view | broadcasting is immutable/borrowed only |
| owned `reshapeView` | owned view or error | never copies |
| borrowed `reshapeView` | borrowed view or error | never copies or erases provenance |
| owned `reshape` | owned view or owned copy | copies only when view reshape is illegal |
| borrowed `reshape` | unavailable | caller must choose `reshapeView` or `reshapeCopy` |
| `reshapeCopy` | owned copy | always materializes, including borrowed input |
| owned `contiguous` | same value or owned copy | returns `this` when already contiguous |
| borrowed `contiguous` | owned copy | removes external aliasing |
| `copy`, `flattenCopy` | owned copy | logical row-major order |
| `mutableCopy` | mutable copy | source is unchanged |
| `freezeCopy` | owned copy | mutable source remains separate |
| `map`, arithmetic, comparisons | owned allocation | eager |
| unary `+` | owned copy | eager snapshot for owned, borrowed, or mutable input |
| array-valued reductions | owned allocation | eager; one multi-axis plan and output; an empty `Axes` still copies |
| `NDArray.build` | owned value | one destination, no final output-sized copy |
| `JvmInterop.copyToArray` / typed-array copy | platform copy | logical row-major order |
| `unsafeBorrow` | borrowed view | later external mutation is observable |

“View” means the result shares the same underlying storage and changes only
shape, strides, or offset. It does not mean the result is necessarily
contiguous or covers the whole buffer.
