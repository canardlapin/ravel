# Scalar indexing optimization results

This directory records the local JDK 22 measurements for rank-specific scalar
reads and mutable updates. The rank-one through rank-four APIs now compute a
physical address directly from the validated layout. The arbitrary-rank
`at(IArray)` and `updateAt(IArray)` APIs are unchanged.

These measurements use JMH 1.37 with two forks, five 500 ms warmup iterations,
seven 500 ms measurement iterations, and the GC profiler. The host is Apple
arm64 on macOS 14.3 with OpenJDK 22+36-2370.

## Reads

| order | side | traversal time | time per read | bytes per traversal | same-host speedup |
|---|---:|---:|---:|---:|---:|
| row-major | 256 | 45.60 us | 0.696 ns | 0.624 B | 22.03x |
| row-major | 1024 | 724.47 us | 0.691 ns | 9.932 B | 21.80x |
| column-major | 256 | 45.02 us | 0.687 ns | 0.617 B | 22.00x |
| column-major | 1024 | 935.67 us | 0.892 ns | 12.808 B | 17.25x |

The normalized allocation is profiler noise for an entire traversal. Dividing
by the number of reads yields about 0.000009-0.000012 B per access. Before this
change, each rank-two read allocated about 152 B for its temporary `IArray`.

## Mutable updates

| order | side | traversal time | time per update | bytes per traversal |
|---|---:|---:|---:|---:|
| row-major | 256 | 24.92 us | 0.380 ns | 24.341 B |
| row-major | 1024 | 376.46 us | 0.359 ns | 29.144 B |
| column-major | 256 | 134.99 us | 2.060 ns | 25.843 B |
| column-major | 1024 | 4026.39 us | 3.840 ns | 78.860 B |

The update measurements include address calculation and value generation. The
larger column-major cost reflects cache-unfriendly writes. Allocation remains
profiler noise when normalized by 65,536 or 1,048,576 updates.

## Correctness and generated code

The focused indexing, rank, mutable, and view suites pass 30 tests on the JVM
and the same 30 tests on Scala.js. They cover contiguous arrays, offset slices,
negative strides, transposes, broadcasts, empty dimensions, wrong arity,
out-of-bounds indices, and mutable view locality.

JVM bytecode for rank-two access contains direct bounds checks and address
arithmetic, with no temporary index-array construction. Optimized Scala.js
inlines the same checks and address calculation, then indexes the matching
typed array directly. Error messages are built only on failure.
