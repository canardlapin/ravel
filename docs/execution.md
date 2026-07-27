# Eager execution and broadcasting

Ravel evaluates ordinary numerical syntax immediately. Built-in operations
allocate one contiguous output. A compound expression such as `x * y + b`
therefore may allocate an intermediate in 1.0. There is no expression graph,
JIT, autodiff tape, or backend registry in `ravel-core`.

Broadcasting aligns trailing axes. Equal dimensions remain unchanged; a
dimension of one expands to the other dimension. In particular, dimensions
zero and one produce zero. This is not implemented as `max(left, right)`.
Expanded operand axes receive stride zero.

The planner validates shapes and offsets, removes length-one axes, and
coalesces adjacent axes only when every operand is contiguous across the
boundary or has zero stride on both sides. It classifies the resulting work as:

- linear contiguous;
- scalar broadcast;
- simple inner-strided;
- fully general strided.

The general loop uses counters and physical offsets. It does not divide a
linear index by dimension sizes.

Built-in numerical operations dispatch once on dtype and then execute a loop
over concrete primitive JVM arrays or Scala.js typed arrays. User callbacks are
different by design: `map`, `zipMap`, and their `*Into` counterparts promise
one callback invocation per logical output element in logical row-major order.
They are never fused or reordered and make no no-boxing promise. Callback
exceptions stop evaluation; a partially filled output is not returned.

## Structural evidence

The 2026-07-27 Phase 3 gate ran 42 JVM and 44 Scala.js tests. JVM `javap`
inspection found primitive `daload`/`dadd`/`dastore` operations in the
`Double` loop and no `DType`, `Numeric`, `Ordering`, iterator, callback, or
generic storage call inside it. The optimized Scala.js `Double` branch performs
one `DoubleStorage` dispatch and then reads and writes `Float64Array` values
directly while maintaining integer counters and physical offsets.

On the gate host, the final descriptive Node benchmark processed approximately
724 million elements/s through the raw Phase 0 loop and 156 million elements/s
through the complete public allocation/planning/addition path. The quantitative
release budget and representative strided result are recorded in
`benchmark-baselines.md`.

The corresponding JVM public dispatcher originally compiled into one method
too large for HotSpot optimization. Splitting dtype dispatch from the
monomorphic methods raised the 65,536-element public addition benchmark from
about 86 to 6,748 operations/s. The release gate therefore inspects method
structure in addition to measuring throughput.
