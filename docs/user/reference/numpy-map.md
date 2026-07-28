# NumPy operation map

Ravel deliberately resembles familiar eager ndarray operations, but it is not
a Scala spelling of all NumPy behavior.

| NumPy | Ravel | Important difference |
|---|---|---|
| `np.zeros((m, n))` | `NDArray.zeros[A](m, n)` | dtype comes from `A` |
| `np.full(shape, x)` | `NDArray.fill(Shape(...), x)` | shape is checked |
| `np.array(values).reshape(shape)` | `NDArray.fromSeq(shape, values)` | exact count required |
| `a[i, j]` | `a(i, j)` | fixed-rank overloads for ranks 1–4 |
| `a[tuple(indices)]` | `a.at(IArray(...))` | dynamic-rank path |
| `a[:, ::2]` | `a.slice(1, Slice.every(2))` | one axis per call |
| `a.T` | `a.transpose` | rank two only |
| `np.transpose(a, axes)` | `a.permuteAxes(axes*)` | checked axis permutation |
| `np.expand_dims(a, axis)` | `a.newAxis(axis)` | rank changes in the type when known |
| `np.squeeze(a, axis)` | `a.squeeze(axis)` | explicit axis required |
| `np.broadcast_to(a, shape)` | `a.broadcastTo(shape)` | immutable/borrowed view |
| `a.reshape(shape)` | `a.reshape(shape)` | may copy; use `reshapeView` to forbid it |
| `np.ascontiguousarray(a)` | `a.contiguous` | owned contiguous values may return themselves |
| `a.astype(dtype)` | `a.cast[B]` | closed primitive dtype family |
| `a.sum(axis)` | `a.sum(axis)` | deterministic Ravel reduction schedule |
| `a.mean(axis, keepdims=True)` | `a.meanKeep(axis)` | rank type is preserved |
| `np.array_equal(a, b)` | `a.sameElements(b)` | `equals` remains reference equality |

Ravel does not include matrix multiplication, factorizations, sparse arrays,
random distributions, I/O, named axes, lazy graphs, GPU execution, or
automatic differentiation. Use a mathematical layer such as
[Gale](https://github.com/canardlapin/gale) for linear algebra.
