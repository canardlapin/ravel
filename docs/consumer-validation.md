# Downstream artifact validation

There is deliberately no `ravel-laws` consumer project. The laws module has no
main-source API or published coordinate; downstream validation resolves only
the core artifacts. A future public conformance kit would require a new adapter
contract and a separate compatibility decision.

The release gate builds two pinned sibling projects against the exact local
`ravel-core` candidate. It does not use source-project dependencies. The check
runs on the JVM and Scala.js and fails if either consumer resolves a different
Ravel version or imports `ravel.internal`.

Run the check by itself with:

```sh
bash scripts/verify-sibling-consumers.sh
```

The script publishes the local JVM and Scala.js candidates unless
`RAVEL_SKIP_PUBLISH=1`. The authoritative release gate sets that flag because
it has already published the same candidate earlier in the run.

## Validated consumers

| Consumer | Pinned revision | Executed evidence |
| --- | --- | --- |
| Gale | `98508f8d36ceedfb3a7cb4ea18807116fa6af66b` | `interopRavelTest` passes 8 tests on the JVM and 8 on Scala.js. Both classpaths contain the exact local candidate. |
| zarr4s | `b7c9840fdd4e5676a58acb4ae516dfdaad9bd177` | The JVM and Scala.js core and Ravel adapter compile and publish locally. A fresh standalone consumer then writes, reads, transforms, rewrites, and verifies a Float32 array on both platforms in offline mode. Both runs produce `transformed = [2, 3, 4, 5, 6, 7]` and resolve the exact local candidate. |

Gale uses public rank-one and rank-two arrays, borrowed arrays, builders,
indexing, and platform interop. Every Gale-to-Ravel and Ravel-to-Gale
conversion is explicitly named `Copy` and creates an independent owner. Its
borrowed-array tests mutate the external JVM array or JavaScript typed array
after conversion and verify that Gale does not observe the mutation.

zarr4s exercises a dynamic-rank storage consumer. Reads return an owned,
canonical `AnyNDArray`; the adapter does not use an unchecked rank cast.
`RavelArraySource.fromCanonical` retains an immutable canonical owner without
copying during source refinement. A non-canonical view must use the explicit
`RavelArraySource.copyOf` materialization. Zarr chunk creation still copies
logical values into the Zarr payload required by the storage API.

## API friction record

| Question | Gale | zarr4s |
| --- | --- | --- |
| Internal Ravel access | None. The gate rejects `ravel.internal` imports. | None. The gate rejects `ravel.internal` imports. |
| Copies | One deliberate copy at every library boundary. | Owned reads allocate their result. Canonical source refinement does not copy; non-canonical input requires an explicitly named copy. |
| Rank casts | None. Conversions accept static `Rank[1]` or `Rank[2]`. | None. Zarr shapes remain `AnyRank` because their rank is runtime data. |
| Missing operations | None for the vector and matrix conversion scope. | None for the whole-array read, transform, and write workflow. |
| Ownership ambiguity | None. Borrowed inputs and copied outputs have separate overloads and tests. | None found. The adapter accepts only immutable owned inputs and returns owned reads. |
| Error recovery | No Ravel exception recovery is required. | The pinned adapter needed a source migration from exception-shaped checked errors to `ShapeError` and `CanonicalLayoutError`. The gate applies the reviewed patch in an isolated checkout and continues to handle failures as `Either` values. |

The zarr4s migration is checked in as
`scripts/consumer-patches/zarr4s-b7c9840-ravel-errors.patch`. Keeping the patch
in Ravel makes this pre-1.0 compatibility break executable and reviewable. It
does not modify the sibling checkout.

## Why image4s is not this gate's artifact consumer

The current image4s build uses sbt `ProjectRef` source dependencies on
`ravel-core`, `ravel-packed`, and `ravel-stencil`. Packed and stencil are
deliberately excluded from Ravel's 1.0 publication matrix. An image4s source
build would therefore test a broader unpublished workspace, not the released
`ravel-core` artifact. The gate uses zarr4s as the second substantial artifact
consumer until image4s can consume the stable module boundary from published
coordinates.
