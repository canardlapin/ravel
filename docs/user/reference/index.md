# Reference

Use these pages for stable summaries:

- [Copy and view behavior](copy-view-table.md)
- [Dtypes and ranks](dtype-rank.md)
- [Failures and recovery](failures.md)
- [NumPy operation map](numpy-map.md)
- [Availability and project scope](status.md)

Generated Scaladoc is split by source module:

- [Core API](https://canardlapin.github.io/ravel/api/)
- [Packed API](https://canardlapin.github.io/ravel/api/packed/)
- [Stencil API](https://canardlapin.github.io/ravel/api/stencil/)
- [Laws API](https://canardlapin.github.io/ravel/api/laws/)

Only the core API has a 1.0 artifact promise. The packed, stencil, and laws
references document experimental source APIs, not published 1.0 artifacts.
The API references are rebuilt from public source on every documentation
deployment. Scala.js API documentation is compiled by the documentation gate
as well; the bundled browser references use JVM output because most public
symbols are shared and JVM-specific interop is directly navigable there.
