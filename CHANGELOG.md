# Changelog

Notable user-facing changes to published `ravel-core` artifacts are recorded
here. Development changes remain under **Unreleased** until an exact `vX.Y.Z`
tag passes the release gate; release sections use the tag date.

## Unreleased

### Fixed

- Avoid per-element boxing for ordinary Scala `map` callbacks from `Int`,
  `Long`, `Float`, and `Double` arrays to `Boolean`, `Int`, `Long`, `Float`, or
  `Double` results on the JVM. A 512 KiB `Double` result now allocates only the
  primitive output plus constant Ravel overhead.
