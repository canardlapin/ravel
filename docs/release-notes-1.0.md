# Ravel 1.0 candidate release notes

## Published artifacts

Ravel 1.0 publishes only the dense-array core:

- `io.github.canardlapin::ravel-core` for the JVM;
- `io.github.canardlapin::ravel-core` through `%%%` for Scala.js.

The concrete artifact IDs are `ravel-core_3` and
`ravel-core_sjs1_3`. `ravel-laws`, `ravel-packed`, and `ravel-stencil` are not
part of the 1.0 publication or compatibility promise. They remain experimental
source modules and may be released later only after their own contracts and
version lines are defined.

The checked-in `ravel-laws` helpers are preliminary internal evidence, not a
stable adapter-driven conformance kit. Core 1.0 attaches no compatibility
promise to those helpers or to the current two constant Discipline properties.

Local `publishLocalSnapshot` follows the same matrix and publishes only the
two core platform artifacts as `1.0.0-SNAPSHOT`.

These notes describe the candidate boundary; they do not claim that a Central
artifact has already been published.
