# Release engineering gates

Ravel is an early 0.1-level project. The tooling below is useful scaffolding,
not evidence that a 1.0 freeze or Maven Central publication is imminent. The
critical product gate remains NumPy semantic parity
([numpy-benchmarks.md](numpy-benchmarks.md)).

## Formatting

Scalafmt is configured in `.scalafmt.conf`. Locally:

```sh
sbt fmt        # rewrite
sbt fmtCheck   # CI gate
```

CI runs `fmtCheck` before tests so formatting churn stays separate from
semantic failures.

## Binary compatibility

`sbt-mima-plugin` is enabled on `ravel-core` and `ravel-laws`. Until a first
tagged Central release exists, `mimaPreviousArtifacts` is empty and
`mimaFailOnNoPrevious` is false, so `sbt mimaCheck` is a no-op scaffolding
gate.

After a first tagged publication someday, set previous artifacts before cutting
follow-on releases.

## Coverage diagnostics

JVM coverage is available without a fail threshold:

```sh
sbt coverageReportJvm
```

Reports land under `target/scala-3.7.4/scoverage-report/`. A recent local
aggregate measured about **73%** statement / **56%** branch coverage on JVM
core+laws tests; treat that as a diagnostic baseline, not a CI threshold.
Scala.js coverage is not part of the gate.

## Publishable coordinates

```sh
sbt verifyPublishArtifacts
bash scripts/verify-publish-artifacts.sh
```

The script checks that generated POMs use `io.github.canardlapin`, the
`ravel-core` / `ravel-laws` artifact ids (JVM and `_sjs1`), Apache-2.0
metadata, and that `ravel-core` does not compile-depend on MUnit or ship Gale
packages.

For sibling Gale verification against local snapshots:

```sh
sbt publishLocalSnapshot
# in the Gale repository:
sbt interopRavelTest
```

`publishLocalSnapshot` forces `1.0.0-SNAPSHOT`, matching Gale's current
`ravelVersion`.

## Combined local gate

```sh
sbt releaseEngineeringGate
bash scripts/numpy-parity-gate.sh
```

`releaseEngineeringGate` runs formatting, MiMa scaffolding, POM packaging, and
`testAll`. Always also run the NumPy parity script before treating numerical
work as done.
