# Release engineering gates

The stable release scope is `ravel-core` for the JVM and Scala.js. The laws,
packed, and stencil modules remain cross-tested source modules and are not
published under the 1.0 compatibility promise.

## Authoritative gate

Run the same command used by protected-branch CI and tag publication:

```sh
bash scripts/release-gate.sh
```

The command fails on candidate worktree changes (ignoring Mote's append-only
operation receipts) and records phase logs, runtime versions, the exact
candidate commit, completion status, and checksums under
`target/release-gate/<commit>/`. On an exact `v*` tag it tests that tag's
version; otherwise it uses `1.0.0-SNAPSHOT`. It verifies:

- the core-only publication matrix;
- formatting, all-module compilation, and core MiMa;
- JVM, Node, real-Chromium, full-link, and representation tests;
- executable documentation and generated API docs;
- exact NumPy semantic parity and the comparison-helper unit tests;
- exact-version JVM and Scala.js POM, binary, source, and API jar contents;
- local publication of both core platforms; and
- fresh external JVM and Scala.js projects resolved only from those local
  artifacts.

Tag publication invokes this exact entry point before `ci-release`, so any
failure prevents signing or publication. CI uploads its receipt directory even
when a phase fails.

## Formatting

Scalafmt is configured in `.scalafmt.conf`. Locally:

```sh
sbt fmt        # rewrite
sbt fmtCheck   # CI gate
```

CI runs `fmtCheck` before tests so formatting churn stays separate from
semantic failures.

## Binary compatibility

The 1.0 compatibility gate covers only `ravel-core`. Until a first tagged
Central release exists, `mimaPreviousArtifacts` is empty and
`mimaFailOnNoPrevious` is false, so `sbt mimaCheck` is a no-op scaffolding
gate.

The exact `1.0.0` tag has no predecessor. The first dynver build after that tag
(`1.0.0+N...`) and every later 1.x build automatically select the JVM and
Scala.js `1.0.0` artifacts as `mimaPreviousArtifacts` and set
`mimaFailOnNoPrevious := true`. `verifyMimaBaselinePolicy` tests this transition
without claiming that the still-unpublished baseline can already be resolved.

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

The build first verifies its eight-row module/platform matrix and writes that
same matrix to `target/release/publication-manifest.tsv`. The script derives
its expected set from the published rows, then checks exact-version JVM and
Scala.js POMs, binaries, source jars, and API jars; coordinates; Apache-2.0
metadata; test dependency scopes; and core package boundaries. Negative tests
prove that omitting a selected platform or its files cannot pass.
`ravel-laws`, `ravel-packed`, and `ravel-stencil` have
`publish / skip := true` and are not 1.0 artifacts.
The current laws helpers remain cross-tested source code, but their two
constant Discipline properties are not presented as a downstream conformance
kit and receive no MiMa baseline.

For sibling Gale verification against local snapshots:

```sh
sbt publishLocalSnapshot
# in the Gale repository:
sbt interopRavelTest
```

`publishLocalSnapshot` forces `1.0.0-SNAPSHOT` for core only, matching Gale's
current `ravelVersion`. It cannot publish experimental modules under that
version. The authoritative gate's fresh JVM and Scala.js consumers resolve in
offline mode after local publication, so they cannot substitute a remote
artifact for the generated candidate coordinate.

## Component diagnostics

```sh
sbt releaseEngineeringGate
```

`releaseEngineeringGate` is the sbt-only subset. The shell entry point is
authoritative because NumPy parity, artifact inspection, and fresh consumer
projects cannot be expressed honestly as sbt-only checks.
