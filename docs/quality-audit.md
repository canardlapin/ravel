# Scala quality assurance audit

## Verdict

Ravel 1.0 has strong local assurance for its numerical storage contract,
cross-platform semantics, ownership, adversarial layouts, and primitive-kernel
structure. Its remaining quality gaps are repository automation around
formatting, coverage diagnostics, and post-1.0 compatibility enforcement.

## Context and applicability

Ravel is a pure, eager numerical library for Scala 3.7.4 on the JVM and
Scala.js. The 1.0 matrix publishes only `ravel-core`; laws, packed, and stencil
remain cross-tested experimental source modules. It has no effect API,
provider registry, concurrent resource lifecycle, Scala Native target, or
matrix algorithm, so Cats Effect laws, backend conformance, convergence tests,
and Scala Native CI are not applicable.

The first 1.0 release has no prior binary baseline. The published 1.0 API will
become the required compatibility baseline for later 1.x releases.

## Scorecard

| Assurance dimension | Rating | Evidence | Gap or rationale |
|---|---|---|---|
| ScalaCheck use and generator quality | Strong | Four cross-platform properties run 250 cases each with one worker over zero, singleton, ordinary shapes, composed views, broadcasting, and mutable locality. Failures retain ScalaCheck seed and shrinking. | Add specialized overflow generators if the layout algebra expands. |
| Reusable law-test module | Experimental | `ravel-laws` contains useful public helpers, but its Discipline surface is not yet a generated downstream conformance kit. | Keep it unpublished for 1.0; parameterize real laws over a downstream adapter before stabilization. |
| Test framework and Discipline integration | Strong | MUnit and MUnit-ScalaCheck run on both platforms. `RavelDiscipline` exposes a `RuleSet`, and the suite executes every property with deterministic single-worker parameters. | No standard typeclass hierarchy requires Cats law suites. |
| Typeclass lawfulness and coherence | Strong | The closed dtype capability givens live in `DType`; compile tests reject unsupported arithmetic and rank evidence. Cast and identity behavior is tested. | The capabilities are closed witnesses, not user-extensible algebra instances. |
| Backend or provider conformance | Not applicable | Ravel has one platform implementation per target and no public backend/provider registry. | Gale owns numerical backend selection. |
| Cross-platform and cross-version CI | Strong for supported targets | Local and remote JVM, Node, real Chromium, representation, and full-optimized Scala.js gates pass on the first published commit; CI uses JDK 21 and Node 22. | Only Scala 3.7.4 is configured. |
| Numerical and computational assurance | Strong | Exact primitive cases, IEEE edge values, empty fibers, deterministic floating merge order, generated layouts, and mutable injectivity are tested. | Ravel does not implement iterative or approximate algorithms. |
| Differential and independent oracles | Strong | Coordinate reference models and direct Scala primitive operations check views, broadcasts, arithmetic, callbacks, and reductions without using optimized kernels. | External numerical libraries would add dependency without a distinct 1.0 oracle. |
| Failure, convergence, and resource contracts | Strong | Bounds, shape, broadcast, slice, layout overflow, noncontiguous reshape, empty reduction, callback failure, and closed-builder failures are asserted. | Convergence and resource lifecycle are not applicable. |
| Work and allocation accounting | Strong | JMH separates setup, raw reusable-output loops, public allocating operations, and representative strided work. GC-normalized allocation and output size are recorded. | Node allocation is structurally inspected rather than reported as a stable byte counter. |
| Compiler discipline | Strong | `-deprecation`, `-feature`, `-unchecked`, `-Wunused:all`, `-Wvalue-discard`, and `-Werror` apply to all projects. | None for the current compiler. |
| Formatting and semantic rewrites | Present | `.scalafmt.conf`, `sbt fmtCheck`, and a CI formatting step. | Keep formatting-only failures separate from semantic failures. |
| Binary and source compatibility | Scaffolded | MiMa covers the publishable `ravel-core` JVM and Scala.js artifacts with empty previous artifacts until Central `1.0.0`. | Point `mimaPreviousArtifacts` at published `1.0.0` before `1.0.1` / `1.1`. |
| Coverage and mutation signal | Diagnostic | `sbt coverageReportJvm` with no fail threshold. | Review exclusions before adopting a minimum. |
| Benchmark and performance evidence | Strong | JMH and optimized Node probes disclose fixtures, runtime versions, throughput, allocations, regression budgets, and the oversized-dispatcher failure. NumPy compute geomean budgets are recorded. | CI currently runs structural proof, not timing thresholds on a dedicated stable runner. |
| Documentation and release evidence | Strong | Copy/view, ownership, casting, reduction, NumPy migration, Gale boundary, artifact POMs, local gates, remote CI, and performance evidence are versioned. POM verify script and release-engineering page landed. | Maven Central publication remains unverified until the signed release succeeds. |

## Strongest evidence

The most useful checks generate legal and degenerate layouts, compare their
logical values with coordinate models, and run the same semantics on JVM and
Scala.js. The browser suite proves that JavaScript interop runs in a real
browser, not only Node. The performance gate also found and corrected a
dispatcher that passed semantic tests and bytecode primitive checks but was too
large for HotSpot to compile.

## Prioritized follow-up

1. After Maven Central `1.0.0`, set MiMa previous artifacts and keep the CI
   `mimaCheck` gate red on accidental binary breaks.
2. Review scoverage exclusions and decide whether a soft minimum belongs in CI.
3. Add a Scala Next consumer compilation job if Ravel promises a wider compiler
   consumption range than Scala 3.7.x.
4. Wire a stable-runner NumPy parity job once the 0.50× compute-geomean target
   is met locally.

Cats, Cats Effect, `sbt-typelevel`, and a public backend abstraction solve no
current Ravel problem and are not recommended for 1.0.

## Verification boundary

Locally executed commands include `sbt testAllFull`, `sbt
representationProof`, platform artifact packaging and eviction reports, JMH
with GC profiling, the optimized Node benchmark, and Gale's `sbt
interopRavelTest`. GitHub Actions run `30264191210` passed the JDK 21, Node 22,
real-Chromium, laws, representation, and optimized-link gates on commit
`7f7c90fe5cda40ca8aaccd038ea0edd71793622e`. No published Maven Central
artifacts or binary-compatibility baseline existed during this audit.
