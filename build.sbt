import org.scalajs.linker.interface.ModuleKind
import org.scalajs.sbtplugin.ScalaJSPlugin
import org.scalajs.sbtplugin.ScalaJSPlugin.autoImport.*
import sbtcrossproject.CrossPlugin.autoImport.*
import scalajscrossproject.ScalaJSCrossPlugin.autoImport.*
import com.typesafe.tools.mima.plugin.MimaPlugin
import com.typesafe.tools.mima.plugin.MimaPlugin.autoImport.*
import laika.ast.Path.Root
import laika.helium.config.{HeliumIcon, IconLink}
import scoverage.ScoverageKeys.*

ThisBuild / organization := "io.github.canardlapin"
ThisBuild / scalaVersion := "3.7.4"
ThisBuild / homepage := Some(url("https://github.com/canardlapin/ravel"))
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/canardlapin/ravel"),
    "scm:git:https://github.com/canardlapin/ravel.git",
    Some("scm:git:git@github.com:canardlapin/ravel.git")
  )
)
ThisBuild / developers := List(
  Developer(
    id = "canardlapin",
    name = "canardlapin",
    email = "307091466+canardlapin@users.noreply.github.com",
    url = url("https://github.com/canardlapin")
  )
)
ThisBuild / scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Wunused:all",
  "-Wvalue-discard",
  "-Werror"
)
ThisBuild / Test / parallelExecution := false
// Pages deployment is owned by .github/workflows/pages.yml. Keep the standalone
// site plugin focused on mdoc/Laika generation rather than a gh-pages branch.
ThisBuild / tlSitePublishBranch := None

// MiMa: no previous artifact until 1.0.0 is published to Central.
// After that release, set previous to "1.0.0" (and later 1.x releases) on core/laws.
ThisBuild / mimaFailOnNoPrevious := false

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "org.scalameta" %%% "munit" % "1.3.0" % Test,
    "org.scalacheck" %%% "scalacheck" % "1.19.0" % Test,
    "org.scalameta" %%% "munit-scalacheck" % "1.3.0" % Test
  )
)

lazy val publishableSettings = Seq(
  mimaPreviousArtifacts := Set.empty
)

lazy val benchmarkSharedSettings = Seq(
  Compile / unmanagedSourceDirectories +=
    (ThisBuild / baseDirectory).value / "modules/benchmarks/shared/src/main/scala",
  Test / unmanagedSourceDirectories +=
    (ThisBuild / baseDirectory).value / "modules/benchmarks/shared/src/test/scala",
  Compile / sourceGenerators += Def.task {
    val output =
      (Compile / sourceManaged).value / "ravel/bench/CrossRuntimeCourtBuild.scala"
    IO.write(
      output,
      s"""package ravel.bench
         |
         |private[bench] object CrossRuntimeCourtBuild:
         |  val scalaVersion: String = "${scalaVersion.value}"
         |""".stripMargin
    )
    Seq(output)
  }.taskValue
)

lazy val core = crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Full)
  .in(file("modules/core"))
  .enablePlugins(MimaPlugin)
  .settings(commonSettings)
  .settings(publishableSettings)
  .settings(
    name := "ravel-core",
    description := "Dense immutable multidimensional arrays for Scala 3 on the JVM and Scala.js.",
    // Diagnostic only; no coverage threshold until exclusions are understood.
    coverageFailOnMinimum := false
  )
  .jsSettings(
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))
  )

lazy val laws = crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Full)
  .in(file("modules/laws"))
  .enablePlugins(MimaPlugin)
  .dependsOn(core)
  .settings(publishableSettings)
  .settings(
    name := "ravel-laws",
    description := "Reusable MUnit, ScalaCheck, and Discipline law bundles for Ravel.",
    libraryDependencies ++= Seq(
      "org.scalameta" %%% "munit" % "1.3.0",
      "org.scalameta" %%% "munit-scalacheck" % "1.3.0",
      "org.typelevel" %%% "discipline-core" % "1.7.0"
    )
  )
  .jsSettings(
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))
  )

/** Generic N-D neighborhood / stencil execution. No image semantics. */
lazy val stencil = crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Full)
  .in(file("modules/stencil"))
  .enablePlugins(MimaPlugin)
  .dependsOn(core)
  .settings(commonSettings)
  .settings(publishableSettings)
  .settings(
    name := "ravel-stencil",
    description := "Generic N-dimensional neighborhood traversal and border-index mapping for Ravel arrays."
  )
  .jsSettings(
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))
  )

/** Sub-byte packed arrays (1/2/4-bit codes). Parallel to NDArray, not a DType. */
lazy val packed = crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Full)
  .in(file("modules/packed"))
  .enablePlugins(MimaPlugin)
  .settings(commonSettings)
  .settings(publishableSettings)
  .settings(
    name := "ravel-packed",
    description := "Endian-independent sub-byte packed arrays with logical sample strides and wordwise one-bit set algebra."
  )
  .jsSettings(
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))
  )

lazy val representationProbeJVM = project
  .in(file("modules/benchmarks/jvm"))
  .enablePlugins(JmhPlugin)
  .dependsOn(core.jvm)
  .settings(commonSettings)
  .settings(benchmarkSharedSettings)
  .settings(
    name := "ravel-representation-probe-jvm",
    publish / skip := true
  )

// Deliberately standalone: the incubating Vector API is a benchmark experiment,
// not part of the root aggregate, the core dependency graph, or any published artifact.
lazy val vectorApiProbeJVM = project
  .in(file("modules/benchmarks/vector-jvm"))
  .enablePlugins(JmhPlugin)
  .settings(
    name := "ravel-vector-api-probe-jvm",
    publish / skip := true,
    Compile / javacOptions ++=
      Seq("-source", "16", "-target", "16", "--add-modules", "jdk.incubator.vector"),
    Jmh / javacOptions ++=
      Seq("-source", "16", "-target", "16", "--add-modules", "jdk.incubator.vector"),
    // The reflection generator runs in a plugin-owned fork that cannot receive
    // module options. ASM inspects bytecode without loading Vector API classes.
    Jmh / generatorType := "asm",
    Compile / run / fork := true,
    Compile / run / javaOptions += "--add-modules=jdk.incubator.vector",
    Jmh / javaOptions += "--add-modules=jdk.incubator.vector"
  )

lazy val representationProbeJS = project
  .in(file("modules/benchmarks/js"))
  .enablePlugins(ScalaJSPlugin)
  .dependsOn(core.js)
  .settings(commonSettings)
  .settings(benchmarkSharedSettings)
  .settings(
    name := "ravel-representation-probe-js",
    publish / skip := true,
    Compile / mainClass := Some("ravel.bench.CrossRuntimeCourt"),
    scalaJSUseMainModuleInitializer := true,
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))
  )

lazy val browserTests = project
  .in(file("modules/browser-tests"))
  .enablePlugins(ScalaJSPlugin)
  .dependsOn(core.js)
  .settings(commonSettings)
  .settings(
    name := "ravel-browser-tests",
    publish / skip := true,
    Test / jsEnv := new jsenv.playwright.PWEnv(
      browserName = "chrome",
      headless = true,
      showLogs = true
    ),
    scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.ESModule))
  )

lazy val docsBundle = taskKey[File](
  "Build the executable guide and bundle JVM Scaladoc under the site /api path."
)

// Public guide inputs live under docs/user. The other docs/ files are internal
// design, benchmark, audit, and release records and must not be rendered.
lazy val docs = project
  .in(file("site"))
  .dependsOn(core.jvm)
  .enablePlugins(TypelevelSitePlugin)
  .settings(
    name := "ravel-docs",
    publish / skip := true,
    mdocIn := file("docs/user"),
    tlSiteApiUrl := Some(url("https://canardlapin.github.io/ravel/api/")),
    tlSiteHelium := tlSiteHelium.value.site.topNavigationBar(
      homeLink = IconLink.internal(Root / "index.md", HeliumIcon.home)
    ),
    docsBundle := {
      val siteOutput = (laikaSite / target).value
      tlSite.value
      val apiOutput = (core.jvm / Compile / doc).value
      val bundledApi = siteOutput / "api"
      IO.delete(bundledApi)
      IO.copyDirectory(apiOutput, bundledApi)
      siteOutput
    }
  )

lazy val root = project
  .in(file("."))
  .aggregate(
    core.jvm,
    core.js,
    laws.jvm,
    laws.js,
    stencil.jvm,
    stencil.js,
    packed.jvm,
    packed.js,
    browserTests,
    representationProbeJVM,
    representationProbeJS
  )
  .settings(
    name := "ravel",
    publish / skip := true
  )

addCommandAlias(
  "compileAll",
  ";coreJVM/compile;coreJS/compile;lawsJVM/compile;lawsJS/compile;stencilJVM/compile;stencilJS/compile;packedJVM/compile;packedJS/compile"
)
addCommandAlias(
  "testAll",
  ";coreJVM/test;coreJS/test;lawsJVM/test;lawsJS/test;stencilJVM/test;stencilJS/test;packedJVM/test;packedJS/test"
)
addCommandAlias(
  "testAllFull",
  ";testAll;browserTests/test;coreJS/Test/fullLinkJS;lawsJS/Test/fullLinkJS"
)
addCommandAlias(
  "representationProof",
  ";coreJVM/test;coreJS/test;representationProbeJVM/test;representationProbeJS/test;representationProbeJS/fullLinkJS"
)
addCommandAlias("fmtCheck", ";scalafmtCheckAll;scalafmtSbtCheck")
addCommandAlias("fmt", ";scalafmtAll;scalafmtSbt")
addCommandAlias(
  "mimaCheck",
  ";coreJVM/mimaReportBinaryIssues;coreJS/mimaReportBinaryIssues;lawsJVM/mimaReportBinaryIssues;lawsJS/mimaReportBinaryIssues"
)
addCommandAlias(
  "coverageReportJvm",
  ";coverage;coreJVM/test;lawsJVM/test;coverageAggregate;coverageOff"
)
addCommandAlias(
  "verifyPublishArtifacts",
  ";coreJVM/makePom;coreJS/makePom;lawsJVM/makePom;lawsJS/makePom;coreJVM/packageBin;lawsJVM/packageBin"
)
addCommandAlias(
  "publishLocalSnapshot",
  """;set ThisBuild / version := "1.0.0-SNAPSHOT"; coreJVM/publishLocal; coreJS/publishLocal; lawsJVM/publishLocal; lawsJS/publishLocal"""
)
addCommandAlias(
  "releaseEngineeringGate",
  ";fmtCheck;mimaCheck;verifyPublishArtifacts;testAll"
)
addCommandAlias(
  "numpyParitySignatures",
  """;representationProbeJVM/runMain ravel.bench.AccessPatternParity --out target/access-patterns/parity/ravel-signatures.json --side 32,64;representationProbeJVM/runMain ravel.bench.OperationMatrixParity --out target/operation-matrix/parity/ravel-signatures.json --side 32,64"""
)
// Compile both platform API surfaces, execute every mdoc example, render Laika,
// validate guide links, and place the JVM API reference in the deployable site.
addCommandAlias("docsCheck", ";coreJS/doc;docs/docsBundle")
