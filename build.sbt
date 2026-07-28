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

lazy val representationProbeJVM = project
  .in(file("modules/benchmarks/jvm"))
  .enablePlugins(JmhPlugin)
  .dependsOn(core.jvm)
  .settings(
    name := "ravel-representation-probe-jvm",
    publish / skip := true
  )

lazy val representationProbeJS = project
  .in(file("modules/benchmarks/js"))
  .enablePlugins(ScalaJSPlugin)
  .dependsOn(core.js)
  .settings(
    name := "ravel-representation-probe-js",
    publish / skip := true,
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
    browserTests,
    representationProbeJVM,
    representationProbeJS
  )
  .settings(
    name := "ravel",
    publish / skip := true
  )

addCommandAlias("compileAll", ";coreJVM/compile;coreJS/compile;lawsJVM/compile;lawsJS/compile")
addCommandAlias("testAll", ";coreJVM/test;coreJS/test;lawsJVM/test;lawsJS/test")
addCommandAlias(
  "testAllFull",
  ";testAll;browserTests/test;coreJS/Test/fullLinkJS;lawsJS/Test/fullLinkJS"
)
addCommandAlias(
  "representationProof",
  ";coreJVM/test;coreJS/test;representationProbeJS/fullLinkJS"
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
  """;representationProbeJVM/runMain ravel.bench.AccessPatternParity --out target/access-patterns/parity/ravel-signatures.json --side 32,64"""
)
// Compile both platform API surfaces, execute every mdoc example, render Laika,
// validate guide links, and place the JVM API reference in the deployable site.
addCommandAlias("docsCheck", ";coreJS/doc;docs/docsBundle")
