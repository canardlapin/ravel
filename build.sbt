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
// The first commit after the 1.0.0 tag receives a dynver 1.0.0+N version, which
// automatically turns the published 1.0.0 artifact into a required baseline.
ThisBuild / mimaFailOnNoPrevious := false

def requiresCore10MimaBaseline(currentVersion: String): Boolean =
  currentVersion.matches(raw"1\.0\.0\+[0-9]+.*") ||
    currentVersion.matches(raw"1\.0\.[1-9][0-9]*(?:[-+].*)?") ||
    currentVersion.matches(raw"1\.[1-9][0-9]*\.[0-9]+(?:[-+].*)?")

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "org.scalameta" %%% "munit" % "1.3.0" % Test,
    "org.scalacheck" %%% "scalacheck" % "1.19.0" % Test,
    "org.scalameta" %%% "munit-scalacheck" % "1.3.0" % Test
  )
)

lazy val coreReleaseSettings = Seq(
  publish / skip := false,
  mimaPreviousArtifacts := {
    if (requiresCore10MimaBaseline(version.value)) {
      Set(projectID.value.withRevision("1.0.0"))
    } else Set.empty
  },
  mimaFailOnNoPrevious := requiresCore10MimaBaseline(version.value)
)

// These modules remain source-visible and cross-tested, but they are not part
// of the ravel-core 1.0 compatibility or publication promise.
lazy val experimentalModuleSettings = Seq(
  publish / skip := true
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
  .settings(coreReleaseSettings)
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
  .settings(experimentalModuleSettings)
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
  .settings(experimentalModuleSettings)
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
  .dependsOn(core)
  .settings(commonSettings)
  .settings(experimentalModuleSettings)
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
  "Build the executable guide and bundle JVM module Scaladoc under the site /api path."
)

lazy val verifyCoreReleaseMatrix = taskKey[Unit](
  "Fail unless only ravel-core is publishable for the 1.0 release line."
)

lazy val releaseModuleMatrix =
  taskKey[Seq[(String, String, Boolean, ModuleID, File, File)]](
    "Single source of module, platform, publication, coordinates, artifact, and target state."
  )

lazy val writePublishedArtifactsManifest = taskKey[File](
  "Write every release-module coordinate and expected artifact target for shell verification."
)

lazy val verifyMimaBaselinePolicy = taskKey[Unit](
  "Verify that the 1.0.0 tag has no predecessor and every later 1.x build requires it."
)

// Public guide inputs live under docs/user. The other docs/ files are internal
// design, benchmark, audit, and release records and must not be rendered.
lazy val docs = project
  .in(file("site"))
  .dependsOn(core.jvm, stencil.jvm, packed.jvm)
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
      val coreApi = (core.jvm / Compile / doc).value
      val moduleApis = Seq(
        "laws" -> (laws.jvm / Compile / doc).value,
        "stencil" -> (stencil.jvm / Compile / doc).value,
        "packed" -> (packed.jvm / Compile / doc).value
      )
      val bundledApi = siteOutput / "api"
      IO.delete(bundledApi)
      IO.copyDirectory(coreApi, bundledApi)
      moduleApis.foreach { case (module, apiOutput) =>
        IO.copyDirectory(apiOutput, bundledApi / module)
      }
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
    publish / skip := true,
    releaseModuleMatrix := Seq(
      (
        "coreJVM",
        "jvm",
        (core.jvm / publish / skip).value,
        (core.jvm / projectID).value,
        (core.jvm / Compile / packageBin / artifactPath).value,
        (core.jvm / crossTarget).value
      ),
      (
        "coreJS",
        "js",
        (core.js / publish / skip).value,
        (core.js / projectID).value,
        (core.js / Compile / packageBin / artifactPath).value,
        (core.js / crossTarget).value
      ),
      (
        "lawsJVM",
        "jvm",
        (laws.jvm / publish / skip).value,
        (laws.jvm / projectID).value,
        (laws.jvm / Compile / packageBin / artifactPath).value,
        (laws.jvm / crossTarget).value
      ),
      (
        "lawsJS",
        "js",
        (laws.js / publish / skip).value,
        (laws.js / projectID).value,
        (laws.js / Compile / packageBin / artifactPath).value,
        (laws.js / crossTarget).value
      ),
      (
        "stencilJVM",
        "jvm",
        (stencil.jvm / publish / skip).value,
        (stencil.jvm / projectID).value,
        (stencil.jvm / Compile / packageBin / artifactPath).value,
        (stencil.jvm / crossTarget).value
      ),
      (
        "stencilJS",
        "js",
        (stencil.js / publish / skip).value,
        (stencil.js / projectID).value,
        (stencil.js / Compile / packageBin / artifactPath).value,
        (stencil.js / crossTarget).value
      ),
      (
        "packedJVM",
        "jvm",
        (packed.jvm / publish / skip).value,
        (packed.jvm / projectID).value,
        (packed.jvm / Compile / packageBin / artifactPath).value,
        (packed.jvm / crossTarget).value
      ),
      (
        "packedJS",
        "js",
        (packed.js / publish / skip).value,
        (packed.js / projectID).value,
        (packed.js / Compile / packageBin / artifactPath).value,
        (packed.js / crossTarget).value
      )
    ),
    verifyCoreReleaseMatrix := {
      val matrix = releaseModuleMatrix.value
      val requiredSkip = Map(
        "coreJVM" -> false,
        "coreJS" -> false,
        "lawsJVM" -> true,
        "lawsJS" -> true,
        "stencilJVM" -> true,
        "stencilJS" -> true,
        "packedJVM" -> true,
        "packedJS" -> true
      )
      val actualModules = matrix.map(_._1).toSet
      if (actualModules != requiredSkip.keySet) {
        sys.error(
          "Release module matrix did not match the required 1.0 module set: " +
            s"actual=${actualModules.toSeq.sorted.mkString(",")}, " +
            s"required=${requiredSkip.keySet.toSeq.sorted.mkString(",")}"
        )
      }
      val mismatches = matrix.collect {
        case (module, _, actual, _, _, _) if actual != requiredSkip(module) =>
          s"$module publish / skip was $actual, required ${requiredSkip(module)}"
      }
      if (mismatches.nonEmpty) {
        sys.error(
          "Invalid ravel-core 1.0 publication matrix:\n" + mismatches.mkString("\n")
        )
      }
      val coreNames = matrix.collect {
        case (module, _, _, coordinates, binaryArtifact, _)
            if module == "coreJVM" || module == "coreJS" =>
          val suffix = s"-${coordinates.revision}.jar"
          val filename = binaryArtifact.getName
          if (!filename.endsWith(suffix)) {
            sys.error(s"Unexpected binary artifact filename for $module: $filename")
          }
          module -> filename.stripSuffix(suffix)
      }
      val invalidNames = coreNames.collect {
        case ("coreJVM", artifactName) if artifactName != "ravel-core_3" =>
          s"coreJVM artifact was $artifactName, required ravel-core_3"
        case ("coreJS", artifactName) if artifactName != "ravel-core_sjs1_3" =>
          s"coreJS artifact was $artifactName, required ravel-core_sjs1_3"
      }
      if (invalidNames.nonEmpty) {
        sys.error("Invalid core artifact names:\n" + invalidNames.mkString("\n"))
      }
      streams.value.log.info(
        "verified core-only 1.0 publication matrix: ravel-core JVM and Scala.js"
      )
    },
    writePublishedArtifactsManifest := {
      verifyCoreReleaseMatrix.value
      val base = (ThisBuild / baseDirectory).value
      val output = target.value / "release" / "publication-manifest.tsv"
      val header = "module\tplatform\tpublished\torganization\tartifact\tversion\ttarget"
      val matrix = releaseModuleMatrix.value
      val rows =
        matrix.map { case (module, platform, skip, coordinates, binaryArtifact, artifactTarget) =>
          val relativeTarget = IO
            .relativize(base, artifactTarget)
            .getOrElse(
              sys.error(s"Artifact target $artifactTarget is outside repository $base")
            )
          val suffix = s"-${coordinates.revision}.jar"
          val filename = binaryArtifact.getName
          if (!filename.endsWith(suffix)) {
            sys.error(s"Unexpected binary artifact filename for $module: $filename")
          }
          val artifactId = filename.stripSuffix(suffix)
          Seq(
            module,
            platform,
            (!skip).toString,
            coordinates.organization,
            artifactId,
            coordinates.revision,
            relativeTarget
          ).mkString("\t")
        }
      val metadata = Seq(
        "# schema=1",
        s"# release_modules=${matrix.size}",
        s"# published_platform_artifacts=${matrix.count(module => !module._3)}"
      )
      IO.writeLines(output, metadata ++ (header +: rows))
      streams.value.log.info(s"wrote release publication manifest to $output")
      output
    },
    verifyMimaBaselinePolicy := {
      val cases = Seq(
        "0.1.0-SNAPSHOT" -> false,
        "1.0.0-M1" -> false,
        "1.0.0-RC2" -> false,
        "1.0.0" -> false,
        "1.0.0+1-next-SNAPSHOT" -> true,
        "1.0.1" -> true,
        "1.1.0-RC1" -> true,
        "2.0.0" -> false
      )
      val failures = cases.collect {
        case (candidate, required) if requiresCore10MimaBaseline(candidate) != required =>
          s"$candidate baseline policy was ${requiresCore10MimaBaseline(candidate)}, required $required"
      }
      if (failures.nonEmpty) {
        sys.error("Invalid ravel-core 1.0 MiMa policy:\n" + failures.mkString("\n"))
      }
      streams.value.log.info(
        "verified MiMa policy: 1.0.0 becomes the required baseline after its exact tag"
      )
    }
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
  ";coreJVM/mimaReportBinaryIssues;coreJS/mimaReportBinaryIssues"
)
addCommandAlias(
  "coverageReportJvm",
  ";coverage;coreJVM/test;lawsJVM/test;coverageAggregate;coverageOff"
)
addCommandAlias(
  "verifyPublishArtifacts",
  ";verifyCoreReleaseMatrix;coreJVM/makePom;coreJS/makePom;coreJVM/packageBin;coreJS/packageBin;coreJVM/Compile/packageSrc;coreJS/Compile/packageSrc;coreJVM/Compile/packageDoc;coreJS/Compile/packageDoc;writePublishedArtifactsManifest"
)
addCommandAlias(
  "publishLocalSnapshot",
  """;verifyCoreReleaseMatrix;set ThisBuild / version := "1.0.0-SNAPSHOT"; coreJVM/publishLocal; coreJS/publishLocal"""
)
addCommandAlias(
  "releaseEngineeringGate",
  ";verifyCoreReleaseMatrix;fmtCheck;compileAll;mimaCheck;verifyMimaBaselinePolicy;testAllFull;representationProof;docsCheck;verifyPublishArtifacts"
)
addCommandAlias(
  "numpyParitySignatures",
  """;representationProbeJVM/runMain ravel.bench.AccessPatternParity --out target/access-patterns/parity/ravel-signatures.json --side 32,64;representationProbeJVM/runMain ravel.bench.OperationMatrixParity --out target/operation-matrix/parity/ravel-signatures.json --side 32,64"""
)
// Compile both platform API surfaces, execute every mdoc example, render Laika,
// validate guide links, and place the JVM API references in the deployable site.
addCommandAlias(
  "docsCheck",
  ";coreJS/doc;lawsJS/doc;stencilJS/doc;packedJS/doc;docs/docsBundle"
)
