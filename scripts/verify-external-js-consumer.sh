#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
candidate_version="${RAVEL_CANDIDATE_VERSION:-1.0.0-SNAPSHOT}"
if [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "invalid RAVEL_CANDIDATE_VERSION: $candidate_version" >&2
  exit 1
fi
consumer_root="$(mktemp -d "${TMPDIR:-/tmp}/ravel-js-consumer.XXXXXX")"
trap 'rm -rf "$consumer_root"' EXIT

if [[ "${RAVEL_SKIP_PUBLISH:-0}" != "1" ]]; then
  cd "$project_root"
  sbt "set ThisBuild / version := \"$candidate_version\"" \
    verifyCoreReleaseMatrix \
    coreJS/publishLocal
fi

mkdir -p "$consumer_root/project" "$consumer_root/src/main/scala"

cat >"$consumer_root/project/build.properties" <<'EOF'
sbt.version=1.11.7
EOF

cat >"$consumer_root/project/plugins.sbt" <<'EOF'
addSbtPlugin("org.scala-js" % "sbt-scalajs" % "1.22.0")
EOF

cat >"$consumer_root/build.sbt" <<EOF
import org.scalajs.linker.interface.ModuleKind

enablePlugins(ScalaJSPlugin)

scalaVersion := "3.7.4"
ThisBuild / offline := true
scalaJSUseMainModuleInitializer := true
scalaJSLinkerConfig ~= (_.withModuleKind(ModuleKind.CommonJSModule))

libraryDependencies +=
  "io.github.canardlapin" %%% "ravel-core" % "$candidate_version"
EOF

cat >"$consumer_root/src/main/scala/JsConsumerProbe.scala" <<'EOF'
import ravel.*
import ravel.DType.given
import ravel.js.JsInterop
import scala.scalajs.js.typedarray.Float64Array

@main def jsConsumerProbe(): Unit =
  val external = new Float64Array(4)
  external(0) = 1.0
  external(1) = 2.0
  external(2) = 3.0
  external(3) = 4.0

  val borrowed = JsInterop.unsafeBorrow(external, Shape(2, 2))
  val flattened = borrowed.reshapeView(Shape(4))
  external(0) = 99.0
  assert(flattened(0) == 99.0)

  val owned = borrowed.reshapeCopy(Shape(4))
  external(1) = 77.0
  assert(owned(1) == 2.0)

  val reduced = borrowed.sumKeep(Axes.require(borrowed.rank, 1))
  assert(reduced.shape == Shape(2, 1))
  assert(reduced(0, 0) == 176.0)
  assert(reduced(1, 0) == 7.0)

  val copied = JsInterop.copyToFloat64Array(owned)
  assert(copied.length == 4)
  assert(copied(0) == 99.0)
  assert(copied(1) == 2.0)

  println("fresh Scala.js consumer probe passed")
EOF

cd "$consumer_root"
sbt run
