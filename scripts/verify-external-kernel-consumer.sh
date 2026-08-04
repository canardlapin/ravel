#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
candidate_version="${RAVEL_CANDIDATE_VERSION:-1.0.0-SNAPSHOT}"
if [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "invalid RAVEL_CANDIDATE_VERSION: $candidate_version" >&2
  exit 1
fi
consumer_root="$(mktemp -d "${TMPDIR:-/tmp}/ravel-kernel-consumer.XXXXXX")"
trap 'rm -rf "$consumer_root"' EXIT

if [[ "${RAVEL_SKIP_PUBLISH:-0}" != "1" ]]; then
  cd "$project_root"
  sbt "set ThisBuild / version := \"$candidate_version\"" \
    verifyCoreReleaseMatrix \
    coreJVM/publishLocal
fi

mkdir -p "$consumer_root/project" "$consumer_root/src/main/scala"

cat >"$consumer_root/project/build.properties" <<'EOF'
sbt.version=1.11.7
EOF

cat >"$consumer_root/build.sbt" <<EOF
scalaVersion := "3.7.4"

libraryDependencies +=
  "io.github.canardlapin" %% "ravel-core" % "$candidate_version"
EOF

cat >"$consumer_root/src/main/scala/KernelProbe.scala" <<'EOF'
import ravel.*
import ravel.DType.given

@main def kernelProbe(): Unit =
  val rank3 =
    NDArray.tabulate[Double](2, 3, 4) { (i, j, k) =>
      i.toDouble * 100.0 + j.toDouble * 10.0 + k.toDouble
    }
  val rank4 =
    NDArray.tabulate[Double](2, 2, 3, 4) { (i, j, k, l) =>
      i.toDouble * 1000.0 + j.toDouble * 100.0 + k.toDouble * 10.0 + l.toDouble
    }
  val canonical3 = CanonicalArray.require(rank3)
  val canonical4 = CanonicalArray.require(rank4)

  assert(canonical3(1, -1, -1) == 123.0)
  assert(canonical3.readLinear(canonical3.size - 1) == 123.0)

  val out3 = NDArray.build[Double, Rank[3]](rank3.shape) { builder =>
    var out = 0
    var i = 0
    while i < 2 do
      var j = 0
      while j < 3 do
        var k = 0
        while k < 4 do
          builder.writeLinear(out, rank3(i, j, k) * 2.0)
          out += 1
          k += 1
        j += 1
      i += 1
  }

  val out4 = NDArray.build[Double, Rank[1]](Shape(canonical4.size)) { builder =>
    var out = 0
    while out < canonical4.size do
      builder.writeLinear(out, canonical4.readLinear(out) + 1.0)
      out += 1
  }

  val mutable = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 2))
  val mutableCanonical = MutableCanonicalArray.require(mutable)
  mutableCanonical.writeLinear(0, 7.0)
  mutableCanonical(1, -1) = 9.0

  assert(out3(1, 2, 3) == 246.0)
  assert(out4(out4.size - 1) == 1124.0)
  assert(mutableCanonical.readLinear(0) == 7.0)
  assert(mutableCanonical.readLinear(3) == 9.0)

  val boundsArePublic =
    try
      rank3(2, 0, 0)
      false
    catch
      case _: InvalidIndex => true
  assert(boundsArePublic)

  println(s"kernel consumer probe passed: ${out3(1, 2, 3) + out4(out4.size - 1)}")
EOF

cd "$consumer_root"
sbt run
