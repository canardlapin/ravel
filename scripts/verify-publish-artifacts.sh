#!/usr/bin/env bash
# Inspect the core-only 1.0 POMs and JVM jar for coordinate sanity.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

fail() {
  echo "verify-publish-artifacts: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing $1"
}

jar_contains() {
  local archive="$1"
  local entry="$2"
  jar tf "$archive" | awk -v entry="$entry" '
    $0 == entry { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

jar_contains_forbidden_package() {
  local archive="$1"
  jar tf "$archive" | awk '
    /(^|\/)(gale|breeze)\// { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

pom_contains() {
  local pom="$1"
  local needle="$2"
  grep -F "$needle" "$pom" >/dev/null || fail "$pom missing '$needle'"
}

newest() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -t $pattern 2>/dev/null | head -n 1 || true
}

candidate_version="${RAVEL_CANDIDATE_VERSION:-}"
if [[ -n "$candidate_version" ]] &&
  [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  fail "invalid RAVEL_CANDIDATE_VERSION: $candidate_version"
fi

if [[ -n "$candidate_version" ]]; then
  core_jvm_pom="$(newest "modules/core/jvm/target/scala-3*/ravel-core_3-$candidate_version.pom")"
  core_js_pom="$(newest "modules/core/js/target/scala-3*/ravel-core_sjs1_3-$candidate_version.pom")"
else
  core_jvm_pom="$(newest 'modules/core/jvm/target/scala-3*/ravel-core_3-*.pom')"
  [[ -n "$core_jvm_pom" ]] || fail "no core JVM POM; run sbt verifyPublishArtifacts first"
  candidate_version="$(basename "$core_jvm_pom")"
  candidate_version="${candidate_version#ravel-core_3-}"
  candidate_version="${candidate_version%.pom}"
  core_js_pom="$(newest "modules/core/js/target/scala-3*/ravel-core_sjs1_3-$candidate_version.pom")"
fi
[[ -n "$core_jvm_pom" ]] || fail "no core JVM POM; run sbt verifyPublishArtifacts first"
[[ -n "$core_js_pom" ]] || fail "no core JS POM; run sbt verifyPublishArtifacts first"

for pom in "$core_jvm_pom" "$core_js_pom"; do
  require_file "$pom"
  pom_contains "$pom" "<groupId>io.github.canardlapin</groupId>"
  pom_contains "$pom" "<url>https://github.com/canardlapin/ravel</url>"
  pom_contains "$pom" "<name>Apache-2.0</name>"
  pom_contains "$pom" "<url>https://www.apache.org/licenses/LICENSE-2.0</url>"
done

pom_contains "$core_jvm_pom" "<artifactId>ravel-core_3</artifactId>"
pom_contains "$core_js_pom" "<artifactId>ravel-core_sjs1_3</artifactId>"

# Core must not depend on test frameworks at compile scope.
if grep -qi '<artifactId>munit' "$core_jvm_pom"; then
  awk '
    /<dependency>/ {dep=1; block=$0; next}
    dep {block=block ORS $0}
    /<\/dependency>/ {
      if (dep && block ~ /munit/ && block !~ /<scope>test<\/scope>/) exit 2
      dep=0; block=""
    }
  ' "$core_jvm_pom" || fail "ravel-core JVM POM has non-test MUnit dependency"
fi

core_jvm_dir="$(dirname "$core_jvm_pom")"
core_js_dir="$(dirname "$core_js_pom")"
core_jar="$core_jvm_dir/ravel-core_3-$candidate_version.jar"
core_sources="$core_jvm_dir/ravel-core_3-$candidate_version-sources.jar"
core_api="$core_jvm_dir/ravel-core_3-$candidate_version-javadoc.jar"
core_js_jar="$core_js_dir/ravel-core_sjs1_3-$candidate_version.jar"
core_js_sources="$core_js_dir/ravel-core_sjs1_3-$candidate_version-sources.jar"
core_js_api="$core_js_dir/ravel-core_sjs1_3-$candidate_version-javadoc.jar"

for artifact in \
  "$core_jar" \
  "$core_sources" \
  "$core_api" \
  "$core_js_jar" \
  "$core_js_sources" \
  "$core_js_api"; do
  require_file "$artifact"
done

for binary_jar in "$core_jar" "$core_js_jar"; do
  if jar_contains_forbidden_package "$binary_jar"; then
    fail "$binary_jar must not contain gale or breeze packages"
  fi
done

jar_contains "$core_sources" 'ravel/NDArray.scala' ||
  fail "JVM sources jar is missing ravel/NDArray.scala"
jar_contains "$core_js_sources" 'ravel/js/JsInterop.scala' ||
  fail "Scala.js sources jar is missing ravel/js/JsInterop.scala"
jar_contains "$core_api" 'ravel/NDArray.html' ||
  fail "JVM API jar is missing ravel/NDArray.html"
jar_contains "$core_js_api" 'ravel/js/JsInterop$.html' ||
  fail "Scala.js API jar is missing ravel/js/JsInterop$.html"

echo "verify-publish-artifacts: OK"
echo "  version $candidate_version"
echo "  $core_jvm_pom"
echo "  $core_js_pom"
echo "  $core_jar"
echo "  $core_js_jar"
