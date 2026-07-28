#!/usr/bin/env bash
# Inspect locally generated publishable POMs and jars for 1.0 coordinate sanity.
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

core_jvm_pom="$(newest 'modules/core/jvm/target/scala-3*/ravel-core_3-*.pom')"
core_js_pom="$(newest 'modules/core/js/target/scala-3*/ravel-core_sjs1_3-*.pom')"
laws_jvm_pom="$(newest 'modules/laws/jvm/target/scala-3*/ravel-laws_3-*.pom')"
laws_js_pom="$(newest 'modules/laws/js/target/scala-3*/ravel-laws_sjs1_3-*.pom')"

[[ -n "$core_jvm_pom" ]] || fail "no core JVM POM; run sbt verifyPublishArtifacts first"
[[ -n "$core_js_pom" ]] || fail "no core JS POM; run sbt verifyPublishArtifacts first"
[[ -n "$laws_jvm_pom" ]] || fail "no laws JVM POM; run sbt verifyPublishArtifacts first"
[[ -n "$laws_js_pom" ]] || fail "no laws JS POM; run sbt verifyPublishArtifacts first"

for pom in "$core_jvm_pom" "$core_js_pom" "$laws_jvm_pom" "$laws_js_pom"; do
  require_file "$pom"
  pom_contains "$pom" "<groupId>io.github.canardlapin</groupId>"
  pom_contains "$pom" "<url>https://github.com/canardlapin/ravel</url>"
  pom_contains "$pom" "<name>Apache-2.0</name>"
  pom_contains "$pom" "<url>https://www.apache.org/licenses/LICENSE-2.0</url>"
done

pom_contains "$core_jvm_pom" "<artifactId>ravel-core_3</artifactId>"
pom_contains "$core_js_pom" "<artifactId>ravel-core_sjs1_3</artifactId>"
pom_contains "$laws_jvm_pom" "<artifactId>ravel-laws_3</artifactId>"
pom_contains "$laws_js_pom" "<artifactId>ravel-laws_sjs1_3</artifactId>"

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

# Laws must compile-depend on ravel-core and MUnit.
pom_contains "$laws_jvm_pom" "<artifactId>ravel-core_3</artifactId>"
pom_contains "$laws_jvm_pom" "<artifactId>munit_3</artifactId>"

core_jar="$(ls -t modules/core/jvm/target/scala-3*/ravel-core_3-*.jar 2>/dev/null \
  | grep -vE 'sources|javadoc|tests' | head -n 1 || true)"
laws_jar="$(ls -t modules/laws/jvm/target/scala-3*/ravel-laws_3-*.jar 2>/dev/null \
  | grep -vE 'sources|javadoc|tests' | head -n 1 || true)"
[[ -n "$core_jar" ]] || fail "missing ravel-core JVM jar"
[[ -n "$laws_jar" ]] || fail "missing ravel-laws JVM jar"

if jar tf "$core_jar" | grep -E 'gale/|breeze/' >/dev/null; then
  fail "ravel-core jar must not contain gale or breeze packages"
fi

echo "verify-publish-artifacts: OK"
echo "  $core_jvm_pom"
echo "  $core_js_pom"
echo "  $laws_jvm_pom"
echo "  $laws_js_pom"
echo "  $core_jar"
echo "  $laws_jar"
