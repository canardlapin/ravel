#!/usr/bin/env bash
# Inspect every artifact selected by the build-emitted publication matrix.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

manifest="${RAVEL_PUBLICATION_MANIFEST:-target/release/publication-manifest.tsv}"
candidate_version="${RAVEL_CANDIDATE_VERSION:-}"

fail() {
  echo "verify-publish-artifacts: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || fail "missing or empty $1"
}

pom_contains() {
  local pom="$1"
  local needle="$2"
  grep -F "$needle" "$pom" >/dev/null || fail "$pom missing '$needle'"
}

jar_contains() {
  local archive="$1"
  local entry="$2"
  jar tf "$archive" | awk -v entry="$entry" '
    $0 == entry { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

jar_contains_suffix() {
  local archive="$1"
  local suffix="$2"
  jar tf "$archive" | awk -v suffix="$suffix" '
    length($0) >= length(suffix) && substr($0, length($0) - length(suffix) + 1) == suffix {
      found = 1
    }
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

verify_dependency_scopes() {
  local pom="$1"
  awk '
    /<dependency>/ { dep = 1; block = $0; next }
    dep { block = block ORS $0 }
    /<\/dependency>/ {
      if (dep && block ~ /<artifactId>[^<]*(munit|scalacheck|discipline)[^<]*<\/artifactId>/ && block !~ /<scope>test<\/scope>/) exit 2
      dep = 0
      block = ""
    }
  ' "$pom" || fail "$pom leaks a test framework outside test scope"
}

if [[ -n "$candidate_version" ]] &&
  [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  fail "invalid RAVEL_CANDIDATE_VERSION: $candidate_version"
fi

require_file "$manifest"

published_count=0
module_count=0
expected_module_count=""
expected_published_count=""
manifest_schema=""
seen_keys="|"

while IFS=$'\t' read -r module platform published organization artifact version artifact_target; do
  case "$module" in
    "# schema="*)
      manifest_schema="${module#\# schema=}"
      continue
      ;;
    "# release_modules="*)
      expected_module_count="${module#\# release_modules=}"
      continue
      ;;
    "# published_platform_artifacts="*)
      expected_published_count="${module#\# published_platform_artifacts=}"
      continue
      ;;
  esac
  if [[ "$module" == "module" ]]; then
    [[ "$platform" == "platform" && "$published" == "published" ]] ||
      fail "invalid manifest header in $manifest"
    continue
  fi

  [[ -n "$module" && -n "$platform" && -n "$published" && -n "$organization" &&
    -n "$artifact" && -n "$version" && -n "$artifact_target" ]] ||
    fail "incomplete publication row in $manifest"
  [[ "$platform" == "jvm" || "$platform" == "js" ]] ||
    fail "$module has unsupported platform '$platform'"
  [[ "$published" == "true" || "$published" == "false" ]] ||
    fail "$module has invalid published flag '$published'"

  key="$module:$platform"
  [[ "$seen_keys" != *"|$key|"* ]] || fail "duplicate publication row $key"
  seen_keys="$seen_keys$key|"
  module_count=$((module_count + 1))

  if [[ -n "$candidate_version" && "$version" != "$candidate_version" ]]; then
    fail "$module manifest version was $version, required $candidate_version"
  fi
  if [[ "$published" != "true" ]]; then
    continue
  fi

  published_count=$((published_count + 1))
  pom="$artifact_target/$artifact-$version.pom"
  binary="$artifact_target/$artifact-$version.jar"
  sources="$artifact_target/$artifact-$version-sources.jar"
  api="$artifact_target/$artifact-$version-javadoc.jar"

  for file in "$pom" "$binary" "$sources" "$api"; do
    require_file "$file"
  done

  pom_contains "$pom" "<groupId>$organization</groupId>"
  pom_contains "$pom" "<artifactId>$artifact</artifactId>"
  pom_contains "$pom" "<version>$version</version>"
  pom_contains "$pom" "<url>https://github.com/canardlapin/ravel</url>"
  pom_contains "$pom" "<name>Apache-2.0</name>"
  pom_contains "$pom" "<url>https://www.apache.org/licenses/LICENSE-2.0</url>"
  verify_dependency_scopes "$pom"

  jar_contains "$binary" 'META-INF/MANIFEST.MF' ||
    fail "$binary is missing META-INF/MANIFEST.MF"
  jar_contains_suffix "$sources" '.scala' || fail "$sources contains no Scala sources"
  jar_contains "$api" 'index.html' || fail "$api is missing index.html"

  if [[ "$artifact" == ravel-core_* ]]; then
    if jar_contains_forbidden_package "$binary"; then
      fail "$binary must not contain gale or breeze packages"
    fi
    jar_contains "$sources" 'ravel/NDArray.scala' ||
      fail "$sources is missing ravel/NDArray.scala"
    jar_contains "$api" 'ravel/NDArray.html' ||
      fail "$api is missing ravel/NDArray.html"
    if [[ "$platform" == "js" ]]; then
      jar_contains "$sources" 'ravel/js/JsInterop.scala' ||
        fail "$sources is missing ravel/js/JsInterop.scala"
      jar_contains "$api" 'ravel/js/JsInterop$.html' ||
        fail "$api is missing ravel/js/JsInterop$.html"
    fi
  fi

  echo "  verified $module $organization:$artifact:$version"
done <"$manifest"

[[ "$manifest_schema" == "1" ]] || fail "$manifest has an unsupported or missing schema"
[[ "$expected_module_count" =~ ^[0-9]+$ ]] ||
  fail "$manifest has no valid release-module count"
[[ "$expected_published_count" =~ ^[0-9]+$ ]] ||
  fail "$manifest has no valid published-artifact count"
[[ "$module_count" -eq "$expected_module_count" ]] ||
  fail "$manifest declared $expected_module_count release modules but contained $module_count"
[[ "$published_count" -eq "$expected_published_count" ]] ||
  fail "$manifest declared $expected_published_count published artifacts but contained $published_count"
[[ "$module_count" -gt 0 ]] || fail "$manifest contains no release modules"
[[ "$published_count" -gt 0 ]] || fail "$manifest selects no published modules"

echo "verify-publish-artifacts: OK"
echo "  manifest $manifest"
echo "  release modules $module_count"
echo "  published platform artifacts $published_count"
