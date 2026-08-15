#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

source_manifest="${RAVEL_PUBLICATION_MANIFEST:-target/release/publication-manifest.tsv}"
candidate_version="${RAVEL_CANDIDATE_VERSION:-1.0.0-SNAPSHOT}"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/ravel-artifact-verifier.XXXXXX")"
trap 'rm -rf "$fixture_root"' EXIT

expect_failure() {
  local label="$1"
  local fixture="$2"
  if RAVEL_CANDIDATE_VERSION="$candidate_version" \
    RAVEL_PUBLICATION_MANIFEST="$fixture" \
    bash scripts/verify-publish-artifacts.sh >"$fixture_root/$label.log" 2>&1; then
    echo "artifact verifier negative test unexpectedly passed: $label" >&2
    exit 1
  fi
}

# A verifier that silently drops one publishable platform is a false green.
awk -F '\t' '$1 != "coreJS"' "$source_manifest" >"$fixture_root/missing-platform.tsv"
expect_failure missing-platform "$fixture_root/missing-platform.tsv"

# Every selected row must point to the exact generated artifact set.
awk -F '\t' -v OFS='\t' -v missing="$fixture_root/missing" '
  $1 == "coreJS" { $7 = missing }
  { print }
' "$source_manifest" >"$fixture_root/missing-artifacts.tsv"
expect_failure missing-artifacts "$fixture_root/missing-artifacts.tsv"

echo "artifact verifier negative tests: OK"
