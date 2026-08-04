#!/usr/bin/env bash
# Authoritative ravel-core candidate gate. Protected-branch CI and tag publication
# both invoke this exact entry point before publication can begin.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

candidate_sha="$(git rev-parse HEAD)"
candidate_version="${RAVEL_CANDIDATE_VERSION:-}"
if [[ -z "$candidate_version" ]]; then
  exact_tag="$(git describe --tags --exact-match 2>/dev/null || true)"
  if [[ "$exact_tag" == v* ]]; then
    candidate_version="${exact_tag#v}"
  else
    candidate_version="1.0.0-SNAPSHOT"
  fi
fi
if [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "release-gate: invalid candidate version: $candidate_version" >&2
  exit 1
fi
receipt_root="${RAVEL_RELEASE_RECEIPT_DIR:-target/release-gate/$candidate_sha}"
mkdir -p "$receipt_root"

dirty_paths="$(git status --porcelain --untracked-files=normal | grep -v '^?? .mote/ops/' || true)"
if [[ -n "$dirty_paths" ]]; then
  echo "release-gate: candidate worktree is not clean" >&2
  echo "$dirty_paths" >&2
  exit 1
fi

run_phase() {
  local phase="$1"
  shift
  echo "release-gate: $phase"
  "$@" 2>&1 | tee "$receipt_root/$phase.log"
}

run_numpy_helper_tests() {
  local numpy_venv="${NUMPY_PARITY_VENV:-target/access-patterns/venv}"
  "$numpy_venv/bin/python" modules/benchmarks/python/test_compare_access_patterns.py
  "$numpy_venv/bin/python" modules/benchmarks/python/test_operation_matrix.py
}

{
  echo "status=running"
  echo "candidate_sha=$candidate_sha"
  echo "candidate_version=$candidate_version"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "shell_java=$(java -version 2>&1 | head -n 1)"
  echo "node=$(node --version)"
  echo "python=$(python3 --version 2>&1)"
  echo "sbt=$(sbt --script-version)"
} >"$receipt_root/manifest.txt"

run_phase sbt \
  sbt -batch \
    verifyCoreReleaseMatrix \
    fmtCheck \
    compileAll \
    mimaCheck \
    testAllFull \
    representationProof \
    docsCheck \
    verifyPublishArtifacts \
    "set ThisBuild / version := \"$candidate_version\"" \
    coreJVM/publishLocal \
    coreJS/publishLocal

run_phase numpy-parity bash scripts/numpy-parity-gate.sh
run_phase numpy-helper-tests run_numpy_helper_tests
run_phase artifact-inspection \
  env RAVEL_CANDIDATE_VERSION="$candidate_version" \
  bash scripts/verify-publish-artifacts.sh
run_phase consumer-jvm \
  env RAVEL_SKIP_PUBLISH=1 RAVEL_CANDIDATE_VERSION="$candidate_version" \
  bash scripts/verify-external-kernel-consumer.sh
run_phase consumer-js \
  env RAVEL_SKIP_PUBLISH=1 RAVEL_CANDIDATE_VERSION="$candidate_version" \
  bash scripts/verify-external-js-consumer.sh

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "status=passed"
} >>"$receipt_root/manifest.txt"

find "$receipt_root" -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 >"$receipt_root/checksums.sha256"

echo "release-gate: PASS $candidate_sha"
echo "release-gate: receipts $receipt_root"
