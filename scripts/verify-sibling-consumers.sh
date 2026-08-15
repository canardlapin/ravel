#!/usr/bin/env bash
# Run pinned, real downstream projects against locally published Ravel artifacts.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
candidate_version="${RAVEL_CANDIDATE_VERSION:-1.0.0-SNAPSHOT}"
gale_revision="98508f8d36ceedfb3a7cb4ea18807116fa6af66b"
zarr_revision="b7c9840fdd4e5676a58acb4ae516dfdaad9bd177"
zarr_version="0.1.0-ravel-consumer-${zarr_revision:0:7}"
work_root="$(mktemp -d "${TMPDIR:-/tmp}/ravel-sibling-consumers.XXXXXX")"
trap 'rm -rf "$work_root"' EXIT

if [[ ! "$candidate_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "invalid RAVEL_CANDIDATE_VERSION: $candidate_version" >&2
  exit 1
fi

if [[ "${RAVEL_SKIP_PUBLISH:-0}" != "1" ]]; then
  cd "$root"
  sbt -batch \
    "set ThisBuild / version := \"$candidate_version\"" \
    verifyCoreReleaseMatrix \
    coreJVM/publishLocal \
    coreJS/publishLocal
fi

materialize_repo() {
  local destination="$1"
  local repository="$2"
  local revision="$3"
  local local_hint="$4"
  if [[ -d "$local_hint/.git" ]] &&
    git -C "$local_hint" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git clone --quiet --no-checkout --shared "$local_hint" "$destination"
    git -C "$destination" checkout --quiet --detach "$revision"
  else
    mkdir -p "$destination"
    git -C "$destination" init --quiet
    git -C "$destination" fetch --quiet --depth 1 "$repository" "$revision"
    git -C "$destination" checkout --quiet --detach FETCH_HEAD
  fi
}

require_local_ravel_classpath() {
  local label="$1"
  local log="$2"
  local artifact="$3"
  awk -v artifact="$artifact" -v version="$candidate_version" '
    index($0, ".ivy2/local") && index($0, artifact) && index($0, version) { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$log" || {
    echo "$label did not resolve $artifact:$candidate_version from local Ivy" >&2
    exit 1
  }
}

scala_root="$(dirname "$root")"
gale_hint="${RAVEL_GALE_CHECKOUT:-$scala_root/gale}"
zarr_hint="${RAVEL_ZARR4S_CHECKOUT:-$scala_root/zarr4s}"
gale_root="$work_root/gale"
zarr_root="$work_root/zarr4s"

materialize_repo \
  "$gale_root" \
  "https://github.com/canardlapin/gale.git" \
  "$gale_revision" \
  "$gale_hint"
materialize_repo \
  "$zarr_root" \
  "https://github.com/canardlapin/zarr4s.git" \
  "$zarr_revision" \
  "$zarr_hint"

echo "sibling-consumers: Gale $gale_revision"
echo "sibling-consumers: zarr4s $zarr_revision"

# Gale's pinned consumer predates a final Ravel version property. Change only
# its isolated build copy so the same script also exercises M/RC/final tags.
awk -v version="$candidate_version" '
  /^lazy val ravelVersion = / {
    print "lazy val ravelVersion = \"" version "\""
    found = 1
    next
  }
  { print }
  END { if (!found) exit 42 }
' "$gale_root/build.sbt" >"$gale_root/build.sbt.next"
mv "$gale_root/build.sbt.next" "$gale_root/build.sbt"

if rg -n '_root_\.ravel\.internal|import ravel\.internal' "$gale_root/interop-ravel"; then
  echo "Gale consumer uses a Ravel internal package" >&2
  exit 1
fi

(
  cd "$gale_root"
  sbt -batch \
    interopRavelTest \
    'show interopRavelJVM / Compile / externalDependencyClasspath' \
    'show interopRavelJS / Compile / externalDependencyClasspath'
) 2>&1 | tee "$work_root/gale.log"

require_local_ravel_classpath Gale "$work_root/gale.log" "ravel-core_3"
require_local_ravel_classpath Gale "$work_root/gale.log" "ravel-core_sjs1_3"

# The pinned adapter used Ravel's former exception-shaped checked errors. Apply
# the reviewed migration to pure public errors in the isolated source tree.
git -C "$zarr_root" apply --unidiff-zero \
  "$root/scripts/consumer-patches/zarr4s-b7c9840-ravel-errors.patch"

if rg -n '_root_\.ravel\.internal|import ravel\.internal' "$zarr_root/interop-ravel"; then
  echo "zarr4s consumer uses a Ravel internal package" >&2
  exit 1
fi

(
  cd "$zarr_root"
  sbt -Dravel.version="$candidate_version" -batch \
    "set ThisBuild / version := \"$zarr_version\"" \
    'set interopRavelJVM / publish / skip := false' \
    'set interopRavelJS / publish / skip := false' \
    coreJVM/publishLocal \
    coreJS/publishLocal \
    interopRavelJVM/publishLocal \
    interopRavelJS/publishLocal
)

consumer_root="$work_root/zarr-consumer"
cp -R "$zarr_root/examples/ravel-standalone-consumer" "$consumer_root"

# The adapter build above populated all transitive dependency caches. Offline
# resolution now proves the standalone project consumes only local artifacts.
awk '
  { print }
  /ThisBuild \/ scalaVersion :=/ { print "ThisBuild / offline := true" }
' "$consumer_root/build.sbt" >"$consumer_root/build.sbt.next"
mv "$consumer_root/build.sbt.next" "$consumer_root/build.sbt"

(
  cd "$consumer_root"
  sbt \
    -Dzarr4s.version="$zarr_version" \
    -Dravel.version="$candidate_version" \
    -batch \
    consumerJVM/compile \
    consumerJS/compile \
    consumerJVM/run \
    consumerJS/run \
    'show consumerJVM / Compile / externalDependencyClasspath' \
    'show consumerJS / Compile / externalDependencyClasspath'
) 2>&1 | tee "$work_root/zarr-consumer.log"

transformed_count="$(grep -cF 'transformed = [2, 3, 4, 5, 6, 7]' "$work_root/zarr-consumer.log")"
[[ "$transformed_count" -eq 2 ]] || {
  echo "zarr4s consumer did not execute the expected workflow on both platforms" >&2
  exit 1
}
require_local_ravel_classpath zarr4s "$work_root/zarr-consumer.log" "ravel-core_3"
require_local_ravel_classpath zarr4s "$work_root/zarr-consumer.log" "ravel-core_sjs1_3"

echo "sibling consumer gates: OK"
