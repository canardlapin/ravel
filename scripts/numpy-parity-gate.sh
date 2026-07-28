#!/usr/bin/env bash
# Critical correctness gate: Ravel public access-pattern signatures must match NumPy.
# Timing/JMH is intentionally excluded; use docs/numpy-benchmarks.md for performance runs.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

sides="${NUMPY_PARITY_SIDES:-32,64}"
out_dir="${NUMPY_PARITY_OUT:-target/access-patterns/parity}"
venv_dir="${NUMPY_PARITY_VENV:-target/access-patterns/venv}"
python_bin="${PYTHON:-python3}"

mkdir -p "$out_dir"

echo "numpy-parity-gate: Ravel signatures (sides=$sides)"
sbt -batch "representationProbeJVM/runMain ravel.bench.AccessPatternParity --out $out_dir/ravel-signatures.json --side $sides"

if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "numpy-parity-gate: creating venv at $venv_dir"
  "$python_bin" -m venv "$venv_dir"
  "$venv_dir/bin/python" -m pip install --upgrade pip
  "$venv_dir/bin/python" -m pip install -r modules/benchmarks/python/requirements.txt
fi

IFS=',' read -r -a side_array <<< "$sides"
numpy_side_args=()
for side in "${side_array[@]}"; do
  numpy_side_args+=(--side "$side")
done

echo "numpy-parity-gate: NumPy signatures-only"
"$venv_dir/bin/python" \
  modules/benchmarks/python/numpy_access_patterns.py \
  "${numpy_side_args[@]}" \
  --signatures-only \
  --out "$out_dir/numpy-signatures.json"

echo "numpy-parity-gate: compare --parity-only"
"$venv_dir/bin/python" \
  modules/benchmarks/python/compare_access_patterns.py \
  --signatures "$out_dir/ravel-signatures.json" \
  --numpy "$out_dir/numpy-signatures.json" \
  --parity-only

echo "numpy-parity-gate: OK"
