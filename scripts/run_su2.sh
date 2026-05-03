#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  direct|adjoint|all)
    mode="${1}"
    ;;
  *)
    echo "Usage: scripts/run_su2.sh direct|adjoint|all" >&2
    exit 2
    ;;
esac

if ! command -v SU2_CFD >/dev/null 2>&1; then
  echo "SU2_CFD was not found on PATH." >&2
  echo "Install SU2 and expose SU2_CFD before rerunning this script." >&2
  echo "See: https://su2code.github.io/docs_v7/Download/" >&2
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${root_dir}/run"
mkdir -p "${run_dir}/direct" "${run_dir}/adjoint"

run_direct() {
  cp "${root_dir}/data/input/inv_NACA0012.cfg" "${run_dir}/direct/"
  cp "${root_dir}/data/input/mesh_NACA0012_inv.su2" "${run_dir}/direct/"
  (
    cd "${run_dir}/direct"
    SU2_CFD inv_NACA0012.cfg | tee direct.log
  )
}

run_adjoint() {
  if [[ ! -f "${run_dir}/direct/restart_flow.dat" ]]; then
    echo "Direct restart file missing. Run direct first." >&2
    exit 1
  fi
  cp "${root_dir}/data/input/inv_NACA0012_adjoint.cfg" "${run_dir}/adjoint/"
  cp "${root_dir}/data/input/mesh_NACA0012_inv.su2" "${run_dir}/adjoint/"
  cp "${run_dir}/direct/restart_flow.dat" "${run_dir}/adjoint/solution_flow.dat"
  (
    cd "${run_dir}/adjoint"
    SU2_CFD inv_NACA0012_adjoint.cfg | tee adjoint.log
  )
}

case "${mode}" in
  direct)
    run_direct
    ;;
  adjoint)
    run_adjoint
    ;;
  all)
    run_direct
    run_adjoint
    ;;
esac
