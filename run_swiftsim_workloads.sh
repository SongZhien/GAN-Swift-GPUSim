#!/usr/bin/env bash
set -euo pipefail

SIM_ROOT="/home/lxz/Swift-GPUSim/swift-sim"
SIM_BIN="${SIM_ROOT}/flex-gpu-origin"
TRACE_ROOT="/home/lxz/traces_disk/swift_traces"
LOG_ROOT="${HOME}/traces_disk/swift_traces"

mkdir -p "${LOG_ROOT}"

export SWIFTSIM_TRACE_PATH="${TRACE_ROOT}"

run_workload() {
  local workload="$1"
  local log_file="${LOG_ROOT}/${workload}_sim.txt"

  echo "[RUN] workload=${workload} log=${log_file}"
  (
    cd "${SIM_ROOT}"
    "${SIM_BIN}" "${workload}" > "${log_file}"
  )
  echo "[DONE] workload=${workload}"
}

run_workload "gpt"
run_workload "bert"
run_workload "llama"
run_workload "qwen"

echo "[ALL_DONE] all workloads finished"
