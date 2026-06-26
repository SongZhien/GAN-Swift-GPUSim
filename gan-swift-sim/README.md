# GAN-Swift-sim

GAN-Swift-sim is a trace-driven GPU simulator. It executes GPU kernel traces and
reports simulated cycle counts, instruction counts, cache behavior, and SM stall
breakdowns for workload-level analysis.

The simulator in this directory is the main simulation component of the
repository. The NVBit tracing utilities are kept separately under
`../tracer_nvbit`.

## Features

- Simulates kernels from SASS and memory trace files.
- Supports automatic discovery of all kernels in a workload trace directory.
- Supports running a fixed number of kernels, an explicit kernel range, or a
  sampled kernel list from a `.g` file.
- Runs multiple kernels concurrently with a configurable concurrency limit.
- Models configurable SM resources, Tensor Core units, uniform datapath units,
  L1/L2 caches, memory partitions, MSHRs, and DRAM-related timing parameters.
- Reports per-kernel and workload-level metrics including GPU cycles, SM cycles,
  issue-active cycles, memory stalls, dependency stalls, unit stalls, and
  scheduler idle cycles.

## Directory Layout

```text
<repo-root>/
├── gan-swift-sim/
│   ├── Makefile
│   ├── README.md
│   ├── *.cpp / *.h
│   ├── gpu.config
│   ├── gpu.config_3080
│   └── gpu_isa_latency.config
└── tracer_nvbit/
```

Important files in `gan-swift-sim`:

- `main.cpp`: command-line parsing, kernel selection, concurrent execution, and
  workload progress output.
- `gpu.h` / `gpu.cpp`: GPU model construction and simulation loop.
- `kernel.h` / `kernel.cpp`: kernel, block, warp, instruction, and memory
  request handling.
- `trace_reader.h` / `trace_reader.cpp`: SASS and memory trace parsing.
- `cache.h` / `cache.cpp`: L1/L2 cache, MSHR, memory partition, and DRAM routing
  logic.
- `config_reader.h` / `config_reader.cpp`: simulator configuration loading.
- `gpu.config`: default GPU architecture and memory-system configuration.
- `gpu_isa_latency.config`: instruction latency and execution-unit mapping.

## Trace Layout

GAN-Swift-sim expects each benchmark to have the following trace layout:

```text
<trace-root>/<benchmark>/traces/kernel-<id>.sass
<trace-root>/<benchmark>/traces/kernel-<id>.mem
```

For example, benchmark `cfd` with kernel `7` should provide:

```text
<trace-root>/cfd/traces/kernel-7.sass
<trace-root>/cfd/traces/kernel-7.mem
```

The simulator reads `kernel-<id>.mem` directly and builds an in-memory index for
per-block memory records. It does not require pre-generated
`kernel-<id>-block-<block-id>.mem` files.

## Trace Root Configuration

The recommended way to select a trace root is to set the trace-path environment
variable before running the simulator:

```bash
export SWIFTSIM_TRACE_PATH=<trace-root>
```

The environment variable name is kept for compatibility with the current source
code. Project-facing documentation and directory names use `GAN-Swift-sim`.

If the environment variable is not set, the simulator falls back to
`DEFAULT_TRACE_PATH` in `gpu.h`. For portable experiments, prefer setting the
environment variable instead of relying on a machine-specific default path.

## Build

Build from the simulator directory so relative configuration paths resolve
correctly:

```bash
cd <repo-root>/gan-swift-sim
make clean
make -j"$(nproc)"
```

The build produces:

```text
<repo-root>/gan-swift-sim/flex-gpu-origin
```

To remove generated objects and the executable:

```bash
cd <repo-root>/gan-swift-sim
make clean
```

## Run

Run all commands from `gan-swift-sim` unless you also adjust the relative config
paths in the source code.

### Run All Kernels in a Benchmark

```bash
cd <repo-root>/gan-swift-sim
export SWIFTSIM_TRACE_PATH=<trace-root>
./flex-gpu-origin <benchmark>
```

The simulator scans:

```text
<trace-root>/<benchmark>/traces/
```

and runs every kernel for which it finds `kernel-<id>.sass` or
`kernel-<id>.mem`.

### Run the First N Kernels

```bash
cd <repo-root>/gan-swift-sim
export SWIFTSIM_TRACE_PATH=<trace-root>
./flex-gpu-origin <benchmark> <kernel_num>
```

This runs `kernel-1` through `kernel-<kernel_num>`.

Example:

```bash
./flex-gpu-origin cfd 300
```

### Run a Kernel Range

```bash
cd <repo-root>/gan-swift-sim
export SWIFTSIM_TRACE_PATH=<trace-root>
./flex-gpu-origin <benchmark> <kernel_begin> <kernel_end>
```

This runs every kernel in the inclusive range
`kernel-<kernel_begin>` through `kernel-<kernel_end>`.

Example:

```bash
./flex-gpu-origin cfd 10 20
```

### Run Kernels Listed in a `.g` File

```bash
cd <repo-root>/gan-swift-sim
export SWIFTSIM_TRACE_PATH=<trace-root>
./flex-gpu-origin <benchmark> --g <kernels.g>
```

The `.g` file may contain one kernel reference per line. The simulator extracts
the numeric kernel id from strings containing `kernel-<id>`.

Valid examples:

```text
kernel-7.trace
kernel-7.trace.xz
path/to/kernel-7-context.trace.xz
```

Only the kernel id is used. The simulator does not read `.trace.xz` content
directly.

## Runtime Output

At workload start:

```text
[WORKLOAD] benchmark=<benchmark> total_kernels=<N> max_concurrent_kernels=<M>
```

For each kernel start:

```text
[START] kernel=<id> started=<started>/<total> finished=<finished>/<total>
```

For each completed kernel:

```text
[DONE] kernel=<id> gpu_sim_cycle=<cycles> gpu_sim_insn=<insts> active_sm_cycles=<cycles> issue_active_sm_cycles=<cycles> memory_stall_sm_cycles=<cycles> dependency_stall_sm_cycles=<cycles> unit_stall_sm_cycles=<cycles> scheduler_idle_sm_cycles=<cycles> total_sm_cycles=<cycles> completed_gpu_cycles=<cycles> finished=<finished>/<total>
```

At workload completion:

```text
[WORKLOAD_DONE] benchmark=<benchmark> finished_kernels=<finished>/<total> completed_gpu_cycles=<cycles> completed_active_sm_cycles=<cycles> completed_issue_active_sm_cycles=<cycles> completed_memory_stall_sm_cycles=<cycles> completed_dependency_stall_sm_cycles=<cycles> completed_unit_stall_sm_cycles=<cycles> completed_scheduler_idle_sm_cycles=<cycles> completed_sm_cycles=<cycles> total_host_time <time> us
```

Additional per-kernel timing lines are printed after each kernel:

```text
preprocess_time <time> us
gpu_cycle_time <time> us
total_time <time> us
kernel_total_host_time <time> us
```

Metric notes:

- `gpu_sim_cycle`: simulated global cycle count for one kernel.
- `gpu_sim_insn`: simulated instruction count for one kernel.
- `active_sm_cycles`: accumulated resident-active SM cycles.
- `issue_active_sm_cycles`: accumulated SM cycles with useful issue activity.
- `memory_stall_sm_cycles`: accumulated cycles attributed to memory stalls.
- `dependency_stall_sm_cycles`: accumulated cycles attributed to dependency
  stalls.
- `unit_stall_sm_cycles`: accumulated cycles attributed to execution-unit
  contention.
- `scheduler_idle_sm_cycles`: accumulated cycles attributed to scheduler idle
  behavior.
- `total_sm_cycles`: accumulated SM cycles across all SMs.
- `total_host_time` and `kernel_total_host_time`: wall-clock runtime on the host
  machine, in microseconds.

## Configuration

The simulator reads configuration files from the current working directory:

```text
./gpu.config
./gpu_isa_latency.config
```

For that reason, run the executable from `gan-swift-sim`.

The default `gpu.config` currently describes an Ampere-style GPU model with:

- `sm_num:46`
- `mem_num:16`
- `l2_cache_sub_partitions:16`
- Tensor pipeline entries such as `TC_units` and `TENSOR_units`
- Uniform datapath entries such as `UDP_units`
- Configurable L1 cache, L2 cache, MSHR, and DRAM parameters

Use `gpu.config_3080` as an alternate starting point when evaluating a different
GPU configuration. After changing configuration values, rebuild if the change is
paired with source-code edits; otherwise rerun the executable from the simulator
directory.

## Kernel Concurrency

The maximum number of concurrently simulated kernels is controlled in
`main.cpp`:

```cpp
static const std::size_t MAX_CONCURRENT_KERNELS = 32;
```

Lower this value if a workload requires too much host memory. Rebuild after
changing the constant:

```bash
cd <repo-root>/gan-swift-sim
make clean
make -j"$(nproc)"
```

## Development Notes

- Keep trace data outside the repository and pass its location through the trace
  root environment variable.
- Avoid committing generated files such as `*.o`, `flex-gpu-origin`, temporary
  logs, and large trace outputs.
- Use relative repository paths in documentation and scripts so the project can
  be moved between machines without editing hard-coded user directories.
- When comparing with hardware counters, record both resident-active metrics and
  issue/stall breakdown metrics; they describe different aspects of execution.
