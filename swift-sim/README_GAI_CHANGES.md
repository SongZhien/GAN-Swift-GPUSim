# GAI Swifi GPU Simulator Changes

This copy is based on `/home/songz/Swift-GPUSim` and is intended for
root-cause accuracy work against real GPU behavior, not only against Accel-Sim.

## Implemented first-stage fixes

| Area | Original problem | Implemented change | Accuracy effect | Runtime effect |
| --- | --- | --- | --- | --- |
| Opcode semantics | The simulator truncated opcodes such as `LDG.E.128`, `IMAD.MOV.U32`, and `HMMA.*` to the base opcode too early. | `inst` now stores both `m_full_opcode` and base `m_opcode`; LD/ST classification uses the full opcode. | Prevents losing cache-space, memory, and ISA suffix semantics at the first decode stage. | Negligible. |
| Tensor pipeline | Tensor instructions were mapped onto SP/INT/DP units. | Added `TENSOR_units` and mapped `HMMA`, `DMMA`, `BMMA`, and `IMMA` to it. | Stops tensor ops from falsely contending with scalar SP/INT pipelines. | Negligible, sometimes faster if false contention is removed. |
| Uniform datapath | Uniform datapath instructions consumed normal INT issue bandwidth. | Added `UDP_units` and moved `R2UR`, `ULDC`, `UIMAD`, `UISETP`, etc. onto it. | Better matches Ampere-style separate uniform datapath behavior. | Negligible. |
| Active-cycle metric | `active_sm_cycles` counted resident-block active cycles, not only cycles with useful instruction issue. | Added per-SM `m_issue_active_cycles` and prints `issue_active_sm_cycles`. | Provides a second metric closer to hardware issue activity and helps explain gaps to NCU `sm_active_cycle`. | Negligible counter overhead. |
| Memory config copy | `gpu_config` copy constructor dropped `memory_config`. | Copy constructor now copies `memory_config`. | Avoids silently default/empty DRAM config after copying. | None. |
| DRAM scheduler config | `scheduler_type:1` fell through to error because `break` was missing. | Added the missing `break`. | Makes FR-FCFS selectable for later DRAM timing work. | None unless FR-FCFS is enabled and implemented. |
| Memory fetch packet | `mem_fetch::m_packet_size` was initialized to zero. | Initialize packet size to `SECTOR_SIZE` and initialize `m_tlb_hit`. | Avoids zero-size interconnect/cache traffic accounting. | Negligible. |
| GPU hardware config | The default config used 82 SMs and 24 memory/L2 subpartitions while the measured RTX3070/NCU setup uses 46 SMs and 16 memory partitions. | Changed `gpu.config` to `sm_num:46`, `mem_num:16`, and `l2_cache_sub_partitions:16`. | Aligns scheduling and cache/memory partitioning with the profiled device. | May increase `gpu_sim_cycle` for kernels that previously used the extra simulated SMs. |
| Stall breakdown | The simulator only printed one coarse `active_sm_cycles` value. | Added `issue_active_sm_cycles`, `memory_stall_sm_cycles`, `dependency_stall_sm_cycles`, `unit_stall_sm_cycles`, and `scheduler_idle_sm_cycles` to per-kernel and workload-done logs. | Makes it possible to identify whether BN/depthwise/GEMM error is dominated by memory, scoreboard/dependency, or pipeline contention. | Low to moderate; classification scans resident warps only on cycles where no instruction issues. |
| Non-blocking global stores | Ordinary global stores were modeled like loads and held warp memory dependencies until the memory system returned a reply. | Added `mem_fetch::m_no_reply`; STG/ST requests still enter cache/L2/DRAM but no longer create pending warp memory replies. | Better matches real GPU store-queue behavior and reduces artificial memory stalls in BN, elementwise, and depthwise-heavy kernels. | Low; memory traffic is still simulated, but fewer warp reply bookkeeping events are generated. |
| L2 MSHR merge replies | L2 MSHR hits were still sent to DRAM, creating duplicate off-chip requests for cache-line/sector misses that should have merged. | L2 MSHR hits now only attach to the outstanding miss; the DRAM return wakes all waiting requests for that MSHR entry and sends one reply per waiting SM/warp. | Reduces systemic overestimation of memory traffic and memory stall in conv/GEMM-style kernels with repeated shared cache-line misses. | Usually faster for memory-heavy kernels because duplicate DRAM traffic is removed. |
| Memory partition indexing | The DRAM path hard-coded `subpid * 2` even though subpartitions per channel are configurable. | Replaced those address calculations with `m_n_sub_partition_per_memory_channel` and initialized the round-robin subpartition index. | Prevents incorrect L2/DRAM routing when the GPU config changes. | None. |
| DRAM model selection | `memory_config::init()` ignored the config file and forced `simple_dram_model = 0`. | The simulator now honors `simple_dram_model` from `gpu.config`. | Makes memory timing experiments reproducible from config rather than silently using the detailed path. | Depends on selected model. |
| Tensor/uniform issue interval | The generic issue formula `32 / unit_width` made one-per-subcore tensor/uniform/branch units look like 32-cycle issue pipes. | Added an issue-interval helper; tensor, uniform, and branch units can issue every cycle while still using their configured latency for dependencies. | Reduces artificial unit stalls in tensor-heavy VGG/ResNet/Inception kernels. | Usually faster for tensor-heavy kernels. |

## Still-needed deeper model work

| Area | Why it is still inaccurate | Suggested next modification | Expected runtime impact |
| --- | --- | --- | --- |
| Scoreboard and operand readiness | Current dependency tracking is a simplified register-string model and does not model real scoreboard stalls, writeback ports, reuse flags, or operand collector pressure. | Add explicit per-warp scoreboard entries, operand-ready events, and structural stalls before issue. | Medium, usually +10% to +30%. |
| Warp scheduler policy | Current scheduling is a simple subcore scan and does not model real scheduler priorities, eligible-warp selection, issue slots, dual-issue restrictions, or stall reasons. | Add scheduler model with eligible/stall buckets and per-cycle scheduler arbitration. | Medium. |
| Memory coalescing and instruction width | Trace-side coalescing is coarse; suffixes such as `.32/.64/.128`, cache operators, local/global/shared spaces, and predication need stronger semantics. | Decode full opcode modifiers and derive transaction count/size/cache behavior from width and address distribution. | Low to medium. |
| DRAM timing | The detailed DRAM timing path is mostly inactive; FIFO behavior dominates. | Re-enable and validate bank, row-buffer, bus, FR-FCFS, write-drain, and return-queue timing. | High for memory-bound workloads, often +30% to +100%. |
| Hardware constants | Current `gpu.config` is only an approximate Ampere-like config. | Calibrate SM count, memory partitions, latencies, occupancy limits, and issue widths to the exact measured GPU. | Low after calibration. |
| Metric mapping | NCU `sm__cycles_active` is not identical to resident-block active time. | Use `issue_active_sm_cycles` plus stall breakdowns, and separately report resident cycles, issue cycles, memory-pending cycles, and barrier cycles. | Low to medium. |

## Current root-cause focus after 2026-05-21 rerun

The latest rerun did use this GAI binary, but the old CSV still compared NCU
against the coarse resident-style `active_sm_cycles`. For `mobilenet-v3`, the
largest signed error contributors were batch-normalization kernels, depthwise
convolution kernels, and GEMM/CUTLASS kernels. The newly added stall counters
are meant to separate those errors before applying more aggressive timing
changes.

| Priority | Area | Concrete next edit |
| --- | --- | --- |
| P0 | CSV/metric plumbing outside this directory | Parse and record the new stall fields from the log, then compare NCU against multiple candidate metrics instead of only resident `active_sm_cycles`. |
| P1 | Memory timing | Use the new `memory_stall_sm_cycles` to calibrate L1/L2/DRAM latency and cache operator behavior per kernel class. |
| P1 | BN/depthwise kernels | Inspect whether these kernels are dominated by memory stalls; if so, fix coalescing, L1 cache policy, and MSHR merge behavior before tuning arithmetic latency. |
| P1 | GEMM/CUTLASS kernels | Calibrate tensor pipeline issue interval/latency using `issue_active_sm_cycles` and `unit_stall_sm_cycles`. |

## Verification

Built with:

```bash
make clean && make -j$(nproc)
```

Smoke tested with:

```bash
env SWIFTSIM_TRACE_PATH=/home/songz/traces_disk/swift_llm_moretoken ./flex-gpu-origin mobilenet-v3 1
```

Observed kernel 1 output after the changes:

```text
gpu_sim_cycle:10520
active_sm_cycles:474652
issue_active_sm_cycles:424186
memory_stall_sm_cycles:156
dependency_stall_sm_cycles:39178
unit_stall_sm_cycles:9784
total_sm_cycles:560652
```
