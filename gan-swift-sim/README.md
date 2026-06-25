# Swift-GPUSim `swift-sim`

这个目录下的程序是一个以 `kernel-N.sass` 和 `kernel-N.mem` 为输入的 trace-driven GPU 模拟器。

当前版本已支持：

- 连续执行 kernel：`./flex-gpu-origin <benchmark> <kernel_num>`
- 区间执行 kernel：`./flex-gpu-origin <benchmark> <kernel_begin> <kernel_end>`
- 通过 `.g` 文件执行指定 kernel 子集：`./flex-gpu-origin <benchmark> --g <kernels.g>`
- 固定并发上限 + 槽位补齐
- 运行中输出：
  - `[WORKLOAD]`
  - `[START]`
  - `[DONE]`
  - `completed_gpu_cycles`

## Trace 路径

程序优先通过环境变量 `SWIFTSIM_TRACE_PATH` 查找 trace；如果没有设置，则回退到 [gpu.h](/home/lxz/Swift-GPUSim/swift-sim/gpu.h) 中的默认路径 `DEFAULT_TRACE_PATH`。

```cpp
#define DEFAULT_TRACE_PATH "/home/lxz/traces_disk/swift_traces/"
```

对于 benchmark `foo`，程序会读取：

```text
<TRACE_PATH>/foo/traces/kernel-N.sass
<TRACE_PATH>/foo/traces/kernel-N.mem
```
当前版本不再在运行时生成 `kernel-N-block-M.mem`。程序会直接基于 `kernel-N.mem` 建立按 block 的轻量索引，并按需读取对应 block 的访存记录。

你可以用两种方式设置环境变量：

### 方式 A：当前 shell 会话内先导出

```bash
export SWIFTSIM_TRACE_PATH=/your/trace/root
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin llama
```

### 方式 B：单条命令前临时设置

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
SWIFTSIM_TRACE_PATH=/your/trace/root ./flex-gpu-origin llama
```

## 必需文件

`swift-sim` 真正需要的 trace 文件是：

- `kernel-N.sass`
- `kernel-N.mem`

可选文件：

- `.g`
  - 只在你想运行某个 kernel 子集时使用

程序不会要求 `kernel-N-block-M.mem`，也不会在运行时生成它们。

程序不会直接读取 `.trace.xz` 内容；如果使用 `.g`，只是从 `.g` 的每一行里提取 `kernel-<id>`。

## 编译

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
make
```

生成的可执行文件是：

```text
/home/lxz/Swift-GPUSim/swift-sim/flex-gpu-origin
```

## 执行方式

### 1. 自动执行该 workload 下全部 kernel

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin <benchmark>
```

示例：

```bash
export SWIFTSIM_TRACE_PATH=/path/to/trace_root
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin llama
```

程序会自动扫描：

```text
$SWIFTSIM_TRACE_PATH/<benchmark>/traces/
```

下面真实存在的 `kernel-N.sass` / `kernel-N.mem`，然后按编号执行全部 kernel。

### 2. 从 `kernel-1` 开始执行前 `N` 个 kernel

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin <benchmark> <kernel_num>
```

示例：

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin cfd 300
```

### 3. 执行区间 kernel

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin <benchmark> <kernel_begin> <kernel_end>
```

示例：

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin cfd 10 20
```

这表示执行：

- `kernel-10`
- `kernel-11`
- ...
- `kernel-20`

### 4. 通过 `.g` 执行采样选中的 kernel

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
./flex-gpu-origin <benchmark> --g <kernels.g>
```

示例：

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
SWIFTSIM_TRACE_PATH=/path/to/trace_root ./flex-gpu-origin cfd --g /path/to/sampled_kernels.g
```

`.g` 文件要求：

- 一行一个 kernel 名
- 行内容可以是：
  - `kernel-7.trace`
  - `kernel-7-ctx_xxx.trace.xz`
  - 其他包含 `kernel-7` 的字符串
- 程序只会提取里面的 kernel 编号 `7`

## 并发执行

最大并发 kernel 数由 [main.cpp](/home/lxz/Swift-GPUSim/swift-sim/main.cpp) 中的常量控制：

```cpp
static const std::size_t MAX_CONCURRENT_KERNELS = 8;
```

当前调度方式不是静态分批，而是：

- 最多同时运行 `MAX_CONCURRENT_KERNELS` 个 kernel
- 某个 kernel 完成后，立即补下一个未执行 kernel

如果你要降低内存压力，可以把它改小，例如改成：

```cpp
static const std::size_t MAX_CONCURRENT_KERNELS = 4;
```

改完后重新编译：

```bash
cd /home/lxz/Swift-GPUSim/swift-sim
make
```

## 运行输出说明

### 运行开始时

```text
[WORKLOAD] benchmark=cfd total_kernels=300 max_concurrent_kernels=8
```

表示：

- 当前 benchmark 名
- 本次一共要运行多少个 kernel
- 当前最大并发数

### 某个 kernel 启动时

```text
[START] kernel=29 started=29/300 finished=12/300
```

表示：

- `kernel=29`：当前启动的 kernel 编号
- `started`：已经启动过多少个 kernel
- `finished`：已经完成多少个 kernel

### 某个 kernel 完成时

```text
[DONE] kernel=29 gpu_sim_cycle=3748 completed_gpu_cycles=17071 finished=4/300
```

表示：

- `gpu_sim_cycle`
  - 这个 kernel 自己的模拟周期数
- `completed_gpu_cycles`
  - 到当前为止，所有已完成 kernel 的 `gpu_sim_cycle` 累计值
- `finished`
  - 已完成 kernel 数

### 每个 kernel 的详细统计

每个 kernel 完成后还会打印：

- `gpu_sim_cycle`
- `gpu_sim_insn`
- `total_sm_cycles`
- `total_thread_inst`
- `total_thread_ldst_inst`
- `total_warp_inst`
- `total_ldst_warp_inst`
- L1 / L2 cache stats
- `total_time`

其中：

- `gpu_sim_cycle`
  - 单个 kernel 的全局模拟周期
- `total_sm_cycles`
  - 所有 SM 累积工作周期，不是 kernel 的全局完成时间
- `total_time`
  - 宿主机上这次仿真实际花费的时间，单位 `us`
  - 不是模拟出来的 GPU cycle

## 迁移到其他服务器

需要确认两件事：

### 1. trace 根目录

最推荐的方式是设置环境变量：

```bash
export SWIFTSIM_TRACE_PATH=/your/trace/root
```

如果不想用环境变量，再修改 [gpu.h](/home/lxz/Swift-GPUSim/swift-sim/gpu.h) 中的：

```cpp
#define DEFAULT_TRACE_PATH "..."
```

### 2. 配置文件位置

[config_reader.h](/home/lxz/Swift-GPUSim/swift-sim/config_reader.h) 当前使用相对路径：

```cpp
./gpu.config
./gpu_isa_latency.config
```

因此通常只要在 `swift-sim` 目录内执行程序即可。
