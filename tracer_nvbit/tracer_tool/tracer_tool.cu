/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bitset>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#define TRACER_VERSION "5"

/*
 * IMPORTANT:
 * The simulator trace parser asserts: reg_srcs_num <= MAX_SRC (parser-side).
 * Many Accel-Sim trace parsers use MAX_SRC=4. To avoid crashing the parser,
 * we cap the number of printed src regs to TRACE_MAX_SRC (output-side only).
 */
#ifndef TRACE_MAX_SRC
#define TRACE_MAX_SRC 4
#endif

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
int lineinfo = 0;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* Use xz to compress the *.trace file */
int xz_compress_trace = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string user_folder = getcwd(NULL, 0);
std::string cwd = getcwd(NULL, 0);
std::string traces_location = cwd + "/traces/";
std::string kernelslist_location = cwd + "/traces/kernelslist";
std::string stats_location = cwd + "/traces/stats.csv";

std::unordered_map<CUcontext, std::string> ctx_kernelslist;
std::unordered_map<CUcontext, std::string> ctx_stats_location;
std::unordered_map<CUcontext, int> ctx_kernelid;
std::unordered_map<CUcontext, FILE *> ctx_resultsFile;
std::unordered_map<CUcontext, std::string> ctx_current_kernel_name;

std::string kernel_ranges = "";

struct KernelRange {
  uint64_t start;
  uint64_t end; // UINT64_MAX means open-ended
  std::vector<std::regex> kernel_name_regexes; // Vector of regexes for multiple patterns
};
std::vector<KernelRange> g_kernel_ranges;
uint64_t g_max_kernel_id = 0;

// -------------------- Kernel latency timing (GPU stream time, ms) --------------------
struct KernelTimingInfo {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUstream stream = 0;
  int kernel_id = -1;          // tracer kernel id (same as in trace/stats)
  bool active = false;         // timing active for current kernel
  bool stream_captured = false; // if kernel launch is captured by graph
};
static std::unordered_map<CUcontext, KernelTimingInfo> ctx_timing;
static bool warned_call_rel_noinc = false;

void parse_kernel_ranges_from_env() {
  g_kernel_ranges.clear();
  g_max_kernel_id = 0;

  const char *env_var = std::getenv("DYNAMIC_KERNEL_RANGE");
  if (!env_var || std::string(env_var).empty()) {
    g_kernel_ranges.push_back({0, 0, {std::regex(".*")}}); // 0 end = trace all
    return;
  }
  std::string input(env_var);
  std::istringstream stream(input);
  std::string token;

  while (stream >> token) {
    if (token.empty())
      continue;

    uint64_t start = 0, end = 0;
    std::vector<std::regex> regexes;

    size_t at_pos = token.find('@');
    std::string range_part, regex_part;

    if (at_pos != std::string::npos) {
      range_part = token.substr(0, at_pos);
      regex_part = token.substr(at_pos + 1);
    } else {
      range_part = token;
    }

    // Parse the range
    if (!range_part.empty()) {
      size_t dash_pos = range_part.find('-');
      if (dash_pos != std::string::npos) {
        std::string start_str = range_part.substr(0, dash_pos);
        std::string end_str = range_part.substr(dash_pos + 1);

        start = std::stoull(start_str);
        if (!end_str.empty()) {
          end = std::stoull(end_str);
        } else {
          end = 0; // open-ended
        }
      } else {
        start = std::stoull(range_part);
        end = start;
      }
    } else {
      // No range → match all IDs
      start = 0;
      end = 0;
    }

    // Parse the regexes
    if (!regex_part.empty()) {
      std::istringstream regex_stream(regex_part);
      std::string regex_token;
      while (std::getline(regex_stream, regex_token, ',')) {
        try {
          regexes.emplace_back(regex_token);
        } catch (const std::regex_error &e) {
          std::cerr << "Invalid regex: " << regex_token << std::endl;
        }
      }
    } else {
      regexes.emplace_back(".*"); // match all kernel names
    }

    g_kernel_ranges.push_back({start, end, regexes});
    if (end > g_max_kernel_id) {
      g_max_kernel_id = end;
    }
  }
}

bool should_trace_kernel(uint64_t kernel_id, const std::string &kernel_name) {
  for (const auto &range : g_kernel_ranges) {
    // Check range for kernel ID
    if (range.end == 0) {
      if (kernel_id >= range.start) {
        // Match any of the regexes for this range
        for (const auto &regex : range.kernel_name_regexes) {
          if (std::regex_match(kernel_name, regex)) {
            return true;
          }
        }
      }
    } else if (kernel_id >= range.start && kernel_id <= range.end) {
      // Match any of the regexes for this range
      for (const auto &regex : range.kernel_name_regexes) {
        if (std::regex_match(kernel_name, regex)) {
          return true;
        }
      }
    }
  }
  return false;
}

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

/* File pointers for the kernels, and stats files */
static FILE *kernelsFile = NULL;
static FILE *statsFile = NULL;
static bool first_call = true;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

/* Spinlock fast forward control */
int enable_spinlock_fast_forward = 0;
int spinlock_iter_to_keep = 0;
// Map from kernel name to spinlock instruction indices
std::map<std::string, std::vector<uint32_t> *> spinlock_instr_map;
std::pair<std::string, std::vector<uint32_t> > parse_spinlock_instructions(const std::string &line);

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(lineinfo, "TRACE_LINEINFO", 0,
              "Include source code line info at the start of each traced line. "
              "The target binary must be compiled with -lineinfo or "
              "--generate-line-info");
  GET_VAR_STR(
      kernel_ranges, "DYNAMIC_KERNEL_RANGE",
      "Specify kernel IDs or ranges to trace. Format:\n"
      "  - Single ID:       \"2\" traces only kernel 2.\n"
      "  - Range:           \"5-8\" traces kernels 5 through 8 (inclusive).\n"
      "  - Open-ended:      \"10-\" traces from kernel 10 onward.\n"
      "  - Multiple ranges: \"2 5-8 10-\" (space-separated).\n"
      "  - With regex:      \"5-8@kernel_a.*,kernel_b.*\" traces kernels 5–8 "
      "with matching names.\n"
      "If unset or empty, all kernels will be traced from the beginning.");
  GET_VAR_INT(
      active_from_start, "ACTIVE_FROM_START", 1,
      "Start instruction tracing from start or wait for cuProfilerStart "
      "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_RANGE options have no "
      "effect");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(
      terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT",
      0, "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0,
              "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(xz_compress_trace, "TRACE_FILE_COMPRESS", 1,
              "Create xz-compressed trace"
              "file");
  GET_VAR_INT(enable_spinlock_fast_forward, "ENABLE_SPINLOCK_FAST_FORWARD", 0,
              "Enable spinlock fast forwarding");
  GET_VAR_INT(spinlock_iter_to_keep, "SPINLOCK_ITER_TO_KEEP", 1,
              "Number of iterations to keep for spinlock fast forwarding");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  active_region = false;
  char *usr_defined_folder = std::getenv("TRACES_FOLDER");
  if (usr_defined_folder != NULL)
    user_folder = usr_defined_folder;
  parse_kernel_ranges_from_env();

  // Read in the spinlock_instructions.txt and build a map from kernel name to
  // spinlock instruction indices
  if (enable_spinlock_fast_forward) {
    std::string spinlock_instr_file =
        user_folder + "/spinlock_detection/spinlock_instructions.txt";
    std::ifstream instr_fs(spinlock_instr_file);
    std::string line;
    while (std::getline(instr_fs, line)) {
      auto [kernel_name, indices] = parse_spinlock_instructions(line);
      spinlock_instr_map[kernel_name] = new std::vector<uint32_t>(indices);
    }
    instr_fs.close();
  }
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose >= 1) {
      printf("Inspecting function %s at address 0x%lx\n",
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      uint32_t line_num = 0;
      // Temporary workaround for a bug in NVBit 1.7.4, which does not correctly
      // handle `call.rel`. Instrumenting this instruction leads to illegal
      // memory access. Refer to:
      // https://github.com/NVlabs/NVBit/issues/142#issue-2911561744
      if (!strcmp(instr->getOpcode(), "CALL.REL.NOINC")) {
        if (!warned_call_rel_noinc) {
          printf("Warning: Ignoring CALL.REL.NOINC (NVBit bug workaround); suppressing duplicate warnings\n");
          warned_call_rel_noinc = true;
        }
        continue;
      }

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      if (verbose >= 2) {
        instr->printDecoded();
      }

      if (lineinfo) {
        char *file_name, *dir_name;
        nvbit_get_line_info(ctx, func, instr->getOffset(), &file_name,
                            &dir_name, &line_num);
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];

      /* check all operands. For now, we ignore constant, TEX, predicates and
       * unified registers. We only report vector regisers */
      int src_oprd[MAX_SRC];
      int srcNum = 0;
      int dst_oprd = -1;
      int mem_oper_idx = -1;
      int num_mref = 0;
      uint64_t imm_value = 0;

      for (int i = 0; i < instr->getNumOperands(); ++i) {
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          assert(srcNum < MAX_SRC);
          src_oprd[srcNum] = instr->getOperand(i)->u.mref.ra_num;
          srcNum++;
          mem_oper_idx++;
          num_mref++;
        } else if (op->type == InstrType::OperandType::REG) {
          if (i == 0) {
            // find dst reg
            dst_oprd = instr->getOperand(0)->u.reg.num;
          } else {
            // find src regs
            assert(srcNum < MAX_SRC);
            src_oprd[srcNum] = instr->getOperand(i)->u.reg.num;
            srcNum++;
          }
        }
        // Add immediate value for DEPBAR instruction
        else if (op->type == InstrType::OperandType::IMM_UINT64) {
          imm_value = instr->getOperand(i)->u.imm_uint64.value;
        }
      }

      do {
        /* insert call to the instrumentation function with its
         * arguments */
        nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);

        /* pass predicate value */
        nvbit_add_call_arg_guard_pred_val(instr);

        /* send opcode and pc */
        nvbit_add_call_arg_const_val32(instr, opcode_id);
        nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());

        /* mem addresses info */
        if (mem_oper_idx >= 0) {
          nvbit_add_call_arg_const_val32(instr, 1);
          assert(num_mref <= 2);
          if (num_mref == 2) { // LDGSTS
            nvbit_add_call_arg_mref_addr64(instr, 1 - mem_oper_idx);
          } else {
            nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
          }
          nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
        } else {
          nvbit_add_call_arg_const_val32(instr, 0);
          nvbit_add_call_arg_const_val64(instr, static_cast<uint64_t>(-1));
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }

        /* reg info */
        nvbit_add_call_arg_const_val32(instr, dst_oprd);
        for (int i = 0; i < srcNum; i++) {
          nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
        }
        for (int i = srcNum; i < MAX_SRC; i++) {
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }
        nvbit_add_call_arg_const_val32(instr, srcNum);

        /* immediate info */
        nvbit_add_call_arg_const_val64(instr, imm_value);

        /* add pointer to channel_dev and other counters*/
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
        nvbit_add_call_arg_const_val64(instr,
                                       (uint64_t)&total_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(
            instr, (uint64_t)&reported_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report);
        /* Add Source code line number for current instr */
        nvbit_add_call_arg_const_val32(instr, (int)line_num);
        /* Add instruction index for current instr (spinlock detection) */
        nvbit_add_call_arg_const_val32(instr, (uint32_t)instr->getIdx());

        mem_oper_idx--;
      } while (mem_oper_idx >= 0);

      cnt++;
    }
  }
}

__global__ void flush_channel() {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  channel_dev.push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  channel_dev.flush();
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                nvbit_api_cuda_t cbid, void *params,
                                bool stream_capture = false,
                                bool build_graph = false) {
  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched.
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
  }

  // Mark if the kernel should be traced
  std::string func_name = std::string(nvbit_get_func_name(ctx, func, true));
  if (active_from_start && should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = true;

  // Terminate tracing if the limit number of kernels is reached
  if (terminate_after_limit_number_of_kernels_reached && g_max_kernel_id != 0 &&
      ctx_kernelid[ctx] > g_max_kernel_id) {
    exit(0);
  }

  // Get launch config for this kernel
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
      cbid == API_CUDA_cuLaunchKernelEx) {
    cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
    gridDimX = p->config->gridDimX;
    gridDimY = p->config->gridDimY;
    gridDimZ = p->config->gridDimZ;
    blockDimX = p->config->blockDimX;
    blockDimY = p->config->blockDimY;
    blockDimZ = p->config->blockDimZ;
    sharedMemBytes = p->config->sharedMemBytes;
    hStream = p->config->hStream;
  } else {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
    gridDimX = p->gridDimX;
    gridDimY = p->gridDimY;
    gridDimZ = p->gridDimZ;
    blockDimX = p->blockDimX;
    blockDimY = p->blockDimY;
    blockDimZ = p->blockDimZ;
    sharedMemBytes = p->sharedMemBytes;
    hStream = p->hStream;
  }

  // Get the number of registers and shared memory size for the kernel
  int nregs;
  CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

  int shmem_static_nbytes;
  CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

  int binary_version;
  CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                                   CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));

  // Instrument the kernel if needed
  instrument_function_if_needed(ctx, func);

  // Enable or disable tracing based on the active region
  if (active_region) {
    nvbit_enable_instrumented(ctx, func, true);
    stop_report = false;
  } else {
    nvbit_enable_instrumented(ctx, func, false);
    stop_report = true;
  }

  if (!stop_report) {
    // Record timing only for kernels we actually trace. Calling CUDA APIs for
    // skipped kernels from inside the callback can deadlock under NVBit.
    KernelTimingInfo &ti = ctx_timing[ctx];

    ti.kernel_id = ctx_kernelid[ctx];
    ti.stream = hStream;
    ti.active = true;
    ti.stream_captured = stream_capture;

    if (ti.start == nullptr) {
      cudaEventCreate(&ti.start);
    }
    if (ti.stop == nullptr) {
      cudaEventCreate(&ti.stop);
    }

    cudaEventRecord(ti.start, (cudaStream_t)hStream);
  }

  // Create the trace file per kernel
  char buffer[2048];
  std::string trace_filename = user_folder + "/traces/";
  sprintf(buffer, "%s/kernel-%d-ctx_0x%lx.trace", trace_filename.c_str(),
          ctx_kernelid[ctx], ctx);

  if (!stop_report) {
    if (!xz_compress_trace) {
      ctx_resultsFile[ctx] = fopen(buffer, "w");
      printf("Writing results to %s\n", buffer);
    } else {
      char cmd_buffer[1039];
      sprintf(cmd_buffer, "xz -1 -T0 > %s.xz", buffer);
      ctx_resultsFile[ctx] = popen(cmd_buffer, "w");
      printf("Writing results to %s.xz\n", buffer);
    }

    // Writing header information
    fprintf(ctx_resultsFile[ctx], "-kernel name = %s\n",
            nvbit_get_func_name(ctx, func, true));
    fprintf(ctx_resultsFile[ctx], "-kernel id = %d\n", ctx_kernelid[ctx]);
    fprintf(ctx_resultsFile[ctx], "-grid dim = (%d,%d,%d)\n", gridDimX,
            gridDimY, gridDimZ);
    fprintf(ctx_resultsFile[ctx], "-block dim = (%d,%d,%d)\n", blockDimX,
            blockDimY, blockDimZ);
    fprintf(ctx_resultsFile[ctx], "-shmem = %d\n",
            shmem_static_nbytes + sharedMemBytes);
    fprintf(ctx_resultsFile[ctx], "-nregs = %d\n", nregs);
    fprintf(ctx_resultsFile[ctx], "-binary version = %d\n", binary_version);
    fprintf(ctx_resultsFile[ctx], "-cuda stream id = %lu\n", (uint64_t)hStream);
    fprintf(ctx_resultsFile[ctx], "-shmem base_addr = 0x%016lx\n",
            (uint64_t)nvbit_get_shmem_base_addr(ctx));
    fprintf(ctx_resultsFile[ctx], "-local mem base_addr = 0x%016lx\n",
            (uint64_t)nvbit_get_local_mem_base_addr(ctx));
    fprintf(ctx_resultsFile[ctx], "-nvbit version = %s\n", NVBIT_VERSION);
    fprintf(ctx_resultsFile[ctx], "-accelsim tracer version = %s\n",
            TRACER_VERSION);
    fprintf(ctx_resultsFile[ctx], "-enable lineinfo = %d\n", lineinfo);
    fprintf(ctx_resultsFile[ctx], "\n");

    fprintf(ctx_resultsFile[ctx],
            "#traces format = [line_num] PC mask dest_num [reg_dests] "
            "opcode src_num "
            "[reg_srcs] mem_width [adrrescompress?] [mem_addresses] "
            "immediate\n");
    fprintf(ctx_resultsFile[ctx], "\n");
  }

  kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
  // This will be a relative path to the traces file
  sprintf(buffer, "kernel-%d-ctx_0x%lx.trace%s", ctx_kernelid[ctx], ctx,
          xz_compress_trace ? ".xz" : "");
  if (!stop_report) {
    fprintf(kernelsFile, buffer);
    fprintf(kernelsFile, "\n");
  }
  fclose(kernelsFile);

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  unsigned blocks = gridDimX * gridDimY * gridDimZ;
  unsigned threads = blockDimX * blockDimY * blockDimZ;

  fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,
          nvbit_get_func_name(ctx, func, true), gridDimX, gridDimY, gridDimZ,
          blocks, blockDimX, blockDimY, blockDimZ, threads);

  fclose(statsFile);

  ctx_kernelid[ctx]++;
  ctx_current_kernel_name[ctx] =
      std::string(nvbit_get_func_name(ctx, func, true));
  recv_thread_receiving = true;
}

static void leave_kernel_launch(CUcontext ctx, CUfunction func) {
  float latency_ms = -1.0f;
  unsigned total_insts_per_kernel = 0;
  unsigned reported_insts_per_kernel = 0;

  if (!stop_report) {
    if (ctx_timing.find(ctx) != ctx_timing.end()) {
      KernelTimingInfo &ti = ctx_timing[ctx];
      if (ti.active && !ti.stream_captured && ti.stop != nullptr) {
        cudaEventRecord(ti.stop, (cudaStream_t)ti.stream);
      }
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("cuda error: %s\n", cudaGetErrorName(err));
    }
    assert(err == cudaSuccess);

    skip_flag = true;
    flush_channel<<<1, 1>>>();
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
    skip_flag = false;

    while (recv_thread_receiving) {
      pthread_yield();
    }

    total_insts_per_kernel = total_dynamic_instr_counter - old_total_insts;
    old_total_insts = total_dynamic_instr_counter;

    reported_insts_per_kernel =
        reported_dynamic_instr_counter - old_total_reported_insts;
    old_total_reported_insts = reported_dynamic_instr_counter;

    if (ctx_timing.find(ctx) != ctx_timing.end()) {
      KernelTimingInfo &ti = ctx_timing[ctx];
      if (ti.active) {
        if (!ti.stream_captured && ti.start != nullptr && ti.stop != nullptr) {
          cudaEventSynchronize(ti.stop);
          cudaEventElapsedTime(&latency_ms, ti.start, ti.stop);
        }
        ti.active = false;
      }
    }
  } else if (ctx_timing.find(ctx) != ctx_timing.end()) {
    ctx_timing[ctx].active = false;
  }

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  fprintf(statsFile, "%d,%d,%.6f", total_insts_per_kernel,
          reported_insts_per_kernel, latency_ms);
  fprintf(statsFile, "\n");
  fclose(statsFile);

  if (!stop_report) {
    if (!xz_compress_trace) {
      fclose(ctx_resultsFile[ctx]);
    } else {
      pclose(ctx_resultsFile[ctx]);
    }
  }

  std::string func_name = std::string(nvbit_get_func_name(ctx, func, true));
  if (active_from_start && !should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = false;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (first_call == true) {
    first_call = false;
    std::string traces_folder = user_folder + "/traces";
    if (mkdir(traces_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ==
        -1) {
      if (errno == EEXIST) {
        // already exists
      } else {
        // something else
        std::cout << "cannot create folder error:" << strerror(errno)
                  << std::endl;
        return;
      }
    }
    kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "w");
    statsFile = fopen(ctx_stats_location[ctx].c_str(), "w");
    fprintf(statsFile,
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts, latency\n");
    fclose(statsFile);
  }

  switch (cbid) {
  // We start with recording kernel launch events
  // Identify all the possible CUDA launch events without stream
  // parameters, they will not get involved with cuda graph
  case API_CUDA_cuLaunch:
  case API_CUDA_cuLaunchGrid: {
    cuLaunch_params *p = (cuLaunch_params *)params;
    CUfunction func = p->f;
    if (!is_exit) {
      enter_kernel_launch(ctx, func, cbid, params, false, false);
    } else {
      leave_kernel_launch(ctx, func);
    }
  } break;

  // kernel launches with stream parameter, they can be used for cuda graph
  case API_CUDA_cuLaunchKernel_ptsz:
  case API_CUDA_cuLaunchKernel:
  case API_CUDA_cuLaunchCooperativeKernel:
  case API_CUDA_cuLaunchCooperativeKernel_ptsz:
  case API_CUDA_cuLaunchKernelEx:
  case API_CUDA_cuLaunchKernelEx_ptsz:
  case API_CUDA_cuLaunchGridAsync: {
    CUfunction func;
    CUstream hStream;

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
      hStream = p->config->hStream;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
               cbid == API_CUDA_cuLaunchKernel ||
               cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
               cbid == API_CUDA_cuLaunchCooperativeKernel) {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      func = p->f;
      hStream = p->hStream;
    } else {
      cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
      func = p->f;
      hStream = p->hStream;
    }

    cudaStreamCaptureStatus streamStatus;
    /* check if the stream is capturing, if yes, do not sync */
    CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
    if (!is_exit) {
      bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
      enter_kernel_launch(ctx, func, cbid, params, stream_capture);
    } else {
      if (streamStatus != cudaStreamCaptureStatusActive) {
        if (verbose >= 1) {
          printf("kernel %s not captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
        leave_kernel_launch(ctx, func);
      } else {
        if (verbose >= 1) {
          printf("kernel %s captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
      }
    }
  } break;

  case API_CUDA_cuGraphAddKernelNode: {
    cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
    CUfunction func = p->nodeParams->func;

    if (!is_exit) {
      // cuGraphAddKernelNode_params->nodeParams is the same as
      // cuLaunchKernel_params up to sharedMemBytes
      enter_kernel_launch(ctx, func, cbid, (void *)p->nodeParams, false, true);
    }
  } break;

  case API_CUDA_cuGraphLaunch: {
    // if we are exiting a cuda graph launch:
    // Wait until the graph is completed using cudaStreamSynchronize()
    if (is_exit) {
      cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      assert(cudaGetLastError() == cudaSuccess);
      /* push a flush channel kernel */
      skip_flag = true;
      flush_channel<<<1, 1, 0, p->hStream>>>();
      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      skip_flag = false;
      assert(cudaGetLastError() == cudaSuccess);
    }
  } break;

  // Now we need to record cuda memcpy events
  case API_CUDA_cuMemcpyHtoD_v2: {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      char buffer[1024];
      kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
      sprintf(buffer, "MemcpyHtoD,0x%016llx,%llu", p->dstDevice, p->ByteCount);
      fprintf(kernelsFile, buffer);
      fprintf(kernelsFile, "\n");
      fclose(kernelsFile);
    }
  } break;

  // For cuProfiler, we need to set the active region accordingly
  case API_CUDA_cuProfilerStart: {
    if (is_exit && !active_from_start) {
      active_region = true;
    }
  } break;

  case API_CUDA_cuProfilerStop: {
    if (is_exit && !active_from_start) {
      active_region = false;
    }
  } break;

  default:
    break;
  }
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {
  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        if (stride != (int)(addrs[s] - addrs[s - 1])) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {
  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

void trim_string(std::string &str) {
  // Remove the leading and trailing spaces
  str.erase(0, str.find_first_not_of(' '));
  str.erase(str.find_last_not_of(' ') + 1);
}

typedef std::map<uint32_t, size_t> counter_t;
typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> warp_key_t;

counter_t create_counter(const std::vector<uint32_t> &indices) {
  counter_t counter;
  for (auto instr_idx : indices) {
    counter[instr_idx] = 0;
  }
  return counter;
}

std::pair<std::string, std::vector<uint32_t> > parse_spinlock_instructions(const std::string &line) {
  std::vector<uint32_t> indices;
  // Each line is of the form: <kernel_id>, <kernel_name>: <indices>
  // Though kernel id is not used
  // -2 for the comma and the space
  size_t name_length = line.find(':') - line.find(',') - 2;
  std::string kernel_name = line.substr(line.find(',') + 2, name_length);
  std::string indices_str = line.substr(line.find(':') + 1);
  trim_string(indices_str);
  std::stringstream ss(indices_str);
  std::string instr_idx;
  while (std::getline(ss, instr_idx, ' ')) {
    indices.push_back(std::stoi(instr_idx));
  }
  return {kernel_name, indices};
}

/* -----------------------------
 *  NEW: Buffer TB->warp->inst lines, then emit parser-compatible TB stream:
 *   #BEGIN_TB
 *   thread block = x,y,z
 *   warp = w
 *   insts = N
 *   <N instruction lines>
 *   ...
 *   #END_TB
 * ----------------------------- */

struct TBKey {
  int x;
  int y;
  int z;
  bool operator==(const TBKey &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct TBKeyHash {
  size_t operator()(const TBKey &k) const {
    size_t hx = (size_t)k.x * 1315423911u;
    size_t hy = (size_t)k.y * 2654435761u;
    size_t hz = (size_t)k.z * 97531u;
    return hx ^ (hy << 1) ^ (hz << 2);
  }
};

struct WarpBuf {
  std::vector<std::string> inst_lines;
};

struct TBBuf {
  std::unordered_map<int, WarpBuf> warps; // warpid_tb -> lines
  bool has_any = false;
};

static inline std::string format_inst_line(const inst_trace_t *ma) {
  std::stringstream ss;

  if (lineinfo) {
    ss << std::dec << ma->line_num << " ";
  }

  // PC and mask
  ss << std::hex;
  ss.width(4);
  ss.fill('0');
  ss << (unsigned)ma->vpc << " ";

  ss.width(8);
  ss.fill('0');
  ss << (unsigned)(ma->active_mask & ma->predicate_mask) << " ";

  // dest
  if (ma->GPRDst >= 0) {
    ss << std::dec << 1 << " "
       << "R" << ma->GPRDst << " ";
  } else {
    ss << std::dec << 0 << " ";
  }

  // opcode
  ss << id_to_opcode_map[ma->opcode_id] << " ";

  // src regs (capped)
  unsigned src_count_all = 0;
  for (int s = 0; s < MAX_SRC; s++) {
    if (ma->GPRSrcs[s] >= 0)
      src_count_all++;
  }
  unsigned src_count = src_count_all;
  if (src_count > (unsigned)TRACE_MAX_SRC)
    src_count = (unsigned)TRACE_MAX_SRC;

  ss << std::dec << src_count << " ";

  unsigned printed = 0;
  for (int s = 0; s < MAX_SRC && printed < (unsigned)TRACE_MAX_SRC; s++) {
    if (ma->GPRSrcs[s] >= 0) {
      ss << "R" << ma->GPRSrcs[s] << " ";
      printed++;
    }
  }

  // mem addresses / compression
  std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
  if (ma->is_mem) {
    std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, '.')) {
      if (!token.empty())
        tokens.push_back(token);
    }

    ss << std::dec << get_datawidth_from_opcode(tokens) << " ";

    bool base_stride_success = false;
    uint64_t base_addr = 0;
    int stride = 0;
    std::vector<long long> deltas;

    if (enable_compress) {
      base_stride_success = base_stride_compress(ma->addrs, mask, base_addr, stride);
      if (!base_stride_success) {
        base_delta_compress(ma->addrs, mask, base_addr, deltas);
      }
    }

    if (base_stride_success && enable_compress) {
      ss << std::dec << (unsigned)address_format::base_stride << " ";
      ss << std::hex << "0x" << base_addr << " ";
      ss << std::dec << stride << " ";
    } else if (!base_stride_success && enable_compress) {
      ss << std::dec << (unsigned)address_format::base_delta << " ";
      ss << std::hex << "0x" << base_addr << " ";
      ss << std::dec;
      for (int i = 0; i < (int)deltas.size(); i++) {
        ss << deltas[i] << " ";
      }
    } else {
      ss << std::dec << (unsigned)address_format::list_all << " ";
      ss << std::hex;
      for (int s = 0; s < 32; s++) {
        if (mask.test(s)) {
          ss << "0x" << ma->addrs[s] << " ";
        }
      }
      ss << std::dec;
    }
  } else {
    ss << std::dec << 0 << " ";
  }

  // immediate
  ss << std::dec << ma->imm;

  return ss.str();
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  // This counter map will keep track of the spinlock instruction
  // count in the current detected spinlock loop for each warp
  std::map<warp_key_t, counter_t> warp_counter_map;

  // Buffer of TBs for current kernel
  std::unordered_map<TBKey, TBBuf, TBKeyHash> tb_bufs;

  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;
    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed */
        if (ma->cta_id_x == -1) {
          // If we were not tracing, just finish.
          if (!stop_report) {
            // Emit parser-compatible TB stream now:
            // Deterministic order for TB keys
            std::vector<TBKey> keys;
            keys.reserve(tb_bufs.size());
            for (auto &kv : tb_bufs) {
              if (kv.second.has_any)
                keys.push_back(kv.first);
            }
            std::sort(keys.begin(), keys.end(),
                      [](const TBKey &a, const TBKey &b) {
                        if (a.x != b.x) return a.x < b.x;
                        if (a.y != b.y) return a.y < b.y;
                        return a.z < b.z;
                      });

            for (auto &k : keys) {
              auto &tb = tb_bufs[k];
              if (!tb.has_any) continue;

              fprintf(ctx_resultsFile[ctx], "#BEGIN_TB\n");
              // IMPORTANT: must match parser sscanf exactly: "thread block = %d,%d,%d"
              fprintf(ctx_resultsFile[ctx], "thread block = %d,%d,%d\n", k.x, k.y, k.z);

              // deterministic warp order
              std::vector<int> warps;
              warps.reserve(tb.warps.size());
              for (auto &wkv : tb.warps) warps.push_back(wkv.first);
              std::sort(warps.begin(), warps.end());

              for (int w : warps) {
                auto &wl = tb.warps[w].inst_lines;
                fprintf(ctx_resultsFile[ctx], "warp = %d\n", w);
                fprintf(ctx_resultsFile[ctx], "insts = %d\n", (int)wl.size());
                for (auto &line : wl) {
                  fprintf(ctx_resultsFile[ctx], "%s\n", line.c_str());
                }
              }

              fprintf(ctx_resultsFile[ctx], "#END_TB\n");
            }
          }

          tb_bufs.clear();
          recv_thread_receiving = false;

          if (enable_spinlock_fast_forward) {
            warp_counter_map.clear();
          }
          break;
        }

        // If current kernel is not traced, ignore all inst records until end marker
        if (stop_report) {
          num_processed_bytes += sizeof(inst_trace_t);
          continue;
        }

        /* Spinlock fast forwarding */
        if (enable_spinlock_fast_forward) {
          warp_key_t warp_key = std::make_tuple(ma->cta_id_x, ma->cta_id_y,
                                                ma->cta_id_z, ma->warpid_tb);
          if (warp_counter_map.find(warp_key) == warp_counter_map.end()) {
            std::vector<uint32_t> &indices =
                *(spinlock_instr_map[ctx_current_kernel_name[ctx]]);
            warp_counter_map[warp_key] = create_counter(indices);
          }

          auto &counter = warp_counter_map[warp_key];

          if (counter.find(ma->instr_idx) != counter.end()) {
            counter[ma->instr_idx]++;
            if (counter[ma->instr_idx] > (size_t)spinlock_iter_to_keep) {
              num_processed_bytes += sizeof(inst_trace_t);
              continue;
            }
          } else {
            for (auto &kv : counter) {
              kv.second = 0;
            }
          }
        }

        // Buffer this instruction line into TB->warp
        TBKey tbk{ma->cta_id_x, ma->cta_id_y, ma->cta_id_z};
        int warp = ma->warpid_tb;

        TBBuf &tb = tb_bufs[tbk];
        tb.has_any = true;

        std::string inst_line = format_inst_line(ma);
        tb.warps[warp].inst_lines.push_back(inst_line);

        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }

  free(recv_buffer);
  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  ctx_current_kernel_name[ctx] = "";
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, ctx);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }

  // destroy cuda events for timing (avoid leaks)
  auto it = ctx_timing.find(ctx);
  if (it != ctx_timing.end()) {
    if (it->second.start) cudaEventDestroy(it->second.start);
    if (it->second.stop) cudaEventDestroy(it->second.stop);
    ctx_timing.erase(it);
  }
}

void nvbit_at_ctx_init(CUcontext ctx) {
  // Everytime we init a context, add the foldername and kernelid to the set
  char buffer[2048];
  sprintf(buffer, "kernelslist_ctx_0x%lx", ctx);
  std::string tmp_kernelslist = user_folder + "/traces/" + buffer;
  ctx_kernelslist[ctx] = tmp_kernelslist;
  sprintf(buffer, "stats_ctx_0x%lx", ctx);
  std::string tmp_stats = user_folder + "/traces/" + buffer;
  ctx_stats_location[ctx] = tmp_stats;
  ctx_kernelid[ctx] = 1;

  // init timing info for this context
  ctx_timing[ctx] = KernelTimingInfo{};
}
