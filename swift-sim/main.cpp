#include <iostream>
#include <fstream>
#include <mutex>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <set>
#include <dirent.h>
#include "gpu.h"

#include <chrono>
#ifdef __GLIBC__
#include <malloc.h>
#endif

typedef std::chrono::high_resolution_clock Clock;

// Maximum number of kernel simulations allowed to run concurrently.
// Tune this value manually to trade off runtime vs. memory usage.
static const std::size_t MAX_CONCURRENT_KERNELS = 32;

static std::mutex g_progress_mutex;
static std::size_t g_total_kernels = 0;
static std::size_t g_started_kernels = 0;
static std::size_t g_finished_kernels = 0;
static long long g_completed_gpu_cycles = 0;
static long long g_completed_active_sm_cycles = 0;
static long long g_completed_sm_cycles = 0;
static long long g_completed_issue_active_sm_cycles = 0;
static long long g_completed_memory_stall_sm_cycles = 0;
static long long g_completed_dependency_stall_sm_cycles = 0;
static long long g_completed_unit_stall_sm_cycles = 0;
static long long g_completed_scheduler_idle_sm_cycles = 0;

struct worker_context {
    std::string benchmark;
    int kernel_id;
};

void *sim_gpu(void *thread_arg);
static void run_single_kernel(int kernel_id, const std::string &benchmark);

static std::vector<int> parse_g_kernel_ids(const std::string &g_path) {
    std::ifstream file(g_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open g file: " << g_path << std::endl;
        exit(1);
    }

    std::vector<int> kernel_ids;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::size_t pos = line.find("kernel-");
        if (pos == std::string::npos) {
            continue;
        }
        pos += 7;
        std::size_t end = pos;
        while (end < line.size() &&
               std::isdigit(static_cast<unsigned char>(line[end]))) {
            end++;
        }
        if (end == pos) {
            continue;
        }
        kernel_ids.push_back(std::stoi(line.substr(pos, end - pos)));
    }

    if (kernel_ids.empty()) {
        std::cerr << "ERROR: no kernel entries found in g file: " << g_path
                  << std::endl;
        exit(1);
    }
    return kernel_ids;
}

static std::vector<int> discover_kernel_ids_from_trace_dir(const std::string &benchmark) {
    const std::string trace_dir = get_trace_root() + benchmark + "/traces";
    DIR *dir = opendir(trace_dir.c_str());
    if (dir == nullptr) {
        std::cerr << "ERROR: cannot open trace directory: " << trace_dir << std::endl;
        exit(1);
    }

    std::set<int> kernel_ids;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        std::size_t pos = name.find("kernel-");
        if (pos != 0) {
            continue;
        }
        pos += 7;
        std::size_t end = pos;
        while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
            end++;
        }
        if (end == pos) {
            continue;
        }
        if (name.find(".sass") == std::string::npos && name.find(".mem") == std::string::npos) {
            continue;
        }
        if (name.find("-block-") != std::string::npos) {
            continue;
        }
        kernel_ids.insert(std::stoi(name.substr(pos, end - pos)));
    }
    closedir(dir);

    if (kernel_ids.empty()) {
        std::cerr << "ERROR: no kernel trace files found in: " << trace_dir << std::endl;
        exit(1);
    }

    return std::vector<int>(kernel_ids.begin(), kernel_ids.end());
}

static void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " <benchmark>\n"
              << "  " << prog << " <benchmark> <kernel_num>\n"
              << "  " << prog << " <benchmark> <kernel_begin> <kernel_end>\n"
              << "  " << prog << " <benchmark> --g <kernelslist.g>\n";
}

int main(int argc, char **argv) {
    auto workload_t1 = Clock::now();
    std::string benchmark;
    std::vector<int> kernel_ids;

    if (argc == 2) {
        benchmark = argv[1];
        kernel_ids = discover_kernel_ids_from_trace_dir(benchmark);
    } else if (argc == 4 && std::string(argv[2]) == "--g") {
        benchmark = argv[1];
        kernel_ids = parse_g_kernel_ids(argv[3]);
    } else if (argc == 3) {
        benchmark = argv[1];
        int kernel_num = std::stoi(argv[2]);
        for (int i = 0; i < kernel_num; i++) {
            kernel_ids.push_back(i + 1);
        }
    } else if (argc == 4) {
        benchmark = argv[1];
        int kernel_begin = std::stoi(argv[2]);
        int kernel_end = std::stoi(argv[3]);
        for (int i = kernel_begin; i <= kernel_end; i++) {
            kernel_ids.push_back(i);
        }
    } else {
        std::cerr << "ERROR: invalid parameters" << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    const std::size_t max_workers = std::max<std::size_t>(
        1, std::min<std::size_t>(MAX_CONCURRENT_KERNELS, kernel_ids.size()));
    {
        std::lock_guard<std::mutex> lock(g_progress_mutex);
        g_total_kernels = kernel_ids.size();
        g_started_kernels = 0;
        g_finished_kernels = 0;
        g_completed_gpu_cycles = 0;
        g_completed_active_sm_cycles = 0;
        g_completed_sm_cycles = 0;
        g_completed_issue_active_sm_cycles = 0;
        g_completed_memory_stall_sm_cycles = 0;
        g_completed_dependency_stall_sm_cycles = 0;
        g_completed_unit_stall_sm_cycles = 0;
        g_completed_scheduler_idle_sm_cycles = 0;
    }
    printf("[WORKLOAD] benchmark=%s total_kernels=%zu max_concurrent_kernels=%zu\n",
           benchmark.c_str(), kernel_ids.size(), max_workers);
    fflush(stdout);

    struct thread_slot {
        pthread_t thread{};
        worker_context ctx;
        bool active = false;
    };

    std::vector<thread_slot> slots(max_workers);
    std::size_t next_kernel_index = 0;
    std::size_t active_threads = 0;

    auto launch_in_slot = [&](thread_slot &slot, int kernel_id) {
        slot.ctx.benchmark = benchmark;
        slot.ctx.kernel_id = kernel_id;
        int rc =
            pthread_create(&slot.thread, nullptr, sim_gpu, (void *)&slot.ctx);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
        slot.active = true;
        active_threads += 1;
    };

    while (next_kernel_index < kernel_ids.size() && active_threads < max_workers) {
        launch_in_slot(slots[active_threads], kernel_ids[next_kernel_index]);
        next_kernel_index += 1;
    }

    while (active_threads > 0) {
        bool progressed = false;
        for (auto &slot : slots) {
            if (!slot.active) {
                continue;
            }
#ifdef __linux__
            int join_rc = pthread_tryjoin_np(slot.thread, nullptr);
            if (join_rc == 0) {
                slot.active = false;
                active_threads -= 1;
                progressed = true;
                if (next_kernel_index < kernel_ids.size()) {
                    launch_in_slot(slot, kernel_ids[next_kernel_index]);
                    next_kernel_index += 1;
                }
            }
#else
            pthread_join(slot.thread, nullptr);
            slot.active = false;
            active_threads -= 1;
            progressed = true;
            if (next_kernel_index < kernel_ids.size()) {
                launch_in_slot(slot, kernel_ids[next_kernel_index]);
                next_kernel_index += 1;
            }
#endif
        }
        if (!progressed) {
            usleep(1000);
        }
    }
    auto workload_t2 = Clock::now();
    std::chrono::nanoseconds workload_t21 = workload_t2 - workload_t1;
    long workload_time =
        std::chrono::duration_cast<std::chrono::microseconds>(workload_t21).count();
    printf("[WORKLOAD_DONE] benchmark=%s finished_kernels=%zu/%zu completed_gpu_cycles=%lld completed_active_sm_cycles=%lld completed_issue_active_sm_cycles=%lld completed_memory_stall_sm_cycles=%lld completed_dependency_stall_sm_cycles=%lld completed_unit_stall_sm_cycles=%lld completed_scheduler_idle_sm_cycles=%lld completed_sm_cycles=%lld total_host_time %ld us\n",
           benchmark.c_str(), g_finished_kernels, g_total_kernels,
           g_completed_gpu_cycles, g_completed_active_sm_cycles,
           g_completed_issue_active_sm_cycles, g_completed_memory_stall_sm_cycles,
           g_completed_dependency_stall_sm_cycles, g_completed_unit_stall_sm_cycles,
           g_completed_scheduler_idle_sm_cycles, g_completed_sm_cycles, workload_time);
    return 0;
}

void *sim_gpu(void *thread_arg) {
    worker_context *ctx = (worker_context *)thread_arg;
    run_single_kernel(ctx->kernel_id, ctx->benchmark);
    return nullptr;
}

static void run_single_kernel(int kernel_id, const std::string &benchmark) {
    auto kernel_t0 = Clock::now();
    thread_data kernel_thread_data;
    kernel_thread_data.kernel_id = kernel_id;
    kernel_thread_data.benchmark = benchmark;
    {
        std::lock_guard<std::mutex> lock(g_progress_mutex);
        g_started_kernels += 1;
        printf("[START] kernel=%d started=%zu/%zu finished=%zu/%zu\n", kernel_id,
               g_started_kernels, g_total_kernels, g_finished_kernels,
               g_total_kernels);
        fflush(stdout);
    }
    std::string trace_path = get_trace_root() + benchmark + "/traces";
    std::ifstream file_sass(trace_path + "/kernel-" + std::to_string(kernel_id) +
                            ".sass");
    std::ifstream file_mem(trace_path + "/kernel-" + std::to_string(kernel_id) +
                           ".mem");
    if (!file_mem.is_open()) {
        std::cerr << "ERROR: mem trace missing: "
                  << trace_path + "/kernel-" + std::to_string(kernel_id) + ".mem"
                  << std::endl;
        pthread_exit(nullptr);
    }
    if (!file_sass.is_open()) {
        std::cerr << "ERROR: sass trace missing: "
                  << trace_path + "/kernel-" + std::to_string(kernel_id) + ".sass"
                  << std::endl;
        pthread_exit(nullptr);
    }
    {
        gpu m_gpu = gpu(benchmark);
        auto preprocess_t0 = Clock::now();
        m_gpu.launch_kernel(kernel_id);
        m_gpu.traceReader->build_block_mem_index(trace_path, kernel_id);
        m_gpu.execute_kernel(kernel_id);
        auto preprocess_t1 = Clock::now();
        auto t1 = Clock::now();
        m_gpu.gpu_cycle((void *)&kernel_thread_data);
        auto t2 = Clock::now();
        std::chrono::nanoseconds preprocess_ns = preprocess_t1 - preprocess_t0;
        std::chrono::nanoseconds t21 = t2 - t1;
        std::chrono::nanoseconds total_ns = t2 - kernel_t0;
        long preprocess_time =
            std::chrono::duration_cast<std::chrono::microseconds>(preprocess_ns).count();
        long kernel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(t21).count();
        long total_kernel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(total_ns).count();
        {
            std::lock_guard<std::mutex> lock(g_progress_mutex);
            g_finished_kernels += 1;
            g_completed_gpu_cycles += m_gpu.get_gpu_sim_cycles();
            g_completed_active_sm_cycles += m_gpu.get_active_sm_cycles();
            g_completed_sm_cycles += m_gpu.get_total_sm_cycles();
            g_completed_issue_active_sm_cycles += m_gpu.get_issue_active_sm_cycles();
            g_completed_memory_stall_sm_cycles += m_gpu.get_memory_stall_sm_cycles();
            g_completed_dependency_stall_sm_cycles += m_gpu.get_dependency_stall_sm_cycles();
            g_completed_unit_stall_sm_cycles += m_gpu.get_unit_stall_sm_cycles();
            g_completed_scheduler_idle_sm_cycles += m_gpu.get_scheduler_idle_sm_cycles();
            printf(
                "[DONE] kernel=%d gpu_sim_cycle=%d gpu_sim_insn=%lld active_sm_cycles=%d issue_active_sm_cycles=%d memory_stall_sm_cycles=%d dependency_stall_sm_cycles=%d unit_stall_sm_cycles=%d scheduler_idle_sm_cycles=%d total_sm_cycles=%d completed_gpu_cycles=%lld completed_active_sm_cycles=%lld completed_issue_active_sm_cycles=%lld completed_memory_stall_sm_cycles=%lld completed_dependency_stall_sm_cycles=%lld completed_unit_stall_sm_cycles=%lld completed_scheduler_idle_sm_cycles=%lld completed_sm_cycles=%lld finished=%zu/%zu\n",
                kernel_id, m_gpu.get_gpu_sim_cycles(), m_gpu.get_gpu_sim_insn(),
                m_gpu.get_active_sm_cycles(),
                m_gpu.get_issue_active_sm_cycles(), m_gpu.get_memory_stall_sm_cycles(),
                m_gpu.get_dependency_stall_sm_cycles(), m_gpu.get_unit_stall_sm_cycles(),
                m_gpu.get_scheduler_idle_sm_cycles(), m_gpu.get_total_sm_cycles(),
                g_completed_gpu_cycles, g_completed_active_sm_cycles,
                g_completed_issue_active_sm_cycles, g_completed_memory_stall_sm_cycles,
                g_completed_dependency_stall_sm_cycles, g_completed_unit_stall_sm_cycles,
                g_completed_scheduler_idle_sm_cycles, g_completed_sm_cycles,
                g_finished_kernels, g_total_kernels);
            fflush(stdout);
        }
        printf("preprocess_time %ld us\n", preprocess_time);
        printf("gpu_cycle_time %ld us\n", kernel_time);
        printf("total_time %ld us\n", total_kernel_time);
        printf("kernel_total_host_time %ld us\n", total_kernel_time);
    }
#ifdef __GLIBC__
    malloc_trim(0);
#endif
}
