//
// Created by 徐向荣 on 2022/6/7.
//

#ifndef TRACE_READER_H
#define TRACE_READER_H

#include <utility>
#include <queue>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <fstream>
#include "kernel.h"

void gen_block_mem_trace(const std::string &trace_path, int kernel_id);



class trace_reader {
public:
    explicit trace_reader(int kernel_id) {
        m_kernel_id = kernel_id;
    }

    void gen_block_mem_trace(const std::string &trace_path, int kernel_id);
    void build_block_mem_index(const std::string &trace_path, int kernel_id);
    const std::vector<std::streampos> *get_block_mem_offsets(int block_id) const;
    const std::string &get_mem_file_path() const;

    void
    read_sass(const std::string &benchmark_path, kernel_info &kernelInfo, std::vector<trace_inst> &trace_insts) const;
    void read_all_block_sass(const std::string &benchmark_path, kernel_info &kernelInfo, std::vector<trace_inst> &trace_insts) const;


private:
    int m_kernel_id;
    bool m_block_mem_index_ready = false;
    std::string m_mem_file_path;
    std::map<int, std::vector<std::streampos>> m_block_mem_offsets;
};

#endif 
