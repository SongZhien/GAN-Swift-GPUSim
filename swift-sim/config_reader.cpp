#include "config_reader.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <queue>
#include <sstream>

namespace {
std::string trim_copy(const std::string &s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string normalize_section_key(const std::string &s) {
    std::string trimmed = trim_copy(s);
    if (trimmed.rfind("##", 0) == 0 && trimmed != "####") {
        return "##" + trim_copy(trimmed.substr(2));
    }
    return trimmed;
}
}

void gpu_config::read_config(const std::string &file_s) {
    std::ifstream file(file_s);
    std::string single_line;
    std::string comments = "//";
    std::string tmp_s = "##";
    std::string end_config = "####";
    while (std::getline(file, single_line)) {
        single_line = trim_copy(single_line);
        if (single_line.empty()) {
            continue;
        }
        if (single_line.compare(0, comments.size(), comments) == 0) {
            continue;
        }
        if (single_line.rfind("##", 0) == 0) {
            std::string section_key = normalize_section_key(single_line);
            if (section_key != "##pipeline_units" &&
                section_key != "##compute_capability" &&
                section_key != "##l1_cache_config" &&
                section_key != "##l2_cache_config" &&
                section_key != "##memory_config" &&
                section_key != "##gpu_isa_latency" &&
                section_key != end_config) {
                continue;
            }
        }
        std::string key, value;
        config_split(single_line, ":", key, value);
        key = normalize_section_key(key);
        value = trim_copy(value);
        if (key == "##pipeline_units") {
            std::getline(file, single_line);
            single_line = trim_copy(single_line);
            config_split(single_line, ":", key, value);
            key = normalize_section_key(key);
            value = trim_copy(value);
            while (single_line.compare(0, tmp_s.size(), tmp_s) != 0) {
                try {
                    m_sm_pipeline_units.insert(std::pair<std::string, int>(key, stoi(value)));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid value '" << value << "' for key '" << key << "' in pipeline_units" << std::endl;
                }
                std::getline(file, single_line);
                single_line = trim_copy(single_line);
                config_split(single_line, ":", key, value);
                key = normalize_section_key(key);
                value = trim_copy(value);
            }
        }
        if (key == "##compute_capability") {
            std::getline(file, single_line);
            single_line = trim_copy(single_line);
            config_split(single_line, ":", key, value);
            key = normalize_section_key(key);
            value = trim_copy(value);
            while (single_line.compare(0, tmp_s.size(), tmp_s) != 0) {
                try {
                    m_compute_capability.insert(std::pair<std::string, int>(key, stoi(value)));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid value '" << value << "' for key '" << key << "' in compute_capability" << std::endl;
                }
                std::getline(file, single_line);
                single_line = trim_copy(single_line);
                config_split(single_line, ":", key, value);
                key = normalize_section_key(key);
                value = trim_copy(value);
            }
        }
        if (key == "##l1_cache_config") {
            std::getline(file, single_line);
            single_line = trim_copy(single_line);
            config_split(single_line, ":", key, value);
            key = normalize_section_key(key);
            value = trim_copy(value);
            while (single_line.compare(0, tmp_s.size(), tmp_s) != 0) {
                l1_cache_config.insert(std::pair<std::string, std::string>(key, value));
                std::getline(file, single_line);
                single_line = trim_copy(single_line);
                config_split(single_line, ":", key, value);
                key = normalize_section_key(key);
                value = trim_copy(value);
            }
        }
        if (key == "##l2_cache_config") {
            std::getline(file, single_line);
            single_line = trim_copy(single_line);
            config_split(single_line, ":", key, value);
            key = normalize_section_key(key);
            value = trim_copy(value);
            while (single_line.compare(0, tmp_s.size(), tmp_s) != 0) {
                l2_cache_config.insert(std::pair<std::string, std::string>(key, value));
                std::getline(file, single_line);
                single_line = trim_copy(single_line);
                config_split(single_line, ":", key, value);
                key = normalize_section_key(key);
                value = trim_copy(value);
            }
        }

        if (key == "##memory_config") {
            std::getline(file, single_line);
            single_line = trim_copy(single_line);
            config_split(single_line, ":", key, value);
            key = normalize_section_key(key);
            value = trim_copy(value);
            while (single_line.compare(0, tmp_s.size(), tmp_s) != 0) {
                try {
                    memory_config.insert(std::pair<std::string, int>(key, stoi(value)));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid value '" << value << "' for key '" << key << "' in memory_config" << std::endl;
                }
                std::getline(file, single_line);
                single_line = trim_copy(single_line);
                config_split(single_line, ":", key, value);
                key = normalize_section_key(key);
                value = trim_copy(value);
            }
        }

        if (key == "##gpu_isa_latency") {
            while (std::getline(file, single_line)) {
                single_line = trim_copy(single_line);
                if (single_line.empty()) {
                    continue;
                }

                if (single_line[0] == '/') {
                    continue; 
                }
                if (single_line.compare(0, end_config.size(), end_config) == 0) {
                    break; 
                }
                if (single_line.length() > 1 && single_line.find(tmp_s) == 0) {
                    continue; 
                }
                std::istringstream is(single_line); 
                std::string str;
                std::queue<std::string> tmp_str; 
                while (is >> str) { 
                    tmp_str.push(str);
                }
                
                if (tmp_str.size() < 3) {
                    continue; // 跳过不完整的行
                }
                
                std::string opcode = tmp_str.front(); 
                tmp_str.pop(); 
                int latency; 
                std::string latency_t = tmp_str.front();
                
                // 安全的整数转换
                if (latency_t == "-") {
                    latency = -1;
                } else {
                    try {
                        latency = std::stoi(latency_t);
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Warning: Invalid latency value '" << latency_t << "' for opcode '" << opcode << "'" << std::endl;
                        latency = 1; // 默认值
                    }
                }
                
                tmp_str.pop(); 
                std::string unit = tmp_str.front(); 
                std::tuple<int, std::string> tuple_t(latency, unit);
                gpu_isa_latency.insert(std::pair<std::string, std::tuple<int, std::string> >(opcode, tuple_t));
            }
            key = end_config;
        }
        if(key!=end_config && !key.empty()){
            m_gpu_config.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    file.close();
}

void
gpu_config::config_split(const std::string &str, const std::string &pattern, std::string &str1, std::string &str2) {
    std::string strs = str + pattern;
    std::size_t pos = strs.find(pattern);
    std::size_t size = strs.size();
    bool bStop = false;
    while (pos != std::string::npos) {
        if (!bStop) {
            str1 = strs.substr(0, pos);
            bStop = true;
        } else {
            str2 = strs.substr(0, pos);
            return;
        }
        strs = strs.substr(pos + 1, size);
        pos = strs.find(pattern);
    }
}
