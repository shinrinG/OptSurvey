#pragma once

#include <vector>
#include <string>

std::vector<std::string> get_subdirs_indir(const std::string& root_dir_name);
std::vector<std::string> get_fnames_indir(const std::string& dir_name, const char* ext = "*.bmp");
bool checkFileExist(const std::string& str);
std::string getDatetimeStr();