#include "SysFind.h"

#include <vector>
#include <string>
#include <Windows.h>
#include <algorithm>
#include <sstream>
#include<fstream>
#include<iostream>
#include <iomanip>

std::vector<std::string> get_subdirs_indir(const std::string& root_dir_name)
{
	std::vector<std::string> vsubdirs;
	HANDLE hFind;
	WIN32_FIND_DATA win32fd;
	std::string search_name = root_dir_name + "\\*";
	hFind = FindFirstFile(search_name.c_str(), &win32fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				std::string f_name(win32fd.cFileName);
				vsubdirs.push_back(f_name);
			}
		} while (FindNextFile(hFind, &win32fd));
		FindClose(hFind);
		std::sort(vsubdirs.begin(), vsubdirs.end());
	}
	vsubdirs.erase(std::remove(vsubdirs.begin(), vsubdirs.end(), "."), vsubdirs.end());
	vsubdirs.erase(std::remove(vsubdirs.begin(), vsubdirs.end(), ".."), vsubdirs.end());
	return vsubdirs;
}

// get filename list
std::vector<std::string> get_fnames_indir(const std::string& dir_name, const char* ext) {
	std::vector<std::string> vfnames;

	WIN32_FIND_DATA win32fd;
	std::ostringstream search_name;
	search_name << dir_name << "\\*" << ext;
	HANDLE hFind = FindFirstFile(search_name.str().c_str(), &win32fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				std::string f_name(win32fd.cFileName);
				vfnames.push_back(f_name);
			}
		} while (FindNextFile(hFind, &win32fd));
		FindClose(hFind);
		std::sort(vfnames.begin(), vfnames.end());
	}
	return vfnames;
}

//CheckFileExists
bool checkFileExist(const std::string& str)
{
	std::ifstream ifs(str);
	return ifs.is_open();
}

std::string getDatetimeStr() {
	time_t t = time(nullptr);
	const tm* localTime = localtime(&t);
	std::stringstream s;
	s << "20" << localTime->tm_year - 100;
	// setw(),setfill()‚Å0‹l‚ß
	s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
	s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
	s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
	s << std::setw(2) << std::setfill('0') << localTime->tm_min;
	s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
	// std::string‚É‚µ‚Ä’l‚ð•Ô‚·
	return s.str();
}