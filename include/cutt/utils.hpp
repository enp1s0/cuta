#ifndef __CUTT_UTILS_HPP__
#define __CUTT_UTILS_HPP__
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace cutt {
using mode_t = std::vector<std::pair<std::string, std::size_t>>;

namespace utils {
inline void insert_mode(
		mode_t& mode,
		const std::string name,
		const std::size_t dim
		) {
	mode.push_back(std::make_pair(name, dim));
}

inline std::size_t get_index(
		const mode_t& mode,
		const std::unordered_map<std::string, std::size_t>& pos
		) {
	std::unordered_map<std::string, std::size_t> stride;
	std::size_t dim_product = 1;
	for (const auto &m : mode) {
		stride.insert(std::make_pair(m.first, dim_product));
		dim_product *= m.second;
	}
	std::size_t index = 0;
	for (const auto& p : pos) {
		index += stride.at(p.first) * p.second;
	}
	return index;
}
} // namespace utils
} // namespace cutt
#endif
