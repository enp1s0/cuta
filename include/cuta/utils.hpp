#ifndef __CUTA_UTILS_HPP__
#define __CUTA_UTILS_HPP__
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace cuta {
using mode_t = std::vector<std::pair<std::string, std::size_t>>;

namespace utils {
inline void insert_mode(
		mode_t& mode,
		const std::string name,
		const std::size_t dim
		) {
	mode.push_back(std::make_pair(name, dim));
}

inline std::vector<std::string> get_dim_names(
		const cuta::mode_t& mode
		) {
	std::vector<std::string> dim_names(mode.size());
	for (std::uint32_t i = 0; i < mode.size(); i++) {
		dim_names[i] = mode[i].first;
	}
	return dim_names;
}

inline std::vector<std::string> get_intersection_dim_names(
		const std::vector<std::string>& a_names,
		const std::vector<std::string>& b_names
		) {
	std::vector<std::string> result;
	for (const auto& a : a_names) {
		for (const auto& b : b_names) {
			if (a == b) {
				result.push_back(a);
				continue;
			}
		}
	}
	return result;
}

inline cuta::mode_t get_intersection_mode(
		const cuta::mode_t& a_mode,
		const std::vector<std::string>& b_names
		) {
	cuta::mode_t result;
	for (const auto& a : a_mode) {
		for (const auto& b : b_names) {
			if (a.first == b) {
				result.push_back(a);
				continue;
			}
		}
	}
	return result;
}

inline cuta::mode_t get_intersection_mode(
		const cuta::mode_t& a_mode,
		const cuta::mode_t& b_mode
		) {
	return cuta::utils::get_intersection_mode(a_mode, cuta::utils::get_dim_names(b_mode));
}

inline std::vector<std::string> get_difference_dim_names(
		const std::vector<std::string>& minuend_names,
		const std::vector<std::string>& subtrahend_names
		) {
	std::vector<std::string> result;
	for (const auto& m : minuend_names) {
		bool add = true;
		for (const auto& s : subtrahend_names) {
			if (m == s) {
				add = false;
				break;
			}
		}
		if (add) {
			result.push_back(m);
		}
	}
	return result;
}

inline cuta::mode_t get_difference_mode(
		const cuta::mode_t& minuend_mode,
		const std::vector<std::string>& subtrahend_names
		) {
	cuta::mode_t result;
	for (const auto& m : minuend_mode) {
		bool add = true;
		for (const auto& s : subtrahend_names) {
			if (m.first == s) {
				add = false;
				break;
			}
		}
		if (add) {
			result.push_back(m);
		}
	}
	return result;
}

inline cuta::mode_t get_difference_mode(
		const cuta::mode_t& minuend_mode,
		const cuta::mode_t& subtrahend_mode
		) {
	return cuta::utils::get_difference_mode(minuend_mode, cuta::utils::get_dim_names(subtrahend_mode));
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

inline std::size_t get_num_elements(
		const mode_t& mode
		) {
	std::size_t num = 1;
	for (const auto& m : mode) {
		num *= m.second;
	}
	return num;
}

template <class T = std::size_t>
inline std::vector<T> get_dim_sizes(const cuta::mode_t& mode) {
	std::vector<T> sizes(mode.size());
	unsigned i = 0;
	for (const auto& m : mode) {
		sizes[i++] = m.second;
	}
	return sizes;
}

template <class T = unsigned>
inline std::vector<T> get_permutation(const cuta::mode_t mode, const std::vector<std::string>& reshaped_order) {
	std::vector<T> permutation;

	for (const auto& o : reshaped_order) {
		unsigned i = 0;
		bool found = false;
		for (const auto m : mode) {
			if (m.first == o) {
				permutation.push_back(i);
				found = true;
				break;
			}
			i++;
		}
		if (!found) {
			throw std::runtime_error("[cuta] Unknown mode name " + o + " (" + __func__ + ")");
		}
	}

	return permutation;
}

inline cuta::mode_t get_permutated_mode(const cuta::mode_t mode, const std::vector<std::string>& reshaped_order) {
	cuta::mode_t permutated_mode;

	for (const auto& o : reshaped_order) {
		unsigned i = 0;
		bool found = false;
		for (const auto m : mode) {
			if (m.first == o) {
				permutated_mode.push_back(m);
				found = true;
				break;
			}
			i++;
		}
		if (!found) {
			throw std::runtime_error("[cuta] Unknown mode name " + o + " (" + __func__ + ")");
		}
	}

	return permutated_mode;
}

inline void print_mode(
		const mode_t& mode,
		const std::string name = ""
		) {
	if (name.length() != 0) {
		std::printf("mode(%s):\n", name.c_str());
	}
	for (unsigned i = 0; i < mode.size(); i++) {
		std::printf("|-\\- %s\n", mode[i].first.c_str());
		std::printf("| %lu\n", mode[i].second);
	}
}

inline std::string get_mode_string(
		const cuta::mode_t& mode
		) {
	std::string string = "";

	for (std::uint32_t i = 0; i < mode.size(); i++) {
		string += mode[i].first + "[" + std::to_string(mode[i].second) + "]";
		if (i + 1 != mode.size()) {
			string += "-";
		}
	}
	return string;
}
} // namespace utils
} // namespace cuta
#endif
