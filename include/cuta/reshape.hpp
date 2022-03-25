#ifndef __CUTT_RESHAPE_HPP__
#define __CUTT_RESHAPE_HPP__
#include <cstdint>
#include <string>
#include <vector>
#include "utils.hpp"

namespace cuta {
inline std::vector<unsigned> get_permutation(const cuta::mode_t mode, const std::vector<std::string>& reshaped_order) {
	std::vector<unsigned> permutation;

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

template <class T>
void reshape(
		T* const dst_ptr,
		const T* const src_ptr,
		const cuta::mode_t& mode,
		const std::vector<std::string>& reshaped_order,
		cudaStream_t cuda_stream = 0
		);
} // namespace cuta
#endif
