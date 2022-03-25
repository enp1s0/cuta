#ifndef __CUTT_RESHAPE_HPP__
#define __CUTT_RESHAPE_HPP__
#include <cstdint>
#include <string>
#include <vector>
#include "utils.hpp"

namespace cuta {
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
