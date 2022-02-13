#ifndef __CUTT_RESHAPE_HPP__
#define __CUTT_RESHAPE_HPP__
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace cutt {
template <class T>
void reshape(
		T* const dst_ptr,
		const T* const src_ptr,
		const std::unordered_map<std::string, std::size_t>& mode,
		const std::vector<std::string>& original_order,
		const std::vector<std::string>& reshaped_order,
		cudaStream_t cuda_stream = 0
		);
} // namespace cutt
#endif
