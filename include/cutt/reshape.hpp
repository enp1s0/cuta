#ifndef __CUTT_RESHAPE_HPP__
#define __CUTT_RESHAPE_HPP__
#include <cstdint>
#include <string>
#include <vector>

namespace cutt {
template <class T>
void reshape(
		T* const dst_ptr,
		const T* const src_ptr,
		const std::vector<std::pair<std::string, std::size_t>>& mode,
		const std::vector<std::string>& reshaped_order,
		cudaStream_t cuda_stream = 0
		);
} // namespace cutt
#endif
