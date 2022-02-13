#ifndef __CUTT_RESHAPE_HPP__
#define __CUTT_RESHAPE_HPP__
#include <cstdint>
#include <vector>
#include <map>

namespace cutt {
template <class T>
void reshape(
		T* const dst_ptr,
		const T* src_ptr,
		const std::map<std::string, unsigned>& mode,
		const std::vector<std::string> original_order,
		const std::vector<std::string> reshaped_order
		);
} // namespace cutt
#endif
