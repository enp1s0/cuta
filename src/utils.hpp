#ifndef __CUTT_SRC_UTILS_HPP__
#define __CUTT_SRC_UTILS_HPP__
#include <stdexcept>
#include <sstream>

#define CUTT_CHECK_ERROR(status) cuta::detail::check_error(status, __FILE__, __LINE__, __func__)
#define CUTT_CHECK_ERROR_M(status, message) cuta::detail::check_error(status, __FILE__, __LINE__, __func__, (message))

namespace cuta {
namespace detail {
inline void check_error(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss << "[cuta error] ";
		ss << cudaGetErrorString(error);
		if(message.length() != 0){
			ss << " : " << message;
		}
		ss << " [" << filename << ":" << line << " in " << funcname << "]";
		throw std::runtime_error(ss.str());
	}
}
} // namespace detail
} // namespace cuta
#endif
