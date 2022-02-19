#ifndef __CUTT_CUTENSOR_HELPERS_HPP__
#define __CUTT_CUTENSOR_HELPERS_HPP__
#include <cuda_fp16.h>
#include <mma.h>
#include <cuComplex.h>
#include <cutensor.h>
#include <vector>
#include <sstream>
#include "utils.hpp"

namespace cutt {
namespace cutensor {
inline void check_error(cutensorStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUTENSOR_STATUS_SUCCESS){
		std::string error_string = cutensorGetErrorString(error);
		std::stringstream ss;
		ss << error_string;
		if(message.length() != 0){
			ss<<" : "<<message;
		}
		ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
#ifndef CUTT_CHECK_ERROR
#define CUTT_CHECK_ERROR(status) cutt::cutensor::check_error(status, __FILE__, __LINE__, __func__)
#endif
#ifndef CUTT_CHECK_ERROR_M
#define CUTT_CHECK_ERROR_M(status, message) cutt::cutensor::check_error(status, __FILE__, __LINE__, __func__, message)
#endif



} // namespace cutensor
} // namespace cutt
#endif
