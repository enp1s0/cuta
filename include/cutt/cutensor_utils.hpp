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

#define CUTT_DATA_TYPE_DEF(type_name, number_type, type_size) \
	template <> inline cudaDataType_t get_data_type<type_name>(){return CUDA_##number_type##_##type_size;}
template <class T>
inline cudaDataType_t get_data_type();
CUTT_DATA_TYPE_DEF(half, R, 16F);
CUTT_DATA_TYPE_DEF(half2, C, 16F);
CUTT_DATA_TYPE_DEF(float, R, 32F);
CUTT_DATA_TYPE_DEF(cuComplex, C, 32F);
CUTT_DATA_TYPE_DEF(double, R, 64F);
CUTT_DATA_TYPE_DEF(cuDoubleComplex, C, 64F);

template <class T>
cutensorComputeType_t get_compute_type();
template <> cutensorComputeType_t get_compute_type<double                       >() {return CUTENSOR_COMPUTE_64F;}
template <> cutensorComputeType_t get_compute_type<float                        >() {return CUTENSOR_COMPUTE_32F;}
template <> cutensorComputeType_t get_compute_type<half                         >() {return CUTENSOR_COMPUTE_16F;}
//template <> cutensorComputeType_t get_compute_type<nvcuda::wmma::precision::tf32>() {return CUTENSOR_COMPUTE_TF32;}
template <> cutensorComputeType_t get_compute_type<__nv_bfloat16                >() {return CUTENSOR_COMPUTE_16BF;}
template <> cutensorComputeType_t get_compute_type<uint32_t                     >() {return CUTENSOR_COMPUTE_32U;}
template <> cutensorComputeType_t get_compute_type<int32_t                      >() {return CUTENSOR_COMPUTE_32I;}
template <> cutensorComputeType_t get_compute_type<uint8_t                      >() {return CUTENSOR_COMPUTE_8U;}
template <> cutensorComputeType_t get_compute_type<int8_t                       >() {return CUTENSOR_COMPUTE_8I;}

inline int get_extent_in_int(
		const std::string str
		) {
	int res = 0;
	for (unsigned i = 0; i < str.length(); i++) {
		res = (res << 1) ^ static_cast<int>(str[i]);
	}
	return res;
}

inline std::vector<int> get_extent_list_in_int(
		const mode_t& mode
		) {
	std::vector<int> vec(mode.size());
	for (unsigned i = 0; i < mode.size(); i++) {
		vec[i] = get_extent_in_int(mode[i].first);
	}
	return vec;
}

template <class T>
inline cutensorTensorDescriptor_t get_descriptor(
		const cutensorHandle_t cutensor_handle,
		const mode_t& mode
		) {
	std::vector<int64_t> extent(mode.size());
	for (unsigned i = 0; i < mode.size(); i++) {
		extent[i] = mode[i].second;
	}
	cutensorTensorDescriptor_t desc;
	cutensorInitTensorDescriptor(
				&cutensor_handle,
				&desc,
				mode.size(),
				extent.data(),
				nullptr,
				get_data_type<T>(),
				CUTENSOR_OP_IDENTITY
				);
	return desc;
}

} // namespace cutensor
} // namespace cutt
#endif
