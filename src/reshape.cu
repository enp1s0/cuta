#include <cuta/reshape.hpp>
#include "utils.hpp"
#include <cuda_fp16.h>
#include <unordered_map>
#include <algorithm>

namespace {
constexpr unsigned max_num_mode = 30;
__constant__ std::size_t c_reshaped_stride[max_num_mode];
__constant__ std::size_t c_reshaped_dim   [max_num_mode];

template <class T>
struct VecType {using type = void; static const unsigned len = 0;};
template <> struct VecType<double> {using type = double2; static const unsigned len = 2;};
template <> struct VecType<float > {using type = float4 ; static const unsigned len = 4;};
template <> struct VecType<half  > {using type = half2  ; static const unsigned len = 2;};

template <class T, class VecT, unsigned VecLen>
__global__ void reshpae_kernel (
		T* const dst_ptr,
		const T* const src_ptr,
		const unsigned num_mode,
		const std::size_t num_elements
		) {
	for (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid * VecLen < num_elements; tid += gridDim.x * blockDim.x) {
		if ((tid + 1) * VecLen < num_elements) {
			const auto v = reinterpret_cast<const VecT*>(src_ptr)[tid];

			for (unsigned j = 0; j < VecLen; j++) {
				auto dst_j = tid * VecLen + j;
				auto dst_i = decltype(tid)(0);
				for (unsigned i = 0; i < num_mode; i++) {
					dst_i += (dst_j % c_reshaped_dim[i]) * c_reshaped_stride[i];
					dst_j /= c_reshaped_dim[i];
				}
				dst_ptr[dst_i] = reinterpret_cast<const float*>(&v)[j];
			}
		} else {
			for (unsigned j = 0; j < num_elements - tid * VecLen; j++) {
				auto dst_j = tid * VecLen + j;
				const auto v = src_ptr[dst_j];
				auto dst_i = decltype(tid)(0);
				for (unsigned i = 0; i < num_mode; i++) {
					dst_i += (dst_j % c_reshaped_dim[i]) * c_reshaped_stride[i];
					dst_j /= c_reshaped_dim[i];
				}
				dst_ptr[dst_i] = v;
			}
		}
	}
}
} // noname namespace

template <class T>
void cuta::reshape(
		T *const dst_ptr,
		const T* const src_ptr,
		const cuta::mode_t& mode,
		const std::vector<std::string>& reshaped_order,
		cudaStream_t cuda_stream) {

	const auto num_mode = mode.size();

	// Validations
	if (num_mode > max_num_mode) {
		throw std::runtime_error("The maximum number of modes is " + std::to_string(max_num_mode) + ". Given " + std::to_string(num_mode) + ".");
	}

	if (num_mode != reshaped_order.size()) {
		throw std::runtime_error("The size of reshaped mode order list is different from mode list.");
	}

	// Check integration
	std::vector<std::string> reshaped_mode_name = reshaped_order;
	std::sort(reshaped_mode_name.begin(), reshaped_mode_name.end());
	std::vector<std::string> mode_name;
	for (const auto& m : mode) {
		if (!std::binary_search(reshaped_mode_name.begin(), reshaped_mode_name.end(), m.first)) {
			throw std::runtime_error("Reshaped mode list does not contain \"" + m.first + "\", which is in the source mode list");
		}
		mode_name.push_back(m.first);
	}
	std::sort(mode_name.begin(), mode_name.end());
	for (const auto& m : reshaped_order) {
		if (!std::binary_search(mode_name.begin(), mode_name.end(), m)) {
			throw std::runtime_error("Unknown mode \"" + m + "\" in reshaped mode list which is not included in the source mode list");
		}
	}

	// Calculate strides
	std::unordered_map<std::string, std::size_t> stride;
	std::size_t dim_product = 1;
	for (const auto& m : mode) {
		stride.insert(std::make_pair(m.first, dim_product));
		dim_product *= m.second;
	}

	std::vector<std::size_t> reshaped_stride(num_mode);
	std::vector<std::size_t> reshaped_dim   (num_mode);

	for (unsigned i = 0; i < num_mode; i++) {
		reshaped_stride[i] = stride[reshaped_order[i]];
		for (const auto& m : mode) {
			if (m.first == reshaped_order[i]) {
				reshaped_dim   [i] = m.second;
			}
		}
	}

	CUTA_CHECK_ERROR(cudaMemcpyToSymbolAsync(c_reshaped_stride, reshaped_stride.data(), sizeof(std::size_t) * num_mode, 0, cudaMemcpyHostToDevice, cuda_stream));
	CUTA_CHECK_ERROR(cudaMemcpyToSymbolAsync(c_reshaped_dim   , reshaped_dim   .data(), sizeof(std::size_t) * num_mode, 0, cudaMemcpyHostToDevice, cuda_stream));

	const unsigned block_size = 512;

	const auto kernel = reshpae_kernel<T, typename VecType<T>::type, VecType<T>::len>;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const unsigned num_sm = prop.multiProcessorCount;

	int grid_size;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, kernel, block_size, 0);
	kernel<<<grid_size * num_sm, block_size, 0, cuda_stream>>>(
			dst_ptr,
			src_ptr,
			num_mode,
			dim_product
			);
}

#define CUTA_RESHAPE_INSTANCE(type) \
template void cuta::reshape<type>(type* const, const type* const, const std::vector<std::pair<std::string, std::size_t>>&, const std::vector<std::string>&, cudaStream_t);
CUTA_RESHAPE_INSTANCE(double);
CUTA_RESHAPE_INSTANCE(float );
CUTA_RESHAPE_INSTANCE(half  );
