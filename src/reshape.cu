#include <cutt/reshape.hpp>
#include "utils.hpp"
#include <cuda_fp16.h>
#include <unordered_map>

namespace {
constexpr unsigned max_num_mode = 30;
__constant__ std::size_t c_reshaped_stride[max_num_mode];
__constant__ std::size_t c_reshaped_dim   [max_num_mode];

template <class T>
__global__ void reshpae_kernel (
		T* const dst_ptr,
		const T* const src_ptr,
		const unsigned num_mode,
		const std::size_t num_elements
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_elements) {
		return;
	}

	const auto v = src_ptr[tid];

	auto dst_j = tid;
	auto dst_i = decltype(tid)(0);
	for (unsigned i = 0; i < num_mode; i++) {
		dst_i += (dst_j % c_reshaped_dim[i]) * c_reshaped_stride[i];
		dst_j /= c_reshaped_dim[i];
	}
	dst_ptr[dst_i] = v;
}
} // noname namespace

template <class T>
void cutt::reshape(
		T *const dst_ptr,
		const T* const src_ptr,
		const std::vector<std::pair<std::string, std::size_t>>& mode,
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

	CUTT_CHECK_ERROR(cudaMemcpyToSymbolAsync(c_reshaped_stride, reshaped_stride.data(), sizeof(std::size_t) * num_mode, 0, cudaMemcpyHostToDevice, cuda_stream));
	CUTT_CHECK_ERROR(cudaMemcpyToSymbolAsync(c_reshaped_dim   , reshaped_dim   .data(), sizeof(std::size_t) * num_mode, 0, cudaMemcpyHostToDevice, cuda_stream));

	const unsigned block_size = 256;
	const auto grid_size = (dim_product + block_size - 1) / block_size;

	reshpae_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
			dst_ptr,
			src_ptr,
			num_mode,
			dim_product
			);
}

#define CUTT_RESHAPE_INSTANCE(type) \
template void cutt::reshape<type>(type* const, const type* const, const std::vector<std::pair<std::string, std::size_t>>&, const std::vector<std::string>&, cudaStream_t);
CUTT_RESHAPE_INSTANCE(double);
CUTT_RESHAPE_INSTANCE(float );
CUTT_RESHAPE_INSTANCE(half  );
