#include <cutt/reshape.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

template <class T>
void reshape_test(
		const std::vector<unsigned> mode_dim
		) {
	std::vector<std::pair<std::string, std::size_t>> mode;
	std::vector<std::string> original_mode;
	for (unsigned i = 0; i < mode_dim.size(); i++) {
		const auto name = "m" + std::to_string(i);
		mode.push_back(std::make_pair(name, mode_dim[i]));

		original_mode.push_back(name);
	}
	std::vector<std::string> reshaped_mode(mode.size());
	std::reverse_copy(original_mode.begin(), original_mode.end(), reshaped_mode.begin());

	std::size_t dim_product = 1;
	for (const auto d : mode_dim) {
		dim_product *= d;
	}

	T* d_original_ptr;
	cudaMalloc(&d_original_ptr, sizeof(T) * dim_product);
	const auto mom = std::sqrt(static_cast<T>(dim_product));
	T* h_original_ptr;
	cudaMallocHost(&h_original_ptr, sizeof(T) * dim_product);
	for (std::size_t i = 0; i < dim_product; i++) {
		h_original_ptr[i] = i / mom;
	}
	cudaMemcpy(d_original_ptr, h_original_ptr, sizeof(T) * dim_product, cudaMemcpyDefault);

	T* d_reshaped_ptr;
	cudaMalloc(&d_reshaped_ptr, sizeof(T) * dim_product);

	cutt::reshape(
			d_original_ptr,
			d_reshaped_ptr,
			mode,
			reshaped_mode
			);

	cudaFree(d_original_ptr);
	cudaFree(d_reshaped_ptr);
	cudaFree(h_original_ptr);
}

int main() {
	reshape_test<float>({1000, 1000, 1000});
}
