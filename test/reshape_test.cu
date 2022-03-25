#include <cuta/reshape.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

constexpr unsigned num_sampling = 100;

template <class T>
void reshape_test(
		const std::vector<unsigned> mode_dim
		) {
	cuta::mode_t original_mode;
	std::vector<std::string> original_mode_order;
	for (unsigned i = 0; i < mode_dim.size(); i++) {
		const auto name = "m" + std::to_string(i);
		cuta::utils::insert_mode(original_mode, name, mode_dim[i]);

		original_mode_order.push_back(name);
	}
	std::vector<std::string> reshaped_mode_order(original_mode.size());
	std::reverse_copy(original_mode_order.begin(), original_mode_order.end(), reshaped_mode_order.begin());

	std::size_t dim_product = cuta::utils::get_num_elements(original_mode);

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

	cuta::reshape(
			d_reshaped_ptr,
			d_original_ptr,
			original_mode,
			reshaped_mode_order
			);
	cuta::utils::print_mode(original_mode, "input");

	// check via sampling
	cuta::mode_t reshaped_mode;
	for (const auto& o : reshaped_mode_order) {
		for (const auto& m : original_mode) {
			if (m.first == o) {
				cuta::utils::insert_mode(reshaped_mode, o, m.second);
			}
		}
	}

	std::mt19937 mt(0);
	unsigned num_correct = 0;
	for (unsigned i = 0; i < num_sampling; i++) {
		std::unordered_map<std::string, std::size_t> pos;
		for (const auto& m : original_mode) {
			std::uniform_int_distribution<std::size_t> dist(0, m.second - 1);
			pos.insert(std::make_pair(m.first, dist(mt)));
		}

		const auto h_v = h_original_ptr[cuta::utils::get_index(original_mode, pos)];

		T d_v;
		cudaMemcpy(&d_v, d_reshaped_ptr + cuta::utils::get_index(reshaped_mode, pos), sizeof(T), cudaMemcpyDefault);

		if (d_v == h_v) {
			num_correct++;
		} else {
			std::printf("%e != %e\n", d_v, h_v);
		}
	}

	cudaFree(d_original_ptr);
	cudaFree(d_reshaped_ptr);
	cudaFree(h_original_ptr);

	// Output result
	std::printf("shape(");
	for (unsigned i = 0; i < mode_dim.size() - 1; i++) {
		std::printf("%u,", mode_dim[i]);
	}
	std::printf("%u): ", mode_dim[mode_dim.size() - 1]);
	std::printf("Test %5u / %5u Passed\n", num_correct, num_sampling);
}

int main() {
	reshape_test<float>({1000, 1000, 1000});
	reshape_test<float>({10, 10, 10});
	reshape_test<float>({2, 1, 1});
}
