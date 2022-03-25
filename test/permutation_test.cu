#include <iostream>
#include <cuta/reshape.hpp>


int main() {
	cuta::mode_t mode;
	cuta::utils::insert_mode(mode, "a", 10);
	cuta::utils::insert_mode(mode, "b", 10);
	cuta::utils::insert_mode(mode, "c", 10);
	cuta::utils::insert_mode(mode, "d", 10);
	cuta::utils::insert_mode(mode, "e", 10);

	std::vector<std::string> perm_order = {"b", "a", "e", "d", "c"};

	{
		const auto permutation = cuta::utils::get_permutation<>(mode, perm_order);

		std::printf("permutation = ");
		for (const auto& v : permutation) {
			std::printf("%u ", v);
		}
		std::printf("\n");
		std::printf("expected    = 1 0 4 3 2\n");
	}
	{
		const auto permutated_mode = cuta::utils::get_permutated_mode(mode, perm_order);
		cuta::utils::print_mode(permutated_mode);
	}
}
