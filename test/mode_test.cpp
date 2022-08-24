#include <iostream>
#include <cuta/utils.hpp>

int main() {
	cuta::mode_t mode_a = {
		{"a", 10},
		{"b", 10},
		{"c", 10},
		{"d", 10},
		{"e", 10},
	};
	cuta::mode_t mode_b = {
		{"f", 10},
		{"b", 10},
		{"g", 10},
		{"d", 10},
		{"h", 10},
	};

	cuta::utils::print_mode(mode_a, "A");
	cuta::utils::print_mode(mode_b, "B");

	cuta::utils::print_mode(cuta::utils::get_intersection_mode(mode_a, mode_b), "A U B");
	cuta::utils::print_mode(cuta::utils::get_difference_mode(mode_a, mode_b)  , "A \\ B");
}
