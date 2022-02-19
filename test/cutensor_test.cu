#include <cutt/cutensor_utils.hpp>

using compute_t = float;

int main() {
	cutt::mode_t mode_A, mode_B, mode_C;

	cutt::utils::insert_mode(mode_A, "a", 10);
	cutt::utils::insert_mode(mode_A, "b", 10);
	cutt::utils::insert_mode(mode_A, "c", 10);

	cutt::utils::insert_mode(mode_B, "a", 10);
	cutt::utils::insert_mode(mode_B, "b", 10);
	cutt::utils::insert_mode(mode_B, "d", 10);

	cutt::utils::insert_mode(mode_C, "c", 10);
	cutt::utils::insert_mode(mode_C, "d", 10);

	compute_t *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, sizeof(compute_t) * cutt::utils::get_num_elements(mode_A));
	cudaMalloc(&d_B, sizeof(compute_t) * cutt::utils::get_num_elements(mode_B));
	cudaMalloc(&d_C, sizeof(compute_t) * cutt::utils::get_num_elements(mode_C));

	cutensorHandle_t cutensor_handle;
	CUTT_CHECK_ERROR(cutensorInit(&cutensor_handle));

	const auto desc_A = cutt::cutensor::get_descriptor<compute_t>(cutensor_handle, mode_A);
	const auto desc_B = cutt::cutensor::get_descriptor<compute_t>(cutensor_handle, mode_B);
	const auto desc_C = cutt::cutensor::get_descriptor<compute_t>(cutensor_handle, mode_C);

	uint32_t alignment_requirement_A;
	CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, d_A, &desc_A, &alignment_requirement_A));
	uint32_t alignment_requirement_B;
	CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, d_B, &desc_B, &alignment_requirement_B));
	uint32_t alignment_requirement_C;
	CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, d_C, &desc_C, &alignment_requirement_C));

	cutensorContractionDescriptor_t desc_contraction;
	CUTT_CHECK_ERROR(cutensorInitContractionDescriptor(&cutensor_handle, &desc_contraction,
				&desc_A, cutt::cutensor::get_extent_list_in_int(mode_A).data(), alignment_requirement_A,
				&desc_B, cutt::cutensor::get_extent_list_in_int(mode_B).data(), alignment_requirement_B,
				&desc_C, cutt::cutensor::get_extent_list_in_int(mode_C).data(), alignment_requirement_C,
				&desc_C, cutt::cutensor::get_extent_list_in_int(mode_C).data(), alignment_requirement_C,
				cutt::cutensor::get_compute_type<compute_t>()));

	cutensorContractionFind_t find;
	CUTT_CHECK_ERROR(cutensorInitContractionFind(&cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

	std::size_t work_size = 0;
	CUTT_CHECK_ERROR(cutensorContractionGetWorkspace(&cutensor_handle, &desc_contraction, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &work_size));

	void* work_mem_ptr;
	cudaMalloc(&work_mem_ptr, work_size);

	cutensorContractionPlan_t plan;
	CUTT_CHECK_ERROR(cutensorInitContractionPlan(&cutensor_handle, &plan, &desc_contraction, &find, work_size));

	const compute_t alpha = 1.0f;
	const compute_t beta = 0.0f;
	CUTT_CHECK_ERROR(cutensorContraction(&cutensor_handle,
				&plan,
				reinterpret_cast<const void*>(&alpha), d_A, d_B,
				reinterpret_cast<const void*>(&beta), d_C, d_C,
				work_mem_ptr, work_size, 0
				));

	cudaFree(work_mem_ptr);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
