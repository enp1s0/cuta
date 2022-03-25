cmake_minimum_required(VERSION 3.18)
project(cuta LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TEST_CUTENSOR_UTILS OFF)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/cuta/*.hpp")

SET(lib_name cuta)
add_library(${lib_name} STATIC
	${SRCDIR}/reshape.cu
	${SRCDIR}/utils.hpp
	${HEADERS}
	)

target_include_directories(${lib_name} PUBLIC ${INCDIR})

target_link_libraries(${lib_name} PRIVATE
	cuda
	)

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER ${INCDIR}/cuta/${library}.hpp)
target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

##########################################################################
# Installing
##########################################################################
install(TARGETS cuta
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/cuta
	)

##########################################################################
# Tests
##########################################################################

# Directory
set(TESTSRCDIR test)

foreach(test reshape permutation)
	set(test_out ${test}.test)
	add_executable(${test_out} ${TESTSRCDIR}/${test}_test.cu ${HEADERS})
	target_include_directories(${test_out} PRIVATE ${INCDIR})
	target_link_libraries(${test_out} PRIVATE cuta)
endforeach()

if (${TEST_CUTENSOR_UTILS})
	foreach(test cutensor)
		set(test_out ${test}.test)
		add_executable(${test_out} ${TESTSRCDIR}/${test}_test.cu ${HEADERS})
		target_include_directories(${test_out} PRIVATE ${INCDIR})
		target_link_libraries(${test_out} PRIVATE cutensor)
	endforeach()
endif()
