set(ARCHES_SECURED ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_ARCHITECTURES "")

set(ITS_ARCH_FOUND FALSE)
foreach(ARCH ${ARCHES_SECURED})
	if (ARCH GREATER_EQUAL 70)
		set(ITS_ARCH_FOUND TRUE)
		list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCH})
	endif()
endforeach()

if(${ITS_ARCH_FOUND})
	add_executable(05_IndependentThreadScheduling
		src/main.cu
		../shared/include/test_scheduling.cuh
	)
	target_include_directories(05_IndependentThreadScheduling PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
	message("Not building sample 05 - CMAKE_CUDA_ARCHITECTURES do not include ITS support (>=70)")
	message("To define more than one architecture, separate them with semicolon (e.g., 60;70)")
endif()