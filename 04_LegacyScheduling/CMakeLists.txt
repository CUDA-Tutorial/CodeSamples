set(ARCHES_SECURED ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_ARCHITECTURES "")

set(LEGACY_ARCH_FOUND FALSE)
foreach(ARCH ${ARCHES_SECURED})
	if (ARCH LESS_EQUAL 60)
		set(LEGACY_ARCH_FOUND TRUE)
		list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCH})
	endif()
endforeach()

if(${LEGACY_ARCH_FOUND})
	add_executable(04_LegacyScheduling
		src/main.cu
		../shared/include/test_scheduling.cuh
	)
	target_include_directories(04_LegacyScheduling PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
	message("Not building sample 04 - CMAKE_CUDA_ARCHITECTURES do not include legacy (<=60)")
	message("To define more than one architecture, separate them with semicolon (e.g., 60;70)")
endif()