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
	add_executable(16_TensorCores
		src/main.cu
		../shared/include/utility.h
	)
	target_include_directories(16_TensorCores PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	if (MSVC)
		target_link_libraries(16_TensorCores cublas.lib)
	else()
		target_link_libraries(16_TensorCores cublas)
	endif()
else()
	message("Not building sample 16 - CMAKE_CUDA_ARCHITECTURES do not include ITS support (>=70)")
	message("To define more than one architecture, separate them with semicolon (e.g., 60;70)")
endif()