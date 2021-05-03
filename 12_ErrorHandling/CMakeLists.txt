add_executable(12_ErrorHandling
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(12_ErrorHandling PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})