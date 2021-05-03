add_executable(10_Events
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(10_Events PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})