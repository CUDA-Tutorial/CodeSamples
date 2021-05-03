add_executable(15_GraphAPI
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(15_GraphAPI PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})