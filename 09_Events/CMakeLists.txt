add_executable(09_Events
	src/main.cu
)

target_include_directories(09_Events PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})