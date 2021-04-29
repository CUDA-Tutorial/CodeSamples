add_executable(10_ErrorHandling
	src/main.cu
)

target_include_directories(10_ErrorHandling PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})