add_executable(10_Streams
	src/main.cu
)

target_include_directories(10_Streams PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})