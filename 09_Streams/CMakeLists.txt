add_executable(09_Streams
	src/main.cu
)

target_include_directories(09_Streams PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})