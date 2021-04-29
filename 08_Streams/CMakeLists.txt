add_executable(08_Streams
	src/main.cu
)

target_include_directories(08_Streams PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})