add_executable(09_Streams
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(09_Streams PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})