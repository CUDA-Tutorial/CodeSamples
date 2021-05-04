add_executable(13_MemoryFences
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(13_MemoryFences PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})