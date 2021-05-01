add_executable(13_ManagedMemory
	src/main.cu
)

target_include_directories(13_ManagedMemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})