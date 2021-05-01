add_executable(14_ManagedMemory
	src/main.cu
)

target_include_directories(14_ManagedMemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})