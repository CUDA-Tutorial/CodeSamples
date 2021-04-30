add_executable(11_ManagedMemory
	src/main.cu
)

target_include_directories(11_ManagedMemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})