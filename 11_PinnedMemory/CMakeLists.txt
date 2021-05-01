add_executable(11_PinnedMemory
	src/main.cu
)

target_include_directories(11_PinnedMemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})