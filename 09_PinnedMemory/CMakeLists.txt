add_executable(09_PinnedMemory
	src/main.cu
)

target_include_directories(09_PinnedMemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})