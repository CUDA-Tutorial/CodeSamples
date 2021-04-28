add_executable(06_MemoryBasics
	src/main.cu
)

target_include_directories(06_MemoryBasics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})