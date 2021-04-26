add_executable(05_MemoryBasics
	src/main.cu
)

target_include_directories(05_MemoryBasics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})