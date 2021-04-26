add_executable(01_HelloGPU
	src/main.cu
)

target_include_directories(01_HelloGPU PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})