add_executable(00_MyGPU
	src/main.cu
)

target_include_directories(00_MyGPU PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})