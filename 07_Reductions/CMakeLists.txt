add_executable(07_Reductions
	src/main.cu
)

target_include_directories(07_Reductions PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})