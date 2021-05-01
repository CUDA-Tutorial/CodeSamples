add_executable(08_Reductions
	../../shared/include/generate_random.h
	src/main.cu
)

target_include_directories(08_Reductions PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})