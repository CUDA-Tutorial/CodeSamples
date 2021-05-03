add_executable(07_Synchronization
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(07_Synchronization PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})