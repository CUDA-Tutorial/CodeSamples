add_executable(16_CooperativeGroups
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(16_CooperativeGroups PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})