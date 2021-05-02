add_executable(17_CooperativeGroups
	../../shared/include/utility.h
	src/main.cu
)

target_include_directories(17_CooperativeGroups PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})