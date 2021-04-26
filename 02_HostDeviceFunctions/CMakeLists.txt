add_executable(02_HostDeviceFunctions
	src/main.cu
)

target_include_directories(02_HostDeviceFunctions PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})