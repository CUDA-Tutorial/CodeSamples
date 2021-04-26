add_executable(03_GridConfiguration
	src/main.cu
)

target_include_directories(03_GridConfiguration PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})