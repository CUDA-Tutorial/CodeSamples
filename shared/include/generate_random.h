#include <random>

void prepareRandomNumbersCPUGPU(unsigned int N, std::vector<float>& vals, float** dValsPtr)
{
	constexpr float target = 42.f;
	// Print expected value, because reference may be off due to floating point (im-)precision
	std::cout << "\nExpected value: " << target * N << "\n" << std::endl;

	// Generate a few random inputs to accumulate
	std::default_random_engine eng(0xcaffe);
	std::normal_distribution<float> dist(target);
	vals.resize(N);
	std::for_each(vals.begin(), vals.end(), [&dist, &eng](float& f) { f = dist(eng); });

	// Allocate some global GPU memory to write the inputs to
	cudaMalloc((void**)dValsPtr, sizeof(float) * N);
	// Expliclity copy the inputs from the CPU to the GPU
	cudaMemcpy(*dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
}