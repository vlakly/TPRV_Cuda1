#include <iostream>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <thread>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUtimer.h"
#include "MATRIX.h"

using namespace std;

#define TESTCOUNT 10
#define threads_in_blockSIZE 32
#define SIZE 512

void getInfoCUDADevice(cudaDeviceProp& prop, int id) {
	printf("CUDA device %i name  - %s\n", id, prop.name);
	printf("CUDA device %i Warp size in threads  - %i\n", id, prop.warpSize);
	printf("CUDA device %i Maximum number of threads per threads_in_block  - %i\n", id, prop.maxThreadsPerthreads_in_block);
	printf("CUDA device %i multiprocessors count  - %i\n", id, prop.multiProcessorCount);
	printf("CUDA device %i Maximum size of each dimension of a threads_in_block  - %i %i %i\n", id, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("CUDA device %i Maximum size of each dimension of a grid  - %i %i %i\n", id, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

__global__ void multiplyElements(uint32_t* mA, uint32_t* mB, uint32_t* res, int size) {
	int bx = threads_in_blockIdx.x;
	int by = threads_in_blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = bx * threads_in_blockSIZE + tx;
	int y = by * threads_in_blockSIZE + ty;

	for (int i = 0; i < size; i++) {
		res[y + x * size] += mA[x * size + i] * mB[y + size * i];
	}
}

uint32_t mat_a[SIZE * SIZE];
uint32_t mat_b[SIZE * SIZE];
uint32_t mat_SCALAR[SIZE * SIZE];
uint32_t mat_CUDA[SIZE * SIZE];

int main() {
	int count;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&count);
	cudaGetDeviceProperties(&prop, count - 1);
	//getInfoCUDADevice(prop, count - 1);

	GpuTimer timer;

	cout << "Matrix size: " << SIZE << "x" << SIZE << "\n";
	double avgTime = 0;
	vector<double> timeResult;

	MAT_fill_random(mat_a, SIZE);
	MAT_fill_random(mat_b, SIZE);
	MAT_fill_empty(mat_SCALAR, SIZE);
	MAT_fill_empty(mat_CUDA, SIZE);

	//cout << "Matrix multiplication test (no CUDA)\n";
	//for (int test = 0; test < TESTCOUNT; test++) {
	//	timer.Start();
	//	MAT_scalar_multiply(mat_SCALAR, mat_a, mat_b, SIZE);
	//	timer.Stop();
	//	avgTime += timer.ElapsedSeconds();
	//	cout << "Attempt " << test + 1 << ", time: " << timer.ElapsedSeconds() << " seconds\n";
	//}
	//cout << "Average time is " << avgTime / TESTCOUNT << " seconds\n\n";
	//timeResult.push_back(avgTime / TESTCOUNT);
	//avgTime = 0;

	int elemSize = SIZE * SIZE * sizeof(uint32_t);
	uint32_t *p_A(NULL), *p_B(NULL), *p_C(NULL);
	cudaMalloc((void**)&p_A, elemSize);
	cudaMalloc((void**)&p_B, elemSize);
	cudaMalloc((void**)&p_C, elemSize);

	cout << "Matrix multiplication test (with CUDA)\n";
	for (int test = 0; test < TESTCOUNT; test++) {

		cudaMemcpy(p_A, mat_a, elemSize, cudaMemcpyHostToDevice);
		cudaMemcpy(p_B, mat_b, elemSize, cudaMemcpyHostToDevice);
		cudaMemcpy(p_C, mat_CUDA, elemSize, cudaMemcpyHostToDevice);

		const dim3 threads_in_block(threads_in_blockSIZE, threads_in_blockSIZE);
		const dim3 blocks_in_grid(SIZE / threads_in_block.x, SIZE / threads_in_block.y);

		timer.Start();
		multiplyElements <<<blocks_in_grid, threads_in_block>>> (p_A, p_B, p_C, SIZE);
		timer.Stop();

		cudaMemcpy(mat_CUDA, p_C, elemSize, cudaMemcpyDeviceToHost);

		avgTime += timer.ElapsedSeconds();
		cout << "Attempt " << test + 1 << ", time: " << timer.ElapsedSeconds() << " seconds\n";
	}
	cout << "Average time is " << avgTime / TESTCOUNT << " seconds\n\n";
	timeResult.push_back(avgTime / TESTCOUNT);
	avgTime = 0;

	//cout << "MA:\n";
	//MAT_print(mat_a, SIZE);
	//cout << "\n";

	//cout << "MB:\n";
	//MAT_print(mat_b, SIZE);
	//cout << "\n";

	//cout << "MS:\n";
	//MAT_print(mat_SCALAR, SIZE);
	//cout << "\n";

	//cout << "MC:\n";
	//MAT_print(mat_CUDA, SIZE);
	//cout << "\n";

	//bool are_equal = MAT_check_equality(mat_SCALAR, mat_CUDA, SIZE);
	//if (are_equal) { cout << "Matrices are equal\n"; }
	//else { cout << "Matrices are NOT equal\n"; }

	//cout << "Results table\n";
	//for (int i = 0; i < timeResult.size() / 2; i++) {
	//	cout << "Multiplication #" << i + 1 << ": " << timeResult.at(2 * i) << " | " << timeResult.at(2 * i + 1) << " | " << timeResult.at(2 * i) / timeResult.at(2 * i + 1) << "\n";
	//}

	cudaFree(p_A);
	cudaFree(p_B);
	cudaFree(p_C);
}