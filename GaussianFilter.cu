
// Vladimir Klyzhko
// 
// CUDA with OpenCV
//
// Processing images with gaussian filter on CPU and GPU
// Source images are stored in ../images/source
// Result images are stored in ../images/result

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUtimer.h"

using namespace std;
using namespace cv;

string path_source = "images/source/";
string path_result = "images/result/";
vector<string> images{ /*"mountain.png", "city.png", "desert.png",*/ "8k.png" };

const int test_count = 1;

const int g_size = 3;
const float CPU_Gauss[g_size * g_size] = {
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
	2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};
__constant__ float GPU_Gauss[g_size * g_size] = {
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
	2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

void CPU_Processing(Mat& input, Mat& output) {
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			float blur_pixel = 0.0;
			for (int x = -g_size / 2; x <= g_size / 2; x++) {
				for (int y = -g_size / 2; y <= g_size / 2; y++) {
					int current_row = i + x;
					int current_col = j + y;

					if (current_row >= 0 && current_row < input.rows && current_col >= 0 && current_col < input.cols) {
						float filter_value = CPU_Gauss[(y + g_size / 2) * g_size + (x + g_size / 2)];
						blur_pixel += input.at<uchar>(i, j) * filter_value;
					}
				}
			}
			output.at<uchar>(i, j) = static_cast <uchar>(blur_pixel);
		}
	}
};
__global__ void GPU_Processing(uchar* input, uchar* output, int rows, int cols) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < rows && j < cols) {
		float blur_pixel = 0.0;
		for (int x = -g_size / 2; x <= g_size / 2; x++) {
			for (int y = -g_size / 2; y <= g_size / 2; y++) {
				int current_row = i + x;
				int current_col = j + y;

				if (current_row >= 0 && current_row < rows && current_col >= 0 && current_col < cols) {
					float filter_value = GPU_Gauss[(y + g_size / 2) * g_size + (x + g_size / 2)];
					blur_pixel += input[current_row * cols + current_col] * filter_value;
				}
			}
		}
		output[i * cols + j] = static_cast<uchar>(blur_pixel);
	}
};

int main() {
	cout << "Test count: " << test_count << "\n\n";

	// mute opencv log
	utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

	GpuTimer timer;
	double t_cpu_average = 0;
	double t_gpu_average = 0;

	auto iter {images.begin()};
	while (iter != images.end()) {
		Mat input = imread(path_source + *iter);
		if (input.empty()) {
			cout << "Image error\n";
			return 0;
		}

		int rows = input.rows;
		int cols = input.cols;
		int image_size = input.size().area();

		cout << "Current image: " << *iter << " (" << input.cols << "x" << input.rows << ")\n";

		uchar* mat_input, * mat_output;

		Mat gray_input;
		cvtColor(input, gray_input, COLOR_BGR2GRAY);

		cudaMalloc((void**)&mat_input, image_size);
		cudaMalloc((void**)&mat_output, image_size);

		int block_size = 32;
		dim3 threads_in_block(block_size, block_size);
		dim3 blocks_in_grid(cols / threads_in_block.x, rows / threads_in_block.y);

		// CPU test
		for (int test = 0; test < test_count; test++) {
			Mat cpu_output = gray_input.clone();
			timer.Start();
			CPU_Processing(gray_input, cpu_output);
			timer.Stop();
			imwrite(path_result + "CPU_" + *iter, cpu_output);
			double t_cpu = timer.ElapsedSeconds();
			t_cpu_average += t_cpu;
		}
		t_cpu_average /= test_count;
		cout << "Average CPU time: " << t_cpu_average << "\n";

		// GPU test
		for (int test = 0; test < test_count; test++) {
			Mat gpu_output = gray_input.clone();
			cudaMemcpy(mat_input, gray_input.data, image_size, cudaMemcpyHostToDevice);
			timer.Start();
			GPU_Processing <<<blocks_in_grid, threads_in_block>>> (mat_input, mat_output, rows, cols);
			timer.Stop();
			cudaMemcpy(gpu_output.data, mat_output, image_size, cudaMemcpyDeviceToHost);
			imwrite(path_result + "GPU_" + *iter, gpu_output);
			double t_gpu = timer.ElapsedSeconds();
			t_gpu_average += t_gpu;
		}
		t_gpu_average /= test_count;
		cout << "Average GPU time: " << t_gpu_average << "\n";
		cout << "CPU time / GPU time = " << t_cpu_average / t_gpu_average << "\n";

		cout << "\n";

		++iter;

		cudaFree(mat_input);
		cudaFree(mat_output);
	}

	return 0;
}
