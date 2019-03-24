#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "matrix_lib.h"

__global__ void cuda_dot(const double *m1, const double *m2, int size_y, int size_v, int size_x, double *new_m) {
	double tmp = 0;
	for (int x = 0; x < size_x; ++x) {
		tmp += m1[(blockIdx.x * size_x) + x] * m2[(x * size_v) + threadIdx.x];
	}
	new_m[(blockIdx.x * size_v) + threadIdx.x] = tmp;
}

__global__ void cuda_mult(const double *m1, const double *m2, double *new_m) {
	new_m[blockIdx.x] = m1[blockIdx.x] * m2[blockIdx.x];
}

extern "C" {

	double *dot(const double *m1, const double *m2, int size_y, int size_v, int size_x) {
		double *new_m 		= (double *)malloc((size_v * size_y) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 		= NULL;
		double *cuda_m2 		= NULL;

		cudaMalloc((void**)&cuda_new_m, (size_y * size_v) * sizeof(double));
		cudaMalloc((void**)&cuda_m1, (size_x * size_y) * sizeof(double));
		cudaMalloc((void**)&cuda_m2, (size_x * size_v) * sizeof(double));

		cudaMemcpy(cuda_m1, m1, (size_x * size_y) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_m2, m2, (size_x * size_v) * sizeof(double), cudaMemcpyHostToDevice);

		cuda_dot<<<size_y, size_v>>>(cuda_m1, cuda_m2, size_y, size_v, size_x, cuda_new_m);
		cudaMemcpy(new_m, cuda_new_m, (size_y * size_v) * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(cuda_new_m);
		cudaFree(cuda_m1);
		cudaFree(cuda_m2);
		return(new_m);
	}

	double *mult(const double *m1, const double *m2, int size) {
		double *new_m 		= (double *)malloc((size) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 		= NULL;
		double *cuda_m2 		= NULL;

		cudaMalloc((void**)&cuda_new_m, size * sizeof(double));
		cudaMalloc((void**)&cuda_m1, size * sizeof(double));
		cudaMalloc((void**)&cuda_m2, size * sizeof(double));

		cudaMemcpy(cuda_m1, m1, (size) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_m2, m2, (size) * sizeof(double), cudaMemcpyHostToDevice);

		cuda_mult<<<size, 1>>>(cuda_m1, cuda_m2, cuda_new_m);
		cudaMemcpy(new_m, cuda_new_m, size * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(cuda_new_m);
		cudaFree(cuda_m1);
		cudaFree(cuda_m2);
		return(new_m);
	}

	double *transpose(const double  *m, int size_y, int size_x)
	{
		double *new_m = (double *)malloc(sizeof(double) * (size_x * size_y));

		for (int y = 0; y < size_y; ++y) {
			for (int x = 0; x < size_x; ++x) {
				new_m[x * size_y + y] = m[y * size_x + x];
			}
		}
		return (new_m);
	}

	double *subtract(const double *m1, const double *m2, int size_y, int size_x) {
		int size = size_x * size_y;
		double *new_m = (double *)malloc(sizeof(double) * size);

		for (int i = 0; i < size; ++i) {
			new_m[i] = m1[i] - m2[i];
		}
		return (new_m);
	}

	double *add(const double *m1, const double *m2, int size_y, int size_x) {
		int size = size_x * size_y;
		double *new_m = (double *)malloc(sizeof(double) * size);

		for (int i = 0; i < size; ++i) {
			new_m[i] = m1[i] + m2[i];
		}
		return (new_m);
	}
}
