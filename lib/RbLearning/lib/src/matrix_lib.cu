#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

__global__ void cuda_dot(double *m1, double *m2, int size_y, int size_v, int size_x, double *new_m) {
 	m2[(threadIdx.y * size_v) + threadIdx.x];
	double tmp = 0;
	for (int x = 0; x < size_x; ++x) {
		tmp += m1[(blockIdx.x * size_x) + x] * m2[(x * size_v) + threadIdx.x];
	}
	new_m[(blockIdx.x * size_v) + threadIdx.x] = tmp;
}

__global__ void cuda_mult(double *m1, double *m2, double *new_m) {
	new_m[blockIdx.x * blockDim.x + threadIdx.x] = m1[blockIdx.x * blockDim.x + threadIdx.x] * m2[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void cuda_sub(double *m1, double *m2, double *new_m) {
	new_m[blockIdx.x * blockDim.x + threadIdx.x] = m1[blockIdx.x * blockDim.x + threadIdx.x] - m2[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void cuda_add(double *m1, double *m2, double *new_m) {
	new_m[blockIdx.x * blockDim.x + threadIdx.x] = m1[blockIdx.x * blockDim.x + threadIdx.x] + m2[blockIdx.x * blockDim.x + threadIdx.x];
}

extern "C" {

	int find_nb_blocks(int size, int max_th) {
		int nb_block = 1;

		while (((float)size / (float)nb_block) > (float)max_th || (size % nb_block) != 0) {
			nb_block += 1;
		}

		return (nb_block);
	}

	double *dot(double *m1, double *m2, int size_y, int size_v, int size_x) {
		double *new_m 		= (double *)malloc((size_v * size_y) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 	= NULL;
		double *cuda_m2 	= NULL;

		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error != 0) {
			fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
			exit(1);
		}

		cudaMalloc((void**)&cuda_new_m, (size_y * size_v) * sizeof(double));
		memset(new_m, 0, (size_y * size_v) * sizeof(double));
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

	double *mult(double *m1, double *m2, int size) {
		double *new_m 		= (double *)malloc((size) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 	= NULL;
		double *cuda_m2 	= NULL;

		dim3 numBlocks(find_nb_blocks(size, 1024));
		dim3 threadsPerBlock(size / numBlocks.x);

		cudaMalloc((void**)&cuda_new_m, size * sizeof(double));
		memset(new_m, 0, size * sizeof(double));
		cudaMalloc((void**)&cuda_m1, size * sizeof(double));
		cudaMalloc((void**)&cuda_m2, size * sizeof(double));

		cudaMemcpy(cuda_m1, m1, (size) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_m2, m2, (size) * sizeof(double), cudaMemcpyHostToDevice);

		cuda_mult<<<numBlocks, threadsPerBlock>>>(cuda_m1, cuda_m2, cuda_new_m);
		cudaMemcpy(new_m, cuda_new_m, size * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(cuda_new_m);
		cudaFree(cuda_m1);
		cudaFree(cuda_m2);
		return(new_m);
	}

	double *transpose(double  *m, int size_y, int size_x)
	{
		double *new_m = (double *)malloc(sizeof(double) * (size_x * size_y));
		memset(new_m, 0, size_x * size_y * sizeof(double));

		for (int y = 0; y < size_y; ++y) {
			for (int x = 0; x < size_x; ++x) {
				new_m[x * size_y + y] = m[y * size_x + x];
			}
		}
		return (new_m);
	}

	double *substract(double *m1, double *m2, int size) {
		double *new_m 		= (double *)malloc((size) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 	= NULL;
		double *cuda_m2 	= NULL;

		dim3 numBlocks(find_nb_blocks(size, 1024));
		dim3 threadsPerBlock(size / numBlocks.x);

		cudaMalloc((void**)&cuda_new_m, size * sizeof(double));
		memset(new_m, 0, size * sizeof(double));
		cudaMalloc((void**)&cuda_m1, size * sizeof(double));
		cudaMalloc((void**)&cuda_m2, size * sizeof(double));

		cudaMemcpy(cuda_m1, m1, (size) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_m2, m2, (size) * sizeof(double), cudaMemcpyHostToDevice);
		
		cuda_sub<<<numBlocks, threadsPerBlock>>>(cuda_m1, cuda_m2, cuda_new_m);
		cudaMemcpy(new_m, cuda_new_m, size * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(cuda_new_m);
		cudaFree(cuda_m1);
		cudaFree(cuda_m2);
		return(new_m);
	}

	double *add(double *m1, double *m2, int size) {
		double *new_m 		= (double *)malloc((size) * sizeof(double));
		double *cuda_new_m 	= NULL;
		double *cuda_m1 	= NULL;
		double *cuda_m2 	= NULL;

    	dim3 numBlocks(find_nb_blocks(size, 1024));
		dim3 threadsPerBlock(size / numBlocks.x);

		cudaMalloc((void**)&cuda_new_m, size * sizeof(double));
		memset(new_m, 0, size * sizeof(double));
		cudaMalloc((void**)&cuda_m1, size * sizeof(double));
		cudaMalloc((void**)&cuda_m2, size * sizeof(double));

		cudaMemcpy(cuda_m1, m1, (size) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_m2, m2, (size) * sizeof(double), cudaMemcpyHostToDevice);

		cuda_add<<<numBlocks, threadsPerBlock>>>(cuda_m1, cuda_m2, cuda_new_m);
		cudaMemcpy(new_m, cuda_new_m, size * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(cuda_new_m);
		cudaFree(cuda_m1);
		cudaFree(cuda_m2);
		return(new_m);
	}
}
