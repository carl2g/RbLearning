#include <stdlib.h>
#include <stdio.h>

float *dot(const float const  *m1, const float const  *m2, int size_y, int size_v, int size_x) {
	float *new_m = malloc(sizeof(float) * (size_v * size_y));
	float tmp = 0;

	for (int y = 0; y < size_y; ++y) {
		for (int v = 0; v < size_v; ++v) {
			tmp = 0;
			for (int x = 0; x < size_x; ++x) {
				tmp += m1[(y * size_x) + x] * m2[(x * size_v) + v];
			}
			new_m[(y * size_v) + v] = tmp;
		}
	}
	return (new_m);
}

float *transpose(const float const  *m, int size_y, int size_x)
{
	float *new_m = malloc(sizeof(float) * (size_x * size_y));

	for (int y = 0; y < size_y; ++y) {
		for (int x = 0; x < size_x; ++x) {
			new_m[x * size_y + y] = m[y * size_x + x];
		}
	}
	return (new_m);
}

float *mult(const float const *m1, const float const *m2, int size_y, int size_x) {
	int size = size_x * size_y;
	float *new_m = malloc(sizeof(float) * size);

	for (int i = 0; i < size; ++i) {
		new_m[i] = m1[i] * m2[i];
	}
	return (new_m);
}

float *subtract(const float const *m1, const float const *m2, int size_y, int size_x) {
	int size = size_x * size_y;
	float *new_m = malloc(sizeof(float) * size);

	for (int i = 0; i < size; ++i) {
		new_m[i] = m1[i] - m2[i];
	}
	return (new_m);
}

float *add(const float const *m1, const float const *m2, int size_y, int size_x) {
	int size = size_x * size_y;
	float *new_m = malloc(sizeof(float) * size);

	for (int i = 0; i < size; ++i) {
		new_m[i] = m1[i] + m2[i];
	}
	return (new_m);
}
