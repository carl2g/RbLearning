#ifndef _MATRIX_LIB
	# define _MATRIX_LIB

extern "C" {
	double *dot(double *m1, double *m2, int size_y, int size_v, int size_x);
	double *mult(double *m1, double *m2, int size);
	double *subtract(double *m1, double *m2, int size);
	double *add(double *m1, double *m2, int size);
	double *transpose(double *m, int size_y, int size_x);
}

#endif /* _MATRIX_LIB */
