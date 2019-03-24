#ifndef _MATRIX_LIB
	# define _MATRIX_LIB

extern "C" {
	double *dot(const double *m1, const double *m2, int size_y, int size_v, int size_x);
	double *mult(const double *m1, const double *m2, int size);
	double *subtract(const double *m1, const double *m2, int size);
	double *add(const double *m1, const double *m2, int size);
	double *transpose(const double *m, int size_y, int size_x);
}

#endif /* _MATRIX_LIB */
