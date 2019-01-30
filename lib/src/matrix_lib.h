#ifndef _MATRIX_LIB
	# define _MATRIX_LIB
 
float *dot(const float const  *m1, const float const  *m2, int size_y, int size_v, int size_x);
float *transpose(const float const  *m, int size_y, int size_x);
float *mult(const float const  *m1, const float const  *m2, int size_y, int size_x);
float *subtract(const float const  *m1, const float const  *m2, int size_y, int size_x);
float *add(const float const  *m1, const float const  *m2, int size_y, int size_x);

#endif /* _MATRIX_LIB */
