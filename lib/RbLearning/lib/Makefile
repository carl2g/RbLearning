CC 		= 	nvcc

NAME 		=	c_lib.so

SRCS 		= 	src/matrix_lib.cu

FLAGS 	= 	--shared --ptxas-options=-v --compiler-options '-fPIC' --linker-options -lm --use_fast_math

RM 		=	rm -rf

MV 		= 	mv

LIB_DIR 	= 	'/usr/lib/RbLearning/'

MKDIR 	= 	mkdir -p

all: 			$(NAME)

$(NAME): 		$(SRCS)
			$(CC) $(FLAGS) $(SRCS) -o $(NAME)

install:
			$(MKDIR) $(LIB_DIR)
			$(MV) $(NAME) $(LIB_DIR)

clean:
			$(RM) $(NAME)

re: 			all

.PHONY: 		all clean re
