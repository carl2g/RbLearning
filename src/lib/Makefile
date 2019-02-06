CC 	= 	gcc

NAME 	=	c_lib.so

SRCS 	= 	src/matrix_lib.c 	\
		src/libmath.c

FLAGS = 	-Ofast -shared -fpic -lm -o

OBJS 	= 	$(SRCS:.c=.o)

RM 	=	rm -rf

all: 		$(NAME)

$(NAME): 	$(OBJS)
		$(CC) $(SRCS) $(FLAGS) $(NAME)

clean:
		$(RM) $(OBJS)

fclean: 	clean
		$(RM) $(NAME)

re: 		fclean all

.PHONY: 	all clean fclean re
