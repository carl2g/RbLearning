CC 	= 	gcc

NAME 	=	matrix_lib.so

SRCS 	= 	src/matrix_lib.c

FLAGS = 	-O3 -shared -fpic -o

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
