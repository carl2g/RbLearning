CC 		= 	gcc

NAME 		=	c_lib.so

SRCS 		= 	src/matrix_lib.c

FLAGS 	= 	-Ofast -shared -fpic -lm -o

OBJS 		= 	$(SRCS:.c=.o)

RM 		=	rm -rf

MV 		= 	mv

LIB_DIR 	= 	'/usr/lib/RbLearning/'

MKDIR 	= 	mkdir -p

all: 			$(NAME)

$(NAME): 		$(OBJS)
			$(CC) $(SRCS) $(FLAGS) $(NAME)

install:
			$(MKDIR) $(LIB_DIR)
			$(MV) $(NAME) $(LIB_DIR)

clean:
			$(RM) $(OBJS)

fclean: 		clean
			$(RM) $(NAME)

re: 			fclean all

.PHONY: 		all clean fclean re
