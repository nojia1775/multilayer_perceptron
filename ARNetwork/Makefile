CXX = g++
CXXFLAGS = -std=c++2a -Wall -Wextra -Werror -g -MMD

SRCS =	linear_algebra/src/Complex.cpp \
		linear_algebra/src/DiffMatrix.cpp \
		neural_network/src/ARNetwork.cpp \
		neural_network/src/Functions.cpp \
		neural_network/src/Json.cpp

OBJS_DIR = obj/

OBJS = $(SRCS:%.cpp=$(OBJS_DIR)%.o)

NAME = arnetwork.a

DEPS = $(OBJS:.o=.d)

all: $(NAME)

$(NAME): $(OBJS)
	ar -rsc $(NAME) $(OBJS)

$(OBJS_DIR)%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I./include

clean:
	rm -fr $(OBJS_DIR)

fclean: clean
	rm -rf $(NAME)

re: fclean all

-include $(DEPS)

.PHONY: all re clean fclean