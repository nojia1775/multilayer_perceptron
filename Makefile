CXX = g++

CXXFLAGS = -std=c++2a -Wall -Wextra -Werror -g -MMD

OBJS_DIR = obj

SRCS_TRAIN = train.cpp

SRCS_SPLIT = split.cpp

SRCS_PRED = prediction.cpp

OBJS_TRAIN = $(SRCS_TRAIN:%.cpp=$(OBJS_DIR)/%.o)

OBJS_SPLIT = $(SRCS_SPLIT:%.cpp=$(OBJS_DIR)/%.o)

OBJS_PRED = $(SRCS_PRED:%.cpp=$(OBJS_DIR)/%.o)

DEPS_SPLIT = $(OBJS_SPLIT:.o=.d)

DEPS_TRAIN = $(OBJS_TRAIN:.o=.d)

DEPS_PRED = $(OBJS_PRED:.o=.d)

NAME_SPLIT = split

NAME_TRAIN = train

NAME_PRED = prediction

all: train split prediction

$(NAME_SPLIT): $(OBJS_SPLIT)
	make -C ARNetwork
	$(CXX) $(CXXFLAGS) $^ ARNetwork/arnetwork.a -o $@

$(NAME_TRAIN): $(OBJS_TRAIN)
	make -C ARNetwork
	$(CXX) $(CXXFLAGS) $^ ARNetwork/arnetwork.a -o $@

$(NAME_PRED): $(OBJS_PRED)
	make -C ARNetwork
	$(CXX) $(CXXFLAGS) $^ ARNetwork/arnetwork.a -o $@
	
$(OBJS_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	make clean -C ARNetwork
	rm -rf $(OBJS_DIR) $(DEPS_PRED) $(DEPS_TRAIN) $(DEPS_SPLIT)

fclean: clean
	make fclean -C ARNetwork
	rm -f $(NAME_PRED) $(NAME_TRAIN) $(NAME_SPLIT) training.csv validation.csv

re: fclean all

-include $(DEPS_SPLIT)
-include $(DEPS_TRAIN)
-include $(DEPS_PRED)

.PHONY: all clean fclean re show