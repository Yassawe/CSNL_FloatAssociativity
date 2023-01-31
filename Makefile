SRCS = $(wildcard *.cpp)
EXECUTABLES = $(patsubst %.cpp,%,$(SRCS))

all: $(EXECUTABLES)

%: %.cpp
	g++ $< -o $@ -I../FloatX/src

.PHONY: clean

clean:
	rm -f $(EXECUTABLES)