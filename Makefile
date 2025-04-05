# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -Wno-unused-result -fopenmp
LDFLAGS = -lm

# Target and sources
TARGET = chronos_bolt
SRCS = chronos_bolt.c

# Default build target
all: $(TARGET)

# Build
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Debug build
debug:
	$(CC) -g -o $(TARGET) $(SRCS) $(LDFLAGS)

# Check for memory leaks
valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET) autogluon-chronos-bolt-small.bin data.csv forecast.csv

# Build and run
run: $(TARGET)
	./$(TARGET) autogluon-chronos-bolt-small.bin data.csv forecast.csv

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all debug valgrind run clean
