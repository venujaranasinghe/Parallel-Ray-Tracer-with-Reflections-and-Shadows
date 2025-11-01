# Use Homebrew GCC which has built-in OpenMP support
CC = /opt/homebrew/bin/gcc-15

CFLAGS = -O3 -fopenmp -lm -Wall
TARGET = raytrace_openmp
SOURCE = raytrace_openmp.c

# Default target
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# Clean target
clean:
	rm -f $(TARGET) *.ppm

# Run target
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Compile the OpenMP ray tracer"
	@echo "  make run      - Compile and run the ray tracer"
	@echo "  make clean    - Remove executable and output files"
	@echo "  make help     - Show this help message"

.PHONY: clean run help