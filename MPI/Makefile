CC = mpicc
CFLAGS = -Wall -O3 -std=c99
LDFLAGS = -lm
TARGET = raytrace_mpi
SOURCE = raytrace_mpi.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET).ppm

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

benchmark: $(TARGET)
	@echo "Running MPI Ray Tracer benchmark with different process counts..."
	@echo "==================================================="
	@echo "1 process:"
	@time mpirun -np 1 ./$(TARGET)
	@echo ""
	@echo "2 processes:"
	@time mpirun -np 2 ./$(TARGET)
	@echo ""
	@echo "4 processes:"
	@time mpirun -np 4 ./$(TARGET)
	@echo ""
	@echo "8 processes:"
	@time mpirun -np 8 ./$(TARGET)

help:
	@echo "Available targets:"
	@echo "  all       - Build the MPI ray tracer"
	@echo "  clean     - Remove built files"
	@echo "  run       - Run with 4 processes"
	@echo "  benchmark - Run performance tests with different process counts"
	@echo "  help      - Show this help message"