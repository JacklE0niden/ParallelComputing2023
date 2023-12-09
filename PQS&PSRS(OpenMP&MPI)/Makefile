CC = mpicxx
CXX = mpicxx
CFLAGS = -Wall -O2 -fopenmp
CXXFLAGS = -Wall -O2 -fopenmp
LDFLAGS = -lm -fopenmp

# CC = mpicxx
# CXX = g++
# CFLAGS = -Wall -O2
# CXXFLAGS = -Wall -fopenmp
# LDFLAGS = -lm -fopenmp
SRC_DIR = src
BUILD_DIR = build

TARGET1 = $(BUILD_DIR)/PSRS_MPI
TARGET2 = $(BUILD_DIR)/PQS_OpenMP

SRC1 = $(SRC_DIR)/PSRS_MPI.cpp
SRC2 = $(SRC_DIR)/PQS_OpenMP.cpp

OBJ1 = $(BUILD_DIR)/PSRS_MPI.o
OBJ2 = $(BUILD_DIR)/PQS_OpenMP.o

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET2): $(OBJ2)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@if echo "$<" | grep -q "PQS_OpenMP"; then \
		$(CXX) $(CXXFLAGS) -c -o $@ $<; \
	else \
		$(CC) $(CFLAGS) -c -o $@ $<; \
	fi

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET1) $(TARGET2) $(TARGET3)