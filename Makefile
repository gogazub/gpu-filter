ONEAPI_SETVARS := /opt/intel/oneapi/setvars.sh
CXX = icpx
CXXFLAGS = -std=c++17 -O2 -fsycl
TARGET := median_filter
SOURCES := main.cpp EasyBMP/EasyBMP.cpp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	. $(ONEAPI_SETVARS) >/dev/null 2>&1 && $(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	. $(ONEAPI_SETVARS) >/dev/null 2>&1 && ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.obj *.pdb *.out
