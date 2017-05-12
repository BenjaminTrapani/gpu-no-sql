NVCC = nvcc -std=c++11 -arch=sm_35 -rdc=true

LIBS = cudart

LIB_PATH = /usr/local/cuda/lib64/

INCLUDE_DIRS=-I../deps/thrust/thrust/ -I../deps/cub/ -I.

GPUDBLIB_PATH=/home/gpu/gpudb/gpu-no-sql/src/

GPUDBCORE_SOURCES = gpudbdriver.cu CPUAggregator.cpp

TOPAPI_SOURCES = $(GPUDBCORE_SOURCES) gpudb-api.cpp DocMap.cpp FilterMap.cpp StringConversion.cpp

gpudb:
	$(NVCC) -g $(GPUDBCORE_SOURCES) gpudbdrivertest.cpp --disable-warnings -L $(LIB_PATH) -l $(LIBS) -o $@ $(INCLUDE_DIRS)

clean:
	rm -rf gpudb

check: gpudb
	export LD_LIBRARY_PATH=$(LIB_PATH) && ./gpudb

topapi:
	$(NVCC) -g $(TOPAPI_SOURCES) gpudb-api-tests.cpp --disable-warnings -L $(LIB_PATH) -l $(LIBS) -o $@ $(INCLUDE_DIRS)

topcheck: topapi
	./topapi
