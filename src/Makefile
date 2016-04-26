NVCC = nvcc -std=c++11

LIBS = cuda

INCLUDE_DIRS=-I../deps/thrust/thrust/ -I../deps/cub/ -I.

GPUDBLIB_PATH=/home/gpu/gpudb/gpu-no-sql/src/

TOPAPI_SOURCES = gpudbdriver.cu gpudb-api.cpp DocMap.cpp FilterMap.cpp StringConversion.cpp

gpudb:
	$(NVCC) -O3 gpudbdriver.cu gpudbdrivertest.cpp --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

clean:
	rm -rf gpudb

check: gpudb
	./gpudb

topapi:
	$(NVCC) -g $(TOPAPI_SOURCES) --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

topcheck: topapi
	./topapi
