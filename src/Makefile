NVCC = nvcc -std=c++11

LIBS = cuda

INCLUDE_DIRS=-I../deps/thrust/thrust/ -I../deps/cub/ -I.

GPUDBLIB_PATH=/home/gpu/gpudb/gpu-no-sql/src/

gpudb:
	$(NVCC) -g gpudbdriver.cu gpudbdrivertest.cpp --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

clean:
	rm -rf gpudb

check: gpudb
	./gpudb

topapi:
	$(NVCC) -g gpudbdriver.cu gpudb-api.cpp --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

topcheck: topapi
	./topapi
