NVCC = nvcc

LIBS = cuda

INCLUDE_DIRS=-I../deps/thrust/thrust/ -I../deps/cub/ -I.

gpudb:
	$(NVCC) -g gpudbdriver.cu --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

clean:
	rm -rf gpudb

check: gpudb
	./gpudb

topapi: gpudb
	gcc -g gpudb-api.cpp -o GPUDB

topcheck: topapi
	./GPUDB
