NVCC = nvcc

LIBS = cuda

INCLUDE_DIRS=-I../deps/thrust/thrust/ -I.

gpudb:
	$(NVCC) gpudbdriver.cu --disable-warnings -l $(LIBS) -o $@ $(INCLUDE_DIRS)

clean:
	rm -rf gpudb

check: gpudb
	./gpudb
