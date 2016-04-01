NVCC = nvcc

LIBS = cuda

gpudb:
	$(NVCC) gpudbdriver.cu -l $(LIBS) -o $@ -I.

clean:
	rm -rf gpudb

check: gpudb
	./gpudb
