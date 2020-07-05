# Location of the CUDA Toolkit
NVCC := nvcc

CCFLAGS := -O2

build: exec1

vectorAdd.o:exec1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

vectorAdd: exec1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./exec1

clean:
	rm -f exec1 *.o
	rm -f exec2 *.o
