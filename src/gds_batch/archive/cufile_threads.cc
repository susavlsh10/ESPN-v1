/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <math.h>

//time measurements
#include <time.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"

/*
 * In this sample program, main thread will allocate 100 MB of GPU memory
 * The entire GPU memory will be populated with data by 10 threads.
 * Each thread will read the data at different offsets.
 */

#define TOGB(x) ((x)/(1024*1024*1024L))
#define GB(x) ((x)*1024*1024*1024L)
#define MB(x) ((x)*1024*1024L)
#define KB(x) ((x)*1024L)

#define NUM_THREADS 10
#define DATA_SIZE 100 //MB
#define IOSIZE KB(4)

long data_size;
long data_thread;
int num_threads;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

typedef struct thread_data
{
	int fd;
	void *devPtr;
	loff_t offset;
    CUfileHandle_t cfr_handle;
}thread_data_t;

static void *thread_fn(void *data)
{
	thread_data_t *t = (thread_data_t *)data;
	int fd = t->fd;
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;
	int ret;
	
	/*
	 * We need to set the CUDA device; threads will not inherit main thread's
	 * CUDA context. In this case, since main thread allocated memory on GPU 0,
	 * we set it explicitly. However, threads have to ensure that they are in
	 * same cuda context as devPtr was allocated.
	 */
	cudaSetDevice(0);
	cudaCheckError();

	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		printf("file register error: %s\n", CUFILE_ERRSTR(status.err));
		close(fd);
		exit(1);
	}

	/*
	 * Every thread is registering buffer at different devPtr address of size 100 MB
	 */
	status = cuFileBufRegister(t->devPtr, data_thread, 0);
	if (status.err != CU_FILE_SUCCESS) {
		printf("Buffer register failed :%s\n", CUFILE_ERRSTR(status.err));
		cuFileHandleDeregister(cfr_handle);
		close(fd);
		exit(1);
	}
	
	int jobs = int(data_thread/IOSIZE);
	off_t devPtr_offset = 0;
	loff_t offsetx = t->offset;
	for (int i =0; i<jobs; i++){
		ret = cuFileRead(cfr_handle, t->devPtr, IOSIZE, offsetx, devPtr_offset);
		if (ret < 0) {
			fprintf(stderr, "cuFileRead failed with ret=%d\n", ret);
			goto err;
		}
		
		/*
		fprintf(stdout, "Read Success from fd %d file-offset %ld readSize %ld to GPU 0 Buffer offset %ld size %ld\n",
				fd, (unsigned long) offsetx, IOSIZE, (unsigned long) offsetx, IOSIZE);
		*/

		devPtr_offset += IOSIZE;
		offsetx += IOSIZE;
	}



err:
	status = cuFileBufDeregister(t->devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		fprintf(stderr, "cuFileBufDeregister failed :%s\n", CUFILE_ERRSTR(status.err));
	}

	cuFileHandleDeregister(cfr_handle);
	close(fd);
	return NULL;
}

void help(void) {
	printf("\n./ <cuFile.exe> <file-path-1> <data-size(MB)> <threads>\n");
}

int main(int argc, char **argv) {

	
	void *devPtr;
	size_t offset = 0;


	if (argc < 4) {
		fprintf(stderr, "Invalid input.\n");
		help();
		exit(1);
	}
    
    
    data_size = MB(atoi(argv[2]));
    num_threads = atoi(argv[3]);
    data_thread = int(data_size/num_threads);

    printf("Data size = %ld\n", data_size);
    printf("Number of threads = %d\n", num_threads);
    printf("Data per thread = %ld\n", data_thread);
    
    pthread_t thread[num_threads];
    thread_data t[num_threads];
	cudaSetDevice(0);
	cudaCheckError();

	cudaMalloc(&devPtr, data_size);
	cudaCheckError();

        for (int i = 0; i < num_threads; i++) {
                int fd  = open(argv[1], O_RDWR | O_DIRECT);
                if (fd < 0) {
                        fprintf(stderr, "Unable to open file %s fd %d\n",
                                        argv[1], fd);
                        for(int j = 0; j < i; j++) {
                                close(t[j].fd);
                        }
                        exit(1);
                }
                t[i].fd = fd;
                t[i].devPtr = devPtr;
		t[i].offset = offset;
		devPtr = (char *)devPtr + data_thread;
		offset += data_thread;
	}
    clock_t start, end;
    double cpu_time_used, read_bandwidth;
	start = clock();
    for (int i = 0; i < num_threads; i++) {
		pthread_create(&thread[i], NULL, &thread_fn, &t[i]);
	}

	for (int i = 0; i < num_threads; i++) {
		pthread_join(thread[i], NULL);
	}
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    read_bandwidth = data_size/(cpu_time_used* GB(1));
    printf("Time taken to read data = %f\n", cpu_time_used);
    printf("Read Bandwidth = %f GB/s\n", read_bandwidth);
	cudaFree(devPtr);
	return 0;
}
