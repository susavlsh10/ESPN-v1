#ifndef _GPUDIRECT_IO_
#define _GPUDIRECT_IO_

#include <cstring>
#include <iostream>

//include this header file
#include "cufile.h"

#include <cuda_runtime.h>

#include "GPUDirect_IO.h"

#include <vector>

#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 256

#define MB(x) ((x)*1024*1024L)
#define GB(x) ((x)*1024*1024*1024L)

class GPUDirect_IO{
    public:
    int NUM_BATCH; 
    int batch_size; 
    int ** fd;
    void ***devPtr;
    int batch_offset;
    unsigned nr;
    size_t size;
    cudaStream_t *streams;
    cudaEvent_t *events;
    CUfileDescr_t **cf_descr;
    CUfileHandle_t **cf_handle;
    CUfileIOParams_t **io_batch_params;
    CUfileBatchHandle_t *batch_id;
    CUfileIOEvents_t io_batch_events[MAX_BATCH_IOS];
    
    const char *TESTFILE;
    CUfileError_t status;
    CUfileError_t errorBatch;
    //constructor
    GPUDirect_IO(int gpuid, char* FILE, size_t size, int batch_size, int NUM_BATCHES);

    //Destructor
    ~GPUDirect_IO();

    //Member functions
    void _prepare_batches();

    void read(std::vector<int>& offsets);

    void out1();
    void out2();
    void out3();
    void out4();
    void _reset_batch();
    void _close();

};

#endif