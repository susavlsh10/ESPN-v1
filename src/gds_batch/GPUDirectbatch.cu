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
/*
 * Sample cuFileBatchIOSubmit Read Test.
 *
 * This sample program reads data from GPU memory to a file using the Batch API's.
 * For verification, input data has a pattern.
 * User can verify the output file-data after write using
 * hexdump -C <filepath>
 * 00000000  ab ab ab ab ab ab ab ab  ab ab ab ab ab ab ab ab  |................|
 */  
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>


#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//include this header file
#include "cufile.h"

#include "cufile_sample_utils.h"

#include <cuda_fp16.h>

#include "utils.h"

//time measurements
#include <time.h>

#include <iomanip>  // Required for std::setprecision

#include <thread>
#include <chrono>
#include <pthread.h>

using namespace std;

#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 128

__global__ void parseBuffer(half ***devPtr, half *CLS_MAT, half *BOW_MAT, int NUM_BATCH, int batch_size) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    //half data;

    if (batch_idx < NUM_BATCH && thread_idx < batch_size) {
        half* buffer = (half*)devPtr[batch_idx][thread_idx];      
        half* cls_vector = buffer;
        half* bow_matrix = buffer + 128;

        // Store CLS vector in CLS_MAT
        int cls_offset = batch_idx * batch_size * 128 + thread_idx * 128;
        for (int i = 0; i < 128; i++) {
            CLS_MAT[cls_offset + i] = cls_vector[i];
        }
        // Store BOW vector in BOW_MAT
        int bow_offset = (batch_idx * batch_size + thread_idx) * 60 * 32;
        for (int i = 0; i < 60; i++) {
            for (int j = 0; j < 32; j++) {
                BOW_MAT[bow_offset + i * 32 + j] = bow_matrix[i * 32 + j];
            }
        }
    }
}


class GPUDirect_IO{
    public:
    int NUM_BATCH; 
    int batch_size; 
    int ** fd;
    //void ***devPtr;
    half ***devPtr;

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
    
    std::string TESTFILE;
    CUfileError_t status;
    CUfileError_t errorBatch;
    //constructor
    GPUDirect_IO(int gpuid, std::string FILE, size_t size, int batch_size, int NUM_BATCHES);

    //Destructor
    ~GPUDirect_IO();

    //Member functions
    void _prepare_batches();

    void read(std::vector<int>& offsets, half* CLS_MAT, half* BOW_MAT);

    void out1();
    void out2();
    void out3();
    void out4();
    void _reset_batch();
    void _close();

};

GPUDirect_IO::GPUDirect_IO(int gpuid, std::string FILE, size_t _size, int _batch_size, int NUM_BATCHES){
    // constructor code
    NUM_BATCH = NUM_BATCHES;
    TESTFILE= FILE;
    size = _size;
    batch_size= _batch_size;

    cf_descr = (CUfileDescr_t**) malloc(NUM_BATCHES * sizeof(CUfileDescr_t *));
    cf_handle = (CUfileHandle_t**) malloc(NUM_BATCHES * sizeof(CUfileHandle_t*));
    io_batch_params = (CUfileIOParams_t**) malloc(NUM_BATCHES * sizeof(CUfileIOParams_t*));
    fd = (int **) malloc(NUM_BATCHES * sizeof(int*));
    
    //devPtr = (half***)malloc(NUM_BATCH * sizeof(half**));
    cudaMallocManaged(&devPtr, NUM_BATCH * sizeof(half**));
    batch_id = (CUfileBatchHandle_t*) malloc(NUM_BATCH * sizeof(CUfileBatchHandle_t));
    streams = (cudaStream_t*)malloc(NUM_BATCH * sizeof(cudaStream_t));
    events = (cudaEvent_t*)malloc(NUM_BATCH * sizeof(cudaEvent_t));
    batch_offset = MAX_BUFFER_SIZE*MAX_BATCH_IOS;

    for ( int j = 0; j < NUM_BATCH; j++){
        fd[j] = (int *) malloc(sizeof(int)* MAX_BATCH_IOS);
        
        //devPtr[j] = (half **) malloc(sizeof(half*)* MAX_BATCH_IOS);
        cudaMallocManaged(&devPtr[j], sizeof(half*)* MAX_BATCH_IOS);
        cf_descr[j] = (CUfileDescr_t *) malloc(sizeof(CUfileDescr_t)* MAX_BATCH_IOS);
        cf_handle[j] = (CUfileHandle_t *) malloc(sizeof(CUfileHandle_t)* MAX_BATCH_IOS);
        io_batch_params[j] = (CUfileIOParams_t *) malloc(sizeof(CUfileIOParams_t)* MAX_BATCH_IOS);

    }    

    check_cudaruntimecall(cudaSetDevice(gpuid));
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cufile driver open error: " << cuFileGetErrorString(status) << std::endl;
            exit(1);
    } 

    if(batch_size > MAX_BATCH_IOS) {
		std::cerr << "Requested batch Size exceeds maximum Batch Size limit:" << MAX_BATCH_IOS << std::endl;
		exit(1);
	}

	// opens file and register file handle
    std::cout << "Opening file and registering file handle" <<std::endl;
    for (int j =0; j< NUM_BATCH; j++){
        memset((void *)cf_descr[j], 0, MAX_BATCH_IOS * sizeof(CUfileDescr_t));
        //std::cout << "memset" <<std::endl;
        for(int i = 0; i < batch_size; i++) {
            //open the file for each io
            //fd[j][i] = open(TESTFILE.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
            fd[j][i] = open(TESTFILE.c_str(), O_RDWR | O_DIRECT, 0664);
            if ((fd[j][i]) < 0) {
                std::cerr << "file open error:" << cuFileGetErrorString(errno) << std::endl;
                out1();
            }
        
            //register file handle
            cf_descr[j][i].handle.fd = fd[j][i];
            cf_descr[j][i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

            status = cuFileHandleRegister(&cf_handle[j][i], &cf_descr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
                close(fd[j][i]);
                fd[j][i] = -1;
                out1();
            }    
        }
    }

    std::cout<<"GDS initialized"<<std::endl;
}

GPUDirect_IO::~GPUDirect_IO(){}

void GPUDirect_IO::_prepare_batches(){
    //preparing gds datastructures

    // allocate cuda memory and register for GDS reads
    for (int j =0; j < NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            //allocate cuda memory *change this to allocate a contigious memory space
            devPtr[j][i] = NULL;
            check_cudaruntimecall(cudaMalloc((void**)&devPtr[j][i], size));
            check_cudaruntimecall(cudaMemset((void*)(devPtr[j][i]), 0xab, size));
            check_cudaruntimecall(cudaStreamSynchronize(0));

            //register device memory

            status = cuFileBufRegister(devPtr[j][i], size, 0);
            if ((status.err) != CU_FILE_SUCCESS) {
                std::cerr << "buffer register failed:"<< cuFileGetErrorString(status) << std::endl;
                out2();
            }
        } 
    }

	//std::cout << "initializing io batch params" << std::endl;
	for (int j =0; j < NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            io_batch_params[j][i].mode = CUFILE_BATCH;
            io_batch_params[j][i].fh = cf_handle[j][i];
            io_batch_params[j][i].u.batch.devPtr_base = devPtr[j][i]; //make memory location contiguous 
            io_batch_params[j][i].u.batch.devPtr_offset = 0;    //change this 
            io_batch_params[j][i].u.batch.size = size;
            io_batch_params[j][i].opcode = CUFILE_READ;
            io_batch_params[j][i].cookie = &io_batch_params[j][i];
        }
    }

    //std::cout << "Setting Up Batch" << std::endl;
    for (int j =0; j < NUM_BATCH; j++){
        errorBatch = cuFileBatchIOSetUp(&batch_id[j], batch_size);
        if((errorBatch.err) != 0) {
            std::cerr << "Error in setting Up Batch" << std::endl;
            out3();
	    }
    }
}

void GPUDirect_IO::read(std::vector<int>& offsets, half* CLS_MAT, half* BOW_MAT){

    // set file offset for batch reads
    for (int j =0; j < NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            io_batch_params[j][i].u.batch.file_offset = offsets[j*batch_size + i];
            //io_batch_params[j][i].u.batch.file_offset = batch_offset*j + i * size;
        }
    }
    
    //submit batch io 
    for (int j = 0; j<NUM_BATCH; j++){
        errorBatch = cuFileBatchIOSubmit(batch_id[j], batch_size, io_batch_params[j], 0);
        if(errorBatch.err != 0) {
            std::cerr << "Error in IO Batch Submit" << std::endl;
            out3();
	    }
    }

    //wait for all data to arrive
    int num_completed = 0;
    for (int j = 0; j < NUM_BATCH; j++){
        nr = 0;
        while(num_completed != batch_size) 
        {
            memset(io_batch_events, 0, sizeof(*io_batch_events));
            nr = batch_size;
            errorBatch = cuFileBatchIOGetStatus(batch_id[j], batch_size, &nr, io_batch_events, NULL);	
            if(errorBatch.err != 0) {
                std::cerr << "Error in IO Batch Get Status" << std::endl;
                out4();
            }
            num_completed += nr;
	    }
    }
    parseBuffer<<<NUM_BATCH, batch_size>>>(devPtr, CLS_MAT, BOW_MAT, NUM_BATCH, batch_size);
}

void GPUDirect_IO::_reset_batch(){

    //reset batch ids
    for(int j=0; j< NUM_BATCH; j++){
        cuFileBatchIODestroy(batch_id[j]);
        errorBatch = cuFileBatchIOSetUp(&batch_id[j], batch_size);
        if((errorBatch.err) != 0) {
            std::cerr << "Error in setting Up Batch" << std::endl;
            out3();
	    }
    }

    //reset buffers
    for(int j = 0; j< NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            status = cuFileBufDeregister(devPtr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "buffer deregister failed:"
                    << cuFileGetErrorString(status) << std::endl;
            }
            
            //check_cudaruntimecall(cudaMemset((void*)(devPtr[j][i]), 0xab, size));
            status = cuFileBufRegister(devPtr[j][i], size, 0);
            if ((status.err) != CU_FILE_SUCCESS) {
                std::cerr << "buffer register failed:"<< cuFileGetErrorString(status) << std::endl;
                out2();
            }
        }
    }
    
}

void GPUDirect_IO::_close(){
    out4();
    out3();
    out2();
    out1();
}

void GPUDirect_IO::out1(){
    // close file
    for(int j = 0; j< NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            if (fd[j][i] > 0) {
                cuFileHandleDeregister(cf_handle[j][i]);
                close(fd[j][i]);
            }
        }
    }

	//std::cout << "cuFileHandleDeregister Done" << std::endl;

	status = cuFileDriverClose();
	//std::cout << "cuFileDriverClose Done" << std::endl;
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}

    for (int j = 0; j < NUM_BATCH; j++){
        free(fd[j]);
        cudaFree(devPtr[j]);
        free(cf_descr[j]);
        free(cf_handle[j]);
        free(io_batch_params[j]);
    }
    cudaFree(devPtr);
    std::cout << "cuFileHandleDeregister Done" << std::endl;
}

void GPUDirect_IO::out2(){
    for(int j = 0; j< NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            check_cudaruntimecall(cudaFree(devPtr[j][i]));
        }
    }
    std::cout << "cudaFree Done" << std::endl;
}

void GPUDirect_IO::out3(){
	// deregister the device memory
    for(int j = 0; j< NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            status = cuFileBufDeregister(devPtr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                //ret = -1;
                std::cerr << "buffer deregister failed:"
                    << cuFileGetErrorString(status) << std::endl;
            }
        }
    }
    std::cout << "cuFile BufDeregsiter Done" << std::endl;
}

void GPUDirect_IO::out4(){
    for(int j=0; j< NUM_BATCH; j++){
        cuFileBatchIODestroy(batch_id[j]);
    }
}

void _print_gds_io(GPUDirect_IO* GDS){
    __half read_ptr[GDS->NUM_BATCH][GDS->batch_size][GDS->size/sizeof(__half)];
    for (int j=0; j< GDS->NUM_BATCH; j++){
        for (int i =0; i < GDS->batch_size; i++){
            cudaMemcpy(read_ptr[j][i], GDS->devPtr[j][i], GDS->size, cudaMemcpyDeviceToHost);
        }
    }

    std::cout<<"Data samples"<<std::endl;
    for (int j =0; j < GDS->NUM_BATCH; j++){
        for (int i =0; i< GDS->batch_size; i++){
            for (long unsigned int k=0; k < GDS->size/sizeof(__half); k++){
                std::cout<<(float)read_ptr[j][i][k]<<" ";
            }
        std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void _print_cls(half* deviceCLSMatrix, half* hostCLSMatrix, int cls_mat_size){
    cudaMemcpy(hostCLSMatrix, deviceCLSMatrix, cls_mat_size * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "CLS matrix " <<std::endl;
    for(int j =0; j <cls_mat_size/128; j++){
        for(int i =0; i< 128; i++){
            std::cout <<(float) hostCLSMatrix[j*128 + i] <<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void _print_cls_pretty(half* deviceCLSMatrix, half* hostCLSMatrix, int cls_mat_size){
    cudaMemcpy(hostCLSMatrix, deviceCLSMatrix, cls_mat_size * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "CLS matrix " <<std::endl;
    const int maxElements = 3;
    int maxRow = 3;
    int rows = cls_mat_size/128;
    int cols = 128;
    if (rows < maxRow)
        maxRow = rows;
    std::cout << std::fixed << std::setprecision(3);
    for(int j =0; j <maxRow; j++){
        for(int i =0; i< maxElements; i++){
            std::cout <<(float) hostCLSMatrix[j*128 + i] <<" ";
        }
        std::cout << "... ";
        std::cout << (float) hostCLSMatrix[j * 128 + 128 - 1] << std::endl;
        //std::cout<<std::endl;
    }
    std::cout<<"...."<<std::endl;
    for(int i =0; i< maxElements; i++){
        std::cout <<(float) hostCLSMatrix[(rows - 1) * cols + i] <<" ";
    }
    std::cout << "... ";
    std::cout << (float) hostCLSMatrix[(rows - 1) * cols + 128 - 1] << std::endl;
    //print out last row

}

void _print_bow(half* deviceBOWMatrix, half* hostBOWMatrix, int bow_mat_size){
    cudaMemcpy(hostBOWMatrix, deviceBOWMatrix, bow_mat_size * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "BOW matrix " <<std::endl;
    for (int k = 0; k < bow_mat_size/(32*60); k++){
        for(int j =0; j < (32*60); j++)
            {
                std::cout <<(float) hostBOWMatrix[k*(32*60) + j] <<" ";
            }
            std::cout<<std::endl;
    }
        std::cout<<std::endl;
}

void _copy_cls(GPUDirect_IO* GDS, half* CLS_MAT, cudaStream_t* stream){
    for (int j=0; j< GDS->NUM_BATCH; j++){
        for (int i =0; i < GDS->batch_size; i++){
            //cudaMemcpy(CLS_MAT+j*128+i*128, (void*)GDS->devPtr[j][i], 256, cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(&CLS_MAT[j*128*128+i*128], (void*)GDS->devPtr[j][i], 256, cudaMemcpyDeviceToDevice, stream[j*GDS->batch_size + i]);
        }
    }
    for(int i = 0; i<GDS->NUM_BATCH*GDS->batch_size; i++){
        cudaStreamSynchronize(stream[i]);
    }
}

void* sleepAndReturn(void*) {
    // Sleep for 10 milliseconds (10000 microseconds)
    usleep(10000);
    std::cout << "Thread is awake!" << std::endl;
    return nullptr;
}

int main(int argc, char *argv[]) {
    
    if(argc < 5) {
        std::cerr << argv[0] << " <filepath> <gpuid> <num batch entries> <num of batches> <io size>"<< std::endl;
        exit(1);
    }

    std::string TESTFILE = argv[1];
    int gpuid = atoi(argv[2]);
    int batch_size = atoi(argv[3]);
    int NUM_BATCH = atoi(argv[4]);
    int io_size =  atoi(argv[5]);
    int loop = 10;    

    size_t size = MAX_BUFFER_SIZE * io_size;
	//time measurements
	clock_t start, end, reset_end;
    //clock_t test_start, test_end;


    
    int arraySize = 1024;
    int blockSize = 4096;
    std::vector<int> offsetArray(arraySize);
    

    // Print the generated offset values
    /*
    for (int i = 0; i < arraySize; ++i) {
        std::cout << i << ": offset = " <<offsetArray[i] << " bytes\n";
    }
    */

   //create CPU and GPU buffers for CLS and BOW matrices
    int num_io = NUM_BATCH * batch_size;
    int cls_mat_size = num_io*128;
    int bow_mat_size = num_io*32*60;

    half* hostCLSMatrix = new half[cls_mat_size];
    half* hostBOWMatrix = new half[bow_mat_size];
    
    // Allocate device memory for the matrix
    half* deviceCLSMatrix;
    cudaMalloc((void**)&deviceCLSMatrix, cls_mat_size * sizeof(half));

    half* deviceBOWMatrix;
    cudaMalloc((void**)&deviceBOWMatrix, bow_mat_size * sizeof(half));

    generateOffsetArray(offsetArray,arraySize,blockSize);

    GPUDirect_IO GDS(gpuid, TESTFILE, size, batch_size, NUM_BATCH); 
    GDS._prepare_batches();

    double average_throughput = 0;
    pthread_t thread;
    for (int k = 0; k<loop; k++){
        
        generateOffsetArray(offsetArray,arraySize,blockSize);
        start = clock();

        // Create a new thread
        if (pthread_create(&thread, nullptr, sleepAndReturn, nullptr) != 0) {
            std::cerr << "Failed to create thread." << std::endl;
            return 1;
        }

        std::cout << "GDS starting " <<std::endl;
        GDS.read(offsetArray, deviceCLSMatrix, deviceBOWMatrix);
        
        std::this_thread::sleep_for(20ms);
        //usleep(10000);
        // Wait for the thread to finish
        if (pthread_join(thread, nullptr) != 0) {
            std::cerr << "Failed to join thread." << std::endl;
            return 1;
    }

        end = clock();
        
        GDS._reset_batch();
        
        reset_end = clock();

        //_print_cls(deviceCLSMatrix, hostCLSMatrix, cls_mat_size);
        _print_cls_pretty(deviceCLSMatrix, hostCLSMatrix, cls_mat_size);
        
        //_print_bow(deviceBOWMatrix, hostBOWMatrix, bow_mat_size);
        //_print_gds_io(&GDS);


        average_throughput += _print_throughput(start, end, reset_end, batch_size, size,  NUM_BATCH);
    }    
    
    printf("\n\n");
    GDS._close();
    average_throughput = average_throughput /loop;

    std::cout << "Average throughput = " << average_throughput << " GB/s "<< std::endl;
    
    // Cleanup
    delete[] hostCLSMatrix;
    delete[] hostBOWMatrix;
    cudaFree(deviceCLSMatrix);
    cudaFree(deviceBOWMatrix);

    return 0;
}








    // create test buffer 
    /*
    half*** devPtr;
    cudaMallocManaged(&devPtr, NUM_BATCH * sizeof(half**));
    for ( int j = 0; j < NUM_BATCH; j++){
        //devPtr[j] = (half **) malloc(sizeof(half*)* MAX_BATCH_IOS);
        cudaMallocManaged(&devPtr[j], sizeof(half*)* MAX_BATCH_IOS);
    }

    for (int j =0; j < NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            //allocate cuda memory *change this to allocate a contigious memory space
            devPtr[j][i] = NULL;
            check_cudaruntimecall(cudaMalloc((void**)&devPtr[j][i], size));
            check_cudaruntimecall(cudaMemset((void*)(devPtr[j][i]), 0, size));
            check_cudaruntimecall(cudaStreamSynchronize(0));
            
            //register device memory
            CUfileError_t status = cuFileBufRegister(devPtr[j][i], size, 0);
            if ((status.err) != CU_FILE_SUCCESS) {
                std::cerr << "buffer register failed:"<< cuFileGetErrorString(status) << std::endl;
            }
            
            
        } 
    }   


    //GDS.devPtr = devPtr;    
    
    //_copy_cls(&GDS, deviceCLSMatrix, stream);
    */

    /*
    half *testDevPtr; half* testCPUPtr; half **GPUPtr;
    cudaMalloc((void**)&testDevPtr, size*num_io);
    cudaMemset((void*)(testDevPtr), (half)11, size*num_io);

    testCPUPtr = (half*)malloc(size*num_io);
    // fill cpu buffer 
    int x = 0;
    for (int i =0; i< num_io*size/sizeof(half); i++){
        if (x < 4096){
            testCPUPtr[i] = x;
            x++;
        }
        else{
            x = 0;
        }
    }
    cudaMemcpy(testDevPtr, testCPUPtr, size*num_io, cudaMemcpyHostToDevice);


        //parseCLSBuffer<<<NUM_BATCH, batch_size, 0 ,stream1>>>(GDS.devPtr, deviceCLSMatrix, NUM_BATCH, batch_size);
        //parseBOWBuffer<<<NUM_BATCH, batch_size, 0 ,stream2>>>(GDS.devPtr, deviceBOWMatrix, NUM_BATCH, batch_size);
        
        // Synchronize streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    */