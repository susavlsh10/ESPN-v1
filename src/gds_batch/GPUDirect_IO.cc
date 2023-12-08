#include <fcntl.h>
//#include <assert.h>
#include <unistd.h>


//#include <cstdlib>
//#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

//include this header file
//#include "cufile.h"
#include "GPUDirect_IO.h"

#include "cufile_sample_utils.h"


#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 256

#define MB(x) ((x)*1024*1024L)
#define GB(x) ((x)*1024*1024*1024L)


GPUDirect_IO::GPUDirect_IO( int gpuid, char* FILE, size_t _size, int _batch_size, int NUM_BATCHES){
    // constructor code
    NUM_BATCH = NUM_BATCHES;
    TESTFILE= FILE;
    size = _size;
    batch_size= _batch_size;

    cf_descr = (CUfileDescr_t**) malloc(NUM_BATCHES * sizeof(CUfileDescr_t *));
    cf_handle = (CUfileHandle_t**) malloc(NUM_BATCHES * sizeof(CUfileHandle_t*));
    io_batch_params = (CUfileIOParams_t**) malloc(NUM_BATCHES * sizeof(CUfileIOParams_t*));
    fd = (int **) malloc(NUM_BATCHES * sizeof(int*));
    devPtr = (void***)malloc(NUM_BATCH * sizeof(void**));
    batch_id = (CUfileBatchHandle_t*) malloc(NUM_BATCH * sizeof(CUfileBatchHandle_t));
    streams = (cudaStream_t*)malloc(NUM_BATCH * sizeof(cudaStream_t));
    events = (cudaEvent_t*)malloc(NUM_BATCH * sizeof(cudaEvent_t));
    batch_offset = MAX_BUFFER_SIZE*MAX_BATCH_IOS;


    //create cuda streams and events
    for (int i = 0; i < NUM_BATCHES; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }


    for ( int j = 0; j < NUM_BATCH; j++){
        fd[j] = (int *) malloc(sizeof(int)* MAX_BATCH_IOS);
        devPtr[j] = (void **) malloc(sizeof(void*)* MAX_BATCH_IOS);
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
            fd[j][i] = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
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

GPUDirect_IO::~GPUDirect_IO(){
    
}

void GPUDirect_IO::_prepare_batches(){
    //preparing gds datastructures

    //std::cout << "Allocating cuda memory and registering" <<std::endl;
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
            io_batch_params[j][i].cookie = & io_batch_params[j][i];
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

void GPUDirect_IO::read(std::vector<int>& offsets){

    for (int j =0; j < NUM_BATCH; j++){
        for(int i = 0; i < batch_size; i++) {
            io_batch_params[j][i].u.batch.file_offset = offsets[j*batch_size + i];
            //io_batch_params[j][i].u.batch.file_offset = batch_offset*j + i * size;
            //io_batch_params[j][i].u.batch.devPtr_base = devPtr[j][i];
        }
    }
    
    for (int j = 0; j<NUM_BATCH; j++){
        //std::cout << "Submitting Batch" << j <<std::endl;
        
        // Record an event on the stream
        //cudaEventRecord(events[j], streams[j]);
        errorBatch = cuFileBatchIOSubmit(batch_id[j], batch_size, io_batch_params[j], 0);
        if(errorBatch.err != 0) {
            std::cerr << "Error in IO Batch Submit" << std::endl;
            out3();
	    }
        //cudaStreamWaitEvent(streams[j], events[j], 0);
    }

    int num_completed = 0;
    for (int j = 0; j < NUM_BATCH; j++){
        nr = 0;
        
        //cudaStreamSynchronize(streams[j]);
        while(num_completed != batch_size) 
        {
            memset(io_batch_events, 0, sizeof(*io_batch_events));
            nr = batch_size;
            errorBatch = cuFileBatchIOGetStatus(batch_id[j], batch_size, &nr, io_batch_events, NULL);	
            if(errorBatch.err != 0) {
                std::cerr << "Error in IO Batch Get Status" << std::endl;
                out4();
            }
            //std::cout << "Got events " << nr << std::endl;
            num_completed += nr;
	    }
        // Synchronize the stream with the event to ensure correct execution order
    }
    //return devPtr;
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
        free(devPtr[j]);
        free(cf_descr[j]);
        free(cf_handle[j]);
        free(io_batch_params[j]);
        cudaStreamDestroy(streams[j]);
        cudaEventDestroy(events[j]);
    
    }
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