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

//include this header file
#include "cufile.h"

#include "cufile_sample_utils.h"

//time measurements
#include <time.h>

using namespace std;

#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 128
#define MAX_NR 16

#define MB(x) ((x)*1024*1024L)
#define GB(x) ((x)*1024*1024*1024L)

int main(int argc, char *argv[]) {
    if(argc < 5) {
                std::cerr << argv[0] << " <filepath> <gpuid> <num batch entries> <num of batches> <io size>"<< std::endl;
                exit(1);
        }
    unsigned int NUM_BATCH = atoi(argv[4]);

    unsigned int io_size =  atoi(argv[5]);

    int *fd[NUM_BATCH];

	ssize_t ret = -1;

    void **devPtr[NUM_BATCH];

	const size_t size = MAX_BUFFER_SIZE * io_size;

    size_t cuda_buffer_size; 

    std::cout << "IO size = " << size << " Bytes" <<std::endl;

    CUfileError_t status;

	const char *TESTFILE;

    CUfileDescr_t *cf_descr[NUM_BATCH];

    CUfileHandle_t *cf_handle[NUM_BATCH];
    CUfileIOParams_t *io_batch_params[NUM_BATCH];

	CUfileIOEvents_t io_batch_events[MAX_BATCH_IOS];

    //float *CPUPtr[NUM_BATCH];

	unsigned int i = 0;
    unsigned int j = 0;
    //unsigned int k = 0;

    for ( j = 0; j < NUM_BATCH; j++){
        fd[j] = (int *) malloc(sizeof(int)* MAX_BATCH_IOS);
        devPtr[j] = (void **) malloc(sizeof(void*)* MAX_BATCH_IOS);
        cf_descr[j] = (CUfileDescr_t *) malloc(sizeof(CUfileDescr_t)* MAX_BATCH_IOS);
        cf_handle[j] = (CUfileHandle_t *) malloc(sizeof(CUfileHandle_t)* MAX_BATCH_IOS);
        io_batch_params[j] = (CUfileIOParams_t *) malloc(sizeof(CUfileIOParams_t)* MAX_BATCH_IOS);

    }    


	unsigned int flags = 0;
    CUstream stream;
	CUfileError_t errorBatch[NUM_BATCH]; 
	CUfileBatchHandle_t batch_id[NUM_BATCH]; 
    CUfileBatchHandle_t batch_id1[NUM_BATCH]; 

	unsigned nr;
	unsigned batch_size;
	unsigned num_completed = 0;
	int batch_offset = MAX_BUFFER_SIZE*MAX_BATCH_IOS;

	//time measurements
	clock_t start, end;
    double cpu_time_used, read_bandwidth;
	long data_size;

	if(argc < 4) {
                std::cerr << argv[0] << " <filepath> <gpuid> <num batch entries>"<< std::endl;
                exit(1);
        }
        memset(&stream, 0, sizeof(CUstream));
        TESTFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "cufile driver open error: "
			<< cuFileGetErrorString(status) << std::endl;
                return -1;
        }
	//std::cout << "opening file " << TESTFILE << std::endl;
	
	batch_size = atoi(argv[3]);
	
	if(batch_size > MAX_BATCH_IOS) {
		std::cerr << "Requested batch Size exceeds maximum Batch Size limit:" << MAX_BATCH_IOS << std::endl;
		return -1;
	}
    cuda_buffer_size = size*batch_size*NUM_BATCH;

	// opens file and register file handle
    for (j =0; j< NUM_BATCH; j++){
        memset((void *)cf_descr[j], 0, MAX_BATCH_IOS * sizeof(CUfileDescr_t));
        for(i = 0; i < batch_size; i++) {
            //open the file for each io
            fd[j][i] = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
            if ((fd[j][i]) < 0) {
                std::cerr << "file open error:"
                << cuFileGetErrorString(errno) << std::endl;
                goto out1;
            }
        
            //register file handle
            cf_descr[j][i].handle.fd = fd[j][i];
            cf_descr[j][i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

            status = cuFileHandleRegister(&cf_handle[j][i], &cf_descr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error:"
                    << cuFileGetErrorString(status) << std::endl;
                close(fd[j][i]);
                fd[j][i] = -1;
                goto out1;
            }    
        }
    }



    for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            devPtr[j][i] = NULL;
            check_cudaruntimecall(cudaMalloc(&devPtr[j][i], size));
            check_cudaruntimecall(cudaMemset((void*)(devPtr[j][i]), 0xab, size));
            check_cudaruntimecall(cudaStreamSynchronize(0));

            status = cuFileBufRegister(devPtr[j][i], size, 0);
            if ((status.err) != CU_FILE_SUCCESS) {
                ret = -1;
                std::cerr << "buffer register failed:"
                    << cuFileGetErrorString(status) << std::endl;
                goto out2;
            }
        } 
    }    

	//std::cout << "writing from device memory" << std::endl;
	
    
    for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            io_batch_params[j][i].mode = CUFILE_BATCH;
            io_batch_params[j][i].fh = cf_handle[j][i];
            io_batch_params[j][i].u.batch.devPtr_base = devPtr[j][i];
            io_batch_params[j][i].u.batch.devPtr_offset = 0;
            io_batch_params[j][i].u.batch.size = size;
            io_batch_params[j][i].opcode = CUFILE_READ;
        }
    }    

    
    //std::cout << "Setting Up Batch" << std::endl;
    for (j =0; j < NUM_BATCH; j++){
        errorBatch[j] = cuFileBatchIOSetUp(&batch_id[j], batch_size);
        if((errorBatch[j].err) != 0) {
            std::cerr << "Error in setting Up Batch" << std::endl;
            goto out3;
	    }
    }

    /* Setting the read offset batch */
    start = clock();
    for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            io_batch_params[j][i].u.batch.file_offset = batch_offset*j + i * size;
        }
    }

	
    
	//std::cout << "Submitting Batch IO" << std::endl;
    for (j = 0; j<NUM_BATCH; j++){
        errorBatch[j] = cuFileBatchIOSubmit(batch_id[j], batch_size, io_batch_params[j], flags);
        if(errorBatch[j].err != 0) {
            std::cerr << "Error in IO Batch Submit" << std::endl;
            goto out3;
	    }

        std::cout<< "Batch " << j << " submitted at " << (double) (clock() - start) / CLOCKS_PER_SEC << " s" <<std::endl;

    }
    
	//std::cout << "Batch IO Submitted" << std::endl;
	//wait for all batches to complete
    for (j = 0; j < NUM_BATCH; j++){
        nr = 0;
        while(num_completed != batch_size) 
        {
            memset(io_batch_events, 0, sizeof(*io_batch_events));
            nr = batch_size;
            errorBatch[j] = cuFileBatchIOGetStatus(batch_id[j], batch_size, &nr, io_batch_events, NULL);	
            if(errorBatch[j].err != 0) {
                std::cerr << "Error in IO Batch Get Status" << std::endl;
                goto out4;
            }
            //std::cout << "Got events " << nr << std::endl;
            num_completed += nr;
	    }
    }

	end = clock();

	//std::cout << "\nBatch IO Get status done got completetions for " << nr << " events" << std::endl;
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	data_size = batch_size * size *NUM_BATCH;
    read_bandwidth = data_size/(cpu_time_used* GB(1));
	printf("Total Data size = %ld MB\n", data_size/MB(1));
    printf("Time taken  = %f\n", cpu_time_used);
    printf("Read Bandwidth = %f GB/s\n", read_bandwidth);


    //std::cout << "Second retrieval" <<std::endl;


out4:
	for(j=0; j< NUM_BATCH; j++){
        cuFileBatchIODestroy(batch_id[j]);
    }


	//Submit Batch IO
	//std::cout << "deregistering device memory" << std::endl;
out3:
	// deregister the device memory

    for(j = 0; j< NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            status = cuFileBufDeregister(devPtr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                ret = -1;
                std::cerr << "buffer deregister failed:"
                    << cuFileGetErrorString(status) << std::endl;
            }
        }
    }    
    
/*
    status = cuFileBufDeregister(Devptr);
    if (status.err != CU_FILE_SUCCESS) {
        ret = -1;
        std::cerr << "buffer deregister failed:"
            << cuFileGetErrorString(status) << std::endl;
*/


	//std::cout << "cuFile BufDeregsiter Done" << std::endl;
out2:
	for(j = 0; j< NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            check_cudaruntimecall(cudaFree(devPtr[j][i]));
        }
    }

	//std::cout << "cudaFree Done" << std::endl;
out1:
	// close file
    for(j = 0; j< NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
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
		ret = -1;
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
	ret = 0;

    for ( j = 0; j < NUM_BATCH; j++){
        free(fd[j]);
        free(devPtr[j]);
        free(cf_descr[j]);
        free(cf_handle[j]);
        free(io_batch_params[j]);
    }  
	return ret;

    }


// Archives 

/*
    for (j =0; j < NUM_BATCH; j++){
        memset((void *)cf_descr[j], 0, MAX_BATCH_IOS * sizeof(CUfileDescr_t));
    }

*/

/*
	for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            cf_descr[j][i].handle.fd = fd[j][i];
            cf_descr[j][i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

            status = cuFileHandleRegister(&cf_handle[j][i], &cf_descr[j][i]);
            if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error:"
                    << cuFileGetErrorString(status) << std::endl;
                close(fd[j][i]);
                fd[j][i] = -1;
                goto out1;
            }
        }
    }

*/

/*
    Devptr = NULL;
    check_cudaruntimecall(cudaMalloc((void**)&Devptr, cuda_buffer_size));
    check_cudaruntimecall(cudaMemset((void*)(Devptr), 0xab, cuda_buffer_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));
    status = cuFileBufRegister(Devptr, cuda_buffer_size, 0);
    if ((status.err) != CU_FILE_SUCCESS) {
        ret = -1;
        std::cerr << "buffer register failed:"
            << cuFileGetErrorString(status) << std::endl;
        goto out2;
    }
    


*/
	// filler

	//std::cout << "registering device memory of size :" << size << std::endl;
	// registers device memory
    
    /*
    for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            status = cuFileBufRegister(devPtr[j][i], size, 0);
            if ((status.err) != CU_FILE_SUCCESS) {
                ret = -1;
                std::cerr << "buffer register failed:"
                    << cuFileGetErrorString(status) << std::endl;
                goto out2;
            }
        }
    }
    
    */

    /*
	for (j =0; j < NUM_BATCH; j++){
        for(i = 0; i < batch_size; i++) {
            io_batch_params[j][i].mode = CUFILE_BATCH;
            io_batch_params[j][i].fh = cf_handle[j][i];
            io_batch_params[j][i].u.batch.devPtr_base = Devptr + batch_offset*j + i * size;
            io_batch_params[j][i].u.batch.devPtr_offset = 0; //batch_offset*j + i * size;
            io_batch_params[j][i].u.batch.size = size;
            io_batch_params[j][i].opcode = CUFILE_READ;
        }
    }
    
    */

    //copy the data back to cpu and display it for checking
    /*
    for (j=0; j< NUM_BATCH; j++){
        CPUPtr[j] = (float*) malloc(size);
        for (k =0; k < batch_size; k++){
            cudaMemcpy(CPUPtr[j], devPtr[j][k], size, cudaMemcpyDeviceToHost);
        }
        
    }    
    
    */


    /*
    std::cout<< "GPU to CPU copy samples. " << std::endl;
    for (k =0; k < NUM_BATCH; k++){
        for (j=0; j<size/sizeof(float); j++){
            std::cout<<CPUPtr[k][j]<<" ";
        }
    }
        
    */

    /*
	errorBatch[0] = cuFileBatchIOSubmit(batch_id[0], batch_size, io_batch_params[0], flags);
    if(errorBatch[0].err != 0) {
        std::cerr << "Error in IO Batch Submit" << std::endl;
        goto out3;
    }    
    */