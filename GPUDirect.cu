#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cufile.h"

#include <cufile_sample_utils.h>

#include <cuda_fp16.h>

#include <iostream>
#include <fcntl.h>

//#include <pybind11/numpy.h>
#include <cmath>
#include "Python.h"

#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 128

__global__ void parseBuffer(at::Half ***devPtr,  at::Half* CLS_MAT, at::Half* BOW_MAT, int NUM_BATCH, int batch_size, int cls_size, int bow_size, int num_bow_vectors) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (batch_idx < NUM_BATCH && thread_idx < batch_size) {
        at::Half* buffer = (at::Half*)devPtr[batch_idx][thread_idx];      
        at::Half* cls_vector = buffer;
        at::Half* bow_matrix = buffer + cls_size;

        // Store CLS vector in CLS_MAT
        int cls_offset = batch_idx * batch_size * cls_size + thread_idx * cls_size;
        for (int i = 0; i < cls_size; i++) {
            CLS_MAT[cls_offset + i] = cls_vector[i];
        }

        // Store BOW vector in BOW_MAT
        int bow_offset = (batch_idx * batch_size + thread_idx) * num_bow_vectors * bow_size;
        for (int i = 0; i < num_bow_vectors; i++) {
            for (int j = 0; j < bow_size; j++) {
                BOW_MAT[bow_offset + i * bow_size + j] = bow_matrix[i * bow_size + j];
            }
        }
    }
}

__global__ void parseMixedBuffer(at::Half ***devPtr, at::Half* CLS_MAT, at::Half* BOW_MAT, int* io_size, int NUM_BATCH, int batch_size, int cls_size, int bow_size, int num_bow_vectors) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (batch_idx < NUM_BATCH && thread_idx < batch_size) {
        int io_idx = batch_idx*batch_size + thread_idx;

        at::Half* buffer = (at::Half*)devPtr[batch_idx][thread_idx];      
        at::Half* cls_vector = buffer;
        at::Half* bow_matrix = buffer + cls_size;

        // Store CLS vector in CLS_MAT
        int cls_offset = batch_idx * batch_size * cls_size + thread_idx * cls_size;
        for (int i = 0; i < cls_size; i++) {
            CLS_MAT[cls_offset + i] = cls_vector[i];
        }

        // Store BOW vector in BOW_MAT
        int bow_offset = (batch_idx * batch_size + thread_idx) * num_bow_vectors * bow_size;
        int bow_vec_needed;
        if (io_size[io_idx] == 0){
            bow_vec_needed = num_bow_vectors;
        }
        else{
            bow_vec_needed = 60;
        }
        for (int i = 0; i < bow_vec_needed; i++) {
            for (int j = 0; j < bow_size; j++) {
                BOW_MAT[bow_offset + i * bow_size + j] = bow_matrix[i * bow_size + j];
            }
        }
    }
}

class GPUDirect_IO{
    public:
    int NUM_BATCH; 
    int batch_size; 
    int ** fd;
    at::Half ***devPtr;

    int batch_offset;
    unsigned nr;
    size_t size;
    
    CUfileDescr_t **cf_descr;
    CUfileHandle_t **cf_handle;
    CUfileIOParams_t **io_batch_params;
    CUfileBatchHandle_t *batch_id;
    CUfileIOEvents_t io_batch_events[MAX_BATCH_IOS];
    
    std::string TESTFILE;
    CUfileError_t status;
    CUfileError_t errorBatch;


    //Embedding dimensions and size
    int cls_size;
    int bow_size;
    int num_bow_vectors;
    int BATCHES_USED;

    //constructor
    GPUDirect_IO(int gpuid, std::string FILE, size_t size, int batch_size, int NUM_BATCHES, int _cls_size, int _bow_size);

    //Destructor
    ~GPUDirect_IO();

    //Member functions
    void _prepare_batches();

    //void read(std::vector<int>& offsets, half* CLS_MAT, half* BOW_MAT);
    void read(torch::Tensor offsets, torch::Tensor CLS_MAT, torch::Tensor BOW_MAT);
    void read_mixed(torch::Tensor offsets, torch::Tensor io_size, torch::Tensor CLS_MAT, torch::Tensor BOW_MAT);

    void out1();
    void out2();
    void out3();
    void out4();
    void _reset_batch();
    void _reset_buffer();
    void _close();

};

GPUDirect_IO::GPUDirect_IO(int gpuid, std::string FILE, size_t _size, int _batch_size, int NUM_BATCHES, int _cls_size, int _bow_size){
    // constructor code
    NUM_BATCH = NUM_BATCHES;    //Max number of batches
    TESTFILE= FILE;             // File to read
    size = _size;               // size of io
    batch_size= _batch_size;    // number of io per batch

    cls_size = _cls_size;
    bow_size = _bow_size;
    num_bow_vectors = (_size - cls_size*2)/(_bow_size*2);
    BATCHES_USED = NUM_BATCHES;
    //std::cout << "Number of BOW vectors required = " << num_bow_vectors << std::endl;


    cf_descr = (CUfileDescr_t**) malloc(NUM_BATCHES * sizeof(CUfileDescr_t *));
    cf_handle = (CUfileHandle_t**) malloc(NUM_BATCHES * sizeof(CUfileHandle_t*));
    io_batch_params = (CUfileIOParams_t**) malloc(NUM_BATCHES * sizeof(CUfileIOParams_t*));
    fd = (int **) malloc(NUM_BATCHES * sizeof(int*));
    
    //devPtr = (half***)malloc(NUM_BATCH * sizeof(half**));
    cudaMallocManaged(&devPtr, NUM_BATCH * sizeof(at::Half**));
    batch_id = (CUfileBatchHandle_t*) malloc(NUM_BATCH * sizeof(CUfileBatchHandle_t));
    //streams = (cudaStream_t*)malloc(NUM_BATCH * sizeof(cudaStream_t));
    //events = (cudaEvent_t*)malloc(NUM_BATCH * sizeof(cudaEvent_t));
    batch_offset = MAX_BUFFER_SIZE*MAX_BATCH_IOS;

    for ( int j = 0; j < NUM_BATCH; j++){
        fd[j] = (int *) malloc(sizeof(int)* MAX_BATCH_IOS);
        
        //devPtr[j] = (half **) malloc(sizeof(half*)* MAX_BATCH_IOS);
        cudaMallocManaged(&devPtr[j], sizeof(at::Half*)* MAX_BATCH_IOS);
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
            fd[j][i] = open(TESTFILE.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
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
            io_batch_params[j][i].u.batch.devPtr_base = devPtr[j][i];
            io_batch_params[j][i].u.batch.devPtr_offset = 0;
            io_batch_params[j][i].u.batch.file_offset = 0; 
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

void GPUDirect_IO::read(torch::Tensor offsets, torch::Tensor CLS_MAT, torch::Tensor BOW_MAT){

    // Release the GIL within this block
    {
        Py_BEGIN_ALLOW_THREADS
        long num_io = offsets.size(0);
        int count = 0;

        //std::cout << "Reading " << num_io << " document embeddings" <<std::endl;

        double num = static_cast<double>(num_io)/batch_size;
        BATCHES_USED= static_cast<int>(std::ceil(num));
        //std::cout << "Number of batches required = " << _num_batches_required << std::endl;
        
        
        //BATCHES_USED = NUM_BATCH;
        //int count = 0;

        // Access the tensor data and print its values
        auto offset_data = offsets.accessor<long, 1>();

        // set file offset for batch reads


        for (int j =0; j < BATCHES_USED; j++){
            for(int i = 0; i < batch_size; i++) {
                if (count < num_io){
                    io_batch_params[j][i].u.batch.file_offset = offset_data[j*batch_size + i];
                }
                else{
                    io_batch_params[j][i].u.batch.file_offset = offset_data[0];
                }
                count++;
            }
        }

        //submit batch io 
        for (int j = 0; j<BATCHES_USED; j++){
            errorBatch = cuFileBatchIOSubmit(batch_id[j], batch_size, io_batch_params[j], 0);
            if(errorBatch.err != 0) {
                std::cerr << "Error in IO Batch Submit" << std::endl;
                out3();
            }
        }

        //wait for all data to arrive
        int num_completed = 0;
        for (int j = 0; j < BATCHES_USED; j++){
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
        at::Half* cls_ptr = CLS_MAT.data_ptr<at::Half>();
        at::Half* bow_ptr = BOW_MAT.data_ptr<at::Half>();
        parseBuffer<<<NUM_BATCH, batch_size>>>(devPtr, cls_ptr, bow_ptr, BATCHES_USED, batch_size, cls_size, bow_size, num_bow_vectors);
        Py_END_ALLOW_THREADS
    }


}

void GPUDirect_IO::read_mixed(torch::Tensor offsets, torch::Tensor io_size, torch::Tensor CLS_MAT, torch::Tensor BOW_MAT){

    // Release the lock
    {
    py::gil_scoped_release release;
    long num_io = offsets.size(0);
    int count = 0;
    double num = static_cast<double>(num_io)/batch_size;
    BATCHES_USED= static_cast<int>(std::ceil(num));
    auto offset_data = offsets.accessor<long, 1>();

    // set file offset for batch reads
    for (int j =0; j < BATCHES_USED; j++){
        for(int i = 0; i < batch_size; i++) {
            if (count < num_io){
                io_batch_params[j][i].u.batch.file_offset = offset_data[j*batch_size + i];
            }
            else{
                io_batch_params[j][i].u.batch.file_offset = offset_data[0];
            }
            count++;
        }
    }

    //submit batch io 
    for (int j = 0; j<BATCHES_USED; j++){
        errorBatch = cuFileBatchIOSubmit(batch_id[j], batch_size, io_batch_params[j], 0);
        if(errorBatch.err != 0) {
            std::cerr << "Error in IO Batch Submit" << std::endl;
            out3();
	    }
    }

    //wait for all data to arrive
    int num_completed = 0;
    for (int j = 0; j < BATCHES_USED; j++){
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
    at::Half* cls_ptr = CLS_MAT.data_ptr<at::Half>();
    at::Half* bow_ptr = BOW_MAT.data_ptr<at::Half>();
    int* io_size_ptr = io_size.data_ptr<int>(); 
    parseMixedBuffer<<<NUM_BATCH, batch_size>>>(devPtr, cls_ptr, bow_ptr, io_size_ptr, BATCHES_USED, batch_size, cls_size, bow_size, num_bow_vectors);
    }
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

}

void GPUDirect_IO::_reset_buffer(){
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

void ClearTensor(torch::Tensor tensor) {

    // Access the underlying CUDA pointer
    at::Half* data_ptr = tensor.data_ptr<at::Half>();

    // Fill the tensor with values (example: fill with 1.0)
    int numel = tensor.numel();
    cudaMemset(data_ptr, 0, sizeof(float) * numel);
}

void printTensor(torch::Tensor tensor) {

    // Get the tensor size
    long size = tensor.size(0);

    // Access the tensor data and print its values
    auto data = tensor.accessor<long, 1>();
    for (int i = 0; i < size; ++i) {
        long value = data[i];
        printf("%ld ", value);
    }
    printf("\n");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ClearTensor", &ClearTensor, "Fill tensor with all 0");
    py::class_<GPUDirect_IO>(m, "GPUDirect_IO"
                                    )
                                    .def(py::init<int, std::string, size_t, int, int, int, int>())
                                    .def("_prepare_batches", &GPUDirect_IO::_prepare_batches)
                                    .def("read", &GPUDirect_IO::read)
                                    .def("read_mixed", &GPUDirect_IO::read_mixed)
                                    .def("_reset_batch", &GPUDirect_IO::_reset_batch)
                                    .def("_reset_buffer", &GPUDirect_IO::_reset_buffer)
                                    .def("_close", &GPUDirect_IO::_close)
                                    ;
}