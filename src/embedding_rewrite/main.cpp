#include <string>
#include "npy_reader.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <half.hpp>

template <typename T>
T* createDynamicArray(int size) {
    return new T[size];
}

int main( int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: program <npy_file_path> <starting rows> <num rows>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    int num_rows, row_num;
    try {
        num_rows = std::stoi(argv[3]);
        row_num = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return 1;
    }

    std::string npy_file = argv[1];

    //std::string npy_file = "/home/grads/s/sls7161/Documents/cpp_kernels-/learn_cnpy/array.npy";

    meta npy_meta = read_npy_metadata(npy_file);

    int Ny = npy_meta.shape[1];

    off_t offset = npy_meta.dataStart + row_num*npy_meta.word_size*npy_meta.shape[1];

    int bytes_to_read = npy_meta.word_size*Ny*num_rows;

    //float* arr;

    //float* arr = new float[npy_meta.shape[0]*npy_meta.shape[1]];

    half_float::half* arr = new half_float::half[num_rows*Ny];
    //float* arr = new float[num_rows*Ny];

    ssize_t nread = pread(npy_meta.file, arr, bytes_to_read, offset);

    //print the loaded data
    
    std::cout<<"Data loaded:" <<std::endl;
    for (int i=0; i <num_rows; i++){
        for (int j =0; j< Ny; j++){
            std::cout<<arr[Ny * i + j]<<" ";
        }
        std::cout<<std::endl;
    }  
    
    delete arr;
    close(npy_meta.file);
}