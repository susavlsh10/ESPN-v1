#include <cuda_runtime.h>
#include "gpu_kernels.h"

__global__ void EmbeddingParse(half*** devPtr, half* CLS_MAT, half* BOW_MAT){
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    // Parse CLS vector
    half* cls_vector = (half*)devPtr[batch_idx][element_idx];
    for (int i = 0; i < 128; i++) {
        CLS_MAT[batch_idx * blockDim.x + element_idx][i] = cls_vector[i];
    }

    // Parse BOW matrix
    half* bow_matrix = (half*)devPtr[batch_idx][element_idx] + 128;
    for (int i = 0; i < 60; i++) {
        for (int j = 0; j < 32; j++) {
            BOW_MAT[batch_idx * blockDim.x + element_idx][i][j] = bow_matrix[i * 32 + j];
        }
    }
}

