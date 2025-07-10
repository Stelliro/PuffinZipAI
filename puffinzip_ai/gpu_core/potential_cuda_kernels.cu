#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h> // For strlen, memcpy in host example
#include <stdlib.h> // For malloc, free in host example

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void countCharInSegmentsKernel(const char* inputData, int* outputCounts, int dataLength, int segmentSize, char targetChar) {
    extern __shared__ int s_charCounts[];

    int tid_in_block = threadIdx.x;
    int block_id = blockIdx.x;
    int threads_per_block = blockDim.x;

    s_charCounts[tid_in_block] = 0;

    int global_char_idx_start = block_id * segmentSize + tid_in_block;
    int stride = threads_per_block;

    for (int i = global_char_idx_start; i < segmentSize * (block_id + 1) && i < dataLength; i += stride) {
        if (inputData[i] == targetChar) {
            s_charCounts[tid_in_block]++;
        }
    }
    __syncthreads();

    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            s_charCounts[tid_in_block] += s_charCounts[tid_in_block + s];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        outputCounts[block_id] = s_charCounts[0];
    }
}

__global__ void rleCompressBlockKernel(const char* inputBlockData, int inputBlockSize,
                                      unsigned char* rleOutputBufferForBlock, int* actualRleOutputSizeForBlock,
                                      int maxRleOutputPerBlock) {
    
    extern __shared__ char s_data[]; 
    
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    for (int i = tid; i < inputBlockSize; i += block_threads) {
        s_data[i] = inputBlockData[i];
    }
    __syncthreads();

    if (tid == 0) {
        int currentRleIdx = 0;
        if (inputBlockSize == 0) {
            *actualRleOutputSizeForBlock = 0;
            return;
        }

        char currentChar = s_data[0];
        unsigned char count = 1;

        for (int i = 1; i < inputBlockSize; ++i) {
            if (s_data[i] == currentChar) {
                count++;
                if (count == 255) { 
                    if (currentRleIdx + 2 > maxRleOutputPerBlock) { break; }
                    rleOutputBufferForBlock[currentRleIdx++] = count;
                    rleOutputBufferForBlock[currentRleIdx++] = (unsigned char)currentChar;
                    count = 0; 
                    if (i + 1 < inputBlockSize) { 
                        currentChar = s_data[i + 1]; 
                    } else if (count > 0) { 
                        
                    }
                }
            } else {
                if (count > 0) {
                    if (currentRleIdx + 2 > maxRleOutputPerBlock) { break; }
                    rleOutputBufferForBlock[currentRleIdx++] = count;
                    rleOutputBufferForBlock[currentRleIdx++] = (unsigned char)currentChar;
                }
                currentChar = s_data[i];
                count = 1;
            }
        }
        
        if (count > 0) {
            if (currentRleIdx + 2 <= maxRleOutputPerBlock) {
                 rleOutputBufferForBlock[currentRleIdx++] = count;
                 rleOutputBufferForBlock[currentRleIdx++] = (unsigned char)currentChar;
            }
        }
        *actualRleOutputSizeForBlock = currentRleIdx;
    }
}


#ifdef COMPILE_STANDALONE_CUDA_TEST_MAIN
void runVectorAddExample() {
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) { printf("Host malloc failed for VectorAdd.\n"); return; }

    for (int i = 0; i < N; ++i) { h_A[i] = (float)i; h_B[i] = (float)(i * 2); }

    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, size); if (err != cudaSuccess) { printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); free(h_A); free(h_B); free(h_C); return; }
    err = cudaMalloc(&d_B, size); if (err != cudaSuccess) { printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); free(h_A); free(h_B); free(h_C); return; }
    err = cudaMalloc(&d_C, size); if (err != cudaSuccess) { printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); cudaFree(d_B); free(h_A); free(h_B); free(h_C); return; }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("VectorAddKernel example sequence complete. First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
         printf("  h_C[%d] = %f (Expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

void runCharCountExample() {
    const char* test_data_host = "AAAAABBBCCCCCDDDDDAAAAAEEEEEOOOOAAAABBB";
    int data_len = strlen(test_data_host);
    if (data_len == 0) { printf("CharCount: Empty test data.\n"); return; }

    int segment_processing_size = 8; 
    int num_segments = (data_len + segment_processing_size - 1) / segment_processing_size;
    char target_char_to_count = 'A';

    char* d_input_data_chars;
    int* d_output_segment_counts;
    cudaMalloc(&d_input_data_chars, data_len * sizeof(char));
    cudaMalloc(&d_output_segment_counts, num_segments * sizeof(int));
    cudaMemcpy(d_input_data_chars, test_data_host, data_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_output_segment_counts, 0, num_segments * sizeof(int));


    int threadsPerBlockForCharCount = 128; 
    size_t dynamicSharedMemBytes = threadsPerBlockForCharCount * sizeof(int);

    countCharInSegmentsKernel<<<num_segments, threadsPerBlockForCharCount, dynamicSharedMemBytes>>>(
        d_input_data_chars, d_output_segment_counts, data_len, segment_processing_size, target_char_to_count);
    cudaDeviceSynchronize();

    int* h_output_segment_counts = (int*)malloc(num_segments * sizeof(int));
    if (!h_output_segment_counts) { printf("Host malloc failed for char counts.\n"); cudaFree(d_input_data_chars); cudaFree(d_output_segment_counts); return; }
    cudaMemcpy(h_output_segment_counts, d_output_segment_counts, num_segments * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nCharacter '%c' counts per %d-char segment from '%s':\n", target_char_to_count, segment_processing_size, test_data_host);
    for(int i=0; i < num_segments; ++i) {
        printf("  Segment %d: %d\n", i, h_output_segment_counts[i]);
    }

    cudaFree(d_input_data_chars);
    cudaFree(d_output_segment_counts);
    free(h_output_segment_counts);
    printf("CountCharInSegmentsKernel example sequence complete.\n");
}

void runRleCompressBlockExample() {
    const char* rle_test_data_host = "AAABBCDDDDEFFFFFGHIJJJJAAA";
    int rle_data_len = strlen(rle_test_data_host);
    if (rle_data_len == 0) { printf("RLE Compress: Empty test data.\n"); return; }

    int blockSizeForRleKernel = rle_data_len; 
    int numBlocksForRle = 1; 

    char* d_rle_input_block;
    unsigned char* d_rle_output_buffer_block;
    int* d_rle_output_size_block;
    
    int maxOutputBufferPerBlock = blockSizeForRleKernel * 2; 

    cudaMalloc(&d_rle_input_block, blockSizeForRleKernel * sizeof(char));
    cudaMalloc(&d_rle_output_buffer_block, maxOutputBufferPerBlock * sizeof(unsigned char));
    cudaMalloc(&d_rle_output_size_block, sizeof(int));

    cudaMemcpy(d_rle_input_block, rle_test_data_host, blockSizeForRleKernel * sizeof(char), cudaMemcpyHostToDevice);
    
    int threadsPerBlockRle = 256; 
    size_t dynamicSharedMemBytesRle = blockSizeForRleKernel * sizeof(char); 

    rleCompressBlockKernel<<<numBlocksForRle, threadsPerBlockRle, dynamicSharedMemBytesRle>>>(
        d_rle_input_block, blockSizeForRleKernel, 
        d_rle_output_buffer_block, d_rle_output_size_block,
        maxOutputBufferPerBlock
    );
    cudaDeviceSynchronize();

    int h_rle_output_size_for_block;
    cudaMemcpy(&h_rle_output_size_for_block, d_rle_output_size_block, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_rle_output_size_for_block > 0) {
        unsigned char* h_rle_output_buffer = (unsigned char*)malloc(h_rle_output_size_for_block * sizeof(unsigned char));
        if (!h_rle_output_buffer) { printf("Host malloc failed for RLE output.\n"); /* Free GPU mem */ return; }
        cudaMemcpy(h_rle_output_buffer, d_rle_output_buffer_block, h_rle_output_size_for_block * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        printf("\nRLE Compression (Block Kernel) for '%s':\n", rle_test_data_host);
        printf("  Compressed Size: %d bytes\n", h_rle_output_size_for_block);
        printf("  Compressed Data (count,char pairs): ");
        for (int k = 0; k < h_rle_output_size_for_block; k += 2) {
            if (k + 1 < h_rle_output_size_for_block) {
                printf("%d%c ", (int)h_rle_output_buffer[k], (char)h_rle_output_buffer[k+1]);
            }
        }
        printf("\n");
        free(h_rle_output_buffer);
    } else {
         printf("\nRLE Compression (Block Kernel) for '%s' resulted in empty or error output.\n", rle_test_data_host);
    }

    cudaFree(d_rle_input_block);
    cudaFree(d_rle_output_buffer_block);
    cudaFree(d_rle_output_size_block);
    printf("RleCompressBlockKernel example sequence complete.\n");
}


int main() {
    runVectorAddExample();
    runCharCountExample();
    runRleCompressBlockExample();
    return 0;
}
#endif