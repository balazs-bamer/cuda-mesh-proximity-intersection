#include "proximity-impl.cuh"
#include <cuda.h>
#include <stdio.h>

__device__ float gFactor;

__global__ void reallyDoFactor(float * const aFactor) {
  int32_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index == 0) {
    printf("reallyDoFactor in %f\n", *aFactor);
    *aFactor *= 2.0f;
    printf("reallyDoFactor out %f\n", *aFactor);
  }
}

__global__ void reallyDoIt(float const * const aTransferIn, float * const aTransferOut) {
  int32_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index == 0) {
    int32_t n = static_cast<int32_t>(aTransferIn[0]);
    float result = 0.0f;
    float const *pointer = aTransferIn + 1;
    while(n > 0) {
      result += *pointer;
      ++pointer;
      --n;
    }
    gFactor = 0.5005f;
    printf("reallyDoIt gFactor %f\n", gFactor);
    reallyDoFactor<<<1,1>>>(&gFactor);
    cudaDeviceSynchronize();
    printf("reallyDoIt pre result %f\n", result);
    *aTransferOut = result * gFactor;
    printf("reallyDoIt end result %f\n", *aTransferOut);
  }
}

float fragor::ProximityImpl::doIt() const {
  std::vector<float> serial;
  serial.reserve(mPara.size() + 1u);
  serial.push_back(static_cast<float>(mPara.size()));
  std::copy(mPara.begin(), mPara.end(), std::back_inserter(serial));
  float *transferIn;
  float *transferOut;
  size_t transferSize = sizeof(float) * serial.size();
  cudaMalloc(&transferIn, transferSize);
  cudaMalloc(&transferOut, sizeof(float));
  cudaMemcpy(transferIn, serial.data(), transferSize, cudaMemcpyHostToDevice);
  reallyDoIt<<<1,1>>>(transferIn, transferOut);
  float result;
  cudaMemcpy(transferOut, &result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(transferIn);
  cudaFree(transferOut);
  return result;
}

// Nsight does not support this feature with static library output. Need to develop here and after it create a CMakeList.
// https://stackoverflow.com/questions/22076052/cuda-dynamic-parallelism-makefile
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples
