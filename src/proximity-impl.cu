#include "proximity-impl.cuh"
#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>

__host__ void fragor::Transfer::init(float *aPointer, std::vector<float> const &aPara) {
  mSize = aPara.size();
  mData = aPointer;
  std::copy(aPara.begin(), aPara.end(), mData);
}

__device__ float gFactor = 1.0f;

__global__ void reallyDoFactor(float * const aFactor) {
  int32_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index == 0) {
    printf("reallyDoFactor in %f\n", *aFactor);
    *aFactor += 1.0f;
    printf("reallyDoFactor out %f\n", *aFactor);
  }
}

__device__ void fragor::Transfer::compute() {
  int32_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index == 0) {
    float result = 0.0f;
    for(uint32_t i = 0; i < mSize; ++i) {
      result += mData[i];
    }
    printf("reallyDoIt gFactor %f\n", gFactor);
    reallyDoFactor<<<1,1>>>(&gFactor);
    cudaDeviceSynchronize();            // cuda::device::current::get().synchronize(); does not work on device
    printf("reallyDoIt pre result %f\n", result);
    mResult = result * gFactor;
    printf("reallyDoIt end result %f\n", mResult);
  }
}

__global__ void reallyDoIt(fragor::Transfer *aTransfer) {
  aTransfer->compute();
}

float fragor::ProximityImpl::doIt() const {
  float result = 0.0f;
  for(uint32_t i = 0; i < cmIterations; ++i) {
    cuda::launch(reallyDoIt, cuda::launch_configuration_t( 1, 1 ), mTransfer.get());
    cuda::device::current::get().synchronize();
    result += mTransfer->getResult();
  }
  return result;
}

// Nsight does not support this feature with static library output. Need to develop here and after it create a CMakeList.
// https://stackoverflow.com/questions/22076052/cuda-dynamic-parallelism-makefile
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples
