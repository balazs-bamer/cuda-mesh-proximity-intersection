#include "proximity-impl.cuh"
#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <iostream>
//#include <cudaProfiler.h>

__host__ void fragor::Transfer::init(float *aPointerHost, float *aPointerDevice, std::vector<float> const &aPara) {
  mSize = aPara.size();
  mDataHost = aPointerHost;
  mDataDevice = aPointerDevice;
  std::copy(aPara.begin(), aPara.end(), mDataHost);
}

__global__ void updateFactor(fragor::Transfer *aTransfer) {
  int32_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index == 0) {
//    printf("reallyDoFactor in %f\n", *aFactor);
    aTransfer->adjust(1.0f);
  //  printf("reallyDoFactor out %f\n", *aFactor);
  }
}

__device__ void fragor::Transfer::compute() {
  uint32_t const threadIndex = threadIdx.x;
  uint32_t actualThreadCount = blockDim.x;
  uint32_t actualStepSize    = 1u;
  while(actualThreadCount > 0u) {
    if(threadIndex < actualThreadCount) {
      uint32_t baseIndex = threadIndex * actualStepSize;
      mDataDevice[baseIndex] += mDataDevice[baseIndex + actualStepSize];
    }
    else { // nothing to do
    }
    actualStepSize <<= 1u;
    actualThreadCount >>= 1u;
  }
  if(threadIndex == 0u) {
    //printf("reallyDoIt gFactor %f\n", gFactor);
    updateFactor<<<1,1>>>(this);
    cudaDeviceSynchronize();            // cuda::device::current::get().synchronize(); does not work on device
//    printf("reallyDoIt pre result %f\n", result);
    mResult = mDataDevice[0] * mFactor;
//    printf("reallyDoIt end result %f\n", mResult);
  }
  else { // nothing to do
  }
}

__global__ void compute(fragor::Transfer *aTransfer) {
  aTransfer->compute();
}

float fragor::ProximityImpl::doIt() const {
  float result = 0.0f;
  std::cout << "SMs: " << mDevice.properties().multiProcessorCount << '\n';
  for(uint32_t i = 0; i < cmIterations; ++i) {
    cuda::launch(compute, cuda::launch_configuration_t( 1, mTransferHost->size() / 2u ), static_cast<Transfer*>(mTransfer.device_side));
    cuda::device::current::get().synchronize();
    result += mTransferHost->getResult();         // Result will be invalid, because the kernel corrupts its input, but don't care now.
  }
  return result;
}

// nvprof fails to profile unified memory, so I stuck with mapped.
// Nsight System is incompatible with my CUDA driver, although it is newer tha the minimum supported.
// Nsight does not support this feature with static library output. Need to develop here and after it create a CMakeList.
// https://stackoverflow.com/questions/22076052/cuda-dynamic-parallelism-makefile
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples
