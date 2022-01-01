/*
 * proximity-impl.cuh
 *
 *  Created on: 2021 ápr. 5
 *      Author: balazs
 */

#include "cuda/runtime_api.hpp"
#include <deque>
#include <vector>

#ifndef FRAGOR_PROXIMITY_IMPL_CUH_
#define FRAGOR_PROXIMITY_IMPL_CUH_

namespace fragor {

class Init final {
public:
  Init() {
    if (cuda::device::count() == 0) {
      throw std::runtime_error("No appropriate NVidia card.");
    }
    else { // nothing to do
    }
  }

  ~Init() = default;
};

class Transfer final {
private:
  uint32_t  mSize;
  float    *mDataHost;
  float    *mDataDevice;
  float     mFactor;
  float     mResult;

public:
             __host__          Transfer() = default; // must be trivially constructible
             __host__          ~Transfer() = default;
             __host__ void     init(float *aPointerHost, float *aPointerDevice, std::vector<float> const &aPara);
  __device__ __host__ uint32_t size() const { return mSize; }
  __device__          void     adjust(float const aIncrement) { mFactor += aIncrement; }
  __device__          void     compute();
  __device__ __host__ float    getResult() { return mResult; }
};

class ProximityImpl final {
private:
  Init                              mInit;
  cuda::device_t                    mDevice;
  cuda::memory::mapped::region_pair mData;
  cuda::memory::mapped::region_pair mTransfer;
  Transfer * const                  mTransferHost;
  uint32_t const                    cmIterations;

public:
  ProximityImpl(std::vector<float> const &aPara, uint32_t const aIterations)
  : mInit()
  , mDevice(cuda::device::current::get())
  , mData(cuda::memory::mapped::allocate(mDevice, aPara.size() * sizeof(float), cuda::memory::cpu_write_combining::with_wc))
  , mTransfer(cuda::memory::mapped::allocate(mDevice, sizeof(Transfer), cuda::memory::cpu_write_combining::without_wc))
  , mTransferHost(static_cast<Transfer*>(mTransfer.host_side))
  , cmIterations(aIterations) {
    mTransferHost->init(static_cast<float*>(mData.host_side), static_cast<float*>(mData.device_side), aPara);
  }

  ~ProximityImpl() {
    cuda::memory::mapped::free(mData);
    cuda::memory::mapped::free(mTransfer);
  }

  float doIt() const;
};

}

#endif /* FRAGOR_PROXIMITY_IMPL_CUH_ */
