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
  float    *mData;
  float     mResult;

public:
             __host__          Transfer() = default; // must be trivially constructible
             __host__          ~Transfer() = default;
             __host__ void     init(float *aPointer, std::vector<float> const &aPara);
  __device__ __host__ uint32_t size() const { return mSize; }
  __device__          void     compute();
  __device__ __host__ float    getResult() { return mResult; }
};

class ProximityImpl final {
private:
  Init                                        mInit;
  cuda::memory::managed::unique_ptr<float[]>  mData;
  cuda::memory::managed::unique_ptr<Transfer> mTransfer;
  uint32_t const                              cmIterations;

public:
  ProximityImpl(std::vector<float> const &aPara, uint32_t const aIterations)
  : mInit()
  , mData(cuda::memory::managed::make_unique<float[]>(aPara.size()))
  , mTransfer(cuda::memory::managed::make_unique<Transfer>())
  , cmIterations(aIterations) {
    mTransfer->init(mData.get(), aPara);
  }

  ~ProximityImpl() = default;

  float doIt() const;
};

}

#endif /* FRAGOR_PROXIMITY_IMPL_CUH_ */
