/*
 * proximity-impl.cuh
 *
 *  Created on: 2021 ápr. 5
 *      Author: balazs
 */

#include <deque>
#include <vector>

#ifndef FRAGOR_PROXIMITY_IMPL_CUH_
#define FRAGOR_PROXIMITY_IMPL_CUH_

namespace fragor {

class ProximityImpl final {
private:
  std::deque<float> mPara;

public:
  ProximityImpl(std::initializer_list<float> const aPara) : mPara(aPara) {  }
  ~ProximityImpl() = default;

  float doIt() const;
};

}

#endif /* FRAGOR_PROXIMITY_IMPL_CUH_ */
