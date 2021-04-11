/*
 * proximity-api.h
 *
 *  Created on: 2021 ápr. 5
 *      Author: balazs
 */

#include <memory>

#ifndef FRAGOR_PROXIMITY_API_H_
#define FRAGOR_PROXIMITY_API_H_

namespace fragor {

class ProximityImpl;

class Proximity final {
private:
  std::unique_ptr<ProximityImpl> mImpl;

public:
  Proximity(std::initializer_list<float> const aPara);
  ~Proximity();

  float doIt() const;
};

}

#endif /* FRAGOR_PROXIMITY_API_H_ */
