#include "proximity-api.h"
#include "proximity-impl.cuh"


fragor::Proximity::Proximity(std::initializer_list<float> const aPara)
: mImpl{ std::make_unique<fragor::ProximityImpl>(aPara)} {
}

fragor::Proximity::~Proximity() = default;

float fragor::Proximity::doIt() const {
  return mImpl->doIt();
}
