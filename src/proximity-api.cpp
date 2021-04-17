#include "proximity-api.h"
#include "proximity-impl.cuh"


fragor::Proximity::Proximity(std::vector<float> const &aPara, uint32_t const aIterations)
: mImpl{ std::make_unique<fragor::ProximityImpl>(aPara, aIterations)} {
}

fragor::Proximity::~Proximity() = default;

float fragor::Proximity::doIt() const {
  return mImpl->doIt();
}
