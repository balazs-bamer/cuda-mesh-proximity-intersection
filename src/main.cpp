#include "proximity-api.h"
#include <iostream>
#include <numeric>

constexpr size_t   cgDataSize = 1u << 10u;
constexpr uint32_t cgIterations = 4u;

int main(int aArgc, char **aArgv) {
  std::vector<float> data(cgDataSize, 0.0f);
  std::iota(data.begin(), data.end(), 1.0f);
  fragor::Proximity proxy(data, cgIterations);
  //fragor::Proximity proxy({1.0f, 2.0f, 3.0f, 0.0f}, 2u);
  std::cout << proxy.doIt() << '\n';
  return 0;
}
