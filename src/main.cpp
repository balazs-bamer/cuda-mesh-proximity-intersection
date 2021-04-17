#include "proximity-api.h"
#include <iostream>

int main(int aArgc, char **aArgv) {
  fragor::Proximity proxy({1.0f, 2.0f, 3.0f}, 10u);
  std::cout << proxy.doIt() << '\n';
  return 0;
}
