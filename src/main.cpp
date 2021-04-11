#include "proximity-api.h"
#include <iostream>

int main(int aArgc, char **aArgv) {
  fragor::Proximity proxy({1.1f, 2.2f, 3.3f});
  std::cout << proxy.doIt() << '\n';
  return 0;
}
