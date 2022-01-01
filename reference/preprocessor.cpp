#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int32_t

#include "../../../repos/stl_reader/stl_reader.h"
#include <map>
#include <array>
#include <deque>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <Eigen/Dense>

using Vertex = Eigen::Vector3f;
using Transform = Eigen::Transform<float, 3, Eigen::Affine>;
using Triangle = std::array<Vertex, 3u>;
using CudaTriangle = Vertex*;
using CudaConstTriangle = Vertex const*;
using Mesh = std::vector<Triangle>;

// usage ./IntersectionReference example23.in 

constexpr float    cgPi                            =    3.1415926539f;
constexpr int32_t  cgSphereSectors                 =    11;
constexpr int32_t  cgSphereBelts                   =    5;
constexpr float    cgMaxSphereRadius               =   17.0f;
constexpr float    cgMaxTriangleSide               = cgMaxSphereRadius * 2.0f;
constexpr float    cgCalculateSpheresInflate       =    0.51f;
constexpr int32_t  cgCalculateSpheresNeighbours    =    3;
constexpr float    cgApproximateLeaveInPlaceFactor =    0.01f;
constexpr uint32_t cgApproximateResultSize         =   64u;       // TODO remove and use total surface to compute this value T = sqrt(s*(s-c)*(s-b)*(s-a))
constexpr int32_t  cgApproximateIterations         = 4444u;       // TODO perhaps depends on the result size
constexpr float    cgApproximateTemperatureFactor  =    0.001f;

Eigen::Matrix3f randomTransform() {
  std::default_random_engine generator;
  generator.seed((std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::time_point::min()).count());
  std::uniform_real_distribution<float> distribution(0.1f, 1.0f);
  Eigen::Matrix3f result;
  for(int32_t i = 0; i < 9; ++i) {
    result(i / 3, i % 3) = distribution(generator);
  }
  return result;
}

std::deque<Triangle> divideLargeTriangles(std::deque<Triangle> &aMesh) {
  std::deque<Triangle> result;
  for(auto const &triangle : aMesh) {
    float maxSide = (triangle[0] - triangle[1]).norm();
    maxSide = std::max(maxSide, (triangle[0] - triangle[2]).norm());
    maxSide = std::max(maxSide, (triangle[1] - triangle[2]).norm());
    int32_t divisor = static_cast<int32_t>(std::ceil(maxSide / cgMaxTriangleSide));
    auto vector01 = (triangle[1] - triangle[0]) / divisor;
    auto vector02 = (triangle[2] - triangle[0]) / divisor;
    auto lineBase = triangle[0];
    auto base0 = lineBase;
    auto base1 = (divisor > 1) ? (base0 + vector01) : triangle[1];
    auto base2 = (divisor > 1) ? (base0 + vector02) : triangle[2];
    for(int i = 0; i < divisor - 1; ++i) {
      for(int j = 0; j < divisor - i - 1; j++) {
        result.push_back({base0, base1, base2});
        auto base1next = base1 + vector02;
        result.push_back({base1, base1next, base2});
        base1 = base1next;
        base0 = base2;
        base2 += vector02;
      }
      result.push_back({base0, base1, base2});
      lineBase += vector01;
      base0 = lineBase;
      base1 = base0 + vector01;
      base2 = base0 + vector02;
    }
    result.push_back({base0, triangle[1], base2});
  }
  return result;
}

auto readMesh(std::string const &aFilename, Eigen::Vector3f aTranslation, Eigen::Matrix3f const &aTransform) {
  std::deque<Triangle> work;
  stl_reader::StlMesh<float, int32_t> mesh(aFilename);
  for(int32_t indexTriangle = 0; indexTriangle < mesh.num_tris(); ++indexTriangle) {
    Triangle triangle;
    for(int32_t indexCorner = 0; indexCorner < 3; ++indexCorner) {
      float const * const coords = mesh.tri_corner_coords(indexTriangle, indexCorner);
      Eigen::Vector3f in;
      for(int32_t i = 0; i < 3; ++i) {
        in(i) = coords[i];
      }
      triangle[indexCorner] = /*aTransform **/ (in + aTranslation);
    }
    work.push_back(triangle);
  }
  std::cout << "bef: " << work.size();
  work = divideLargeTriangles(work);
  std::cout << " aft: " << work.size() << '\n';
  return Mesh(work.cbegin(), work.cend());
}

auto readMesh(char const * const aFilename, Eigen::Matrix3f const &aTransform) {
  std::ifstream in(aFilename);
  std::string filename1, filename2;
  Eigen::Vector3f translation;
  in >> filename1 >> filename2 >> translation(0) >> translation(1) >> translation(2);
  std::cout << filename1 << '\n' << filename2 << '\n';
  return std::pair(readMesh(filename1, {0.0f, 0.0f, 0.0f}, aTransform), readMesh(filename2, translation, aTransform));
}

void writeMesh(Mesh const &aMesh1, Mesh const &aMesh2, char const * const aFilename) {
  std::ofstream out(aFilename);
  out << "solid Exported from Blender-2.82 (sub 7)\n";
  for(auto const & triangle : aMesh1) {
    out << "facet normal 0.000000 0.000000 0.000000\nouter loop\n";
    for(auto const & vertex : triangle) {
      out << "vertex " << vertex(0) << ' ' << vertex(1) << ' ' << vertex(2) << '\n';
    }
    out << "endloop\nendfacet\n";
  }
  for(auto const & triangle : aMesh2) {
    out << "facet normal 0.000000 0.000000 0.000000\nouter loop\n";
    for(auto const & vertex : triangle) {
      out << "vertex " << vertex(0) << ' ' << vertex(1) << ' ' << vertex(2) << '\n';
    }
    out << "endloop\nendfacet\n";
  }
  out << "endsolid Exported from Blender-2.82 (sub 7)\n";
}

float calculateDistanceSum(std::unordered_set<uint32_t> const &aIndices, std::deque<Vertex> const &aVertices) {
  std::vector<Vertex> selection;
  selection.reserve(cgApproximateResultSize);
  for(auto const &i : aIndices) {
    selection.push_back(aVertices[i]);
  }
  float sum;
  for(uint32_t i = 0u; i < cgApproximateResultSize; ++i) {
    for(uint32_t j = 0u; j < i; ++j) {
      auto diffSquared = (selection[i] - selection[j]).squaredNorm();
      sum += -1.0f / diffSquared;
    }
  }
  return sum;
}

std::vector<Vertex> approximate(Mesh const aMesh, float const aMedianSideSizeHarmonic) {
  std::deque<Vertex> all;
  std::deque<Vertex> reference;
  float limit = aMedianSideSizeHarmonic * cgApproximateLeaveInPlaceFactor;
  for(auto const &triangle : aMesh) {
    for(auto const &vertex : triangle) {
      bool was = false;
      for(auto const &ref : reference) {
        if((ref - vertex).norm() < limit) {
          was = true;
          break;
        }
        else { // nothing to do
        }
      }
      if(!was) {
        reference.push_back(vertex);
      }
      else { // nothing to do
      }
    }
  }
  std::default_random_engine generator;
  generator.seed((std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::time_point::min()).count());
  std::uniform_int_distribution<uint32_t> distributionAll(0, reference.size() - 1);
  std::uniform_int_distribution<uint32_t> distributionSubset(0, cgApproximateResultSize - 1);
  std::uniform_real_distribution<float> distributionFloat(0.0f, 1.0f);
  std::unordered_set<uint32_t> actualIndices;
  std::cout << "mash size (tri): " << aMesh.size() << "  ref size (vert): " << reference.size();
  size_t iterationsNeeded = 0u;                            // TODO slow and fails if reference.size() <~ cgApproximateResultSize
  while(actualIndices.size() < cgApproximateResultSize) {
    uint32_t randomIndex = distributionAll(generator);
    if(actualIndices.find(randomIndex) == actualIndices.end()) {
      actualIndices.insert(randomIndex);
    }
    else { // nothing to do
    }
    ++iterationsNeeded;
  }
  std::cout << " iter needed: " << iterationsNeeded << '\n';
  std::unordered_set<uint32_t> bestIndices = actualIndices;
  float actualDistancesSum = calculateDistanceSum(actualIndices, reference);
  float bestDistancesSum = actualDistancesSum;
  float initialTemperature = cgApproximateTemperatureFactor / aMedianSideSizeHarmonic / aMedianSideSizeHarmonic * reference.size() * reference.size();
  for(int32_t i = 0u; i < cgApproximateIterations; ++i) {
    auto candidateIndices = actualIndices;
    uint32_t toRemoveIndex = distributionSubset(generator);
    auto thisOne = candidateIndices.begin();
    for(uint32_t j = 0u; j < toRemoveIndex; ++j) {
      ++thisOne;
    }
    candidateIndices.erase(thisOne);
    while(true) {
      uint32_t toInsert = distributionAll(generator);
      if(candidateIndices.find(toInsert) == candidateIndices.end()) {
        candidateIndices.insert(toInsert);
        break;
      }
      else { // nothing to do
      }
    }
    float candidateDistancesSum = calculateDistanceSum(candidateIndices, reference);
    if(candidateDistancesSum > bestDistancesSum) {
      bestDistancesSum = candidateDistancesSum;
      bestIndices = candidateIndices;
    }
    else { // nothing to do
    }
    float diff = actualDistancesSum - candidateDistancesSum;
    float temperature = initialTemperature / (i + 1);
    if(diff < 0.0f || distributionFloat(generator) < ::expf(-diff / temperature)) {
      actualIndices = candidateIndices;
      actualDistancesSum = candidateDistancesSum;
    }
    else { // nothing to do
    }
  }
  std::vector<Vertex> result;
  result.reserve(cgApproximateResultSize);
  for(auto const &i : bestIndices) {
    result.push_back(reference[i]);
  }
  return result;
}

std::vector<float> calculateSpheres(std::vector<Vertex> const &aPoints) {
  Vertex aabb1{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  Vertex aabb2{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
  for(auto const &point : aPoints) {
    for(size_t i = 0u; i < 3u; ++i) {
      aabb1[i] = std::min(point[i], aabb1[i]);
      aabb2[i] = std::max(point[i], aabb2[i]);
    }
  }
  float max = (aabb2 - aabb1).norm();
  
  std::vector<float> result;
  result.reserve(aPoints.size());
  std::vector<float> distances(aPoints.size() - 1u);
  for(auto const &point: aPoints) {
    distances.clear();
    for(auto const &other: aPoints) {
      if(other != point) {
        distances.push_back(-(point - other).norm());
      }
      else { // nothing to do
      }
    }
    std::make_heap(distances.begin(), distances.end());
    float nth;
    for(int32_t i = 0; i < cgCalculateSpheresNeighbours; ++i) {
      std::pop_heap(distances.begin(), distances.end());
      nth = -distances.back();
      distances.pop_back();
    }
    result.push_back(nth * cgCalculateSpheresInflate);
  }
  return result;
}

auto harmonic(Triangle const &aTriangle) {
  float a = (aTriangle[0] - aTriangle[1]).norm();
  float b = (aTriangle[1] - aTriangle[2]).norm();
  float c = (aTriangle[2] - aTriangle[0]).norm();
  return std::pair{ 3.0f / (1.0f / a + 1.0f / b + 1.0f / c), std::max({a, b, c})};
}

auto calculateMedianSideSizeHarmonicAndMaxSide(Mesh const &aMesh1, Mesh const &aMesh2) {
  std::deque<float> harmonics;
  float maxSide1 = 0.0f;
  for(auto const &item1 : aMesh1) {
    auto [harm, max] = harmonic(item1);
    harmonics.push_back(harm);
    maxSide1 = std::max(maxSide1, max);
  }
  float maxSide2 = 0.0f;
  for(auto const &item2 : aMesh2) {
    auto [harm, max] = harmonic(item2);
    harmonics.push_back(harm);
    maxSide2 = std::max(maxSide2, max);
  }
  std::sort(harmonics.begin(), harmonics.end());
  float median = harmonics[harmonics.size() / 2u];
  std::cout << "1: " << aMesh1.size() << " 2: " << aMesh2.size() << " med: " << median << " max: " << maxSide1 << ' ' << maxSide2 << '\n';
  return std::pair(median, std::min(maxSide1, maxSide2)); // In general case, if the meshes are farther apart than the less of the maximum sides, they definitely don't intersect.
}

Mesh toTetras(std::vector<Vertex> const& aPoints, float const aMedianSizeHarmonic) {
  Mesh result;
  float size = aMedianSizeHarmonic / 5.0f;
  float cogShift = size / 4.0f;
  for(auto const &point : aPoints) {
    Vertex corner0(-cogShift, -cogShift, -cogShift);
    Vertex corner1(size - cogShift, -cogShift, -cogShift);
    Vertex corner2(-cogShift, size - cogShift, -cogShift);
    Vertex corner3(-cogShift, -cogShift, size - cogShift);
    result.push_back({point + corner0, point + corner1, point + corner2});
    result.push_back({point + corner0, point + corner1, point + corner3});
    result.push_back({point + corner0, point + corner2, point + corner3});
    result.push_back({point + corner1, point + corner2, point + corner3});
  }
  return result;
}

Mesh getUnitSphere() {
  Mesh result;
  result.reserve(cgSphereBelts * 2 * cgSphereSectors);
  float sectorAngleHalf = cgPi / cgSphereSectors;
  float sectorAngleFull = sectorAngleHalf * 2.0f;
  float beltAngle       = cgPi / (cgSphereBelts + 1.0f);
  float bias = 0.0f;
  float beltAngleUp = 0.0f;
  float beltAngleMiddle = beltAngle;
  float beltAngleDown = 2.0f * beltAngle;
  float beltRadiusUp = 0.0f;
  float beltRadiusMiddle = std::sin(beltAngleMiddle);
  float beltRadiusDown = std::sin(beltAngleDown);
  float beltZup = 1.0f;
  float beltZmiddle = std::cos(beltAngleMiddle);
  float beltZdown = std::cos(beltAngleDown);
  for(int32_t belt = 0; belt < cgSphereBelts; ++belt) {
    float sectorAngleUpDown = bias + sectorAngleHalf;
    float sectorAngleMiddle1 = bias + 0.0f;
    float sectorAngleMiddle2 = bias + sectorAngleFull;
    for(int32_t sector = 0; sector < cgSphereSectors; ++sector) {
      Vertex corner1(beltRadiusUp * std::sin(sectorAngleUpDown), beltRadiusUp * std::cos(sectorAngleUpDown), beltZup);
      Vertex corner2(beltRadiusMiddle * std::sin(sectorAngleMiddle1), beltRadiusMiddle * std::cos(sectorAngleMiddle1), beltZmiddle);
      Vertex corner3(beltRadiusMiddle * std::sin(sectorAngleMiddle2), beltRadiusMiddle * std::cos(sectorAngleMiddle2), beltZmiddle);
      result.push_back({corner1, corner2, corner3});
      corner1 = {beltRadiusDown * std::sin(sectorAngleUpDown), beltRadiusDown * std::cos(sectorAngleUpDown), beltZdown};
      result.push_back({corner2, corner3, corner1});
      sectorAngleUpDown += sectorAngleFull;
      sectorAngleMiddle1 = sectorAngleMiddle2;
      sectorAngleMiddle2 += sectorAngleFull;
    }
    beltAngleUp = beltAngleMiddle;
    beltAngleMiddle = beltAngleDown;
    beltAngleDown += beltAngle;
    beltRadiusUp = beltRadiusMiddle;
    beltRadiusMiddle = beltRadiusDown;
    beltRadiusDown = std::sin(beltAngleDown);
    beltZup = beltZmiddle;
    beltZmiddle = beltZdown;
    beltZdown = std::cos(beltAngleDown);
    bias += sectorAngleHalf;
  }
  return result;
}

Mesh toSpheres(std::vector<Vertex> const& aPoints, std::vector<float> const& aRadii) {
  static auto unitSphere = getUnitSphere();
  Mesh result;
  result.reserve(unitSphere.size() * aPoints.size());
  for(size_t i = 0u; i < aPoints.size(); ++i) {
    auto &point = aPoints[i];
    auto &radius = aRadii[i];
    for(auto const &face : unitSphere) {
      result.push_back({face[0] * radius + point, face[1] * radius + point, face[2] * radius + point});
    }
  }
  return result;
}

auto calculateDistance(std::vector<Vertex> const &aApproximate1, std::vector<Vertex> const &aApproximate2) {
  float distance = std::numeric_limits<float>::max();
  for(auto const &vertex1 : aApproximate1) {
    for(auto const &vertex2 : aApproximate2) {
      distance = std::min(distance, (vertex1 - vertex2).norm());
    }
  }
  return distance;
}

void check(Mesh const &aMesh1, Mesh const &aMesh2) {
  auto [medianSideSizeHarmonic, maxSide] = calculateMedianSideSizeHarmonicAndMaxSide(aMesh1, aMesh2);
  auto approximate1 = approximate(aMesh1, medianSideSizeHarmonic);
  auto approximate2 = approximate(aMesh2, medianSideSizeHarmonic);
  auto sphereSizes1 = calculateSpheres(approximate1);
  auto sphereSizes2 = calculateSpheres(approximate2);
  for(auto i : sphereSizes1) {
    std::cout << i << ' ';
  }
  for(auto i : sphereSizes2) {
    std::cout << i << ' ';
  }
  std::cout << '\n';
  auto distance = calculateDistance(approximate1, approximate2);
  std::cout << "dist: " << distance << '\n';
  writeMesh(toTetras(approximate1, medianSideSizeHarmonic), toTetras(approximate2, medianSideSizeHarmonic), "points.stl");
  writeMesh(toSpheres(approximate1, sphereSizes1), toSpheres(approximate2, sphereSizes2), "spheres.stl");
}

int main(int argc, char **argv) {
  int ret = 0;
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filenameIn>\n";
    ret = 1;
  }
  else {
    try {
      auto transform = randomTransform();
      auto [mesh1, mesh2] = readMesh(argv[1], transform);
      writeMesh(mesh1, mesh2, "out.stl");
      check(mesh1, mesh2);
    }
    catch(std::exception &e) {
      ret = 2;
    }
  }
  return ret;
}
