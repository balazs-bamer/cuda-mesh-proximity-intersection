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

// Exact intersection calculation using triangles.

using Vertex = Eigen::Vector3f;
using Quaternion = Eigen::Quaternion<float>;
using Transform = Eigen::Transform<float, 3, Eigen::Affine>;
using Triangle = std::array<Vertex, 3u>;
using CudaTriangle = Vertex*;
using CudaConstTriangle = Vertex const*;
using Mesh = std::vector<Triangle>;

// usage ./IntersectionReference example23.in 

constexpr float    cgPi                            =    3.1415926539f;
constexpr int32_t  cgSphereSectors                 =    11;
constexpr int32_t  cgSphereBelts                   =    5;
constexpr float    cgCalculateSpheresInflate       =    0.51f;
constexpr int32_t  cgCalculateSpheresNeighbours    =    3;
constexpr float    cgApproximateLeaveInPlaceFactor =    0.01f;
constexpr uint32_t cgApproximateResultSize         =   32u;
constexpr int32_t  cgApproximateIterations         = 2222u;
constexpr float    cgApproximateTemperatureFactor  =    0.01f;
constexpr float    cgEpsilonDistanceFromSideFactor =    0.001f;
constexpr float    cgEpsilonPlaneIntersectionSine  =    0.001f;
constexpr uint32_t cgSignumZero     = 0u;
constexpr uint32_t cgSignumPlus     = 1u;
constexpr uint32_t cgSignumMinus    = 2u;
constexpr uint32_t cgSignumShift0   = 0u;
constexpr uint32_t cgSignumShift1   = 2u;
constexpr uint32_t cgSignumShift2   = 4u;
constexpr uint32_t cgSignumAllZero  = (cgSignumZero  << cgSignumShift0) | (cgSignumZero  << cgSignumShift1) | (cgSignumZero  << cgSignumShift2);
constexpr uint32_t cgSignumAllPlus  = (cgSignumPlus  << cgSignumShift0) | (cgSignumPlus  << cgSignumShift1) | (cgSignumPlus  << cgSignumShift2);
constexpr uint32_t cgSignumAllMinus = (cgSignumMinus << cgSignumShift0) | (cgSignumMinus << cgSignumShift1) | (cgSignumMinus << cgSignumShift2);

constexpr uint32_t cgSignumSelect0a = (cgSignumPlus  << cgSignumShift0) | (cgSignumMinus << cgSignumShift1) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect0b = (cgSignumMinus << cgSignumShift0) | (cgSignumPlus  << cgSignumShift1) | (cgSignumPlus  << cgSignumShift2);
constexpr uint32_t cgSignumSelect0c = (cgSignumZero  << cgSignumShift0) | (cgSignumPlus  << cgSignumShift1) | (cgSignumPlus  << cgSignumShift2);
constexpr uint32_t cgSignumSelect0d = (cgSignumZero  << cgSignumShift0) | (cgSignumMinus << cgSignumShift1) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect0e = (cgSignumPlus  << cgSignumShift0) | (cgSignumZero  << cgSignumShift1) | (cgSignumZero  << cgSignumShift2);
constexpr uint32_t cgSignumSelect0f = (cgSignumMinus << cgSignumShift0) | (cgSignumZero  << cgSignumShift1) | (cgSignumZero  << cgSignumShift2);
constexpr uint32_t cgSignumSelect0g = (cgSignumZero  << cgSignumShift0) | (cgSignumPlus  << cgSignumShift1) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect0h = (cgSignumZero  << cgSignumShift0) | (cgSignumMinus << cgSignumShift1) | (cgSignumPlus  << cgSignumShift2);

constexpr uint32_t cgSignumSelect1a = (cgSignumPlus  << cgSignumShift1) | (cgSignumMinus << cgSignumShift0) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect1b = (cgSignumMinus << cgSignumShift1) | (cgSignumPlus  << cgSignumShift0) | (cgSignumPlus  << cgSignumShift2);
constexpr uint32_t cgSignumSelect1c = (cgSignumZero  << cgSignumShift1) | (cgSignumPlus  << cgSignumShift0) | (cgSignumPlus  << cgSignumShift2);
constexpr uint32_t cgSignumSelect1d = (cgSignumZero  << cgSignumShift1) | (cgSignumMinus << cgSignumShift0) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect1e = (cgSignumPlus  << cgSignumShift1) | (cgSignumZero  << cgSignumShift0) | (cgSignumZero  << cgSignumShift2);
constexpr uint32_t cgSignumSelect1f = (cgSignumMinus << cgSignumShift1) | (cgSignumZero  << cgSignumShift0) | (cgSignumZero  << cgSignumShift2);
constexpr uint32_t cgSignumSelect1g = (cgSignumZero  << cgSignumShift1) | (cgSignumPlus  << cgSignumShift0) | (cgSignumMinus << cgSignumShift2);
constexpr uint32_t cgSignumSelect1h = (cgSignumZero  << cgSignumShift1) | (cgSignumMinus << cgSignumShift0) | (cgSignumPlus  << cgSignumShift2);

// Otherwise select 2, no need for checking and thus no constants.

constexpr uint32_t calculateSignum(float const distances[3], float const aEpsilonDistanceFromSide) noexcept {  // TODO optim kibontja a ciklust, de ciklus miatt nincs reg
  uint32_t result = 0u;
  for(int32_t i = 0; i < 3; ++i) {
    int32_t tmp = (distances[i] > aEpsilonDistanceFromSide ? cgSignumPlus : cgSignumZero);
    tmp = (distances[i] < - aEpsilonDistanceFromSide ? cgSignumMinus : tmp);
    result |= tmp << (cgSignumShift1 * i);
  }
  return result;
}

// TODO consider Eigen3 for CUDA register usage. It indexes even for dot product.
// Consider substituting it with simple custom classes if local memory does not fit L1.

void calculateIntersectionParameter(
  CudaConstTriangle const aShape
, Eigen::Vector3f const &aIntersectionVector
, float const aDistanceCornerNfromOtherPlane[3]
, uint32_t const aSignumShapeFromOtherPlane
, float &aIntersectionParameterA
, float &aIntersectionParameterB) noexcept {
  int32_t indexCommon, indexA, indexB;
  if(aSignumShapeFromOtherPlane == cgSignumSelect0a
  || aSignumShapeFromOtherPlane == cgSignumSelect0b
  || aSignumShapeFromOtherPlane == cgSignumSelect0c
  || aSignumShapeFromOtherPlane == cgSignumSelect0d
  || aSignumShapeFromOtherPlane == cgSignumSelect0e
  || aSignumShapeFromOtherPlane == cgSignumSelect0f
  || aSignumShapeFromOtherPlane == cgSignumSelect0g
  || aSignumShapeFromOtherPlane == cgSignumSelect0h) {
    indexCommon = 0; indexA = 1; indexB = 2;
  }
  else if(aSignumShapeFromOtherPlane == cgSignumSelect1a
  || aSignumShapeFromOtherPlane == cgSignumSelect1b
  || aSignumShapeFromOtherPlane == cgSignumSelect1c
  || aSignumShapeFromOtherPlane == cgSignumSelect1d
  || aSignumShapeFromOtherPlane == cgSignumSelect1e
  || aSignumShapeFromOtherPlane == cgSignumSelect1f
  || aSignumShapeFromOtherPlane == cgSignumSelect1g
  || aSignumShapeFromOtherPlane == cgSignumSelect1h) {
    indexCommon = 1; indexA = 0; indexB = 2;
  }
  else {
    indexCommon = 2; indexA = 1; indexB = 0;
  }
  float vertexProjections[3];
  for(int32_t i = 0; i < 3; ++i) {
    vertexProjections[i] = aIntersectionVector.dot(aShape[i]);
  }
  aIntersectionParameterA =
   vertexProjections[indexA]
 + (vertexProjections[indexCommon] - vertexProjections[indexA])
 * aDistanceCornerNfromOtherPlane[indexA]
 / (aDistanceCornerNfromOtherPlane[indexA] - aDistanceCornerNfromOtherPlane[indexCommon]);
  aIntersectionParameterB = 
   vertexProjections[indexB]
 + (vertexProjections[indexCommon] - vertexProjections[indexB])
 * aDistanceCornerNfromOtherPlane[indexB]
 / (aDistanceCornerNfromOtherPlane[indexB] - aDistanceCornerNfromOtherPlane[indexCommon]);
}

bool doesIntersectInAngle(CudaConstTriangle const aShape1, CudaConstTriangle const aShape2, bool const aEpsilonDistanceFromSide) noexcept { // Entry point
  // TODO consider perhaps min(harmonic mean of sides or area/perimeter of the triangles) for basis of epsilon
  bool result = false;
  Eigen::Vector3f shape1normal = (aShape1[1] - aShape1[0]).cross(aShape1[2] - aShape1[0]);
  Eigen::Vector3f shape2normal = (aShape2[1] - aShape2[0]).cross(aShape2[2] - aShape2[0]);
  shape1normal.normalize();
  shape2normal.normalize();
  float distanceCornerNofShape1FromPlane2[3];
  float distanceCornerNofShape2FromPlane1[3];
  for(int32_t i = 0; i < 3; ++i) {
    distanceCornerNofShape1FromPlane2[i] = shape2normal.dot(aShape1[i] - aShape2[0]);
    distanceCornerNofShape2FromPlane1[i] = shape1normal.dot(aShape2[i] - aShape1[0]);
  }
  uint32_t signumShape1FromPlane2 = calculateSignum(distanceCornerNofShape1FromPlane2, aEpsilonDistanceFromSide); // These contain info about relation of each point and the other plane.
  uint32_t signumShape2FromPlane1 = calculateSignum(distanceCornerNofShape2FromPlane1, aEpsilonDistanceFromSide);
  if(signumShape1FromPlane2 == cgSignumAllPlus || signumShape1FromPlane2 == cgSignumAllMinus || signumShape2FromPlane1 == cgSignumAllPlus || signumShape2FromPlane1 == cgSignumAllMinus) {
    // Nothing to do: one triangle is completely on the one side of the other's plane
  }
  else {
    Eigen::Vector3f intersectionVector = shape1normal.cross(shape2normal);
    if(fabs(intersectionVector.norm()) > cgEpsilonPlaneIntersectionSine && signumShape1FromPlane2 != cgSignumAllZero && signumShape2FromPlane1 != cgSignumAllZero) { // Real intersection, planes are not identical, and both triangles touch the common line.
      intersectionVector.normalize();
      float intersectionParameterAshape1;
      float intersectionParameterBshape1;
      float intersectionParameterAshape2;
      float intersectionParameterBshape2;
      calculateIntersectionParameter(aShape1, intersectionVector, distanceCornerNofShape1FromPlane2, signumShape1FromPlane2, intersectionParameterAshape1, intersectionParameterBshape1); // The two parameters will contain the locations of the touching point.
      calculateIntersectionParameter(aShape2, intersectionVector, distanceCornerNofShape2FromPlane1, signumShape2FromPlane1, intersectionParameterAshape2, intersectionParameterBshape2);
      if(intersectionParameterAshape1 > intersectionParameterBshape1) {
        std::swap(intersectionParameterAshape1, intersectionParameterBshape1);
      }
      else { // nothing to do
      }
      if(intersectionParameterAshape2 > intersectionParameterBshape2) {
        std::swap(intersectionParameterAshape2, intersectionParameterBshape2);
      }
      else { // nothing to do
      }
      if(intersectionParameterAshape1 - aEpsilonDistanceFromSide <= intersectionParameterAshape2 && intersectionParameterAshape2 <= intersectionParameterBshape1 + aEpsilonDistanceFromSide // Epsilons make possible to check for triangles with corner-corner or corner-edge touch.
      || intersectionParameterAshape1 - aEpsilonDistanceFromSide <= intersectionParameterBshape2 && intersectionParameterBshape2 <= intersectionParameterBshape1 + aEpsilonDistanceFromSide
      || intersectionParameterAshape2 - aEpsilonDistanceFromSide <= intersectionParameterAshape1 && intersectionParameterAshape1 <= intersectionParameterBshape2 + aEpsilonDistanceFromSide
      || intersectionParameterAshape2 - aEpsilonDistanceFromSide <= intersectionParameterBshape1 && intersectionParameterBshape1 <= intersectionParameterBshape2 + aEpsilonDistanceFromSide) {
        result = true;
      }
      else { // Nothing to do
      }
    }
    else { // Nothing to do, because we ignore coplanar triangles.
    }     //  Meshes are assumed to be continuous, so other triangles will yield the necessary checks.
  }
  return result;
}

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

auto readMesh(std::string const &aFilename, Eigen::Vector3f aTranslation, Eigen::Matrix3f const &aTransform) {
  Mesh result;
  stl_reader::StlMesh<float, int32_t> mesh(aFilename);
  result.reserve(mesh.num_tris());
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
    result.push_back(triangle);
  }
  return result;
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

void intersect(Mesh const &aMesh1, Mesh const &aMesh2, float const aMedianSideSizeHarmonic) {
  size_t intersectionCount = 0u;
  for(auto const &item1 : aMesh1) {
    for(auto const &item2 : aMesh2) {
      intersectionCount += (doesIntersectInAngle(item1.data(), item2.data(), aMedianSideSizeHarmonic * cgEpsilonDistanceFromSideFactor) ? 1u : 0u);
    }
  }
  std::cout << "intersects: " << intersectionCount << '\n';
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
  std::cout << "ref: " << reference.size() << '\n';
  std::default_random_engine generator;
  generator.seed((std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::time_point::min()).count());
  std::uniform_int_distribution<uint32_t> distributionAll(0, reference.size() - 1);
  std::uniform_int_distribution<uint32_t> distributionSubset(0, cgApproximateResultSize - 1);
  std::uniform_real_distribution<float> distributionFloat(0.0f, 1.0f);
  std::unordered_set<uint32_t> actualIndices;
  while(actualIndices.size() < cgApproximateResultSize) {
    uint32_t randomIndex = distributionAll(generator);
    if(actualIndices.find(randomIndex) == actualIndices.end()) {
      actualIndices.insert(randomIndex);
    }
    else { // nothing to do
    }
  }
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
  auto distance = calculateDistance(approximate1, approximate2);
  std::cout << "dist: " << distance << '\n';
  if(distance < maxSide) {
    intersect(aMesh1, aMesh2, medianSideSizeHarmonic);
  }
  else { // nothing to do
  }
  writeMesh(toTetras(approximate1, medianSideSizeHarmonic), toTetras(approximate2, medianSideSizeHarmonic), "points.stl");
  writeMesh(toSpheres(approximate1, sphereSizes1), toSpheres(approximate2, sphereSizes2), "spheres.stl");
}

int main(int argc, char **argv) {
  int ret = 0;
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filenameIn> [filenameOut]\n";
    ret = 1;
  }
  else {
    try {
      auto transform = randomTransform();
      auto [mesh1, mesh2] = readMesh(argv[1], transform);
      if(argc >= 3) {
        writeMesh(mesh1, mesh2, argv[2]);
      }
      else { // nothing to do
      }
      check(mesh1, mesh2);
    }
    catch(std::exception &e) {
      ret = 2;
    }
  }
  return ret;
}
