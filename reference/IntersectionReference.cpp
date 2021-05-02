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
#include <Eigen/Dense>

using Vertex = Eigen::Vector3f;
using Triangle = std::array<Vertex, 3u>;
using CudaTriangle = Vertex*;
using CudaConstTriangle = Vertex const*;
using Mesh = std::vector<Triangle>;

constexpr int32_t  cgApproximateInitialPointsPerDimension =  8;
constexpr float    cgApproximateMaxJumpLimitFactor        =  1.0f;
constexpr float    cgApproximateLeaveInPlaceFactor        =  0.01f;
constexpr float    cgApproximateRate                      =  0.5f;
constexpr size_t   cgApproximateFinalPointCount           = 32u;
constexpr float    cgEpsilonDistanceFromSideFactor        =  0.001f;
constexpr float    cgEpsilonPlaneIntersectionSine         =  0.001f;
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
      triangle[indexCorner] = aTransform * (in + aTranslation);
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

float harmonic(Triangle const &aTriangle) {
  float a = (aTriangle[0] - aTriangle[1]).norm();
  float b = (aTriangle[1] - aTriangle[2]).norm();
  float c = (aTriangle[2] - aTriangle[0]).norm();
  return 3.0f / (1.0f / a + 1.0f / b + 1.0f / c);
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

auto approximate(Mesh const aMesh, float const aMedianSideSizeHarmonic) {
  std::deque<Vertex> all;
  std::deque<Vertex> reference;
  Vertex min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  Vertex max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
  float limit = aMedianSideSizeHarmonic * cgApproximateLeaveInPlaceFactor;
  for(auto const &triangle : aMesh) {
    for(auto const &vertex : triangle) {
      for(size_t i = 0u; i < 3u; ++i) {
        min(i) = std::min(min(i), vertex(i));
        max(i) = std::max(max(i), vertex(i));
      }
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
  float dx = (max(0) - min(0)) / (cgApproximateInitialPointsPerDimension - 1);
  float dy = (max(1) - min(1)) / (cgApproximateInitialPointsPerDimension - 1);
  float dz = (max(2) - min(2)) / (cgApproximateInitialPointsPerDimension - 1);
  float x = min(0);
  for(int32_t i = 0; i < cgApproximateInitialPointsPerDimension; ++i) {
    float y = min(1);
    for(int32_t j = 0; j < cgApproximateInitialPointsPerDimension; ++j) {
      float z = min(2);
      for(int32_t k = 0; k < cgApproximateInitialPointsPerDimension; ++k) {
        all.push_back(Vertex(x, y, z));
        z += dz;
      }
      y += dy;
    }
    x += dx;
  }
  float largestJump = aMedianSideSizeHarmonic * cgApproximateMaxJumpLimitFactor * 2.0f;
  while(largestJump > aMedianSideSizeHarmonic * cgApproximateMaxJumpLimitFactor) {
    largestJump = 0.0f;
    for(auto &point : all) {
      Vertex displacement(0.0f, 0.0f, 0.0f);
      bool arrived = false;
      float minDistance = std::numeric_limits<float>::max();
      for(auto const &ref : reference) {
        Vertex attraction = ref - point;
        float distance = attraction.norm();
        if(distance < limit) {
          arrived = true;
          break;
        }
        else {
          minDistance = std::min(minDistance, distance);
          displacement += attraction / (distance * distance);
        }
      }
      float howMuch = displacement.norm();
      if(!arrived && howMuch > limit) {
        float jump = minDistance * cgApproximateRate;
        point += displacement / howMuch * jump;
        largestJump = std::max(largestJump, jump);
      }
      else { // nothing to do
      }
    }
    std::cout << "jump: " << largestJump << '\n';
  }
  std::map<float, Vertex> sorted;
  for(auto const &point : all) {
    float distance = std::numeric_limits<float>::max();
    for(auto const &ref : reference) {
      distance = std::min(distance, (ref - point).norm());
    }
    sorted.emplace(std::pair{distance, point});
  }
  std::vector<Vertex> result;
  result.reserve(cgApproximateFinalPointCount);
  std::for_each_n(sorted.begin(), cgApproximateFinalPointCount, [&result](auto const &item){ result.push_back(item.second); });
  return result;
}

float calculateMedianSideSizeHarmonic(Mesh const &aMesh1, Mesh const &aMesh2) {
  std::deque<float> harmonics;
  float average = 0.0f;
  for(auto const &item1 : aMesh1) {
    auto h = harmonic(item1);
    harmonics.push_back(h);
    average += h;
  }
  for(auto const &item2 : aMesh2) {
    auto h = harmonic(item2);
    harmonics.push_back(h);
    average += h;
  }
  average /= harmonics.size();
  std::sort(harmonics.begin(), harmonics.end());
  float median = harmonics[harmonics.size() / 2u];
  std::cout << "1: " << aMesh1.size() << " 2: " << aMesh2.size() << " min:" << harmonics.front() << " avg: " << average << " med: " << harmonics[harmonics.size() / 2u] << " max: " << harmonics.back() << '\n';
  return median;
}

Mesh toMesh(std::vector<Vertex> const aPoints, float const aMedianSizeHarmonic) {
  Mesh result;
  float size = aMedianSizeHarmonic / 10.0f;
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

void check(Mesh const &aMesh1, Mesh const &aMesh2) {
  float medianSideSizeHarmonic = calculateMedianSideSizeHarmonic(aMesh1, aMesh2);
  auto approximate1 = approximate(aMesh1, medianSideSizeHarmonic);
  auto approximate2 = approximate(aMesh2, medianSideSizeHarmonic);
  writeMesh(toMesh(approximate1, medianSideSizeHarmonic), toMesh(approximate2, medianSideSizeHarmonic), "points.stl");
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
