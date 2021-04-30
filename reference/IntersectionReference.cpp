#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int32_t

#include "../../../repos/stl_reader/stl_reader.h"
#include <array>
#include <deque>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>


using Triangle = std::array<Eigen::Vector3f, 3u>;
using CudaTriangle = Eigen::Vector3f*;
using CudaConstTriangle = Eigen::Vector3f const*;
using Triangles = std::deque<Triangle>;

constexpr float    cgEpsilonDistanceFromSide       = 0.01f;       // TODO consider if uniform epsilon suits all needs.
constexpr float    cgEpsilonPlaneIntersectionSine = 0.00001f;
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

constexpr uint32_t calculateSignum(float const distances[3]) noexcept {
  uint32_t result = 0u;
  for(int32_t i = 0; i < 3; ++i) {
    int32_t tmp = (distances[i] > cgEpsilonDistanceFromSide ? cgSignumPlus : cgSignumZero);
    tmp = (distances[i] < -cgEpsilonDistanceFromSide ? cgSignumMinus : tmp);
    result |= tmp << (cgSignumShift1 * i);
  }
  return result;
}

// TODO consider Eigen3 for CUDA register usage. Probably it does not use indexing inside.

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

bool hasCommonPoint(CudaConstTriangle const aShape1, CudaConstTriangle const aShape2) noexcept { // Entry point for common point check.
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
  uint32_t signumShape1FromPlane2 = calculateSignum(distanceCornerNofShape1FromPlane2); // These contain info about relation of each point and the other plane.
  uint32_t signumShape2FromPlane1 = calculateSignum(distanceCornerNofShape2FromPlane1);
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
      if(intersectionParameterAshape1 - cgEpsilonDistanceFromSide <= intersectionParameterAshape2 && intersectionParameterAshape2 <= intersectionParameterBshape1 + cgEpsilonDistanceFromSide // Epsilons make possible to check for triangles with corner-corner or corner-edge touch.
      || intersectionParameterAshape1 - cgEpsilonDistanceFromSide <= intersectionParameterBshape2 && intersectionParameterBshape2 <= intersectionParameterBshape1 + cgEpsilonDistanceFromSide
      || intersectionParameterAshape2 - cgEpsilonDistanceFromSide <= intersectionParameterAshape1 && intersectionParameterAshape1 <= intersectionParameterBshape2 + cgEpsilonDistanceFromSide
      || intersectionParameterAshape2 - cgEpsilonDistanceFromSide <= intersectionParameterBshape1 && intersectionParameterBshape1 <= intersectionParameterBshape2 + cgEpsilonDistanceFromSide) {
std::cout << "on intersecting line\n";
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
//  generator.seed((std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::time_point::min()).count());
  std::uniform_real_distribution<float> distribution(0.1f, 1.0f);
  Eigen::Matrix3f result;
  for(int32_t i = 0; i < 9; ++i) {
    result(i / 3, i % 3) = distribution(generator);
  }
  return result;
}

Triangles readTriangles(char const * const aFilename, Eigen::Matrix3f const &aTransform) {
  Triangles result;
  stl_reader::StlMesh<float, int32_t> mesh(aFilename);
  for(int32_t indexTriangle = 0; indexTriangle < mesh.num_tris(); ++indexTriangle) {
      Triangle triangle;
      for(int32_t indexCorner = 0; indexCorner < 3; ++indexCorner) {
          float const * const coords = mesh.tri_corner_coords(indexTriangle, indexCorner);
          Eigen::Vector3f in;
          for(int32_t i = 0; i < 3; ++i) {
            in(i) = coords[i];
          }
          triangle[indexCorner] = aTransform * in;
      }
      result.push_back(triangle);
  }
  return result;
}

void writeTriangles(Triangles const &aTriangles, char const * const aFilename) {
  std::ofstream out(aFilename);
  out << "solid Exported from Blender-2.82 (sub 7)\n";
  for(auto const & triangle : aTriangles) {
//    Eigen::Vector3f normal = (triangle[1] - triangle[0]).cross(aShape2[2] - aShape2[0]);
//  shape1normal.normalize();
    out << "facet normal 0.000000 0.000000 0.000000\nouter loop\n";
    for(auto const & vertex : triangle) {
      out << "vertex " << vertex(0) << ' ' << vertex(1) << ' ' << vertex(2) << '\n';
    }
    out << "endloop\nendfacet\n";
  }
  out << "endsolid Exported from Blender-2.82 (sub 7)\n";
}

void check(Triangles const &aTriangles) {
  for(int32_t i = 0; i < aTriangles.size(); ++i) {
    for(int32_t j = i + 1; j < aTriangles.size(); ++j) {
      if(hasCommonPoint(aTriangles[i].data(), aTriangles[j].data())) {
        std::cout << "Has common point: " << i << ' ' << j << "\n\n";
      }
      else { // nothing to do
      }
    }
  }
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
      auto triangles = readTriangles(argv[1], transform);
      if(argc >= 3) {
        writeTriangles(triangles, argv[2]);
      }
      else { // nothing to do
      }
      check(triangles);
    }
    catch(std::exception &e) {
      ret = 2;
    }
  }
  return ret;
}
