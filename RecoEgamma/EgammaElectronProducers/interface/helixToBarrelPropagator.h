#include <iostream>
#include <cmath>

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

typedef double TmpType;
typedef Basic2DVector<TmpType> Vector;
typedef GlobalPoint PositionType;
typedef GlobalVector DirectionType;
enum Solution { bothSol, bestSol, onlyPos };

// For sqr
template <typename T>
inline T sqr(const T& t) {
  return t * t;
};

// For solving quad equ.
struct QuadEquationSolver {

  double first;
  double second;
  bool hasSolution;

  QuadEquationSolver(double A, double B, double C) {
    double D = B * B - 4 * A * C;
    if (D < 0)
      hasSolution = false;
    else {
      hasSolution = true;
      auto q = -0.5 * (B + std::copysign(std::sqrt(D), B));
      first = q / A;
      second = C / q;
    }
  }
};

Vector chooseSolution(
    const Vector& d1,
    const Vector& d2,
    const PositionType& startingPos,
    const DirectionType& startingDir, 
    bool& theSolExists) {
  
  Vector theD;

  auto momProj1 = startingDir.x() * d1.x() + startingDir.y() * d1.y();
  auto momProj2 = startingDir.x() * d2.x() + startingDir.y() * d2.y();

    int propSign =1 ;
    if (momProj1 * momProj2 < 0) {
      // if different signs return the positive one
      theSolExists = true;
      theD = (momProj1 * propSign > 0) ? d1 : d2;
    } else if (momProj1 * propSign > 0) {
      // if both positive, return the shortest
      theSolExists = true;
      theD = (d1.mag2() < d2.mag2()) ? d1 : d2;
    } else
      theSolExists = false;

  return theD;
};

void helixBarrelCylinderCrossing(const PositionType& startingPos, const DirectionType& startingDir, 
                                  double rho, double radius, bool& theSolExists,
                                  GlobalPoint& x, GlobalVector& p, double& s){

  // Cylinder must be barrel (x,y = 0,0)
  Solution sol = bothSol;
  PositionType thePos;
  DirectionType theDir;
  double theS;
  PositionType thePos1;
  PositionType thePos2;

  double R = radius;
  const double sraightLineCutoff = 1.e-7;

  if (fabs(rho) * R < sraightLineCutoff && fabs(rho) * startingPos.perp() < sraightLineCutoff) {
    printf("STRAIGHT LINE");
    theSolExists = false;
    return;
    // Should add implementation 
  }

  double R2cyl = R * R;
  double pt = startingDir.perp();

  // center of helix in global coords?
  double center_x = startingPos.x() - startingDir.y() / (pt * rho);
  double center_y = startingPos.y() + startingDir.x() / (pt * rho);
 
  double p2 = startingPos.perp2();
  bool solveForX;
  double B, C, E, F;

  if (fabs(center_x) > fabs(center_y)) {
    solveForX = false;
    E = (R2cyl - p2) / (2. * center_x);
    F = center_y / center_x;
    B = 2. * (startingPos.y() - F * startingPos.x() - E * F);
    C = 2. * E * startingPos.x() + E * E + p2 - R2cyl;
  } else {
    solveForX = true;
    E = (R2cyl - p2) / (2. * center_y);
    F = center_x / center_y;
    B = 2. * (startingPos.x() - F * startingPos.y() - E * F);
    C = 2. * E * startingPos.y() + E * E + p2 - R2cyl;
  }

  QuadEquationSolver eq(1 + F * F, B, C);
  if (!eq.hasSolution) {
    theSolExists = false;
    return;
  }

  double d1_x,d1_y, d2_x,d2_y;

  if (solveForX) {
    d1_x = eq.first;
    d1_y = E - F * eq.first;
    d2_x = eq.second;
    d2_y = E - F * eq.second;

  } else {
    d1_x = E - F * eq.first;
    d1_y = eq.first;
    d2_x = E - F * eq.second;
    d2_y = eq.second;
  }

  Vector theD;
  Vector d1(d1_x,d1_y);
  Vector d2(d2_x,d2_y);

  int theActualDir = 1; // Along momentum propagating from tracker to ECAL
  int propDir = 1; 

  theD = chooseSolution(d1, d2, startingPos, startingDir,theSolExists);
  if (!theSolExists)
    return;

  float ipabs = 1.f / startingDir.mag();
  float sinTheta = float(pt) * ipabs;
  float cosTheta = startingDir.z() * ipabs;

  // -------

  auto dMag = theD.mag();
  float tmp = 0.5f * float(dMag * rho);
  if (std::abs(tmp) > 1.f)
    tmp = std::copysign(1.f, tmp);
  theS = theActualDir * 2.f * std::asin(tmp) / (float(rho) * sinTheta);
  thePos = GlobalPoint(startingPos.x() + theD.x(), startingPos.y() + theD.y(), startingPos.z() + theS * cosTheta);


  if (theS < 0)
    tmp = -tmp;
  auto sinPhi = 2.f * tmp * sqrt(1.f - tmp * tmp);
  auto cosPhi = 1.f - 2.f * tmp * tmp;
  theDir = DirectionType(startingDir.x() * cosPhi - startingDir.y() * sinPhi,
                         startingDir.x() * sinPhi + startingDir.y() * cosPhi,
                         startingDir.z());

  // Fix
  s = theS;
  x = thePos;
  p = theDir;

  if (sol != bothSol)
    return;

  // redundant 
  int theActualDir1 = propDir == 1 ? 1 : -1;
  int theActualDir2 = propDir == 1 ? 1 : -1;

  auto dMag1 = d1.mag();
  auto tmp1 = 0.5f * dMag1 * float(rho);
  if (std::abs(tmp1) > 1.f)
    tmp1 = std::copysign(1.f, tmp1);
  auto theS1 = theActualDir1 * 2.f * std::asin(tmp1) / (rho * sinTheta);
  thePos1 = GlobalPoint(startingPos.x() + d1.x(), startingPos.y() + d1.y(), startingPos.z() + theS1 * cosTheta);

  auto dMag2 = d2.mag();
  auto tmp2 = 0.5f * dMag2 * float(rho);
  if (std::abs(tmp2) > 1.f)
    tmp2 = std::copysign(1.f, tmp2);
  auto theS2 = theActualDir2 * 2.f * std::asin(tmp2) / (float(rho) * sinTheta);
  thePos2 = GlobalPoint(startingPos.x() + d2.x(), startingPos.y() + d2.y(), startingPos.z() + theS2 * cosTheta);

};
