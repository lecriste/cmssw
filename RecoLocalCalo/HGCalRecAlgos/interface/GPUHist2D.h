#include <memory>
#include <cmath>
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"

#ifndef GPUHist2D_h
#define GPUHist2D_h

template <class T, int xDim, int yDim, int max_depth> struct histogram2D{

  histogram2D(float xMin, float xMax, float yMin, float yMax)
  {
    limits_[0] = xMin;
    limits_[1] = xMax;
    limits_[2] = yMin;
    limits_[3] = yMax;
    data_.resize(xDim*yDim);
    for(int i = 0; i < xDim*yDim; i++)
      data_[i].reset();
    xBinSize_ = (xMax-xMin)/xDim;
    yBinSize_ = (yMax-yMin)/yDim;
  }

  __host__
  bool fillBin(float x, float y, T idx)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    
    if(data_[xBin + yBin*xDim].push_back_unsafe(idx) != -1)
      return true;
    else
      return false;
  }

  __device__
  bool fillBinGPU(float x, float y, T idx)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
 
    if(data_[xBin + yBin*xDim].push_back(idx) != -1)
      return true;
   else
     return false;
  
  }
  
  __host__ __device__
  int computeXBinIndex(float x) const
  {
    int xIndex = floor((std::abs(x) - limits_[0]) / xBinSize_);
    return xIndex;
  }

  __host__ __device__
  int computeYBinIndex(float y) const
  {
    int yIndex = floor((y + limits_[3]) / yBinSize_);
    return yIndex;
  } 

  __host__ __device__
  int getBinIdx(float x, float y) const
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    return xBin + yBin*xDim;
  }

  __host__ __device__
  GPU::VecArray<T, max_depth> getBinContent(float x, float y) const
  {

    GPU::VecArray<T, max_depth> empty;

    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);

    if( xBin + yBin*xDim < (uint)data_.size() ) { //fixme: cast is ok?
      return data_[xBin + yBin*xDim];
    }
    else {
      return empty;
    }

  }

  __host__ __device__
  GPU::VecArray<T, max_depth> getBinContent(uint xBin, uint yBin) const
  {

    GPU::VecArray<T, max_depth> empty;

    if( xBin + yBin*xDim < (uint)data_.size() ) { //fixme: cast is ok?
      return data_[xBin + yBin*xDim];
    }
    else {
      return empty;
    }

  }

  __host__ __device__
  std::array<int,4> searchBox(float xMin, float xMax, float yMin, float yMax)
  {
    int xBinMin = xMin <= limits_[0] ? computeXBinIndex(limits_[0]) : computeXBinIndex(xMin);
    int xBinMax = xMax >= limits_[1] ? computeXBinIndex(limits_[1]) : computeXBinIndex(xMax);
    int yBinMin = yMin <= limits_[2] ? computeYBinIndex(limits_[2]) : computeYBinIndex(yMin);
    int yBinMax = yMax >= limits_[3] ? computeYBinIndex(limits_[3]) : computeYBinIndex(yMax);
    std::array<int, 4> box;
    box[0] = xBinMin;
    box[1] = xBinMax - xBinMin + 1;
    box[2] = yBinMin;
    box[3] = yBinMax - yBinMin+ 1;
    return box;
  }

  inline constexpr int size() const { return data_.size(); }
  inline constexpr GPU::VecArray<T, max_depth>& operator[](int i) { return data_[i]; }

  GPU::VecArray<GPU::VecArray<T, max_depth>, xDim*yDim> data_;
  
  float limits_[4];
  float xBinSize_ = 0.0;
  float yBinSize_ = 0.0;
  
};
  
#endif
