//#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include <math.h>

using namespace BinnerGPU;

namespace HGCalRecAlgos{


static const unsigned int lastLayerEE = 28;
static const unsigned int lastLayerFH = 40;

__global__ void kernel_compute_density(Histo2D* theHist,LayerRecHitsGPU* theHits,float* delta_c,double* density)
{
   density[0] = 0.0;
   size_t binIndex = threadIdx.x;
   size_t binSize = theHist->data_[binIndex].size(); 
   for (unsigned int j = 0; j < binSize; j++) 
   {
      int idOne = (theHist->data_[binIndex])[j];
      for (unsigned int i = j; i < binSize; i++)   
      {
       int idTwo = (theHist->data_[binIndex])[i];
       
       //const double dx = (*theHits)[idOne].x - (*theHits)[idTwo].x;
       //const double dy = (*theHits)[idOne].y - (*theHits)[idTwo].y;
       double distanceGPU = 100.0;//sqrt(dx*dx + dy*dy);
       if(distanceGPU < delta_c[0])
	{
	  //(*theHits)[idOne].rho += (*theHits)[idTwo].weight;
	  //density[0] = max(density[0],(*theHits)[idOne].rho);		
	} 
       
      }
   }
//     for (unsigned int j = 0; j < found_size; j++) {
//      if (distance(nd[i].data, found[j].data) < delta_c) {
//        nd[i].data.rho += found[j].data.weight;
//        maxdensity = std::max(maxdensity, nd[i].data.rho);
//      }
//     }
    
}//kernel



/*    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits)
        return;

    float eta = dInputData[rechitLocation].eta;
    float phi = dInputData[rechitLocation].phi;
    unsigned int index = dInputData[rechitLocation].index;

    dOutputData->fillBinGPU(eta, phi, index);

}
*/

double calculateLocalDensityGPU(BinnerGPU::Histo2D theHist, const LayerRecHitsGPU theHits, const unsigned int layer,std::vector<double> vecDeltas_){

  double maxdensity = 0.;
  float delta_c; // maximum search distance (critical distance) for local
                 // density calculation
  
 if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];
  std::cout<<delta_c;


  Histo2D *dInputHist;
  LayerRecHitsGPU* dInputRecHits;
  float* dDelta_c;
  double *dDensity,*hDensity;
  int numBins = theHist.data_.size();

  cudaMalloc(&dInputHist, sizeof(Histo2D));
  cudaMalloc(&dInputRecHits, sizeof(RecHitGPU)*theHits.size());
  cudaMalloc(&dDelta_c, sizeof(float));
  cudaMalloc(&dDensity,sizeof(double)*numBins);
  
  cudaMemset(dDensity, 0x00, sizeof(double)*numBins);

  cudaMemcpy(dDelta_c, &delta_c, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dInputHist, &theHist, sizeof(Histo2D), cudaMemcpyHostToDevice);
  cudaMemcpy(dInputRecHits, &theHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyHostToDevice);
  // Call the kernel
  const dim3 blockSize(numBins,1,1);
  const dim3 gridSize(1,1,1);
  kernel_compute_density <<<gridSize,blockSize>>>(dInputHist, dInputRecHits, dDelta_c,dDensity);
  
    // Copy result back!/
  cudaMemcpy(dDensity, &hDensity, sizeof(double)*numBins, cudaMemcpyDeviceToHost);

    // Free all the memory
  cudaFree(dDensity);
  cudaFree(dInputHist);
  cudaFree(dDelta_c);
  cudaFree(dInputRecHits);
  
  for(int j = 0; j< numBins; j++)
  {
   if (maxdensity < hDensity[j]) 
	maxdensity = hDensity[j];
  }

  return maxdensity;

}//calcualteLocalDensity

__device__ double distance2GPU(const RecHitGPU pt1, const RecHitGPU pt2){   //distance squared
  const double dx = pt1.x - pt2.x;
  const double dy = pt1.y - pt2.y;
  return (dx*dx + dy*dy);
} 

__global__ void kenrel_compute_distance_ToHigher(
    RecHitGPU* nd,
    size_t* rs, 
    int* nearestHigher,
    const double* max_dist2
){
  size_t oi = threadIdx.x + 1;

  {
    double dist2 = *max_dist2;
    unsigned int i = rs[oi];
    // we only need to check up to oi since hits
    // are ordered by decreasing density
    // and all points coming BEFORE oi are guaranteed to have higher rho
    // and the ones AFTER to have lower rho
    for (unsigned int oj = 0; oj < oi; ++oj) {
      unsigned int j = rs[oj];
      double tmp = distance2GPU(nd[i], nd[j]);
      if (tmp <= dist2) { // this "<=" instead of "<" addresses the (rare) case
                          // when there are only two hits
        dist2 = tmp;
        *nearestHigher = j;
      }
    }
    nd[i].delta = sqrt(dist2);
    // this uses the original unsorted hitlist
    nd[i].nearestHigher = *nearestHigher;
  }
}

void launch_kenrel_compute_distance_ToHigher(
  std::vector<RecHitGPU>& nd,
  std::vector<size_t>& rs,
  int& nearestHigher,
  const double max_dist2
){
  RecHitGPU* g_nd;
  size_t* g_rs; 
  int* g_nearestHigher;
  double* g_max_dist2;

  cudaMalloc(&g_nd, sizeof(RecHitGPU)*nd.size());
  cudaMalloc(&g_rs, sizeof(size_t)*rs.size());
  cudaMalloc(&g_nearestHigher,sizeof(int));
  cudaMalloc(&g_max_dist2, sizeof(double));

  cudaMemcpy(g_nd,            &nd[0],            sizeof(RecHitGPU)*nd.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(g_rs,            &rs[0],            sizeof(size_t)*rs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(g_nearestHigher, &nearestHigher, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g_max_dist2,     &max_dist2,     sizeof(double), cudaMemcpyHostToDevice);

  const dim3 blockSize(nd.size()-1,1,1);
  const dim3 gridSize(1,1,1);
  kenrel_compute_distance_ToHigher <<<gridSize,blockSize>>>(g_nd, g_rs, g_nearestHigher,g_max_dist2);

  cudaMemcpy(&nd[0],             g_nd,            sizeof(RecHitGPU)*nd.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&rs[0],             g_rs,            sizeof(size_t)*rs.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&nearestHigher,  g_nearestHigher, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(g_nd);
  cudaFree(g_rs);
  cudaFree(g_nearestHigher);
  cudaFree(g_max_dist2);
}

}//namespace
