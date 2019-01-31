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


}//namespace
