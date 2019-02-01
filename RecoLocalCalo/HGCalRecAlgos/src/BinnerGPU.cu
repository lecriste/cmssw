#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include <math.h>

#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"


namespace BinnerGPU {


  __global__ void kernel_compute_histogram(RecHitGPU*dInputData, Histo2D *dOutputData, const size_t numRechits) {

    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits)
        return;

    float x = dInputData[rechitLocation].x;
    float y = dInputData[rechitLocation].y;
   
    dOutputData->fillBinGPU(x, y, rechitLocation);

  }


  float minEta = 1.6;
  float maxEta = 3.0;
  float minPhi = -M_PI;
  float maxPhi = M_PI;

  float minX = -500.0, minY = -500.0;
  float maxX = 500.0, maxY = 500.0;

  Histo2D computeBins(std::vector<RecHitGPU> layerData) {
    Histo2D hOutputData(minX, maxX, minY, maxY);

    // Allocate memory and put data into device
    Histo2D *dOutputData;
    RecHitGPU* dInputData;
    cudaMalloc(&dOutputData, sizeof(Histo2D));
    cudaMalloc(&dInputData, sizeof(RecHitGPU)*layerData.size());
    cudaMemcpy(dInputData, layerData.data(), sizeof(RecHitGPU)*layerData.size(), cudaMemcpyHostToDevice);
    cudaMemset(dOutputData, 0x00, sizeof(Histo2D));
    cudaMemcpy(dOutputData, &hOutputData, sizeof(Histo2D), cudaMemcpyHostToDevice);
    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(ceil(layerData.size()/1024.0),1,1);
    kernel_compute_histogram <<<gridSize,blockSize>>>(dInputData, dOutputData, layerData.size());

    // Copy result back!
    cudaMemcpy(&hOutputData, dOutputData, sizeof(Histo2D), cudaMemcpyDeviceToHost);
  
    // Free all the memory
    cudaFree(dOutputData);
    cudaFree(dInputData);

    
    return hOutputData;
  }

}
