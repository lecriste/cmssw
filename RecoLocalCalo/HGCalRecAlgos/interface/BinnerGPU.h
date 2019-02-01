#include <memory>
#include <iostream>
#include <vector>

#ifndef Binner_GPU_h
#define Binner_GPU_h

#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"

struct RecHitGPU { 
        unsigned int index;

        double x;
	double y;

	double eta;
        double phi;
	
        double weight;
	double rho;
        double delta;
   
        int nearestHigher;
   
        bool isBorder;
        bool isHalo;
   
        int clusterIndex;
   
        float sigmaNoise;
        float thickness;   
 	
	bool operator > (const RecHitGPU& rhs) const {
                return (rho > rhs.rho);
        }
};

typedef std::vector<RecHitGPU>       LayerRecHitsGPU;
typedef std::vector<LayerRecHitsGPU> HgcRecHitsGPU  ;

namespace BinnerGPU {
    // eta_width = 0.05
    // phi_width = 0.05
    // 2*pi/0.05 = 125
    // 1.4/0.05 = 28
    // 20 (as heuristic)

const int X_BINS=50;
const int Y_BINS=50;
const int MAX_DEPTH=100;

typedef histogram2D<int, X_BINS, Y_BINS, MAX_DEPTH> Histo2D;

Histo2D computeBins(LayerRecHitsGPU layerData);
}


#endif
