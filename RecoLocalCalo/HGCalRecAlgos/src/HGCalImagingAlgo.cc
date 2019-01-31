#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "tbb/task_arena.h"
#include "tbb/tbb.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"
#include <chrono>

using namespace BinnerGPU;

void HGCalImagingAlgo::populate(const HGCRecHitCollection &hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut

  if (dependSensor_) {
    // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
    // once
    computeThreshold();
  }

  std::vector<bool> firstHit(2 * (maxlayer + 1), true);
  std::vector<unsigned int> layerCounters(2 * (maxlayer + 1),0);

  for (unsigned int i = 0; i < hits.size(); ++i) {

    const HGCRecHit &hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layer = rhtools_.getLayerWithOffset(detid);
    float thickness = 0.f;
    // set sigmaNoise default value 1 to use kappa value directly in case of
    // sensor-independent thresholds
    float sigmaNoise = 1.f;
    if (dependSensor_) {
      thickness = rhtools_.getSiThickness(detid);
      int thickness_index = rhtools_.getSiThickIndex(detid);
      if (thickness_index == -1)
        thickness_index = 3;
      double storedThreshold = thresholds_[layer - 1][thickness_index];
      sigmaNoise = v_sigmaNoise_[layer - 1][thickness_index];

      if (hgrh.energy() < storedThreshold)
        continue; // this sets the ZS threshold at ecut times the sigma noise
                  // for the sensor
    }
    if (!dependSensor_ && hgrh.energy() < ecut_)
      continue;

    // map layers from positive endcap (z) to layer + maxlayer+1 to prevent
    // mixing up hits from different sides
    layer += int(rhtools_.zside(detid) > 0) * (maxlayer + 1);

    // determine whether this is a half-hexagon
    bool isHalf = rhtools_.isHalfCell(detid);
    const GlobalPoint position(rhtools_.getPosition(detid));

    // here's were the KDNode is passed its dims arguments - note that these are
    // *copied* from the Hexel
    points_[layer].emplace_back(
        Hexel(hgrh, detid, isHalf, sigmaNoise, thickness, &rhtools_),
        position.x(), position.y());
    
    RecHitGPU hit;
    hit.index = layerCounters[layer]; 
    hit.x = position.x();
    hit.y = position.y();
    hit.eta = std::fabs(position.eta());
    hit.phi = position.phi();
    hit.weight = hgrh.energy();
    hit.rho = 0.0;
 
    recHitsGPU[layer].push_back(hit);//{recHitCounter, position.x(),position.y(), position.eta(),position.phi(),hgrh.energy(),0.0});
    
    layerCounters[layer]++;

    // for each layer, store the minimum and maximum x and y coordinates for the
    // KDTreeBox boundaries
    if (firstHit[layer]) {
      minpos_[layer][0] = position.x();
      minpos_[layer][1] = position.y();
      maxpos_[layer][0] = position.x();
      maxpos_[layer][1] = position.y();
      firstHit[layer] = false;
    } else {
      minpos_[layer][0] = std::min((float)position.x(), minpos_[layer][0]);
      minpos_[layer][1] = std::min((float)position.y(), minpos_[layer][1]);
      maxpos_[layer][0] = std::max((float)position.x(), maxpos_[layer][0]);
      maxpos_[layer][1] = std::max((float)position.y(), maxpos_[layer][1]);
    }

  } // end loop hits

   int count = 0;
   for(auto layer: recHitsGPU) {
     std::cout<<"Layer "<<(count++)<<" Rechits: "<<layer.size()<<std::endl;
   }
}
// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
void HGCalImagingAlgo::makeClusters() {

  std::cout << "- makeClusters() starts" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // For each layer, assign all RecHits to a bin of a Histo2D
  std::vector< BinnerGPU::Histo2D > histosGPU;

  for (auto&layer: recHitsGPU) {
    histosGPU.push_back( BinnerGPU::computeBins(layer) );
    //std::cout<<BinnerGPU::computeBins(layer).data_[0].size() << std::endl;
  }
 
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";

  layerClustersPerLayer_.resize(2 * maxlayer + 2);
  // assign all hits in each layer to a cluster core or halo
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer + 2), [&](size_t i) {
      KDTreeBox bounds(minpos_[i][0], maxpos_[i][0], minpos_[i][1], maxpos_[i][1]);
      KDTree hit_kdtree;
      hit_kdtree.build(points_[i], bounds);

      unsigned int actualLayer =
          i > maxlayer
              ? (i - (maxlayer + 1))
              : i; // maps back from index used for KD trees to actual layer
  	    std::cout<< "======= Layer "<< i ;
      double maxdensity = calculateLocalDensity(
          points_[i], hit_kdtree, actualLayer); // also stores rho (energy
                                               // density) for each point (node)
      double maxdensityBin = calculateLocalDensityCPU(histosGPU[i],recHitsGPU[i],i);
      
      //double maxdensityBin = 0.0;
      std::cout << "LocalD: " << maxdensity << " LocalDBin: " << maxdensityBin << std::endl;
      std::cout << "Original : " << points_[i].size()<< std::endl; 
      size_t numBins = histosGPU[i].data_.size();
      int allrechitbinned = 0;
      for (unsigned int ii = 0; ii < numBins; ii++)
      {
        double tmpDensity = 0.0;
        size_t binSize = histosGPU[i].data_[ii].size();
        allrechitbinned = allrechitbinned + binSize;
     
      } 
      std::cout << "Histo - Num Bins : " << numBins << " Entries : " << allrechitbinned<<std::endl;
      // calculate distance to nearest point with higher density storing
      // distance (delta) and point's index
      calculateDistanceToHigher(points_[i]);
      calculateDistanceToHigherGPU(recHitsGPU[i]);

      findAndAssignClusters(points_[i], hit_kdtree, maxdensity, bounds,
                            actualLayer, layerClustersPerLayer_[i]);

      findAndAssignClustersGPU(maxdensity, actualLayer,
			       recHitsGPU[i], histosGPU[i], 
			       layerClustersPerLayer_[i]);

    });
  });

}

std::vector<reco::BasicCluster> HGCalImagingAlgo::getClusters(bool doSharing) {

  reco::CaloID caloID = reco::CaloID::DET_HGCAL_ENDCAP;
  std::vector<std::pair<DetId, float>> thisCluster;
  for (auto &clsOnLayer : layerClustersPerLayer_) {
    for (unsigned int i = 0; i < clsOnLayer.size(); ++i) {
      double energy = 0;
      Point position;

      if (doSharing) {

        std::vector<unsigned> seeds = findLocalMaximaInCluster(clsOnLayer[i]);
        // sharing found seeds.size() sub-cluster seeds in cluster i

        std::vector<std::vector<double>> fractions;
        // first pass can have noise it in
        shareEnergy(clsOnLayer[i], seeds, fractions);

        // reset and run second pass after vetoing seeds
        // that result in trivial clusters (less than 2 effective cells)

        for (unsigned isub = 0; isub < fractions.size(); ++isub) {
          double effective_hits = 0.0;
          double energy =
              calculateEnergyWithFraction(clsOnLayer[i], fractions[isub]);
          Point position =
              calculatePositionWithFraction(clsOnLayer[i], fractions[isub]);

          for (unsigned ihit = 0; ihit < fractions[isub].size(); ++ihit) {
            const double fraction = fractions[isub][ihit];
            if (fraction > 1e-7) {
              effective_hits += fraction;
              thisCluster.emplace_back(clsOnLayer[i][ihit].data.detid,
                                       fraction);
            }
          }

          if (verbosity_ < pINFO) {
            std::cout << "\t******** NEW CLUSTER (SHARING) ********"
                      << std::endl;
            std::cout << "\tEff. No. of cells = " << effective_hits
                      << std::endl;
            std::cout << "\t     Energy       = " << energy << std::endl;
            std::cout << "\t     Phi          = " << position.phi()
                      << std::endl;
            std::cout << "\t     Eta          = " << position.eta()
                      << std::endl;
            std::cout << "\t*****************************" << std::endl;
          }
          clusters_v_.emplace_back(energy, position, caloID, thisCluster,
                                  algoId_);
          thisCluster.clear();
        }
      } else {
        position = calculatePosition(clsOnLayer[i]); // energy-weighted position
        //   std::vector< KDNode >::iterator it;
        for (auto &it : clsOnLayer[i]) {
          energy += it.data.isHalo ? 0. : it.data.weight;
          // use fraction to store whether this is a Halo hit or not
          thisCluster.emplace_back(it.data.detid, (it.data.isHalo ? 0.f : 1.f));
        }
        if (verbosity_ < pINFO) {
          std::cout << "******** NEW CLUSTER (HGCIA) ********" << std::endl;
          std::cout << "Index          " << i << std::endl;
          std::cout << "No. of cells = " << clsOnLayer[i].size() << std::endl;
          std::cout << "     Energy     = " << energy << std::endl;
          std::cout << "     Phi        = " << position.phi() << std::endl;
          std::cout << "     Eta        = " << position.eta() << std::endl;
          std::cout << "*****************************" << std::endl;
        }
        clusters_v_.emplace_back(energy, position, caloID, thisCluster, algoId_);
        thisCluster.clear();
      }
    }
  }
  return clusters_v_;
}

math::XYZPoint
HGCalImagingAlgo::calculatePosition(std::vector<KDNode> &v) const {
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  unsigned int v_size = v.size();
  unsigned int maxEnergyIndex = 0;
  float maxEnergyValue = 0;
  bool haloOnlyCluster = true;

  // loop over hits in cluster candidate building up weight for
  // energy-weighted position calculation and determining the maximum
  // energy hit in case this is a halo-only cluster
  for (unsigned int i = 0; i < v_size; i++) {
    if (!v[i].data.isHalo) {
      haloOnlyCluster = false;
      total_weight += v[i].data.weight;
      x += v[i].data.x * v[i].data.weight;
      y += v[i].data.y * v[i].data.weight;
      z += v[i].data.z * v[i].data.weight;
    } else {
      if (v[i].data.weight > maxEnergyValue) {
        maxEnergyValue = v[i].data.weight;
        maxEnergyIndex = i;
      }
    }
  }

  if (!haloOnlyCluster) {
    if (total_weight != 0) {
      auto inv_tot_weight = 1. / total_weight;
      return math::XYZPoint(x * inv_tot_weight, y * inv_tot_weight,
                            z * inv_tot_weight);
    }
  } else if (v_size > 0) {
    // return position of hit with maximum energy
    return math::XYZPoint(v[maxEnergyIndex].data.x, v[maxEnergyIndex].data.y,
                          v[maxEnergyIndex].data.z);
  }
  return math::XYZPoint(0, 0, 0);
}

double HGCalImagingAlgo::calculateLocalDensity(std::vector<KDNode> &nd,
                                               KDTree &lp,
                                               const unsigned int layer) const {

  double maxdensity = 0.;
  float delta_c; // maximum search distance (critical distance) for local
                 // density calculation
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];
  
  std::cout << " - delta : " << delta_c << std::endl;
  // for each node calculate local density rho and store it
  for (unsigned int i = 0; i < nd.size(); ++i) {
    // speec up search by looking within +/- delta_c window only
    KDTreeBox search_box(nd[i].dims[0] - delta_c, nd[i].dims[0] + delta_c,
                         nd[i].dims[1] - delta_c, nd[i].dims[1] + delta_c);
    std::vector<KDNode> found;
    lp.search(search_box, found);
    const unsigned int found_size = found.size();
  //  std::cout << i << " ";
int counter = 0;
    for (unsigned int j = 0; j < found_size; j++) {
 
      if (distance(nd[i].data, found[j].data) < delta_c) {
        nd[i].data.rho += found[j].data.weight;
        maxdensity = std::max(maxdensity, nd[i].data.rho);
	counter++;      
      }

    } // end loop found
   std::cout << i << " has " << counter << " neighbours - energy : " << nd[i].data.weight << " - rho " << nd[i].data.rho <<std::endl;

  }   // end loop nodes
  return maxdensity;
}

double
HGCalImagingAlgo::calculateDistanceToHigher(std::vector<KDNode> &nd) const {

  // sort vector of Hexels by decreasing local density
  std::vector<size_t> rs = sorted_indices(nd);

  double maxdensity = 0.0;
  int nearestHigher = -1;

  if (!rs.empty())
    maxdensity = nd[rs[0]].data.rho;
  else
    return maxdensity; // there are no hits
  double dist2 = 0.;
  // start by setting delta for the highest density hit to
  // the most distant hit - this is a convention

  for (auto &j : nd) {
    double tmp = distance2(nd[rs[0]].data, j.data);
    if (tmp > dist2)
      dist2 = tmp;
  }
  nd[rs[0]].data.delta = std::sqrt(dist2);
  nd[rs[0]].data.nearestHigher = nearestHigher;

  // now we save the largest distance as a starting point
  const double max_dist2 = dist2;
  const unsigned int nd_size = nd.size();

  for (unsigned int oi = 1; oi < nd_size;
       ++oi) { // start from second-highest density
    dist2 = max_dist2;
    unsigned int i = rs[oi];
    // we only need to check up to oi since hits
    // are ordered by decreasing density
    // and all points coming BEFORE oi are guaranteed to have higher rho
    // and the ones AFTER to have lower rho
    for (unsigned int oj = 0; oj < oi; ++oj) {
      unsigned int j = rs[oj];
      double tmp = distance2(nd[i].data, nd[j].data);
      if (tmp <= dist2) { // this "<=" instead of "<" addresses the (rare) case
                          // when there are only two hits
        dist2 = tmp;
        nearestHigher = j;
      }
    }
    nd[i].data.delta = std::sqrt(dist2);
    nd[i].data.nearestHigher =
        nearestHigher; // this uses the original unsorted hitlist
  }
  return maxdensity;
}

double
HGCalImagingAlgo::calculateDistanceToHigherGPU(std::vector<RecHitGPU> &nd) const {

  // sort vector of Hexels by decreasing local density
  std::vector<size_t> rs = sorted_indices(nd);
  //for (size_t n = 0; n < rs.size(); n++)
    //std::cout <<rs[n] <<" ";
  //std::cout <<std::endl;
  //exit(0);

  double maxdensity = 0.0;
  int nearestHigher = -1;

  if (!rs.empty())
    maxdensity = nd[rs[0]].rho;
  else
    return maxdensity; // there are no hits
  double dist2 = 0.;
  // start by setting delta for the highest density hit to
  // the most distant hit - this is a convention

  for (auto &j : nd) {
    double tmp = distance2GPU(nd[rs[0]], j);
    if (tmp > dist2)
      dist2 = tmp;
  }
  nd[rs[0]].delta = std::sqrt(dist2);
  nd[rs[0]].nearestHigher = nearestHigher;

  // now we save the largest distance as a starting point
  const double max_dist2 = dist2;
  const unsigned int nd_size = nd.size();

  for (unsigned int oi = 1; oi < nd_size;
       ++oi) { // start from second-highest density
    dist2 = max_dist2;
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
        nearestHigher = j;
      }
    }
    nd[i].delta = std::sqrt(dist2);
    nd[i].nearestHigher =
        nearestHigher; // this uses the original unsorted hitlist
  }
  return maxdensity;
}

int HGCalImagingAlgo::findAndAssignClusters(
    std::vector<KDNode> &nd, KDTree &lp, double maxdensity, KDTreeBox &bounds,
    const unsigned int layer,
    std::vector<std::vector<KDNode>> &clustersOnLayer) const {

  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...

  unsigned int nClustersOnLayer = 0;
  float delta_c; // critical distance
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  std::vector<size_t> rs =
      sorted_indices(nd); // indices sorted by decreasing rho
  std::vector<size_t> ds =
      sort_by_delta(nd); // sort in decreasing distance to higher

  const unsigned int nd_size = nd.size();
  for (unsigned int i = 0; i < nd_size; ++i) {

    if (nd[ds[i]].data.delta < delta_c)
      break; // no more cluster centers to be looked at
    if (dependSensor_) {

      float rho_c = kappa_ * nd[ds[i]].data.sigmaNoise;
      if (nd[ds[i]].data.rho < rho_c)
        continue; // set equal to kappa times noise threshold

    } else if (nd[ds[i]].data.rho * kappa_ < maxdensity)
      continue;

    nd[ds[i]].data.clusterIndex = nClustersOnLayer;
    if (verbosity_ < pINFO) {
      std::cout << "Adding new cluster with index " << nClustersOnLayer
                << std::endl;
      std::cout << "Cluster center is hit " << ds[i] << std::endl;
    }
    nClustersOnLayer++;
  }

  // at this point nClustersOnLayer is equal to the number of cluster centers -
  // if it is zero we are  done
  if (nClustersOnLayer == 0)
    return nClustersOnLayer;

  // assign remaining points to clusters, using the nearestHigher set from
  // previous step (always set except
  // for top density hit that is skipped...)
  for (unsigned int oi = 1; oi < nd_size; ++oi) {
    unsigned int i = rs[oi];
    int ci = nd[i].data.clusterIndex;
    if (ci ==
        -1) { // clusterIndex is initialised with -1 if not yet used in cluster
      nd[i].data.clusterIndex = nd[nd[i].data.nearestHigher].data.clusterIndex;
    }
  }

  // make room in the temporary cluster vector for the additional clusterIndex
  // clusters
  // from this layer
  if (verbosity_ < pINFO) {
    std::cout << "resizing cluster vector by " << nClustersOnLayer << std::endl;
  }
  clustersOnLayer.resize(nClustersOnLayer);

  // assign points closer than dc to other clusters to border region
  // and find critical border density
  std::vector<double> rho_b(nClustersOnLayer, 0.);
  lp.clear();
  lp.build(nd, bounds);
  // now loop on all hits again :( and check: if there are hits from another
  // cluster within d_c -> flag as border hit
  for (unsigned int i = 0; i < nd_size; ++i) {
    int ci = nd[i].data.clusterIndex;
    bool flag_isolated = true;
    if (ci != -1) {
      KDTreeBox search_box(nd[i].dims[0] - delta_c, nd[i].dims[0] + delta_c,
                           nd[i].dims[1] - delta_c, nd[i].dims[1] + delta_c);
      std::vector<KDNode> found;
      lp.search(search_box, found);

      const unsigned int found_size = found.size();
      for (unsigned int j = 0; j < found_size;
           j++) { // start from 0 here instead of 1
        // check if the hit is not within d_c of another cluster
        if (found[j].data.clusterIndex != -1) {
          float dist = distance(found[j].data, nd[i].data);
          if (dist < delta_c && found[j].data.clusterIndex != ci) {
            // in which case we assign it to the border
            nd[i].data.isBorder = true;
            break;
          }
          // because we are using two different containers, we have to make sure
          // that we don't unflag the
          // hit when it finds *itself* closer than delta_c
          if (dist < delta_c && dist != 0. &&
              found[j].data.clusterIndex == ci) {
            // in this case it is not an isolated hit
            // the dist!=0 is because the hit being looked at is also inside the
            // search box and at dist==0
            flag_isolated = false;
          }
        }
      }
      if (flag_isolated)
        nd[i].data.isBorder =
            true; // the hit is more than delta_c from any of its brethren
    }
    // check if this border hit has density larger than the current rho_b and
    // update
    if (nd[i].data.isBorder && rho_b[ci] < nd[i].data.rho)
      rho_b[ci] = nd[i].data.rho;
  } // end loop all hits

  // flag points in cluster with density < rho_b as halo points, then fill the
  // cluster vector
  for (unsigned int i = 0; i < nd_size; ++i) {
    int ci = nd[i].data.clusterIndex;
    if (ci != -1) {
      if (nd[i].data.rho <= rho_b[ci])
        nd[i].data.isHalo = true;
      clustersOnLayer[ci].push_back(nd[i]);
      if (verbosity_ < pINFO) {
        std::cout << "Pushing hit " << i << " into cluster with index " << ci
                  << std::endl;
      }
    }
  }

  // prepare the offset for the next layer if there is one
  if (verbosity_ < pINFO) {
    std::cout << "moving cluster offset by " << nClustersOnLayer << std::endl;
  }
  return nClustersOnLayer;
}

// GPU version
int HGCalImagingAlgo::findAndAssignClustersGPU(
    double maxdensity,  const unsigned int layer,
    LayerRecHitsGPU &layerRecHitsGPU, const BinnerGPU::Histo2D histoGPU,   
    std::vector<std::vector<KDNode>> &clustersOnLayer) const {
    //std::vector<KDNode> &nd, KDTree &lp, double maxdensity, KDTreeBox &bounds,

  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...

  unsigned int nClustersOnLayer = 0;
  float delta_c; // critical distance
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  // std::vector<size_t> rs =
  //     sorted_indices(layerRecHitsGPU); // indices sorted by decreasing rho
  // std::vector<size_t> ds =
  //     sort_by_delta(layerRecHitsGPU); // sort in decreasing distance to higher

  GPU::VecArray<int, BinnerGPU::MAX_DEPTH> rs, ds;
  uint nds = 0;
  uint nrs = 0;
  //const unsigned int layerRecHitsGPU_size = layerRecHitsGPU.size();

  // Loop over bins from binsGPU
  GPU::VecArray<int, BinnerGPU::MAX_DEPTH> indicesRH;
  LayerRecHitsGPU binRecHitsGpu;
  RecHitGPU recHitGpu;
  //
  //if (verbosity_ < pINFO)  //fixme
  std::cout << "-- Start loop over bins from binsGPU" << std::endl;

  for(unsigned int iEta = 0; iEta < BinnerGPU::ETA_BINS; ++iEta) {
    for(unsigned int iPhi = 0; iPhi < BinnerGPU::PHI_BINS; ++iPhi) {

      // Get list of RecHit indices from the 2d histogram
      indicesRH = histoGPU.getBinContent(iEta, iPhi);

      // Build a vector containing all the RecHitGPU from the current 2D bin
      //binRecHitsGpu = 0;

      // Sort the RecHits from the current 2D bin
      rs = sorted_indices(layerRecHitsGPU, indicesRH); // indices sorted by decreasing rho
      ds = sort_by_delta( layerRecHitsGPU, indicesRH); // sort in decreasing distance to higher
      //
      nrs = (uint)rs.size(); // fixme: cast is ok?
      nds = (uint)ds.size(); // fixme: cast is ok?

      // First step: find cluster seeds by looping over RecHis sorted by decreasing rho
      std::cout << "---- Bin (iEta,iPhi)=(" << iEta 
		<< "," << iPhi << ")" << std::endl
		<< "---- Start looping over nds=" << nds 
		<< " RecHits (sorted by decreasing rho)" << std::endl
		<< "---- By the way, there are nrs=" << nrs 
		<< " RecHits (sorted by derceasing distance to nearest local max)" << std::endl;
      //
      for(uint idxRh = 0; idxRh < nds; ++idxRh) { 

	// Get the RecHit corresponding to this index
	if( idxRh < layerRecHitsGPU.size() ) {
	  recHitGpu = layerRecHitsGPU[idxRh];
	}
	else {
	  //if (verbosity_ < pINFO) {
	  std::cout << "---- Error: RecHit with index #" << idxRh
		    << " not found! Continue to next index record." << std::endl
		    << "delta_c = " << delta_c << std::endl;
	  //}
	  continue;
	}

	// Stop searching for cluster seeds when distance to nearest local maximum
	// is below threshold delta_c
	if (recHitGpu.delta < delta_c)
	  break;

	// Apply a lower cut on the energy density 
	if (dependSensor_) { // cut based on a multiple of the noise
	  float rho_c = kappa_ * recHitGpu.sigmaNoise;
	  if (recHitGpu.rho < rho_c)
	    continue; 
	} 
	else if (recHitGpu.rho * kappa_ < maxdensity)
	  continue; // cut based on maximal local density

	recHitGpu.clusterIndex = nClustersOnLayer;
	//if (verbosity_ < pINFO) { //fixme
	std::cout << "Adding new cluster with index " << nClustersOnLayer
		  << std::endl;
	  //std::cout << "Cluster center is hit " << ds[i] << std::endl;
	  //	}
	nClustersOnLayer++;

      } // end loop over MAX_DEPTH
    } // end loop PHI_BINS
  } // end loop ETA_BINS

  /*
  // at this point nClustersOnLayer is equal to the number of cluster centers -
  // if it is zero we are  done
  if (nClustersOnLayer == 0)
    return nClustersOnLayer;

  // assign remaining points to clusters, using the nearestHigher set from
  // previous step (always set except
  // for top density hit that is skipped...)
  for (unsigned int oi = 1; oi < layerRecHitsGPU_size; ++oi) {
    unsigned int i = rs[oi];
    int ci = layerRecHitsGPU[i].clusterIndex;
    if (ci ==
        -1) { // clusterIndex is initialised with -1 if not yet used in cluster
      layerRecHitsGPU[i].clusterIndex = layerRecHitsGPU[layerRecHitsGPU[i].nearestHigher].clusterIndex;
    }
  }

  // make room in the temporary cluster vector for the additional clusterIndex
  // clusters
  // from this layer
  if (verbosity_ < pINFO) {
    std::cout << "resizing cluster vector by " << nClustersOnLayer << std::endl;
  }
  clustersOnLayer.resize(nClustersOnLayer);

  // assign points closer than dc to other clusters to border region
  // and find critical border density
  std::vector<double> rho_b(nClustersOnLayer, 0.);
  lp.clear();
  lp.build(layerRecHitsGPU, bounds); //fixme: all the region below needs to be fixed //ND
  // now loop on all hits again :( and check: if there are hits from another
  // cluster within d_c -> flag as border hit
  for (unsigned int i = 0; i < layerRecHitsGPU_size; ++i) {
    int ci = layerRecHitsGPU[i].clusterIndex;
    bool flag_isolated = true;
    if (ci != -1) {
      KDTreeBox search_box(nd[i].dims[0] - delta_c, nd[i].dims[0] + delta_c,
                           nd[i].dims[1] - delta_c, nd[i].dims[1] + delta_c);
      std::vector<KDNode> found;
      lp.search(search_box, found);

      const unsigned int found_size = found.size();
      for (unsigned int j = 0; j < found_size;
           j++) { // start from 0 here instead of 1
        // check if the hit is not within d_c of another cluster
        if (found[j].clusterIndex != -1) {
          float dist = distance(found[j], nd[i]);
          if (dist < delta_c && found[j].clusterIndex != ci) {
            // in which case we assign it to the border
            nd[i].isBorder = true;
            break;
          }
          // because we are using two different containers, we have to make sure
          // that we don't unflag the
          // hit when it finds *itself* closer than delta_c
          if (dist < delta_c && dist != 0. &&
              found[j].clusterIndex == ci) {
            // in this case it is not an isolated hit
            // the dist!=0 is because the hit being looked at is also inside the
            // search box and at dist==0
            flag_isolated = false;
          }
        }
      }
      if (flag_isolated)
        nd[i].isBorder =
            true; // the hit is more than delta_c from any of its brethren
    }
    // check if this border hit has density larger than the current rho_b and
    // update
    if (nd[i].isBorder && rho_b[ci] < nd[i].rho)
      rho_b[ci] = nd[i].rho;
  } // end loop all hits

  // flag points in cluster with density < rho_b as halo points, then fill the
  // cluster vector
  for (unsigned int i = 0; i < nd_size; ++i) {
    int ci = nd[i].clusterIndex;
    if (ci != -1) {
      if (nd[i].rho <= rho_b[ci])
        nd[i].isHalo = true;
      clustersOnLayer[ci].push_back(nd[i]);
      if (verbosity_ < pINFO) {
        std::cout << "Pushing hit " << i << " into cluster with index " << ci
                  << std::endl;
      }
    }
  }

  // prepare the offset for the next layer if there is one
  if (verbosity_ < pINFO) {
    std::cout << "moving cluster offset by " << nClustersOnLayer << std::endl;
  }
	*/

  std::cout << "-- End findAndAssignClusters()." << std::endl;

  return nClustersOnLayer;
}

// find local maxima within delta_c, marking the indices in the cluster
std::vector<unsigned>
HGCalImagingAlgo::findLocalMaximaInCluster(const std::vector<KDNode> &cluster) {
  std::vector<unsigned> result;
  std::vector<bool> seed(cluster.size(), true);
  float delta_c = 2.;

  for (unsigned i = 0; i < cluster.size(); ++i) {
    for (unsigned j = 0; j < cluster.size(); ++j) {
      if (i != j and distance(cluster[i].data, cluster[j].data) < delta_c) {
        if (cluster[i].data.weight < cluster[j].data.weight) {
          seed[i] = false;
          break;
        }
      }
    }
  }

  for (unsigned i = 0; i < cluster.size(); ++i) {
    if (seed[i] && cluster[i].data.weight > 5e-4) {
      // seed at i with energy cluster[i].weight
      result.push_back(i);
    }
  }

  // Found result.size() sub-clusters in input cluster of length cluster.size()

  return result;
}

math::XYZPoint HGCalImagingAlgo::calculatePositionWithFraction(
    const std::vector<KDNode> &hits, const std::vector<double> &fractions) {
  double norm(0.0), x(0.0), y(0.0), z(0.0);
  for (unsigned i = 0; i < hits.size(); ++i) {
    const double weight = fractions[i] * hits[i].data.weight;
    norm += weight;
    x += weight * hits[i].data.x;
    y += weight * hits[i].data.y;
    z += weight * hits[i].data.z;
  }
  math::XYZPoint result(x, y, z);
  result /= norm;
  return result;
}

double HGCalImagingAlgo::calculateEnergyWithFraction(
    const std::vector<KDNode> &hits, const std::vector<double> &fractions) {
  double result = 0.0;
  for (unsigned i = 0; i < hits.size(); ++i) {
    result += fractions[i] * hits[i].data.weight;
  }
  return result;
}

void HGCalImagingAlgo::shareEnergy(
    const std::vector<KDNode> &incluster, const std::vector<unsigned> &seeds,
    std::vector<std::vector<double>> &outclusters) {
  std::vector<bool> isaseed(incluster.size(), false);
  outclusters.clear();
  outclusters.resize(seeds.size());
  std::vector<Point> centroids(seeds.size());
  std::vector<double> energies(seeds.size());

  if (seeds.size() == 1) { // short circuit the case of a lone cluster
    outclusters[0].clear();
    outclusters[0].resize(incluster.size(), 1.0);
    return;
  }

  // saving seeds

  // create quick seed lookup
  for (unsigned i = 0; i < seeds.size(); ++i) {
    isaseed[seeds[i]] = true;
  }

  // initialize clusters to be shared
  // centroids start off at seed positions
  // seeds always have fraction 1.0, to stabilize fit
  // initializing fit
  for (unsigned i = 0; i < seeds.size(); ++i) {
    outclusters[i].resize(incluster.size(), 0.0);
    for (unsigned j = 0; j < incluster.size(); ++j) {
      if (j == seeds[i]) {
        outclusters[i][j] = 1.0;
        centroids[i] = math::XYZPoint(incluster[j].data.x, incluster[j].data.y,
                                      incluster[j].data.z);
        energies[i] = incluster[j].data.weight;
      }
    }
  }

  // run the fit while we are less than max iterations, and clusters are still
  // moving
  const double minFracTot = 1e-20;
  unsigned iter = 0;
  const unsigned iterMax = 50;
  double diff = std::numeric_limits<double>::max();
  const double stoppingTolerance = 1e-8;
  const auto numberOfSeeds = seeds.size();
  auto toleranceScaling =
      numberOfSeeds > 2 ? (numberOfSeeds - 1) * (numberOfSeeds - 1) : 1;
  std::vector<Point> prevCentroids;
  std::vector<double> frac(numberOfSeeds), dist2(numberOfSeeds);
  while (iter++ < iterMax && diff > stoppingTolerance * toleranceScaling) {
    for (unsigned i = 0; i < incluster.size(); ++i) {
      const Hexel &ihit = incluster[i].data;
      double fracTot(0.0);
      for (unsigned j = 0; j < numberOfSeeds; ++j) {
        double fraction = 0.0;
        double d2 = (std::pow(ihit.x - centroids[j].x(), 2.0) +
                     std::pow(ihit.y - centroids[j].y(), 2.0) +
                     std::pow(ihit.z - centroids[j].z(), 2.0)) /
                    sigma2_;
        dist2[j] = d2;
        // now we set the fractions up based on hit type
        if (i == seeds[j]) { // this cluster's seed
          fraction = 1.0;
        } else if (isaseed[i]) {
          fraction = 0.0;
        } else {
          fraction = energies[j] * std::exp(-0.5 * d2);
        }
        fracTot += fraction;
        frac[j] = fraction;
      }
      // now that we have calculated all fractions for all hits
      // assign the new fractions
      for (unsigned j = 0; j < numberOfSeeds; ++j) {
        if (fracTot > minFracTot || (i == seeds[j] && fracTot > 0.0)) {
          outclusters[j][i] = frac[j] / fracTot;
        } else {
          outclusters[j][i] = 0.0;
        }
      }
    }

    // save previous centroids
    prevCentroids = std::move(centroids);
    // finally update the position of the centroids from the last iteration
    centroids.resize(numberOfSeeds);
    double diff2 = 0.0;
    for (unsigned i = 0; i < numberOfSeeds; ++i) {
      centroids[i] = calculatePositionWithFraction(incluster, outclusters[i]);
      energies[i] = calculateEnergyWithFraction(incluster, outclusters[i]);
      // calculate convergence parameters
      const double delta2 = (prevCentroids[i] - centroids[i]).perp2();
      diff2 = std::max(delta2, diff2);
    }
    // update convergance parameter outside loop
    diff = std::sqrt(diff2);
  }
}

void HGCalImagingAlgo::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors, while the last one, number 3 (the
  // fourth) will address the Scintillators. This change will support both
  // geometries at the same time.

  if (initialized_)
    return; // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;
  const unsigned maxNumberOfThickIndices = 3;
  dummy.resize(maxNumberOfThickIndices + 1, 0); // +1 to accomodate for the Scintillators
  thresholds_.resize(maxlayer, dummy);
  v_sigmaNoise_.resize(maxlayer, dummy);

  for (unsigned ilayer = 1; ilayer <= maxlayer; ++ilayer) {
    for (unsigned ithick = 0; ithick < maxNumberOfThickIndices; ++ithick) {
      float sigmaNoise =
          0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
          (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
      thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
      v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
    }
    float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer];
    thresholds_[ilayer - 1][maxNumberOfThickIndices] = ecut_ * scintillators_sigmaNoise;
    v_sigmaNoise_[ilayer -1][maxNumberOfThickIndices] = scintillators_sigmaNoise;
  }

}

double HGCalImagingAlgo::calculateLocalDensityCPU(Histo2D hist, LayerRecHitsGPU hits, const unsigned int layer) const
{
   float delta_c; // maximum search distance (critical distance) for local
                 // density calculation
   if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
   else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
   else
    delta_c = vecDeltas_[2];
   
   double maxdensity = 0.0;
   
   size_t numBins = hist.data_.size();
   
   for (unsigned int i = 0; i < numBins; i++) 
   {
   	double tmpDensity = 0.0;
	size_t binSize = hist.data_[i].size();
	
 	for (unsigned int j = 0; j < binSize; j++)
   	{	
		int counter = 0;
      		int idOne = hist.data_[i][j];
	
      		for (unsigned int k = j; k < binSize; k++)
      		{	 
       			int idTwo = hist.data_[i][k];
		//std::cout << idTwo << " - ";	
			if(distanceGPU(hits[idOne],hits[idTwo]) < delta_c)
       			{
				
			       counter++;
				hits[idOne].rho += hits[idTwo].weight;
				tmpDensity = std::max(tmpDensity,hits[idOne].rho);
       			}
      		}
		std::cout << "layer : " << layer << " - " << idOne << " has " << counter << " neighbours - energy : " << hits[idOne].weight << " - rho " << hits[idOne].rho <<std::endl;
   	}
	maxdensity = std::max(maxdensity,tmpDensity);
   }

   return maxdensity;
}




