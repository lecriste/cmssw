#ifndef RECOHGCAL_TICL_TRACKSTERSPCA_H
#define RECOHGCAL_TICL_TRACKSTERSPCA_H

#include "FWCore/Framework/interface/Event.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include <vector>

namespace ticl {
  void assignPCAtoTracksters(std::vector<Trackster> &,
                             const std::vector<reco::CaloCluster> &,
                             double,
                             const edm::Event* iEvent = 0,
                             const edm::EDGetTokenT<HGCRecHitCollection>* hgcalRecHitsEEToken_ = 0,
                             const edm::EDGetTokenT<HGCRecHitCollection>* hgcalRecHitsFHToken_ = 0,
                             const edm::EDGetTokenT<HGCRecHitCollection>* hgcalRecHitsBHToken_ = 0,
                             const hgcal::RecHitTools* rhtools_ = 0,
                             bool energyWeight = true);

  void fillHitMap(std::map<DetId, const HGCRecHit*>&,
                  const HGCRecHitCollection&,
                  const HGCRecHitCollection&,
                  const HGCRecHitCollection&);

  std::vector<std::pair<DetId,HGCRecHit>> getRecHitsFromTrackster(const Trackster tks, const std::vector<reco::CaloCluster> &layerClusters,
                                                                  std::map<DetId, const HGCRecHit*>& hitMap);

}
#endif
