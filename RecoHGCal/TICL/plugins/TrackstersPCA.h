#ifndef RECOHGCAL_TICL_TRACKSTERSPCA_H
#define RECOHGCAL_TICL_TRACKSTERSPCA_H

#include "FWCore/Framework/interface/Event.h"
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
                             bool energyWeight = true);

  void fillHitMap(std::map<DetId, const HGCRecHit*>&,
                  const HGCRecHitCollection&,
                  const HGCRecHitCollection&,
                  const HGCRecHitCollection&);
}
#endif
