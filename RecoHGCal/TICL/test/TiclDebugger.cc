//
// Original Author:  Marco Rovere
//         Created:  Fri May  1 07:21:02 CEST 2020
//
//
//
// system include files
#include <memory>
#include <iostream>
#include <numeric>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
//
// class declaration
//

class TiclDebugger : public edm::one::EDAnalyzer<> {
public:
  explicit TiclDebugger(const edm::ParameterSet&);
  ~TiclDebugger() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::InputTag trackstersMerge_;
  const edm::InputTag tracks_;
  const edm::InputTag caloParticles_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersMergeToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticlesToken_;
};

TiclDebugger::TiclDebugger(const edm::ParameterSet& iConfig)
    : trackstersMerge_(iConfig.getParameter<edm::InputTag>("trackstersMerge")),
      tracks_(iConfig.getParameter<edm::InputTag>("tracks")),
      caloParticles_(iConfig.getParameter<edm::InputTag>("caloParticles")) {
  edm::ConsumesCollector&& iC = consumesCollector();
  trackstersMergeToken_ = iC.consumes<std::vector<ticl::Trackster>>(trackstersMerge_);
  tracksToken_ = iC.consumes<std::vector<reco::Track>>(tracks_);
  caloParticlesToken_ = iC.consumes<std::vector<CaloParticle>>(caloParticles_);
}

TiclDebugger::~TiclDebugger() {}

void TiclDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  static const char* particle_kind[] = {"gam", "e", "mu", "pi0", "h", "h0", "?", "!"};
  using namespace edm;
  using std::begin;
  using std::end;
  using std::iota;
  using std::sort;

  edm::Handle<std::vector<ticl::Trackster>> trackstersMergeH;

  iEvent.getByToken(trackstersMergeToken_, trackstersMergeH);
  auto const& tracksters = *trackstersMergeH.product();
  std::vector<int> sorted_tracksters_idx(tracksters.size());
  iota(begin(sorted_tracksters_idx), end(sorted_tracksters_idx), 0);
  sort(begin(sorted_tracksters_idx), end(sorted_tracksters_idx), [&tracksters](int i, int j) {
    return tracksters[i].raw_energy() > tracksters[j].raw_energy();
  });

  edm::Handle<std::vector<reco::Track>> tracksH;
  iEvent.getByToken(tracksToken_, tracksH);
  const auto& tracks = *tracksH.product();

  edm::Handle<std::vector<CaloParticle>> caloParticlesH;
  iEvent.getByToken(caloParticlesToken_, caloParticlesH);
  auto const& caloParticles = *caloParticlesH.product();
  std::vector<std::pair<int, float>> bestCPMatches;

  auto bestCaloParticleMatches = [&](const ticl::Trackster& t) -> void {
    bestCPMatches.clear();
    auto idx = 0;
    auto separation = 0.;
    for (auto const& cp : caloParticles) {
      separation = reco::deltaR2(t.barycenter(), cp.momentum());
      if (separation < 0.05) {
        bestCPMatches.push_back(std::make_pair(idx, separation));
      }
      ++idx;
    }
  };

  for (auto const& t : sorted_tracksters_idx) {
    auto const& trackster = tracksters[t];
    auto const& probs = trackster.id_probabilities();
    // Sort probs in descending order
    std::vector<int> sorted_probs_idx(probs.size());
    iota(begin(sorted_probs_idx), end(sorted_probs_idx), 0);
    sort(begin(sorted_probs_idx), end(sorted_probs_idx), [&probs](int i, int j) { return probs[i] > probs[j]; });

    std::cout << "\nTrksIdx: " << t << "\n bary: " << trackster.barycenter()
              << " baryEta: " << trackster.barycenter().eta() << " baryPhi: " << trackster.barycenter().phi()
              << "\n raw_energy: " << trackster.raw_energy() << " raw_em_energy: " << trackster.raw_em_energy()
              << "\n raw_pt: " << trackster.raw_pt() << " raw_em_pt: " << trackster.raw_em_pt()
              << "\n seedIdx: " << trackster.seedIndex() << "\n Probs: ";
    for (auto p_idx : sorted_probs_idx) {
      std::cout << "(" << particle_kind[p_idx] << "):" << probs[p_idx] << " ";
    }
    std::cout << "\n time: " << trackster.time() << "+/-" << trackster.timeError() << std::endl
              << " cells: " << trackster.vertices().size() << " average usage: "
              << std::accumulate(
                     std::begin(trackster.vertex_multiplicity()), std::end(trackster.vertex_multiplicity()), 0.) /
                     trackster.vertex_multiplicity().size()
              << std::endl;
    std::cout << " outInHopsPerformed: " << trackster.outInHopsPerformed() << std::endl;
    std::cout << " link connections: " << trackster.edges().size() << std::endl;
    if (trackster.seedID().id() != 0) {
      auto const& track = tracks[trackster.seedIndex()];
      std::cout << " Seeding Track:" << std::endl;
      std::cout << "   p: " << track.p() << " pt: " << track.pt()
                << " charge: " << track.charge() << " eta: " << track.eta()
                << " outerEta: " << track.outerEta() << " phi: " << track.phi() << " outerPhi: " << track.outerPhi()
                << " missingOuterHits: " << track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS)
                << std::endl;
    }
    bestCaloParticleMatches(trackster);
    if (!bestCPMatches.empty()) {
      std::cout << " Best CaloParticles Matches:" << std::endl;;
      for (auto const& i : bestCPMatches) {
        auto const & cp = caloParticles[i.first];
        std::cout << "   " << i.first << "(" << i.second << "):" << cp.pdgId() << "|"
                  << cp.simClusters().size() << "|"
                  << cp.energy() << "|" << cp.pt() << "  " << std::endl;
      }
      std::cout << std::endl;
    }
  }
}

void TiclDebugger::beginJob() {}

void TiclDebugger::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TiclDebugger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersMerge", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  descriptions.add("ticlDebugger", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(TiclDebugger);
