// This producer converts a list of TICLCandidates to a list of PFCandidates.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

class PFTICLProducer : public edm::global::EDProducer<> {
public:
  PFTICLProducer(const edm::ParameterSet&);
  ~PFTICLProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // parameters
  const bool useTimingQuality_;
  const float timingQualityThreshold_;

  // inputs
  const edm::EDGetTokenT<edm::View<TICLCandidate>> ticl_candidates_;
  const edm::EDGetTokenT<edm::ValueMap<float>> srcTrackTime_, srcTrackTimeError_, srcTrackTimeQuality_;
};

DEFINE_FWK_MODULE(PFTICLProducer);

PFTICLProducer::PFTICLProducer(const edm::ParameterSet& conf)
    : useTimingQuality_(conf.existsAs<edm::InputTag>("trackTimeQualityMap")),
      timingQualityThreshold_(useTimingQuality_ ? conf.getParameter<double>("timingQualityThreshold") : -99.),
      ticl_candidates_(consumes<edm::View<TICLCandidate>>(conf.getParameter<edm::InputTag>("ticlCandidateSrc"))),
      srcTrackTime_(consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeValueMap"))),
      srcTrackTimeError_(consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeErrorMap"))),
      srcTrackTimeQuality_(useTimingQuality_
                           ? consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeQualityMap"))
                           : edm::EDGetTokenT<edm::ValueMap<float>>())
 {
  produces<reco::PFCandidateCollection>();
}

void PFTICLProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ticlCandidateSrc", edm::InputTag("ticlCandidateFromTracksters"));
  desc.add<edm::InputTag>("trackTimeValueMap", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("trackTimeErrorMap", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("trackTimeQualityMap", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<double>("timingQualityThreshold", 0.5);
  descriptions.add("pfTICLProducer", desc);
}

void PFTICLProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  //get TICLCandidates
  edm::Handle<edm::View<TICLCandidate>> ticl_cand_h;
  evt.getByToken(ticl_candidates_, ticl_cand_h);
  const auto ticl_candidates = *ticl_cand_h;
  edm::Handle<edm::ValueMap<float>> trackTimeH, trackTimeErrH, trackTimeQualH;
  evt.getByToken(srcTrackTime_, trackTimeH);
  evt.getByToken(srcTrackTimeError_, trackTimeErrH);
  if (useTimingQuality_) {
    evt.getByToken(srcTrackTimeQuality_, trackTimeQualH);
  }

  auto candidates = std::make_unique<reco::PFCandidateCollection>();

  std::cout << "ticl_candidates size " << ticl_candidates.size() << std::endl ;
  for (const auto& ticl_cand : ticl_candidates) {
    const auto abs_pdg_id = std::abs(ticl_cand.pdgId());
    const auto charge = ticl_cand.charge();
    const auto& four_mom = ticl_cand.p4();

    reco::PFCandidate::ParticleType part_type;
    switch (abs_pdg_id) {
      case 11:
        part_type = reco::PFCandidate::e;
        break;
      case 13:
        part_type = reco::PFCandidate::mu;
        break;
      case 22:
        part_type = reco::PFCandidate::gamma;
        break;
      case 130:
        part_type = reco::PFCandidate::h0;
        break;
      case 211:
        part_type = reco::PFCandidate::h;
        break;
      // default also handles neutral pions (111) for the time being (not yet foreseen in PFCandidate)
      default:
        part_type = reco::PFCandidate::X;
    }

    candidates->emplace_back(charge, four_mom, part_type);

    auto& candidate = candidates->back();
    if (candidate.charge()) {  // otherwise PFCandidate throws
      // Construct edm::Ref from edm::Ptr. As of now, assumes type to be reco::Track. To be extended (either via
      // dynamic type checking or configuration) if additional track types are needed.
      reco::TrackRef ref(ticl_cand.trackPtr().id(), int(ticl_cand.trackPtr().key()), &evt.productGetter());
      candidate.setTrackRef(ref);
    }

    auto time = ticl_cand.time();
    auto timeE = ticl_cand.timeError();
    // Compute weighted average between HGCAL and MTD timing if available
    std::cout << "candidate.charge() " << candidate.charge() << std::endl ;
    if (candidate.charge()) {
      std::cout << "\ntrackTimeQualH= " << (*trackTimeQualH)[candidate.trackRef()] << ", timingQualityThreshold_= " << timingQualityThreshold_ << std::endl;
      const bool assocQuality = (*trackTimeQualH)[candidate.trackRef()] > timingQualityThreshold_;
      if (assocQuality) {
        const auto timeHGC = time;
        const auto timeEHGC = timeE;
        const auto timeMTD = (*trackTimeH)[candidate.trackRef()];
        const auto timeEMTD = (*trackTimeErrH)[candidate.trackRef()];
        std::cout <<"\nBefore average: timeHGC= " << timeHGC << ", timeEHGC= " << timeEHGC << ", timeMTD= " << timeMTD << ", timeEMTD= " << timeEMTD << std::endl;

        timeE = 1 / (pow(timeEHGC,-2) + pow(timeEMTD,-2));
        time = (timeHGC/pow(timeEHGC,2) + timeMTD/pow(timeEMTD,2)) * timeE;
      }
    }
    candidate.setTime(time, timeE);
  }

  evt.put(std::move(candidates));
}
