// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"

// form HGC Validator code
//#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"


#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

//ROOT includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include <TFile.h>
#include <TROOT.h>
#include "TBranch.h"
#include <string>
#include <vector>
#include "TSystem.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include "TH1.h"
#include <algorithm>
#include "Math/GenVector/VectorUtil.h"

struct caloparticle {
  int   pdgid_;
  float energy_;
  float pt_;
  float eta_;
  float phi_;
  float recable_energy_;
  std::vector<DetId> rechitdetid_;
  std::vector<float> rechitenergy_;
  
};

struct layercluster {
  float energy_;
  float eta_;
  float phi_;
  float x_;
  float y_;
  float z_;
  int   nrechits_;
  int   layer_;
  int   idx2Trackster_;
};

struct trackster {
  int   idx_;
  int   type_; // pdgid
  float pt_;
  float energy_;
  float eta_;
  float phi_;
  float x_;
  float y_;
  float z_;
  float sig_[3];
  float pcaeigval_[3];
  float pcasig_[3];
  float pcaEigVect0_eta_;
  float pcaEigVect0_phi_;
  float cpenergy_;
  int   cppdgid_;
  float baryAxis_cos_;
};



//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.
      

class Performance : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit Performance(const edm::ParameterSet&);
  ~Performance();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void fillHitMap(std::map<DetId, const HGCRecHit*>& hitMap, 
			  const HGCRecHitCollection& rechitsEE, 
			  const HGCRecHitCollection& rechitsFH,
			  const HGCRecHitCollection& rechitsBH) const;
  std::vector<ticl::Trackster> getTracksterCollection(const int pdgId_,
						      std::vector<ticl::Trackster> trackstersEM,
						      std::vector<ticl::Trackster> trackstersMIP,
						      std::vector<ticl::Trackster> trackstersHAD,
						      std::vector<ticl::Trackster> trackstersTrk
						      );
  float getDr(float eta1, float phi1, float eta2, float phi2);  
  void findClosestCP2Trkster(std::vector<caloparticle> cps_, ticl::Trackster trackster,
			     float &minDrTrksterCP, unsigned int &cp_idx);   
  void findClosestTrkster2CP(caloparticle cps_, std::vector<ticl::Trackster> tracksters,
			     float &minDrTrksterCP, unsigned int &trkster_idx);   
  std::vector<int> matchRecHit2CPRecHits(DetId detid_, std::vector<DetId> rechitdetid_);
  //  std::vector<layercluster> getLCEnergyFromCP(ticl::Trackster trkster, int idx2Trkster, float &trksterEnFromCP_, 
  //					      caloparticle cp, std::map<DetId, const HGCRecHit*>& hitMap);

  std::shared_ptr<hgcal::RecHitTools> recHitTools;
  //const hgcal::RecHitTools recHitTools;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<CaloParticle> >       caloParticlesToken_;
  edm::EDGetTokenT<HGCRecHitCollection>              hgcalRecHitsEEToken_; 
  edm::EDGetTokenT<HGCRecHitCollection>              hgcalRecHitsFHToken_;
  edm::EDGetTokenT<HGCRecHitCollection>              hgcalRecHitsBHToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     emTrksterToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     trkTrksterToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     hadTrksterToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     mipTrksterToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     mergeTrksterToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection>      hgcalLayerClustersToken_;
  edm::EDGetTokenT<edm::ValueMap<std::pair<float,float>>> hgcalLayerClusterTimeToken_;

  
  TTree *tree = new TTree("tree","tree");

  edm::RunNumber_t irun;
  edm::EventNumber_t ievent;
  edm::LuminosityBlockNumber_t ilumiblock;
  edm::Timestamp itime;

  size_t run, event, lumi, time;
  float weight;  

  int   cp_pdgid[2];
  float cp_e[2];
  float cp_pt[2];
  float cp_eta[2];
  float cp_phi[2];
  float cp_recable_e[2];
  float cp_clusterized_e[2];

  int   n_ts;
  int   n_ts_dr0p2;
  float ts_energy;
  float ts_pt;
  float ts_eta;
  float ts_phi;
  float ts_z;
  float ts_sig[3];
  float ts_pcaeigval[3];
  float ts_pcaEigVect0_eta;
  float ts_pcaEigVect0_phi;
  float ts_pcaBaryEigVect0_cos;
  float ts_pcasig[3];

  /*std::vector<int>   ts_idx;
  std::vector<int>   ts_type;
  std::vector<float> ts_energy;
  std::vector<float> ts_eta;
  std::vector<float> ts_phi;
  std::vector<float> ts_x;
  std::vector<float> ts_y;
  std::vector<float> ts_z;
  std::vector<float> ts_pcaeigval0;
  std::vector<float> ts_pcasig0;
  std::vector<float> ts_cpenergy;
  std::vector<float> ts_cppdgid;*/
  
  /*  int   ts_idx;
  int   ts_type;
  float ts_energy;
  float ts_eta;
  float ts_phi;
  float ts_x;
  float ts_y;
  float ts_z;
  float ts_pcaeigval0;
  float ts_pcasig0;
  float ts_pcaeigval1;
  float ts_pcasig1;
  float ts_pcaeigval2;
  float ts_pcasig2;
  float ts_cpenergy;
  int   ts_cppdgid;*/

  float EBary_cp_eta[2];
  float EBary_cp_phi[2];
  float EAxis_cp_eta[2];
  float EAxis_cp_phi[2];
};  
  
  
  //
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Performance::Performance(const edm::ParameterSet& iConfig)
 :
  caloParticlesToken_(consumes<std::vector<CaloParticle> >(iConfig.getParameter<edm::InputTag>("caloParticles"))),
  hgcalRecHitsEEToken_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("hgcalRecHitsEE"))),
  hgcalRecHitsFHToken_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("hgcalRecHitsFH"))),
  hgcalRecHitsBHToken_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("hgcalRecHitsBH"))),
  emTrksterToken_(consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("emTrkster"))),
  trkTrksterToken_(consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("trkTrkster"))),
  hadTrksterToken_(consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("hadTrkster"))),
  mipTrksterToken_(consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("mipTrkster"))),  
  mergeTrksterToken_(consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("mergeTrkster"))),  
  hgcalLayerClustersToken_(consumes<reco::CaloClusterCollection>(iConfig.getParameter<edm::InputTag>("hgcalLayerClusters"))),
  hgcalLayerClusterTimeToken_(consumes<edm::ValueMap<std::pair<float,float>>>(iConfig.getParameter<edm::InputTag>("layerClusterTime")))
{

  recHitTools.reset(new hgcal::RecHitTools());
  //now do what ever initialization is needed
  usesResource("TFileService");
  edm::Service<TFileService> file;
  
  tree = file->make<TTree>("tree","hgc analyzer");
  
  tree->Branch("run"    , &run    , "run/I"    );
  tree->Branch("event"  , &event  , "event/I"  );
  tree->Branch("lumi"   , &lumi   , "lumi/I"   );
  tree->Branch("weight" , &weight , "weight/F" );

  for (int i=0; i<2; ++i) {
    std::string cpth = "cp"+std::to_string(i);
    // calo particle
    tree->Branch((cpth+"_pdgid").c_str()         , &(cp_pdgid[i])         , (cpth+"_pdgid/I").c_str() );
    tree->Branch((cpth+"_e").c_str()             , &(cp_e[i])             , (cpth+"_e/F").c_str() );
    tree->Branch((cpth+"_pt").c_str()            , &(cp_pt[i])            , (cpth+"_pt/F").c_str() );
    tree->Branch((cpth+"_eta").c_str()           , &(cp_eta[i])           , (cpth+"_eta/F").c_str() );
    tree->Branch((cpth+"_phi").c_str()           , &(cp_phi[i])           , (cpth+"_phi/F").c_str() );
    tree->Branch((cpth+"_recable_e").c_str()     , &(cp_recable_e[i])     , (cpth+"_recable_e/F").c_str() );
    tree->Branch((cpth+"_clusterized_e").c_str() , &(cp_clusterized_e[i]) , (cpth+"_clusterized_e/F").c_str() );

    // trackster
    tree->Branch(("EBary_"+cpth+"_eta").c_str(), &(EBary_cp_eta[i]));
    tree->Branch(("EBary_"+cpth+"_phi").c_str(), &(EBary_cp_phi[i]));
    tree->Branch(("EAxis_"+cpth+"_eta").c_str(), &(EAxis_cp_eta[i]));
    tree->Branch(("EAxis_"+cpth+"_phi").c_str(), &(EAxis_cp_phi[i]));
  }

  // trackster
  tree->Branch("n_ts"       , &n_ts);
  tree->Branch("n_ts_dr0p2" , &n_ts_dr0p2);
  tree->Branch("ts_energy"  , &ts_energy);
  tree->Branch("ts_pt"      , &ts_pt);
  tree->Branch("ts_eta"     , &ts_eta);
  tree->Branch("ts_phi"     , &ts_phi);
  tree->Branch("ts_z"       , &ts_z);
  tree->Branch("ts_sig"     , &ts_sig, "ts_sig[3]/F");

  tree->Branch("ts_pcaeigval", &ts_pcaeigval, "ts_pcaeigval[3]/F");
  tree->Branch("ts_pcaEigVect0_eta", &ts_pcaEigVect0_eta);
  tree->Branch("ts_pcaEigVect0_phi", &ts_pcaEigVect0_phi);
  tree->Branch("ts_pcaBaryEigVect0_cos", &ts_pcaBaryEigVect0_cos);
  tree->Branch("ts_pcasig", &ts_pcasig, "ts_pcasig[3]/F");

  /*
  tree->Branch("ts_type"       , &ts_type);
  tree->Branch("ts_energy"     , &ts_energy);
  tree->Branch("ts_eta"        , &ts_eta);
  tree->Branch("ts_phi"        , &ts_phi);
  tree->Branch("ts_x"          , &ts_x);
  tree->Branch("ts_y"          , &ts_y);
  tree->Branch("ts_z"          , &ts_z);
  tree->Branch("ts_pcaeigval0" , &ts_pcaeigval0);
  tree->Branch("ts_pcasig0"    , &ts_pcasig0);
  tree->Branch("ts_cpenergy"   , &ts_cpenergy);
  tree->Branch("ts_cppdgid"    , &ts_cppdgid);
  */ /*
  tree->Branch("ts_type"       , &ts_type       , "ts_type/I");
  tree->Branch("ts_energy"     , &ts_energy     , "ts_energy/F");
  tree->Branch("ts_eta"        , &ts_eta        , "ts_eta/F");
  tree->Branch("ts_phi"        , &ts_phi        , "ts_phi/F");
  tree->Branch("ts_x"          , &ts_x          , "ts_x/F");
  tree->Branch("ts_y"          , &ts_y          , "ts_y/F");
  tree->Branch("ts_z"          , &ts_z          , "ts_z/F");
  tree->Branch("ts_pcaeigval0" , &ts_pcaeigval0 , "ts_pcaeigval0/F");
  tree->Branch("ts_pcasig0"    , &ts_pcasig0    , "ts_pcasig0/F");
  tree->Branch("ts_pcaeigval1" , &ts_pcaeigval1 , "ts_pcaeigval1/F");
  tree->Branch("ts_pcasig1"    , &ts_pcasig1    , "ts_pcasig1/F");
  tree->Branch("ts_pcasig2"    , &ts_pcasig2    , "ts_pcasig2/F");
  tree->Branch("ts_pcaeigval2" , &ts_pcaeigval2 , "ts_pcaeigval2/F");
  tree->Branch("ts_cpenergy"   , &ts_cpenergy   , "ts_cpenergy/F");
  tree->Branch("ts_cppdgid"    , &ts_cppdgid    , "ts_cppdgid/I");
  */
}


Performance::~Performance()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------

void Performance::fillHitMap(std::map<DetId, const HGCRecHit*>& hitMap,
                                const HGCRecHitCollection& rechitsEE,
                                const HGCRecHitCollection& rechitsFH,
                                const HGCRecHitCollection& rechitsBH) const {
  hitMap.clear();
  for (const auto& hit : rechitsEE) {
    hitMap.emplace(hit.detid(), &hit);
  }

  for (const auto& hit : rechitsFH) {
    hitMap.emplace(hit.detid(), &hit);
  }

  for (const auto& hit : rechitsBH) {
    hitMap.emplace(hit.detid(), &hit);
  }
} // end of Performance::fillHitMap


std::vector<ticl::Trackster> Performance::getTracksterCollection(const int pdgId_, 
							 std::vector<ticl::Trackster> trackstersEM,
							 std::vector<ticl::Trackster> trackstersMIP,
							 std::vector<ticl::Trackster> trackstersHAD,
							 std::vector<ticl::Trackster> trackstersTrk) {

  std::vector<ticl::Trackster> localTracksters; localTracksters.clear();

  if(  abs(pdgId_) == 22 )                         { localTracksters = trackstersEM;  }
  if(  abs(pdgId_) == 13 )                         { localTracksters = trackstersMIP; }
  if(  abs(pdgId_) == 311)                         { localTracksters = trackstersHAD; }
  if( (abs(pdgId_) == 211) || (abs(pdgId_) == 11)) { localTracksters = trackstersTrk; }

  return localTracksters;
} // end of Performance::getTracksterCollection


float Performance::getDr(float eta1, float phi1, float eta2, float phi2) { 
  return sqrt( (eta1-eta2)*(eta1-eta2) + (phi1-phi2)*(phi1-phi2) ); 
}


void Performance::findClosestCP2Trkster(std::vector<caloparticle> cps_, ticl::Trackster trackster, 
					float &minDrTrksterCP, unsigned int &cp_idx) {
  for (unsigned int icp = 0; icp<cps_.size(); ++icp) {
    float tmpdr_ = getDr(cps_.at(icp).eta_,cps_.at(icp).phi_,trackster.barycenter.eta(),trackster.barycenter.phi());
    if (tmpdr_<minDrTrksterCP) { minDrTrksterCP = tmpdr_; cp_idx = icp; }
  } // end of looping over the CPs 
}// end of findClosestCP2Trkster


void Performance::findClosestTrkster2CP(caloparticle cp_, std::vector<ticl::Trackster> tracksters, 
					float &minDrTrksterCP, unsigned int &trkster_idx) {
  
  for (unsigned int itrkster = 0; itrkster<tracksters.size(); ++itrkster) {
    float tmpdr_ = getDr(cp_.eta_,cp_.phi_,tracksters.at(itrkster).barycenter.eta(),tracksters.at(itrkster).barycenter.phi());
    //    float tmpen_ = (abs(tracksters.at(itrkster).raw_energy - cp_.energy_)/(cp_.energy_))*100.;
    //if ( (tmpdr_<minDrTrksterCP) && (tmpen_<minEnTrksterCP) ) { 
    if ( (tmpdr_<minDrTrksterCP) ) { 
      minDrTrksterCP = tmpdr_; 
      trkster_idx = itrkster; 
    }
  } // end of looping over the Tracksters

}// end of findClosestCP2Trkster

std::vector<int> Performance::matchRecHit2CPRecHits(DetId detid_, std::vector<DetId> rechitdetid_) {
  std::vector<int> matchedIdxs; matchedIdxs.clear();
  for (unsigned int i0=0; i0<rechitdetid_.size(); ++i0) {
    if (detid_ == rechitdetid_[i0]) { matchedIdxs.push_back(i0); }
  }
  //std::vector<DetId>::iterator it = std::find(rechitdetid_.begin(), rechitdetid_.end(), detid_);
  //if (it != rechitdetid_.end()) { index = std::distance(rechitdetid_.begin(), it); }
  return matchedIdxs;
} // end of matchRecHit2CPRecHits



/*
std::vector<layercluster> Performance::getLCEnergyFromCP(ticl::Trackster trkster, int idx2Trkster, float &trksterEnFromCP_,
						 caloparticle cp, std::map<DetId, const HGCRecHit*>& hitMap) {

  // get the layer clusters [LC] of the trackster
  edm::PtrVector<reco::BasicCluster> lcs = trkster.clusters();
  std::vector<layercluster> tmp_lcs; tmp_lcs.reserve(lcs.size());

  for (const auto& it_lc : lcs) {
    
    float lcEnFromCP_ = 0.;
    int   lcPurity_   = -1;
    int   layer_ = -1;

    // loop over the RecHits of the LC    
    const std::vector<std::pair<DetId, float>> &hf = it_lc->hitsAndFractions();    
    for (unsigned int j = 0; j < hf.size(); j++) {

      const DetId detid_ = hf[j].first;
      layer_ = recHitTools->getLayerWithOffset(detid_);
      std::map<DetId,const HGCRecHit *>::const_iterator itcheck = hitMap.find(detid_);

      if (itcheck != hitMap.end()) {
	const HGCRecHit *hit = itcheck->second;
	
	bool matchRecHit2CP = false; 
	if (std::find(cp.rechitdetid_.begin(), cp.rechitdetid_.end(), detid_) != cp.rechitdetid_.end()) { matchRecHit2CP = true; } 
	
	if (matchRecHit2CP) {
	  lcEnFromCP_      += hit->energy()*(hf[j].second);
	  trksterEnFromCP_ += hit->energy()*(hf[j].second);
	}
      } // end of itcheck != hitMap.end() check
    } // end of looping over the rechits of the LC
    

    // is this hit matched to a cp -> store the energy
    // decide the type of the lc after all rechits
    // store the energy of this rechit in general
   
    layercluster tmp_lc; 
    tmp_lc.energy_        = it_lc->energy();
    tmp_lc.eta_           = it_lc->eta();
    tmp_lc.phi_           = it_lc->phi();
    tmp_lc.x_             = it_lc->position().x();    
    tmp_lc.y_             = it_lc->position().y();    
    tmp_lc.z_             = it_lc->position().z();    
    tmp_lc.nrechits_      = it_lc->hitsAndFractions().size();
    tmp_lc.layer_         = layer_;
    tmp_lc.idx2Trackster_ = idx2Trkster;

    tmp_lcs.push_back(tmp_lc);
  } // end of looping over the LCs

  return tmp_lcs;
} // end of getLCEnergyFromCP

*/
void Performance::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

   edm::Handle<HGCRecHitCollection> recHitHandleEE;
   iEvent.getByToken(hgcalRecHitsEEToken_, recHitHandleEE);

   edm::Handle<HGCRecHitCollection> recHitHandleFH;
   iEvent.getByToken(hgcalRecHitsFHToken_, recHitHandleFH);

   edm::Handle<HGCRecHitCollection> recHitHandleBH;
   iEvent.getByToken(hgcalRecHitsBHToken_, recHitHandleBH);

   edm::Handle<std::vector<ticl::Trackster>> emTrksterHandle;
   iEvent.getByToken(emTrksterToken_, emTrksterHandle);
   //const std::vector<ticl::Trackster> &emTrksters = *emTrksterHandle;

   edm::Handle<std::vector<ticl::Trackster>> trkTrksterHandle;
   iEvent.getByToken(trkTrksterToken_, trkTrksterHandle);
   //const std::vector<ticl::Trackster> &trkTrksters = *trkTrksterHandle;

   edm::Handle<std::vector<ticl::Trackster>> hadTrksterHandle;
   iEvent.getByToken(hadTrksterToken_, hadTrksterHandle);
   //const std::vector<ticl::Trackster> &hadTrksters = *hadTrksterHandle;

   edm::Handle<std::vector<ticl::Trackster>> mipTrksterHandle;
   iEvent.getByToken(mipTrksterToken_, mipTrksterHandle);
   //const std::vector<ticl::Trackster> &mipTrksters = *mipTrksterHandle;

   edm::Handle<std::vector<ticl::Trackster>> mergeTrksterHandle;
   iEvent.getByToken(mergeTrksterToken_, mergeTrksterHandle);
   const std::vector<ticl::Trackster> &mergeTrksters = *mergeTrksterHandle;

   edm::Handle<std::vector<CaloParticle>> CaloParticles;
   iEvent.getByToken(caloParticlesToken_, CaloParticles);
   const CaloParticleCollection& cps = *CaloParticles;

   edm::Handle<reco::CaloClusterCollection> layerClusterHandle;
   iEvent.getByToken(hgcalLayerClustersToken_, layerClusterHandle);
   //const reco::CaloClusterCollection &lcs = *layerClusterHandle;

   std::map<DetId, const HGCRecHit*> hitMap;
   fillHitMap(hitMap, *recHitHandleEE, *recHitHandleFH, *recHitHandleBH);
   //std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster> > detIdToCaloParticleId_Map;

   edm::Handle<edm::ValueMap<std::pair<float,float>>> lcTimeHandle;
   iEvent.getByToken(hgcalLayerClusterTimeToken_, lcTimeHandle);
   //const auto& lcTime = *lcTimeHandle;

   // init vars
   recHitTools->getEventSetup(iSetup);
   

   // get CaloParticles
   std::vector<caloparticle> caloparticles; //std::cout << " #cps = " << cps.size() << "\n";
   for (auto it_cp = cps.begin(); it_cp != cps.end(); ++it_cp) {
     const CaloParticle& cp = *it_cp;      
    
     if ( (cp.eventId().event() != 0) || (cp.eventId().bunchCrossing()!=0) ) { continue; }

     //std::cout << "\nCP " << std::distance(cps.begin(), it_cp) << std::endl ;
     //std::cout << "eta = " << cp.eta() << std::endl ;
     //std::cout << "phi = " << cp.phi() << std::endl ;

     caloparticle tmpcp_;
     tmpcp_.pdgid_  = cp.pdgId();
     tmpcp_.energy_ = cp.energy();
     tmpcp_.pt_     = cp.pt();
     tmpcp_.eta_    = cp.eta();
     tmpcp_.phi_    = cp.phi();

     caloparticles.push_back(tmpcp_);
   } // end of looping over the calo particles

   // Dummy CaloParticle with a z-parallel direction
   GlobalVector zVersor( 0, 0, 1 );
   caloparticle zCP;
   zCP.pdgid_  = 0;
   zCP.energy_ = 0;
   zCP.pt_     = 0;
   zCP.eta_    = zVersor.eta();
   zCP.phi_    = zVersor.phi();

   // loop over the merged clusters
   /*
   for (auto it_trkster = mergeTrksters.begin(); it_trkster != mergeTrksters.end(); ++it_trkster) {
     std::cout << "\n\nMT " << std::distance(mergeTrksters.begin(), it_trkster) << std::endl ;
     std::cout << "E = " << it_trkster->raw_energy << std::endl ;
     std::cout << "eta = " << it_trkster->barycenter.eta() << std::endl ;
     std::cout << "phi = " << it_trkster->barycenter.phi() << std::endl ;
     std::cout << "axis eta = " << it_trkster->eigenvectors[0].eta() << std::endl ;
     std::cout << "axis phi = " << it_trkster->eigenvectors[0].phi() << std::endl ;
   }
   */

   // get the relevant trackster collection
   //   int cp_pdgid_ = 0; if (caloparticles.size()>0) { cp_pdgid_ = caloparticles.at(0).pdgid_; }
   //   std::vector<ticl::Trackster> tracksters = getTracksterCollection(cp_pdgid_,emMCs, mipMCs, hadMCs);


   // keep tracksters [and the corresponding LC] that
   // have at least some ammount of energy from the CP
   //   std::vector<trackster> trksterCollection; trksterCollection.clear();
   
   // get the most energetic trackster
   float maxE = 0.;
   unsigned int trkster_idx = 999;
   int n_ts_dr0p2_ = 0;
   for (auto it_trkster = mergeTrksters.begin(); it_trkster != mergeTrksters.end(); ++it_trkster) {
     float trksterE = it_trkster->raw_energy ;
     if (trksterE > maxE) {
       maxE = trksterE;
       trkster_idx = std::distance(mergeTrksters.begin(), it_trkster);
     }
   }
   //std::cout << "\nMost energetic trackster idx: " << trkster_idx << std::endl;

   trackster trkster_; 
   if (mergeTrksters.size()>0) {   
     trkster_.energy_ = mergeTrksters.at(trkster_idx).raw_energy;
     trkster_.pt_     = abs(mergeTrksters.at(trkster_idx).raw_pt);
     trkster_.eta_    = mergeTrksters.at(trkster_idx).barycenter.eta();
     trkster_.phi_    = mergeTrksters.at(trkster_idx).barycenter.phi();
     trkster_.z_      = abs(mergeTrksters.at(trkster_idx).barycenter.z());
     trkster_.pcaEigVect0_eta_ = mergeTrksters.at(trkster_idx).eigenvectors[0].eta();
     trkster_.pcaEigVect0_phi_ = mergeTrksters.at(trkster_idx).eigenvectors[0].phi();
     trkster_.baryAxis_cos_ = ROOT::Math::VectorUtil::CosTheta(mergeTrksters.at(trkster_idx).barycenter, mergeTrksters.at(trkster_idx).eigenvectors[0]);
     for (int i=0; i<3; ++i) {
       trkster_.sig_[i] = mergeTrksters.at(trkster_idx).sigmas[i];
       trkster_.pcaeigval_[i] = mergeTrksters.at(trkster_idx).eigenvalues[i];
       trkster_.pcasig_[i] = mergeTrksters.at(trkster_idx).sigmasPCA[i];
     }
   }

   /*
   std::cout << "new event: mergeTrksters \n";
   for (const auto& it_trkster : mergeTrksters ) {
     std::cout << it_trkster.raw_energy << " " << it_trkster.barycenter.eta() <<  " " << it_trkster.barycenter.phi() << "\n";
     }*/
   /*
   std::cout << "new event: emTrksters \n";
   for (const auto& it_trkster : emTrksters ) {
     std::cout << it_trkster.raw_energy << " " << it_trkster.barycenter.eta() <<  " " << it_trkster.barycenter.phi() << "\n";
     }*/


   /*


   // loop over the caloparticles and then find the closest trackster to it
   for (unsigned int icp = 0; icp<caloparticles.size(); ++icp) {

     // get the relevant tracksterCollection based on the cp_id
     std::vector<ticl::Trackster> tracksters = getTracksterCollection(caloparticles.at(icp).pdgid_,emTrksters, mipTrksters, hadTrksters, trkTrksters);

     // find the closest trackster 
     float minDrTrksterCP = 999.;
     unsigned int trkster_idx = 999;
     findClosestTrkster2CP(caloparticles.at(icp), tracksters, minDrTrksterCP, trkster_idx);
     if (minDrTrksterCP>maxDrTrackserCP) { continue; } // skip if DR(Trackster,CP)>maxDrTracksterCP
     
     // get the en from CP, create LC, etc...
     float trksterEnFromCP_ = 0.;
     //std::vector<layercluster> lcs = getLCEnergyFromCP(tracksters.at(trkster_idx), trkster_idx, trksterEnFromCP_, caloparticles.at(icp), hitMap); 
     //if ( (trksterEnFromCP_/caloparticles.at(icp).energy_) < trackstersEnFromCP) { continue; }
     //     if ( tracksters.at(trkster_idx).raw_energy<(0.5*(caloparticles.at(icp).energy_)) ) { continue; }
     
     trackster trkster_; 
     trkster_.idx_        = trkster_idx;
     trkster_.type_       = -1;
     trkster_.energy_     = tracksters.at(trkster_idx).raw_energy;
     trkster_.eta_        = abs(tracksters.at(trkster_idx).barycenter.eta());
     trkster_.phi_        = tracksters.at(trkster_idx).barycenter.phi();
     trkster_.x_          = tracksters.at(trkster_idx).barycenter.x();
     trkster_.y_          = tracksters.at(trkster_idx).barycenter.y();
     trkster_.z_          = abs(tracksters.at(trkster_idx).barycenter.z());
     trkster_.pcaeigval0_ = tracksters.at(trkster_idx).eigenvalues.at(0);
     trkster_.pcasig0_    = tracksters.at(trkster_idx).sigmas.at(0);
     trkster_.cpenergy_   = caloparticles.at(icp).energy_;
     trkster_.cppdgid_    = caloparticles.at(icp).pdgid_;
   
     trksterCollection.push_back(trkster_);
     //lcCollection.insert(lcCollection.end(),lcs.begin(),lcs.end());
    
   } // end of looping over the caloparticles

   // fill the tree
   for (unsigned int ilc=0; ilc<lcCollection.size(); ++ilc) {
     lc_energy.push_back(lcCollection.at(ilc).energy_);
     lc_eta.push_back(lcCollection.at(ilc).eta_);
     lc_phi.push_back(lcCollection.at(ilc).phi_);
     lc_x.push_back(lcCollection.at(ilc).x_);
     lc_y.push_back(lcCollection.at(ilc).y_);
     lc_z.push_back(lcCollection.at(ilc).z_);
     lc_nrechits.push_back(lcCollection.at(ilc).nrechits_);
     lc_layer.push_back(lcCollection.at(ilc).layer_);
     lc_idx2trackster.push_back(lcCollection.at(ilc).idx2Trackster_);
   }
        
   for (unsigned int its=0; its<trksterCollection.size(); ++its) {

     if (its>0) {continue; }

     run   = (size_t)irun;
     event = (size_t)ievent;
     lumi  = (size_t)ilumiblock;
     weight = 1.;

     ts_idx.push_back(trksterCollection.at(its).idx_);
     ts_type.push_back(-1);
     ts_energy.push_back(trksterCollection.at(its).energy_);
     ts_eta.push_back(trksterCollection.at(its).eta_);
     ts_phi.push_back(trksterCollection.at(its).phi_);
     ts_x.push_back(trksterCollection.at(its).x_);
     ts_y.push_back(trksterCollection.at(its).y_);
     ts_z.push_back(trksterCollection.at(its).z_);
     ts_pcaeigval0.push_back(trksterCollection.at(its).pcaeigval0_);
     ts_pcasig0.push_back(trksterCollection.at(its).pcasig0_);
     ts_cpenergy.push_back(trksterCollection.at(its).cpenergy_);
     ts_cppdgid.push_back(trksterCollection.at(its).cppdgid_);
     
     ts_idx = trksterCollection.at(its).idx_;
     ts_type = -1;
     ts_energy = trksterCollection.at(its).energy_;
     ts_eta = trksterCollection.at(its).eta_;
     ts_phi = trksterCollection.at(its).phi_;
     ts_x = trksterCollection.at(its).x_;
     ts_y = trksterCollection.at(its).y_;
     ts_z = trksterCollection.at(its).z_;
     ts_pcaeigval0 = trksterCollection.at(its).pcaeigval0_;
     ts_pcasig0 = trksterCollection.at(its).pcasig0_;
     ts_cpenergy = trksterCollection.at(its).cpenergy_;
     ts_cppdgid = trksterCollection.at(its).cppdgid_;

   }
   */

   if (mergeTrksters.size()>0) {
     n_ts       = mergeTrksters.size();
     n_ts_dr0p2 = n_ts_dr0p2_;
     ts_energy  = trkster_.energy_;
     ts_pt      = trkster_.pt_;
     ts_eta     = trkster_.eta_;
     ts_phi     = trkster_.phi_;
     ts_z       = trkster_.z_;
     memcpy(&ts_sig, trkster_.sig_, sizeof trkster_.sig_);
     memcpy(&ts_pcaeigval, trkster_.pcaeigval_, sizeof trkster_.pcaeigval_);
     ts_pcaEigVect0_eta = trkster_.pcaEigVect0_eta_;
     ts_pcaEigVect0_phi = trkster_.pcaEigVect0_phi_;
     ts_pcaBaryEigVect0_cos = trkster_.baryAxis_cos_;
     memcpy(&ts_pcasig, trkster_.pcasig_, sizeof trkster_.pcasig_);
   }

   // get the first two CPs in the list and store them
   for (unsigned int i=0; i<2; ++i) {
     if (caloparticles.size() > i) {
       cp_pdgid[i]     = caloparticles.at(i).pdgid_;
       cp_e[i]         = caloparticles.at(i).energy_;
       cp_pt[i]        = caloparticles.at(i).pt_;
       cp_eta[i]       = caloparticles.at(i).eta_;
       cp_phi[i]       = caloparticles.at(i).phi_;
       cp_recable_e[i] = caloparticles.at(i).recable_energy_;
       //cp_clusterized_e = clusterized_energy_;
     } else {
       cp_pdgid[i]     = zCP.pdgid_;
       cp_e[i]         = zCP.energy_;
       cp_pt[i]        = zCP.pt_;
       cp_eta[i]       = zCP.eta_;
       cp_phi[i]       = zCP.phi_;
       cp_recable_e[i] = zCP.recable_energy_;
       //cp_clusterized_e = clusterized_energy_;
       //
     }

     if (mergeTrksters.size() > 0) {
       EBary_cp_eta[i] = ts_eta - cp_eta[i];
       EBary_cp_phi[i] = ts_phi - cp_phi[i];
       EAxis_cp_eta[i] = ts_pcaEigVect0_eta - cp_eta[i];
       EAxis_cp_phi[i] = ts_pcaEigVect0_phi - cp_phi[i];
     }
   }

   tree->Fill();

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void
Performance::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
Performance::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Performance::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Performance);
