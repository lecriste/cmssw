#include "DQMOffline/PFTau/plugins/PFMETDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// -- Constructor
//
PFMETDQMAnalyzer::PFMETDQMAnalyzer(const edm::ParameterSet& parameterSet)  
  
{
  pSet_                = parameterSet;
  inputLabel_          = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_          = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_      = pSet_.getParameter<std::string>("BenchmarkLabel"); 

  pfMETMonitor_.setParameters(parameterSet);  

  myMET_ = consumes< edm::View<reco::MET> >(inputLabel_);
  myMatchedMET_ = consumes< edm::View<reco::MET> >(matchLabel_);


  std::string folder = benchmarkLabel_ ;

  // moved from beginJob
  // moved to bookHistograms
  //Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  //std::string path = "ParticleFlow/" + benchmarkLabel_;
  subsystemname_ = "ParticleFlow" ;
  eventInfoFolder_ = subsystemname_ + "/" + folder ;
  //Benchmark::DQM_->setCurrentFolder(path.c_str());
  //edm::LogInfo("PFJMETDQMAnalyzer") << " PFMETDQMAnalyzer::beginJob " <<"Histogram Folder path set to "<< path;
  // moved to bookHistograms
  //pfMETMonitor_.setup(pSet_);  
  nBadEvents_ = 0;

}

// the beginJob and endJob transitions are not triggered anymore
/*
//
// -- BeginJob
//
void PFMETDQMAnalyzer::beginJob() {

  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  std::string path = "ParticleFlow/" + benchmarkLabel_;
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  edm::LogInfo("PFMETDQMAnalyzer") << " PFMETDQMAnalyzer::beginJob " <<"Histogram Folder path set to "<< path;
  pfMETMonitor_.setup(pSet_);  
  nBadEvents_ = 0;
}
*/

//
// -- BookHistograms
//
void PFMETDQMAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & /* iRun */,
					    edm::EventSetup const & /* iSetup */ )
{
  //In STEP1 the direct access to the DQMStore is forbidden.
  //Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  //Benchmark::DQM_->setCurrentFolder(path.c_str());
  ibooker.setCurrentFolder(eventInfoFolder_) ;
  //edm::LogInfo("PFCandidateDQMAnalyzer") << " PFCandidateDQMAnalyzer::beginJob " << "Histogram Folder path set to " << path ;
  edm::LogInfo("PFMETDQMAnalyzer") << " PFMETDQMAnalyzer::beginJob " << "Histogram Folder path set to " << eventInfoFolder_;

  //pfMETMonitor_.setup(pSet_);
  pfMETMonitor_.setup(ibooker, pSet_);
}

//
// -- Analyze
//
void PFMETDQMAnalyzer::analyze(edm::Event const& iEvent, 
			       edm::EventSetup const& iSetup) {
//void PFMETDQMAnalyzer::analyze(DQMStore::IBooker & ibooker,                                                edm::Event const& iEvent, 				                         edm::EventSetup const& iSetup) {
  edm::Handle< edm::View<reco::MET> > metCollection;
  iEvent.getByToken(myMET_, metCollection);   
  
  edm::Handle< edm::View<reco::MET> > matchedMetCollection; 
  iEvent.getByToken(myMatchedMET_, matchedMetCollection);

  if (metCollection.isValid() && matchedMetCollection.isValid()) {
    float maxRes = 0.0;
    float minRes = 99.99;
    pfMETMonitor_.fillOne( (*metCollection)[0], (*matchedMetCollection)[0], minRes, maxRes);    
    //pfMETMonitor_.fillOne( (*metCollection)[0], (*matchedMetCollection)[0], minRes, maxRes, pSet_);   

    /* 
    edm::ParameterSet skimPS = pSet_.getParameter<edm::ParameterSet>("SkimParameter");
    if ( (skimPS.getParameter<bool>("switchOn")) && 
         (nBadEvents_ <= skimPS.getParameter<int32_t>("maximumNumberToBeStored")) ) {
      if ( minRes < skimPS.getParameter<double>("lowerCutOffOnResolution")) {
	//storeBadEvents(iEvent,minRes);
	storeBadEvents(ibooker, iEvent,minRes);
        nBadEvents_++;
      } else if (maxRes > skimPS.getParameter<double>("upperCutOffOnResolution")) {
        nBadEvents_++;
	//storeBadEvents(iEvent,maxRes);
	storeBadEvents(ibooker, iEvent,maxRes);
      }
    }
    */

  }
}

//void PFMETDQMAnalyzer::storeBadEvents(edm::Event const& iEvent, float& val) {
void PFMETDQMAnalyzer::storeBadEvents(DQMStore::IBooker & ibooker, edm::Event const& iEvent, float& val) {
  unsigned int runNb  = iEvent.id().run();
  unsigned int evtNb  = iEvent.id().event();
  unsigned int lumiNb = iEvent.id().luminosityBlock();
  
  std::string path = "ParticleFlow/" + benchmarkLabel_ + "/BadEvents";
  //In STEP1 the direct access to the DQMStore is forbidden
  //Benchmark::DQM_->setCurrentFolder(path.c_str());
  ibooker.setCurrentFolder(eventInfoFolder_) ;
  std::ostringstream eventid_str;
  eventid_str << runNb << "_"<< evtNb << "_" << lumiNb;

  //MonitorElement* me = Benchmark::DQM_->get(path + "/" + eventid_str.str());
  /*
  MonitorElement* me = ibooker.get(path + "/" + eventid_str.str());
  if (me) me->Reset();
  else {
    //me = Benchmark::DQM_->bookFloat(eventid_str.str());
    me = ibooker.bookFloat(eventid_str.str());
  }
  me->Fill(val); 
  */
  MonitorElement* me = ibooker.bookFloat(eventid_str.str()); 
  me->Fill(val); 

}

// the beginJob and endJob transitions are not triggered anymore
/*
//
// -- EndJob
// 
void PFMETDQMAnalyzer::endJob() {
}
*/

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (PFMETDQMAnalyzer) ;
