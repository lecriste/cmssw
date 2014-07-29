#ifndef __DQMOffline_PFTau_PFJetDQMAnalyzer__
#define __DQMOffline_PFTau_PFJetDQMAnalyzer__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DQMOffline/PFTau/interface/PFJetMonitor.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//class PFJetDQMAnalyzer: public edm::EDAnalyzer {
class PFJetDQMAnalyzer: public DQMEDAnalyzer {
 public:
  
  PFJetDQMAnalyzer(const edm::ParameterSet& parameterSet);
  
 private:
  void analyze(edm::Event const&, edm::EventSetup const&);
  //void analyze(DQMStore::IBooker & ibooker, edm::Event const&, edm::EventSetup const&);
  //void beginJob() ;
  //void endJob();

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  //void storeBadEvents(edm::Event const&, float& val);
  void storeBadEvents(DQMStore::IBooker & ibooker, edm::Event const&, float& val);

  edm::EDGetTokenT< edm::View<reco::Jet> > myJet_;
  edm::EDGetTokenT< edm::View<reco::Jet> > myMatchedJet_;
  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;
  
  PFJetMonitor pfJetMonitor_;

  edm::ParameterSet pSet_;
  std::string eventInfoFolder_;
  std::string subsystemname_;

  int nBadEvents_;
};

#endif 
