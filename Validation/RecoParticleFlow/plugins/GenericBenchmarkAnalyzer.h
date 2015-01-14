#ifndef GENERICBENCHMARKANALYZER_H
#define GENERICBENCHMARKANALYZER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007
// extension: Leo Neuhaus & Joanna Weng 09.2008

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <map>

class GenericBenchmarkAnalyzer: public DQMEDAnalyzer, public GenericBenchmark {
public:

  explicit GenericBenchmarkAnalyzer(const edm::ParameterSet&);
  virtual ~GenericBenchmarkAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override ;


 private:

  // Inputs from Configuration File
  std::string outputFile_;
  edm::EDGetTokenT< edm::View<reco::Candidate> > myTruth_;
  edm::EDGetTokenT< edm::View<reco::Candidate> > myReco_;
  edm::InputTag inputTruthLabel_;
  edm::InputTag inputRecoLabel_;
  std::string benchmarkLabel_;
  std::string path_ ;
  bool startFromGen_;
  bool plotAgainstRecoQuantities_;
  bool onlyTwoJets_;
  double recPt_cut;
  double minEta_cut;
  double maxEta_cut;
  double deltaR_cut;
  float minDeltaEt_;
  float maxDeltaEt_;
  float minDeltaPhi_;
  float maxDeltaPhi_;
  bool doMetPlots_;
};

#endif // GENERICBENCHMARKANALYZER_H
