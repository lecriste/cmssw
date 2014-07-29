#include "DQMOffline/PFTau/plugins/MatchMETBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/MET.h"

using namespace reco;
using namespace edm;
using namespace std;

MatchMETBenchmarkAnalyzer::MatchMETBenchmarkAnalyzer(const edm::ParameterSet& parameterSet) : 
//MatchMETBenchmarkAnalyzer::MatchMETBenchmarkAnalyzer(DQMStore::IBooker& b, const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  MatchMETBenchmark( (Benchmark::Mode) parameterSet.getParameter<int>("mode") )
{
  matchedInputLabel_ = parameterSet.getParameter<edm::InputTag>("MatchCollection");
//  setRange( parameterSet.getParameter<double>("ptMin"),
//	    parameterSet.getParameter<double>("ptMax"),
//	    -0.1, 0.1, // range in eta for MET. 
//	    parameterSet.getParameter<double>("phiMin"),
//	    parameterSet.getParameter<double>("phiMax") );

  myColl_ = consumes< View<MET> >(inputLabel_);
  myMatchColl_ = consumes< View<MET> >(matchedInputLabel_);

}

// the beginJob and endJob transitions are not triggered anymore
/*
void 
MatchMETBenchmarkAnalyzer::beginJob()
{
  BenchmarkAnalyzer::beginJob();
  setup();
}
*/

void MatchMETBenchmarkAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & iRun,
					    edm::EventSetup const & iSetup )
{
  // moved from beginJob
  //BenchmarkAnalyzer::beginJob();
  //setup();
  BenchmarkAnalyzer::bookHistograms(ibooker, iRun, iSetup);
  setup(ibooker);
 
}

void 
MatchMETBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  
  
  Handle< View<MET> > collection; 
  iEvent.getByToken(myColl_, collection);

  Handle< View<MET> > matchedCollection; 
  iEvent.getByToken(myMatchColl_, matchedCollection);

  fillOne( (*collection)[0] , (*matchedCollection)[0]);
}

// the beginJob and endJob transitions are not triggered anymore
/*
void MatchMETBenchmarkAnalyzer::endJob() {
}
*/
