#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <iostream>

using namespace std;

BenchmarkAnalyzer::BenchmarkAnalyzer(const edm::ParameterSet& parameterSet)
{

  inputLabel_      = parameterSet.getParameter<edm::InputTag>("InputCollection");
  benchmarkLabel_  = parameterSet.getParameter<std::string>("BenchmarkLabel"); 

  std::string folder = benchmarkLabel_ ;

  // moved from beginJob
  // moved to bookHistograms
  //Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  /*  
      if(!Benchmark::DQM_) {
      throw "Please initialize the DQM service in your cfg";
      }
  */
  //// part of the following could be put in the base class
  //string path = "PFTask/" + benchmarkLabel_ ; 
  subsystemname_ = "ParticleFlow" ;
  eventInfoFolder_ = subsystemname_ + "/" + folder ;
  //Benchmark::DQM_->setCurrentFolder(path.c_str());
  //cout<<"path set to "<<path<<endl;

}

// the beginJob and endJob transitions are not triggered anymore
/*
void 
BenchmarkAnalyzer::beginJob()
{  
  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  if(!Benchmark::DQM_) {
    throw "Please initialize the DQM service in your cfg";
  }

  // part of the following could be put in the base class
  string path = "PFTask/" + benchmarkLabel_ ; 
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  cout<<"path set to "<<path<<endl;
}
*/

//
// -- BookHistograms
//
void BenchmarkAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & /* iRun */,
					    edm::EventSetup const & /* iSetup */ )
{
  //In STEP1 the direct access to the DQMStore is forbidden
  //Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  //Benchmark::DQM_->setCurrentFolder(path.c_str());
  ibooker.setCurrentFolder(eventInfoFolder_) ;
  //cout<<"path set to "<<path<<endl;
  cout << "path set to " << eventInfoFolder_ << endl;

}
