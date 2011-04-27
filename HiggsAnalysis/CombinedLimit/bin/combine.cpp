#include "../interface/Combine.h"
#include <TString.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <RooRandom.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <boost/program_options.hpp>
#include "../interface/ProfileLikelihood.h"
#include "../interface/Hybrid.h"
#include "../interface/HybridNew.h"
#include "../interface/BayesianFlatPrior.h"
#include "../interface/MarkovChainMC.h"
#include "../interface/FeldmanCousins.h"
#include "../interface/ProfilingTools.h"
#include <map>

using namespace std;

int main(int argc, char **argv) {
  using namespace boost;
  namespace po = boost::program_options;

  string name;
  string datacard, dataset;
  int iMass;
  string whichMethod, whichHintMethod;
  int runToys;
  int    seed;
  string toysFile;

  vector<string> librariesToLoad;

  Combine combiner;

  map<string, LimitAlgo *> methods;
  algo = new Hybrid(); methods.insert(make_pair(algo->name(), algo));
  algo = new ProfileLikelihood(); methods.insert(make_pair(algo->name(), algo));
  algo = new BayesianFlatPrior(); methods.insert(make_pair(algo->name(), algo));
  algo = new MarkovChainMC();  methods.insert(make_pair(algo->name(), algo));
  algo = new HybridNew();  methods.insert(make_pair(algo->name(), algo));
  algo = new FeldmanCousins();  methods.insert(make_pair(algo->name(), algo));
  
  string methodsDesc("Method to extract upper limit. Supported methods are: ");
  for(map<string, LimitAlgo *>::const_iterator i = methods.begin(); i != methods.end(); ++i) {
    if(i != methods.begin()) methodsDesc += ", ";
    methodsDesc += i->first;
  }
  
  po::options_description desc("Main options");
  desc.add_options()
    ("datacard,d", po::value<string>(&datacard), "Datacard file (can also be specified directly without the -d or --datacard)")
    ("method,M",      po::value<string>(&whichMethod)->default_value("ProfileLikelihood"), methodsDesc.c_str())
    ("verbose,v",  po::value<int>(&verbose)->default_value(1), "Verbosity level (-1 = very quiet; 0 = quiet, 1 = verbose, 2+ = debug)")
    ("help,h", "Produce help message")
    ;
  combiner.statOptions().add_options()
    ("toys,t", po::value<int>(&runToys)->default_value(0), "Number of Toy MC extractions")
    ("seed,s", po::value<int>(&seed)->default_value(123456), "Toy MC random seed")
    ("hintMethod,H",  po::value<string>(&whichHintMethod)->default_value(""), "Run first this method to provide a hint on the result")
    ;
  combiner.ioOptions().add_options()
    ("name,n",     po::value<string>(&name)->default_value("Test"), "Name of the job, affects the name of the output tree")
    ("mass,m",     po::value<int>(&iMass)->default_value(120), "Higgs mass to store in the output tree")
    ("dataset,D",  po::value<string>(&dataset)->default_value("data_obs"), "Name of the dataset for observed limit")
    ("saveToys",   "Save results of toy MC or other intermediate results")
    ("toysFile",   po::value<string>(&toysFile)->default_value(""), "Read toy mc or other intermediate results from this file")
    ;
  combiner.miscOptions().add_options()
    ("igpMem", "Setup support for memory profiling using IgProf")
    ("LoadLibrary,L", po::value<vector<string> >(&librariesToLoad), "Load library through gSystem->Load(...). Can specify multiple libraries using this option multiple times")
    ;
  desc.add(combiner.statOptions());
  desc.add(combiner.ioOptions());
  desc.add(combiner.miscOptions());
  po::positional_options_description p;
  p.add("datacard", -1);
  po::variables_map vm, vm0;

  // parse the first time, using only common options and allow unregistered options 
  try{
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm0);
    po::notify(vm0);
  } catch(std::exception &ex) {
    cerr << "Invalid options: " << ex.what() << endl;
    cout << "Invalid options: " << ex.what() << endl;
    cout << "Use combine --help to get a list of all the allowed options"  << endl;
    return 999;
  } catch(...) {
    cerr << "Unidentified error parsing options." << endl;
    return 1000;
  }

  // if help, print help
  if(vm0.count("help")) {
    cout << "Usage: combine datacard [options]\n";
    cout << desc;
    map<string, LimitAlgo *>::const_iterator i;
    for(i = methods.begin(); i != methods.end(); ++i) {
        cout << i->second->options() << "\n";
    }
    return 0;
  }

  // now search for algo, and add option
  map<string, LimitAlgo *>::const_iterator it_algo = methods.find(whichMethod);
  if (it_algo == methods.end()) {
    cerr << "Unsupported method: " << whichMethod << endl;
    cout << "Use combine --help to get a list of all the allowed methods and options"  << endl;
    return 1003;
  } 
  desc.add(it_algo->second->options());  

  // parse the first time, now include options of the algo but not unregistered ones
  try{
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
  } catch(std::exception &ex) {
    cerr << "Invalid options: " << ex.what() << endl;
    cout << "Invalid options: " << ex.what() << endl;
    cout << "Use combine --help to get a list of all the allowed options"  << endl;
    return 999;
  } catch(...) {
    cerr << "Unidentified error parsing options." << endl;
    return 1000;
  }

  if(datacard == "") {
    cerr << "Missing datacard file" << endl;
    cout << "Usage: combine [options]\n";
    cout << "Use combine --help to get a list of all the allowed methods and options"  << endl;
    return 1002;
  }

  try {
    combiner.applyOptions(vm);
  } catch (std::exception &ex) {
    cerr << "Error when configuring the combiner:\n\t" << ex.what() << std::endl;
    return 2001;
  }

  algo = it_algo->second;
  try {
      algo->applyOptions(vm);
  } catch (std::exception &ex) {
      cerr << "Error when configuring the algorithm " << whichMethod << ":\n\t" << ex.what() << std::endl;
      return 2002;
  }
  cout << ">>> method used to compute upper limit is " << whichMethod << endl;

  if (!whichHintMethod.empty()) {
      map<string, LimitAlgo *>::const_iterator it_hint = methods.find(whichHintMethod);
      if (it_hint == methods.end()) {
          cerr << "Unsupported hint method: " << whichHintMethod << endl;
          cout << "Usage: combine [options]\n";
          cout << "Use combine --help to get a list of all the allowed methods and options"  << endl;
          return 1003;
      } 
      hintAlgo = it_hint->second;
      hintAlgo->applyDefaultOptions();
      cout << ">>> method used to hint where the upper limit is " << whichHintMethod << endl;
  }
  
  std::cout << ">>> random number generator seed is " << seed << std::endl;
  RooRandom::randomGenerator()->SetSeed(seed); 

  TString massName = TString::Format("mH%d.", iMass);
  TString toyName  = "";  if (runToys > 0 || seed != 123456 || vm.count("saveToys")) toyName  = TString::Format("%d.", seed);
  TString fileName = "higgsCombine" + name + "."+whichMethod+"."+massName+toyName+"root";
  TFile *test = new TFile(fileName, "RECREATE"); outputFile = test;
  TTree *t = new TTree("limit", "limit");
  int syst, iToy, iSeed, iChannel; 
  double mass, limit, limitErr; 
  t->Branch("limit",&limit,"limit/D");
  t->Branch("limitErr",&limitErr,"limitErr/D");
  t->Branch("mh",   &mass, "mh/D");
  t->Branch("syst", &syst, "syst/I");
  t->Branch("iToy", &iToy, "iToy/I");
  t->Branch("iSeed", &iSeed, "iSeed/I");
  t->Branch("iChannel", &iChannel, "iChannel/I");
  t->Branch("t_cpu",   &t_cpu_,  "t_cpu/F");
  t->Branch("t_real",  &t_real_, "t_real/F");
  
  //if (vm.count("saveToys")) writeToysHere = new RooWorkspace("toys","toys"); 
  if (vm.count("saveToys")) writeToysHere = test->mkdir("toys","toys"); 
  if (toysFile != "")       readToysFromHere = TFile::Open(toysFile.c_str());
  
  syst = withSystematics;
  mass = iMass;
  iSeed = seed;
  iChannel = 0;

  // if you have libraries, it's time to load them now
  for (vector<string>::const_iterator lib = librariesToLoad.begin(), endlib = librariesToLoad.end(); lib != endlib; ++lib) {
    gSystem->Load(lib->c_str());
  }

  if (vm.count("igpMem")) setupIgProfDumpHook();

  try {
     combiner.run(datacard, dataset, limit, limitErr, iToy, t, runToys);
  } catch (std::exception &ex) {
     cerr << "Error when running the combination:\n\t" << ex.what() << std::endl;
     test->Close();
     return 3001;
  }
  
  test->WriteTObject(t);
  test->Close();

  for(map<string, LimitAlgo *>::const_iterator i = methods.begin(); i != methods.end(); ++i)
    delete i->second;
}


