#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "RecoEgamma/EgammaElectronProducers/interface/helixToBarrelPropagator.h"

class ElectronNHitSeedProducerNew : public edm::global::EDProducer<> {
public:
  ElectronNHitSeedProducerNew(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  reco::ElectronSeedCollection acceptThisSeed(TrajectorySeed, const edm::Ref<std::vector<reco::SuperCluster> >, reco::ElectronSeedCollection) const ;

  static BoundCylinder& barrel();
  static BoundDisk& negativeEtaEndcap();
  static BoundDisk& positiveEtaEndcap();

  static BoundCylinder* initBarrel();
  static BoundDisk* initPositive();
  static BoundDisk* initNegative();

  static const ReferenceCountingPointer<BoundCylinder> theBarrel_;
  static const ReferenceCountingPointer<BoundDisk> theNegativeEtaEndcap_;
  static const ReferenceCountingPointer<BoundDisk> thePositiveEtaEndcap_;

  static constexpr float epsilon = 0.001;
  /** Hard-wired numbers defining the surfaces on which the crystal front faces lie. */
  static constexpr float barrelRadius = 129.f;       // p81, p50, ECAL TDR
  static constexpr float barrelHalfLength = 270.9f;  // p81, p50, ECAL TDR
  static constexpr float endcapRadius = 171.1f;      // fig 3.26, p81, ECAL TDR
  static constexpr float endcapZ = 320.5f;           // fig 3.26, p81, ECAL TDR
  
private:
 
  const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::EDPutTokenT<reco::ElectronSeedCollection> putToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken;
  edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
};

ElectronNHitSeedProducerNew::ElectronNHitSeedProducerNew(const edm::ParameterSet& pset)
  :   initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
      magFieldToken_(esConsumes()), // Is it constant magnetic field?
      putToken_{produces<reco::ElectronSeedCollection>()},
      geomToken(esConsumes())
{
  superClustersTokens_ = consumes(pset.getParameter<edm::InputTag>("superClusters"));
}

BoundCylinder& ElectronNHitSeedProducerNew::barrel() { return *ElectronNHitSeedProducerNew::theBarrel_; }
BoundDisk& ElectronNHitSeedProducerNew::negativeEtaEndcap() { return *ElectronNHitSeedProducerNew::theNegativeEtaEndcap_; }
BoundDisk& ElectronNHitSeedProducerNew::positiveEtaEndcap() { return *ElectronNHitSeedProducerNew::thePositiveEtaEndcap_; }

BoundCylinder* ElectronNHitSeedProducerNew::initBarrel() {
  Surface::RotationType rot;  // unit rotation matrix
  return new Cylinder(
      barrelRadius,
      Surface::PositionType(0, 0, 0),
      rot,
      new SimpleCylinderBounds(barrelRadius - epsilon, barrelRadius + epsilon, -barrelHalfLength, barrelHalfLength));
}

BoundDisk* ElectronNHitSeedProducerNew::initPositive() {
  Surface::RotationType rot;  // unit rotation matrix
  return new BoundDisk(
      Surface::PositionType(0, 0, endcapZ),
      rot,
      new SimpleDiskBounds(0, endcapRadius, -epsilon, epsilon));
}


BoundDisk* ElectronNHitSeedProducerNew::initNegative() {
  Surface::RotationType rot;  // unit rotation matrix
  return new BoundDisk(
      Surface::PositionType(0, 0, -endcapZ),
      rot,
      new SimpleDiskBounds(0, endcapRadius, -epsilon, epsilon));
}

const ReferenceCountingPointer<BoundCylinder>  ElectronNHitSeedProducerNew::theBarrel_ = initBarrel(); 
const ReferenceCountingPointer<BoundDisk> ElectronNHitSeedProducerNew::thePositiveEtaEndcap_ = initPositive();
const ReferenceCountingPointer<BoundDisk> ElectronNHitSeedProducerNew::theNegativeEtaEndcap_ = initNegative();   

void ElectronNHitSeedProducerNew::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
  desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
  descriptions.add("electronNHitSeedProducerNew", desc);
}

struct trajStateOnSurface{
  GlobalPoint aX = {0,0,0};
  GlobalVector aP = {0,0,0};
  int aCharge = 0;
};

trajStateOnSurface testpropagation(trajStateOnSurface initTsos, float radius){

  trajStateOnSurface tsos;

  tsos.aX = initTsos.aX;
  tsos.aP = initTsos.aP;
  return tsos;
}


void ElectronNHitSeedProducerNew::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  //std::cout << "\n\n NEW EVENT \n " ;
  reco::ElectronSeedCollection eleSeeds{};
  
  auto const& magField = iSetup.getData(magFieldToken_);
  const TrackerGeometry* theG = &iSetup.getData(geomToken);

  PropagatorWithMaterial forwardPropagator_ =  PropagatorWithMaterial(alongMomentum, 0.000511, &magField);

  //LOOP on initial seeds
  for (auto& initialSeedRef : iEvent.get(initialSeedsToken_)) {

    PTrajectoryStateOnDet state1 = initialSeedRef.startingState();
    
    //std::cout<<"---  N hits in the seed: " << initialSeedRef.nHits()<<"  Starting state position: " << initialSeedRef.startingState().parameters().position()<<"  Starting state direction: " << initialSeedRef.startingState().parameters().momentum()<<"----------"<<std::endl;
    //for (size_t iHit = 0; iHit < initialSeedRef.nHits(); iHit++) {
    //  auto const& recHit = *(initialSeedRef.recHits().begin() + iHit);
    //  std::cout<<"---- rechit Global position : "<<recHit.globalPosition()<<"  ---------- rechit Local position : "<<recHit.localPosition()<<"-------------"<<std::endl;
    //}


    DetId detId1(state1.detId());
    TrajectoryStateOnSurface tsos1 = trajectoryStateTransform::transientState(state1, &(theG->idToDet(detId1)->surface()), &iSetup.getData(magFieldToken_));
    TrajectoryStateOnSurface tsos2 = trajectoryStateTransform::transientState(state1, &(theG->idToDet(detId1)->surface()), &iSetup.getData(magFieldToken_));

    // ---------------------------------------------------------------------------
    // Pass all the needed variables to perform the propagation 
    //trajStateOnSurface oldTsos;
    //oldTsos.aX = tsos1.globalPosition();
    //oldTsos.aP = tsos1.globalMomentum();
    //auto newTsos = testpropagation(oldTsos,10.);
    //std::cout<<"  New tsos   "<< newTsos.aX;
    // ---------------------------------------------------------------------------
      

    // ------------------------------------------------------------------------------------
    // Original new algo 

    bool runCPUvesrion_ = true;

    if(runCPUvesrion_){

      if (!tsos1.isValid()) 
        continue;

      TrajectoryStateOnSurface stateAtECAL_ = forwardPropagator_.propagate(tsos1, ElectronNHitSeedProducerNew::barrel());

      //if(stateAtECAL_.isValid())
      //  std::cout<<" Original propagator position : "<< stateAtECAL_.globalPosition() <<"  and eta " << stateAtECAL_.globalPosition().eta() << std::endl;

      if (!stateAtECAL_.isValid() || (stateAtECAL_.isValid() && fabs(stateAtECAL_.globalPosition().eta()) > 1.479)) {
        continue; // ONLY BARREL
        if (tsos1.globalPosition().eta() > 0.) {
          stateAtECAL_ = forwardPropagator_.propagate(tsos1, positiveEtaEndcap());
          //if (stateAtECAL_.isValid())
          //  std::cout<<" Original propagator position at endcap : "<< stateAtECAL_.globalPosition() <<"  and eta " << stateAtECAL_.globalPosition().eta() << std::endl;
        } 
        else {
          stateAtECAL_ = forwardPropagator_.propagate(tsos1, negativeEtaEndcap());
        }
      }
      //if (stateAtECAL_.isValid())
      //  std::cout<<" Original propagator position at endcap : "<< stateAtECAL_.globalPosition() <<"  and eta " << stateAtECAL_.globalPosition().eta() << std::endl;
      if (!stateAtECAL_.isValid()) {
        continue;
      }
      else { //stateAtECAL_ is valid

        // CMSSW CPU implementation 
        for (auto& superClusRef : iEvent.get(superClustersTokens_)) {
          //	int nClus=superClusRef->clustersSize(); IF needed separate cuts for different nClus
          float sc_et = superClusRef->energy() / std::cosh(superClusRef->position().eta());
          double deltar2 =
          reco::deltaR2(stateAtECAL_.globalPosition().eta(),
          stateAtECAL_.globalPosition().phi(), superClusRef->seed()->position().eta(), superClusRef->position().phi());
          double pTratio= sc_et / stateAtECAL_.globalMomentum().perp();

          //  was E-5
          //  eventually these hard-coded values will go to config file, just testing now..
          if (sc_et <= 20.0) { //low pT
            if (deltar2<1E-5 && pTratio>0.8 && pTratio<1.2) { // these cuts need optimisation
              eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
            }
          }
          //////
          // was E-5
          else if (sc_et>20.0 && sc_et<=50.0) { //medium pT -> relax pTratio cut 
            if (deltar2<1E-5 && pTratio>0.4 && pTratio<1.6) {
              eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
            }
          }
          //////
          else if (sc_et>50.0) { //high pT -> no pTratio cut
            if (deltar2<1E-4) {
              eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
            }
          }
        } // Loop over SCs
      } //stateAtECAL_ is valid
    }
    else{
      // ----------------  NEW   --------------------------------------------------------------------
      // Try out the helixBarrelCylinderCrossing propagator
      if (!tsos2.isValid()) continue;
      double rho = tsos2.transverseCurvature();
      bool theSolExists = false;
      GlobalPoint x = {0,0,0}; // position
      GlobalVector p = {0,0,0}; //momentun
      double s=0; //path length
      helixBarrelCylinderCrossing(tsos2.globalPosition(),tsos2.globalMomentum(),rho,barrelRadius,theSolExists,x,p,s);
      //std::cout<<"  theSolExists : "<<theSolExists<< std::endl;
      //std::cout<<" Pos : "<< x  <<"  Path Length : "<< s << " Direction : "<< p << std::endl;
      // --------------------------------------------------------------------------------------------------
      if(!(fabs(x.z())<endcapZ))
        continue;
      if(!theSolExists)
        continue;

      // New CPU implementation
      // ---------------------------------------------------------------------------
      for (auto& superClusRef : iEvent.get(superClustersTokens_)) {
        //	int nClus=superClusRef->clustersSize(); IF needed separate cuts for different nClus
        float sc_et = superClusRef->energy() / std::cosh(superClusRef->position().eta());
        double deltar2 =
        reco::deltaR2(x.eta(),
        x.phi(), superClusRef->seed()->position().eta(), superClusRef->position().phi());
        double pTratio= sc_et / p.perp();

        //  eventually these hard-coded values will go to config file, just testing now..
        if (sc_et <= 20.0) { //low pT
          if (deltar2<1E-5 && pTratio>0.8 && pTratio<1.2) { // these cuts need optimisation
            eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
          }
        }
        //////
        else if (sc_et>20.0 && sc_et<=50.0) { //medium pT -> relax pTratio cut 
          if (deltar2<1E-5 && pTratio>0.4 && pTratio<1.6) {
            eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
          }
        }
        //////
        else if (sc_et>50.0) { //high pT -> no pTratio cut
          if (deltar2<1E-4 ) {
            eleSeeds=acceptThisSeed( initialSeedRef,  superClusRef,  eleSeeds);
          }
        }
      } // loop over SC ends
      // ------------------------------------------------------------------------------------
    }
  } // Loop on initial seed ends
  std::cout << "New: eleSeeds size " << eleSeeds.size() << std::endl;
  iEvent.emplace(putToken_, std::move(eleSeeds));
}

reco::ElectronSeedCollection ElectronNHitSeedProducerNew::acceptThisSeed(
						 TrajectorySeed initialSeedRef, const edm::Ref<std::vector<reco::SuperCluster> > superClusRef,
						 reco::ElectronSeedCollection eleSeeds) const {
  //std::cout << "BEGIN: eleSeeds size in acceptThisSeed function " << eleSeeds.size() << std::endl;
  reco::ElectronSeed eleSeed(initialSeedRef);
  reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
  eleSeed.setCaloCluster(caloClusRef);
  eleSeed.setCaloCluster(caloClusRef);
  eleSeeds.push_back(eleSeed); // accept this initial seed as final seed
  //std::cout << "END: eleSeeds size in acceptThisSeed function " << eleSeeds.size() << std::endl;
  return eleSeeds;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronNHitSeedProducerNew);
