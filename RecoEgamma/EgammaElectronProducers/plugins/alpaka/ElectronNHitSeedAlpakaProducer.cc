#include <Eigen/Core>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/EleSeedSoA.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperclusterDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/SuperclusterHostCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/EleSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/EleSeedHostCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/ParametrizedEngine/interface/ParabolicParametrizedMagneticField.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

// Additional includes for testing / comparing implementations
#include "PixelMatchingAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/ftsFromVertexToPointPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixBarrelPlaneCrossingByCircle.h"
#include "DataFormats/EgammaReco/interface/EleRelPointPairPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing2Order.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixForwardPlaneCrossing.h"

#include "DataFormats/EgammaReco/interface/Plane.h"

using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	class ElectronNHitSeedAlpakaProducer : public global::EDProducer<> {
	public:
		ElectronNHitSeedAlpakaProducer(const edm::ParameterSet& pset): 
			deviceToken_{produces()},
			size_{pset.getParameter<int32_t>("size")},
			initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
			beamSpotToken_(consumes(pset.getParameter<edm::InputTag>("beamSpot"))),
			magFieldToken_(esConsumes()),
			geomToken(esConsumes())
			{
				superClustersTokens_ = consumes(pset.getParameter<edm::InputTag>("superClusters"));
			}

		void produce(edm::StreamID sid, device::Event& event, device::EventSetup const& iSetup) const override {

			auto vprim_ = event.get(beamSpotToken_).position();
			GlobalPoint vprim(vprim_.x(), vprim_.y(), vprim_.z());
			Vector3f vertex{vprim.x(),vprim.y(),vprim.z()};

			// NEW EVENT 

			std::cout<<" -----> NEW EVENT with vprim : "<<vprim_<<std::endl;

			// Get MagField ESProduct for comparing & Geom ESProduct 
			auto const& magField = iSetup.getData(magFieldToken_);
			//const TrackerGeometry* theG = &iSetup.getData(geomToken);


			PropagatorWithMaterial backwardPropagator_ = PropagatorWithMaterial(oppositeToMomentum, 0.000511, &magField);

			reco::SuperclusterHostCollection hostProductSCs{int(event.get(superClustersTokens_).size()), event.queue()};
			reco::SuperclusterDeviceCollection deviceProductSCs{int(event.get(superClustersTokens_).size()), event.queue()};

			reco::EleSeedHostCollection hostProductSeeds{int(event.get(initialSeedsToken_).size()), event.queue()};
			reco::EleSeedDeviceCollection deviceProductSeeds{int(event.get(initialSeedsToken_).size()), event.queue()};

			auto& viewSCs = hostProductSCs.view();
			auto& viewSeeds = hostProductSeeds.view();

			////////////////////////////////////////////////////////////////////
			// Fill in SOAs
			// Technically should separate in different producers that create the SoAs
			// Info on SoAs : https://github.com/cms-sw/cmssw/blob/master/DataFormats/SoATemplate/README.md


			/////////////////////////////////////////////////////////////
			// Can I use these maps for proper conversion back to legacy?
			/////////////////////////////////////////////////////////////
			// In order to write out a reduced collection of matched seeds?
			// Might want to create some sort of assiciation SoA

			std::map<int, reco::SuperClusterRef> superClusterRefMap_;
			std::map<int, TrajectorySeed> seedRefMap_;

			int i = 0;
	        	for (auto& superClusRef : event.get(superClustersTokens_))
			{
				viewSCs[i].id() =  i;
				viewSCs[i].scSeedTheta() =  superClusRef->seed()->position().theta();
				viewSCs[i].scPhi() = superClusRef->position().phi();
				viewSCs[i].scR() = superClusRef->position().r();
				viewSCs[i].scEnergy() = superClusRef->energy();
    
				// Filling in a map with the whole object
				superClusterRefMap_[i] = superClusRef;

				++i;
			}

			// Printing the contents of the map to check if it's filled properly
			// for (const auto& entry : superClusterRefMap_) {
			// 		std::cout << "Index " << entry.first << ": " << std::endl;
			// 		std::cout << "  theta: " << entry.second->seed()->position().theta();
			// }

			// This is so ugly :( 
			// To figure out is there is another way to bulid this SoA
			i = 0;
			for (auto& initialSeedRef : event.get(initialSeedsToken_)) 
			{	
				// Filling in a map with the whole object
    				seedRefMap_[i] = initialSeedRef;  

				// Fill in the view
				viewSeeds[i].nHits() = initialSeedRef.nHits();
				viewSeeds[i].id() = i;		
				viewSeeds[i].isMatched() = 0;		
				viewSeeds[i].matchedScID() = -1;

				auto const& recHit = *(initialSeedRef.recHits().begin() + 0);  
				viewSeeds[i].detectorID().x() = recHit.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? 1: 0;
				viewSeeds[i].isValid().x() = recHit.isValid();


				viewSeeds[i].hitPosX().x() = recHit.globalPosition().x();
				viewSeeds[i].hitPosY().x() = recHit.globalPosition().y();
				viewSeeds[i].hitPosZ().x() = recHit.globalPosition().z();
				viewSeeds[i].surfPosX().x() = recHit.det()->surface().position().x();
				viewSeeds[i].surfPosY().x() = recHit.det()->surface().position().y();
				viewSeeds[i].surfPosZ().x() = recHit.det()->surface().position().z();
				viewSeeds[i].surfRotX().x() = recHit.det()->surface().rotation().z().x();
				viewSeeds[i].surfRotY().x() = recHit.det()->surface().rotation().z().y();
				viewSeeds[i].surfRotZ().x() = recHit.det()->surface().rotation().z().z();

				auto const& recHit1 = *(initialSeedRef.recHits().begin() + 1);  
				viewSeeds[i].detectorID().y() = recHit1.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? 1: 0;
				viewSeeds[i].isValid().y() = recHit1.isValid();
				viewSeeds[i].hitPosX().y() = recHit1.globalPosition().x();
				viewSeeds[i].hitPosY().y() = recHit1.globalPosition().y();
				viewSeeds[i].hitPosZ().y() = recHit1.globalPosition().z();
				viewSeeds[i].surfPosX().y() = recHit1.det()->surface().position().x();
				viewSeeds[i].surfPosY().y() = recHit1.det()->surface().position().y();
				viewSeeds[i].surfPosZ().y() = recHit1.det()->surface().position().z();
				viewSeeds[i].surfRotX().y() = recHit1.det()->surface().rotation().z().x();
				viewSeeds[i].surfRotY().y() = recHit1.det()->surface().rotation().z().y();
				viewSeeds[i].surfRotZ().y() = recHit1.det()->surface().rotation().z().z();

				if(initialSeedRef.nHits()>2){
					auto const& recHit2 = *(initialSeedRef.recHits().begin() + 2);  
					viewSeeds[i].detectorID().z() = recHit2.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? 1: 0;
					viewSeeds[i].isValid().z() = recHit2.isValid();
					viewSeeds[i].hitPosX().z() = recHit2.globalPosition().x();
					viewSeeds[i].hitPosY().z() = recHit2.globalPosition().y();
					viewSeeds[i].hitPosZ().z() = recHit2.globalPosition().z();
					viewSeeds[i].surfPosX().z() = recHit2.det()->surface().position().x();
					viewSeeds[i].surfPosY().z() = recHit2.det()->surface().position().y();
					viewSeeds[i].surfPosZ().z() = recHit2.det()->surface().position().z();
					viewSeeds[i].surfRotX().z() = recHit2.det()->surface().rotation().z().x();
					viewSeeds[i].surfRotY().z() = recHit2.det()->surface().rotation().z().y();
					viewSeeds[i].surfRotZ().z() = recHit2.det()->surface().rotation().z().z();		
				}
				else{
					viewSeeds[i].detectorID().z() = 0;
					viewSeeds[i].isValid().z() = 0;
					viewSeeds[i].hitPosX().z() = 0;
					viewSeeds[i].hitPosY().z() = 0;
					viewSeeds[i].hitPosZ().z() = 0;
					viewSeeds[i].surfPosX().z() = 0;
					viewSeeds[i].surfPosY().z() = 0;
					viewSeeds[i].surfPosZ().z() = 0;
					viewSeeds[i].surfRotX().z() = 0;
					viewSeeds[i].surfRotY().z() = 0;
					viewSeeds[i].surfRotZ().z() = 0;		
				}
				
				++i;
			}

			alpaka::memcpy(event.queue(), deviceProductSCs.buffer(), hostProductSCs.buffer());
			alpaka::memcpy(event.queue(), deviceProductSeeds.buffer(), hostProductSeeds.buffer());

			// Print the SoA 
			// algo_.printEleSeeds(event.queue(), deviceProductSeeds);
			// algo_.printSCs(event.queue(), deviceProductSCs);

			// Matching algorithm
			algo_.matchSeeds(event.queue(), deviceProductSeeds, deviceProductSCs,vertex(0),vertex(1), vertex(2));

			alpaka::memcpy(event.queue(), hostProductSeeds.buffer(), deviceProductSeeds.buffer());

			auto& view = hostProductSeeds.view();
			// for (int i = 0; i < view.metadata().size(); ++i) {
			// 	if(view[i].isMatched()>0){
			// 		std::cout << "Seed " << i << ":" << std::endl;
			// 		std::cout << "  nHits: " << view[i].nHits() << std::endl;
			// 		std::cout << "  isMatched: " << view[i].isMatched() << std::endl;
			// 		std::cout << "  matchedScID: " << view[i].matchedScID() << std::endl;

			// 		// Print hit position details
			// 		for (int j = 0; j < 3; ++j) {
			// 			std::cout << "  Hit " << j << " position: "
			// 					<< " x: " << view[i].hitPosX().x() << ", "
			// 					<< " y: " << view[i].hitPosY().x() << ", "
			// 					<< " z: " << view[i].hitPosZ().x() << std::endl;
			// 		}
			// 		std::cout << std::endl;
			// 	}
			// }

			reco::ElectronSeedCollection eleSeeds{};

			for (int i = 0; i < view.metadata().size(); ++i) {
				if (view[i].isMatched() > 0) {
					int matchedScID = view[i].matchedScID();
					auto scIter = superClusterRefMap_.find(matchedScID);
			 		std::cout << "  matchedScID: " << view[i].matchedScID() << std::endl;

					if (scIter != superClusterRefMap_.end()) {
						const reco::SuperClusterRef& superClusRef = scIter->second;
						auto seedIter = seedRefMap_.find(view[i].id());
						if (seedIter != seedRefMap_.end()) {
							const TrajectorySeed& matchedSeed = seedIter->second;
							reco::ElectronSeed eleSeed(matchedSeed);
							reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
							eleSeed.setCaloCluster(caloClusRef);
							eleSeeds.emplace_back(eleSeed);
						}
					} else {
						std::cerr << "No SuperCluster found for SC ID " << matchedScID << std::endl;
					}
				}
			}
			std::cout << "New eleSeeds size " << eleSeeds.size() << std::endl;
			superClusterRefMap_.clear();
			seedRefMap_.clear();

			// Shouldnt I get some sort of print out from the GPU?
      		//alpaka::wait(event.queue()); 


			// For testing developments wrt legacy implementations
	        for (auto& superClusRef : event.get(superClustersTokens_)) 
			{
				float x = superClusRef->position().r() * sin(superClusRef->seed()->position().theta()) * cos(superClusRef->position().phi());
				float y = superClusRef->position().r() * sin(superClusRef->seed()->position().theta()) * sin(superClusRef->position().phi());
				float z = superClusRef->position().r() * cos(superClusRef->seed()->position().theta());
				GlobalPoint center(x, y, z);
				float theMagField = magField.inTesla(center).mag();
				Vector3f position{x,y,z};
				GlobalPoint sc(GlobalPoint::Polar(superClusRef->seed()->position().theta(),  //seed theta
                                                superClusRef->position().phi(),    //supercluster phi
                                                superClusRef->position().r()));    //supercluster r
			
				for (int charge : {1,-1}) 
				{
					auto freeTS = trackingTools::ftsFromVertexToPoint(magField, sc, vprim, superClusRef->energy(), 1);
					auto initialTrajState = TrajectoryStateOnSurface(freeTS, *PerpendicularBoundPlaneBuilder{}(freeTS.position(), freeTS.momentum()));

					auto newfreeTS = ftsFromVertexToPointPortable::ftsFromVertexToPoint(position, vertex, superClusRef->energy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(position));			
					Vector3f testposition = {newfreeTS.position(0),newfreeTS.position(1),newfreeTS.position(2)};
					Vector3f testmomentum = {newfreeTS.momentum(0),newfreeTS.momentum(1),newfreeTS.momentum(2)};

					auto transverseCurvature = [](const Vector3f& p, int charge, const float& magneticFieldZ) -> float {
							return -2.99792458e-3f * (charge / sqrt(p(0) * p(0) + p(1) * p(1))) * magneticFieldZ;  
					};

					int notValid_old = 0;
					int notValid_new = 0;
					int seeds = 0;
					for (auto& initialSeedRef : event.get(initialSeedsToken_)) 
					{			
						++seeds;
						auto const& recHit = *(initialSeedRef.recHits().begin() + 0);  

						if(!recHit.isValid())      
							continue;

						auto state = backwardPropagator_.propagate(initialTrajState, recHit.det()->surface());

						if(!state.isValid())      
							++notValid_old;

						Vector3f recHitpos{recHit.globalPosition().x(),recHit.globalPosition().y(),recHit.globalPosition().z()};
						Vector3f surfPosition{recHit.det()->surface().position().x(),recHit.det()->surface().position().y(),recHit.det()->surface().position().z()};
						Vector3f surfRotation{recHit.det()->surface().rotation().z().x(),recHit.det()->surface().rotation().z().y(),recHit.det()->surface().rotation().z().z()};
						printf("\nsurfPosition ALPAKA : (%f, %f, %f)", surfPosition.x(), surfPosition.y(), surfPosition.z());
						printf("\nsurfRotation ALPAKA : (%f, %f, %f)", surfRotation.x(), surfRotation.y(), surfRotation.z());					
						Vector3f x2{0,0,0};
						Vector3f p2{0,0,0};
						double rho = 0.;
						double s = 0.;					
						bool theSolExists = false;

						PlanePortable::Plane<Vector3f> plane{surfPosition,surfRotation};
						rho = transverseCurvature(testmomentum,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(position));

						constexpr float small = 1.e-6;  // for orientation of planes
						auto u = plane.normalVector();
						if (std::abs(u(2)) < small) {
							// HelixBarrelPlaneCrossing,
							Propagators::helixBarrelPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,x2,p2,s);
						} 
						else if ((std::abs(u(0)) < small) && (std::abs(u(1)) < small)) 
						{
							// forward plane HelixForwardPlaneCrossing
							Propagators::helixForwardPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,plane,s,x2,p2,theSolExists);
						} 
						else {
							// arbitrary plane HelixArbitraryPlaneCrossing
							Propagators::helixBarrelPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,x2,p2,s);
							//Propagators::helixArbitraryPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,plane,s,x2,p2,theSolExists); // Should check if there is a logic bug - giving similar results but also non valid solutions
						}

						if(!theSolExists)
							++notValid_new;

						if(!state.isValid())
							continue;

						if(!theSolExists)
							continue;

						p2.normalize(); 
						p2*= testmomentum.norm();

						if(false){
							std::cout<<" New "<< rho <<"  and old "<<freeTS.transverseCurvature() <<std::endl;
							std::cout<<" recHit.det()->surface().position() "<<recHit.det()->surface().position()<<" rotation? "<<recHit.det()->surface().rotation().z()<<std::endl;
							std::cout<<" Print out legacy fts position "<< freeTS.position() <<"  and new implementation one : "<<testposition(0) <<" "<<testposition(1)<<" "<<testposition(2)<<std::endl;
							std::cout<<" Print out legacy fts momentum "<< freeTS.momentum() <<"  and new implementation one : "<<testmomentum(0) <<" "<<testmomentum(1)<<" "<<testmomentum(2) <<std::endl;
							std::cout<<" initialTrajState pos "<< initialTrajState.globalPosition()<<"  and momentum  "<<initialTrajState.globalMomentum()<<std::endl;
							std::cout<<" surfPosition " <<surfPosition(0)<<" "<<surfPosition(1)<<" "<<surfPosition(2)<<std::endl;
							std::cout<<" surfRotation " <<surfRotation(0)<<" "<<surfRotation(1)<<" "<<surfRotation(2)<<std::endl;
							std::cout<<" pt = startingDir.head(2).norm() "<< testmomentum.head(2).norm() << " and the equivalent "<< initialTrajState.globalMomentum().perp() <<std::endl;
							std::cout<<" test plane stuff  norm vec"<< plane.normalVector() << "recHit.det()->surface().normalVector "<<recHit.det()->surface().normalVector()<<std::endl;
							std::cout<<" test plane stuff localZ "<< -plane.localZ(testposition) << "recHit.det()->surface().normalVector "<< -recHit.det()->surface().localZ(GlobalPoint(initialTrajState.globalPosition()))<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().position()<<"   and new "<< x2(0) <<" "<< x2(1) << " "<< x2(2)<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().momentum()<<"   and new "<< p2(0) <<" "<< p2(1) << " "<< p2(2)<<std::endl;
							std::cout<<" The path length is : "<< s << std::endl;
							EleRelPointPair pointPair(recHit.globalPosition(), state.globalParameters().position(), vprim);
							EleRelPointPairPortable::EleRelPointPair<Vector3f> pair(recHitpos,x2,vertex);
							printf("Old point pair dZ %lf, dPerp %lf, and dPhi %lf\n",pointPair.dZ(),pointPair.dPerp(),pointPair.dPhi());
							printf("New point pair dZ %lf, dPerp %lf, and dPhi %lf \n",pair.dZ(),pair.dPerp(),pair.dPhi());
						}

						if(false){
							std::cout<<" Initial: "<< state.globalParameters().position()<<"   and new "<< x2(0) <<" "<< x2(1) << " "<< x2(2)<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().momentum()<<"   and new "<< p2(0) <<" "<< p2(1) << " "<< p2(2)<<std::endl;
						}
					}

					if(false){
						std::cout<<"Number of seeds: "<<seeds<<" Propagation notValid_old "<< notValid_old << "  notValid_new " <<notValid_new<<std::endl;
						printf("Print out legacy fts position %f and new fts position  and %lf \n",freeTS.position().x(), testposition(0));
						printf("For SC i=%d Energy is :%f , theta is :%f,  r is : %f \n",i,superClusRef->energy(),superClusRef->seed()->position().theta(),superClusRef->position().r()) ;
						printf(" view %lf ", viewSCs[i].scR());
						printf("Magnetic field full  = %f and the parabolic approximation %f ", theMagField, magneticFieldParabolicPortable::magneticFieldAtPoint(position));
					}
				}
			}

			event.emplace(deviceToken_, std::move(deviceProductSeeds));
		}

		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
			edm::ParameterSetDescription desc;
			desc.add<int32_t>("size");
			desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
			desc.add<edm::InputTag>("beamSpot", {"hltOnlineBeamSpot"});
  			desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
			descriptions.addWithDefaultLabel(desc);
		}

	private:
		const device::EDPutToken<reco::EleSeedDeviceCollection> deviceToken_;
		const int32_t size_;
		const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
		const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
		edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
		const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken;
		edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
		PixelMatchingAlgo const algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(ElectronNHitSeedAlpakaProducer);

