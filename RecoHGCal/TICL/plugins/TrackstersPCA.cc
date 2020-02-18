#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackstersPCA.h"
#include "TPrincipal.h"

#include <iostream>

void ticl::assignPCAtoTracksters(std::vector<Trackster> & tracksters,
    const std::vector<reco::CaloCluster> &layerClusters, double z_limit_em, bool energyWeight) {
  TPrincipal pca(3, "");
  LogDebug("TrackstersPCA") << "-------" << std::endl;
  for (auto &trackster : tracksters) {
    pca.Clear();
    trackster.raw_energy = 0.;
    trackster.raw_em_energy = 0.;
    trackster.raw_pt = 0.;
    trackster.raw_em_pt = 0.;
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      auto fraction = 1.f / trackster.vertex_multiplicity[i];
      trackster.raw_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
      if (std::abs(layerClusters[trackster.vertices[i]].z()) <= z_limit_em)
        trackster.raw_em_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
      double point[3] = {
        layerClusters[trackster.vertices[i]].x(),
        layerClusters[trackster.vertices[i]].y(),
        layerClusters[trackster.vertices[i]].z()
      };
      if (!energyWeight)
        pca.AddRow(&point[0]);
    }

    float weights_sum = 0.f;
    if (energyWeight) {
      for (size_t i = 0; i < trackster.vertices.size(); ++i) {
        float weight = 1.f;
        auto fraction = 1.f / trackster.vertex_multiplicity[i];
        if (trackster.raw_energy)
          weight = (trackster.vertices.size() / trackster.raw_energy) * layerClusters[trackster.vertices[i]].energy() * fraction;
        weights_sum += weight;
        double pointE[3] = {
          weight * layerClusters[trackster.vertices[i]].x(),
          weight * layerClusters[trackster.vertices[i]].y(),
          weight * layerClusters[trackster.vertices[i]].z()
        };
        pca.AddRow(&pointE[0]);
      }
    }

    pca.MakePrincipals();
    trackster.barycenter = ticl::Trackster::Vector((*(pca.GetMeanValues()))[0],
        (*(pca.GetMeanValues()))[1],
        (*(pca.GetMeanValues()))[2]);
    trackster.eigenvalues[0] = (float)(*(pca.GetEigenValues()))[0];
    trackster.eigenvalues[1] = (float)(*(pca.GetEigenValues()))[1];
    trackster.eigenvalues[2] = (float)(*(pca.GetEigenValues()))[2];
    trackster.eigenvectors[0] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][0],
        (*(pca.GetEigenVectors()))[1][0],
        (*(pca.GetEigenVectors()))[2][0] );
    trackster.eigenvectors[1] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][1],
        (*(pca.GetEigenVectors()))[1][1],
        (*(pca.GetEigenVectors()))[2][1] );
    trackster.eigenvectors[2] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][2],
        (*(pca.GetEigenVectors()))[1][2],
        (*(pca.GetEigenVectors()))[2][2] );
    if (trackster.eigenvectors[0].z() * trackster.barycenter.z() < 0.0) {
      trackster.eigenvectors[0] = -ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][0],
          (*(pca.GetEigenVectors()))[1][0],
          (*(pca.GetEigenVectors()))[2][0] );
    }
    trackster.sigmas[0] = (float)(*(pca.GetSigmas()))[0];
    trackster.sigmas[1] = (float)(*(pca.GetSigmas()))[1];
    trackster.sigmas[2] = (float)(*(pca.GetSigmas()))[2];
    auto norm = std::sqrt(trackster.eigenvectors[0].Unit().perp2());
    trackster.raw_pt = norm * trackster.raw_energy;
    trackster.raw_em_pt = norm * trackster.raw_em_energy;
    const auto & mean = *(pca.GetMeanValues());
    const auto & eigenvectors = *(pca.GetEigenVectors());
    const auto & eigenvalues = *(pca.GetEigenValues());
    const auto & sigmas = *(pca.GetSigmas());
    LogDebug("TrackstersPCA") << "Trackster characteristics: " << std::endl;
    LogDebug("TrackstersPCA") << "Size: " << trackster.vertices.size() << std::endl;
    LogDebug("TrackstersPCA") << "Mean: " << mean[0] << ", " << mean[1] << ", " << mean[2] << std::endl;
    LogDebug("TrackstersPCA") << "Weights sum:" << weights_sum << std::endl;
    LogDebug("TrackstersPCA") << "EigenValues: " << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2]  << std::endl;
    LogDebug("TrackstersPCA") << "EigeVectors 1: " << eigenvectors(0, 0) << ", " << eigenvectors(1, 0) << ", " << eigenvectors(2, 0) <<std::endl;
    LogDebug("TrackstersPCA") << "EigeVectors 2: " << eigenvectors(0, 1) << ", " << eigenvectors(1, 1) << ", " << eigenvectors(2, 1) <<std::endl;
    LogDebug("TrackstersPCA") << "EigeVectors 3: " << eigenvectors(0, 2) << ", " << eigenvectors(1, 2) << ", " << eigenvectors(2, 2) <<std::endl;
    LogDebug("TrackstersPCA") << "Sigmas: " << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2] << std::endl;
  }
}
