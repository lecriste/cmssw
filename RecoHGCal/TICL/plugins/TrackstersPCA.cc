#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackstersPCA.h"
#include "TPrincipal.h"

#include <iostream>

void ticl::assignPCAtoTracksters(std::vector<Trackster> & tracksters,
    const std::vector<reco::CaloCluster> &layerClusters, double z_limit_em, bool energyWeight) {
  TPrincipal pca(3, "D"); // D option is necessary to use the GetRow method
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
      if (!energyWeight) {
        double point[3] = {
          layerClusters[trackster.vertices[i]].x(),
          layerClusters[trackster.vertices[i]].y(),
          layerClusters[trackster.vertices[i]].z()
        };
        pca.AddRow(&point[0]);
      }
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
    double sigmasPCA[3] = {0};
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      double p[3];
      pca.X2P(pca.GetRow(i), p);
      sigmasPCA[0] += (p[0] - trackster.barycenter.x()) * (p[0] - trackster.barycenter.x());
      sigmasPCA[1] += (p[1] - trackster.barycenter.y()) * (p[1] - trackster.barycenter.y());
      sigmasPCA[2] += (p[2] - trackster.barycenter.z()) * (p[2] - trackster.barycenter.z());
    }
    for (size_t i=0; i<3; ++i) {
      sigmasPCA[i] = std::sqrt(sigmasPCA[i]/trackster.vertices.size());
      trackster.sigmas[i] = (float)(*(pca.GetSigmas()))[i];
      trackster.sigmasPCA[i] = sigmasPCA[i];
      trackster.eigenvalues[i] = (float)(*(pca.GetEigenValues()))[i];
      trackster.eigenvectors[i] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][i],
        (*(pca.GetEigenVectors()))[1][i],
        (*(pca.GetEigenVectors()))[2][i] );
    }
    if (trackster.eigenvectors[0].z() * trackster.barycenter.z() < 0.0) {
      trackster.eigenvectors[0] = -ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][0],
          (*(pca.GetEigenVectors()))[1][0],
          (*(pca.GetEigenVectors()))[2][0] );
    }
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
    LogDebug("TrackstersPCA") << "SigmasPCA: " << sigmasPCA[0] << ", " << sigmasPCA[1] << ", " << sigmasPCA[2] << std::endl;

    std::cout << "Trackster characteristics: " << std::endl;
    std::cout << "Size: " << trackster.vertices.size() << std::endl;
    std::cout << "Energy: " << trackster.raw_energy << std::endl;
    std::cout << "Mean: " << mean[0] << ", " << mean[1] << ", " << mean[2] << std::endl;
    std::cout << "Weights sum:" << weights_sum << std::endl;
    std::cout << "EigenValues: " << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2]  << std::endl;
    std::cout << "EigeVectors 1: " << eigenvectors(0, 0) << ", " << eigenvectors(1, 0) << ", " << eigenvectors(2, 0) <<std::endl;
    std::cout << "EigeVectors 2: " << eigenvectors(0, 1) << ", " << eigenvectors(1, 1) << ", " << eigenvectors(2, 1) <<std::endl;
    std::cout << "EigeVectors 3: " << eigenvectors(0, 2) << ", " << eigenvectors(1, 2) << ", " << eigenvectors(2, 2) <<std::endl;
    std::cout << "Sigmas: " << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2] << std::endl;
    std::cout << "SigmasPCA: " << sigmasPCA[0] << ", " << sigmasPCA[1] << ", " << sigmasPCA[2] << std::endl;
  }
}
