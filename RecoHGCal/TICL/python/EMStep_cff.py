import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2,
    algo_number = 8,
    iteration_label = "EM",
    LayerClustersInputMask = "trackstersMIP"
)

# CA - PATTERN RECOGNITION

trackstersEM = _trackstersProducer.clone(
    original_mask = "trackstersMIP",
    filtered_mask = cms.InputTag("filteredLayerClustersEM", "EM"),
    seeding_regions = "ticlSeedingGlobal",
    missing_layers = 2,
    min_clusters_per_ntuplet = 10,
    min_cos_theta = 0.94, # ~20 degrees
    min_cos_pointing = 0.7
)

# MULTICLUSTERS

multiClustersFromTrackstersEM = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "trackstersEM"
)

EMStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersEM
    ,trackstersEM
    ,multiClustersFromTrackstersEM)

