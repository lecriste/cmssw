import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersDummy = _filteredLayerClustersProducer.clone(
  clusterFilter = "ClusterFilterByAlgoAndSize",
  min_cluster_size = 2, #inclusive
  algo_number = 8,
  iteration_label = "Dummy"
)

# CA - PATTERN RECOGNITION

ticlTrackstersDummy = _trackstersProducer.clone(
  filtered_mask = cms.InputTag("filteredLayerClustersDummy", "Dummy"),
  seeding_regions = "ticlSeedingGlobal",
  filter_on_categories = [2, 4], # filter muons and charged hadrons
  pid_threshold = 0.0, # 0 means: do not filter
  skip_layers = 0,
  min_layers_per_trackster = 10,
  min_cos_theta = -1., # Fully inclusive
  min_cos_pointing = -1., # Fully inclusive
  max_delta_time = -1.,
  algo_verbosity = 2,
  oneTracksterPerTrackSeed = False,
  promoteEmptyRegionToTrackster = False,
  itername = "DUMMY"
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersDummy = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersDummy"
)

ticlDummyStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersDummy
    ,ticlTrackstersDummy
    ,ticlMultiClustersFromTrackstersDummy)

