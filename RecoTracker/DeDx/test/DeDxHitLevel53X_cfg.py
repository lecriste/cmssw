import FWCore.ParameterSet.Config as cms

process = cms.Process("DEDX")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START53_V14::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValJpsiMM/GEN-SIM-RECO/v2/00000/C475A7F8-352A-E211-91A9-001A92971B68.root'
   )
)


process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")


process.dedxHitInfo               = cms.EDProducer("HSCPDeDxInfoProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
    ProbabilityMode    = cms.untracked.string("Accumulation"),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    UseCalibration     = cms.bool(False),
    calibrationPath    = cms.string("file:Gains.root"),
    ShapeTest          = cms.bool(True),
)


process.p = cms.Path( process.offlineBeamSpot * process.TrackRefitter * process.dedxHitInfo)


process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep SiStripClusteredmNewDetSetVector_generalTracks_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracks_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_dedxHitInfo_*_*",
    ),
    fileName = cms.untracked.string('DEDX.root')
)

process.endPath = cms.EndPath(process.OUT)




