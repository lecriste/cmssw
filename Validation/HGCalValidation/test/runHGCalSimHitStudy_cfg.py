import FWCore.ParameterSet.Config as cms

geometry = 'D35'
geometry = 'D41'
geometry = 'D46'

if geometry == 'D35':
    from Configuration.Eras.Era_Phase2C4_timing_layer_bar_cff import Phase2C4_timing_layer_bar
    process = cms.Process('PROD',Phase2C4_timing_layer_bar)
    process.load('Configuration.Geometry.GeometryExtended2026D35_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D35Reco_cff')
elif geometry == 'D41':
    from Configuration.Eras.Era_Phase2C8_timing_layer_bar_cff import Phase2C8_timing_layer_bar
    process = cms.Process('PROD',Phase2C8_timing_layer_bar)
    process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
elif geometry == 'D46':
    from Configuration.Eras.Era_Phase2C9_timing_layer_bar_cff import Phase2C9_timing_layer_bar
    process = cms.Process('PROD',Phase2C9_timing_layer_bar)
    process.load('Configuration.Geometry.GeometryExtended2026D46_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Validation.HGCalValidation.hgcSimHitStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        #'file:step1_29034.root',
        #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_11_0_0_pre7/RelValSingleMuPt10/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v1_2026D41noPU-v1/10000/EF277881-D8DF-784C-B68E-938872CB39D0.root', # not accessible
        'root://cmsxrootd.fnal.gov///store/relval/CMSSW_11_0_0_pre7/RelValSingleMuPt10/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v1_2026D41noPU-v1/10000/62AEE413-64C4-3241-AE73-466BCD8EE252.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hgcSimHit'+geometry+'tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.hgcalSimHitStudy)
