#! /usr/bin/env python

import sys
import os
import commands
import optparse

from Validation.RecoTrack.plotting.validation import Sample, Validation

### parsing input options
def parseOptions():

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    parser.add_option('', '--Obj', dest='OBJ',  type='string', default='SimHits', help='Object to run. Options are: Geometry, SimHits, Digis, RecHits, Calibrations, CaloParticles, hgcalLayerClusters')
    parser.add_option('', '--html-validation-name', dest='HTMLVALNAME', type='string', default='', help='Could be either be hgcalLayerClusters or hgcalMultiClusters')
    parser.add_option('-d', '--download', action='store_true', dest='DOWNLOAD', default=False, help='Download DQM files from RelVals')
    parser.add_option('-y', '--dry-run', action='store_true', dest='DRYRUN', default=False, help='perform a dry run (no jobs are lauched).')

    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

parseOptions()

### processing the external os commands
def processCmd(cmd, quite = 0):
    print cmd
    status, output = commands.getstatusoutput(cmd)
    if (status !=0 and not quite):
        print 'Error in processing command:\n   ['+cmd+']'
        print 'Output:\n   ['+output+'] \n'
    return output

### putype
def putype(t):
    if "_pmx" in NewRelease:
        if "_pmx" in RefRelease:
            return {"default": "pmx"+t}
        return {"default": t, NewRelease: "pmx"+t}
    return t

### Reference release
RefRelease='CMSSW_11_0_0_pre12' #None#'CMSSW_10_6_0_patch2 

### Relval release 
NewRelease='CMSSW_11_0_0_pre13'

#phase2samples_test = [
#    Sample("RelValQCDPt20toInf", midfix="MuEnrichPt15_14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU")
#]

#Some closeby relvals crashed for CMSSW_11_0_0_pre13_phase2 campaign

phase2samples_noPU = [
#    Sample("RelValCloseByParticleGun_CE_H_Fine_300um", dqmVersion="0002", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
#    Sample("RelValCloseByParticleGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValCloseByParticleGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag="_2026D49noPU")
    ]
'''
    Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuPt10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuPt100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuPt1000", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuFlatPt2To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuFlatPt0p7To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleTauFlatPt2To150", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValQCD_Pt20toInfMuEnrichPt15", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValQCD_Pt15To7000_Flat", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZTT", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZEE", midfix="14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValB0ToKstarMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToEleEle", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToJpsiGamma", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToJpsiPhi_mumuKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToPhiPhi_KKKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValDisplacedMuPt30To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValDisplacedMuPt2To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValDisplacedMuPt10To30", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTauToMuMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    #Sample("RelValMinBias", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValH125GGgluonfusion", midfix="14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValNuGun", scenario="2026D49", appendGlobalTag="_2026D49noPU")
'''


'''
phase2samples_noPU = [
    #Sample("RelValMinBias", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

    Sample("RelValElectronGunPt2To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValMuGunPt0p7To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValMuGunPt2To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValPhotonGunPt8To150", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValPiGunPt0p7To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTauGunPt2To150", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

    Sample("RelValB0ToKstarMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToEleEle", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToJpsiGamma", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToJpsiPhi_mumuKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValBsToPhiPhi_KKKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

    #Sample("RelValB0ToKstarMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    #Sample("RelValBsToEleEle", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    #Sample("RelValBsToMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    #Sample("RelValBsToJpsiGamma", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    #Sample("RelValBsToJpsiPhi_mumuKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    #Sample("RelValBsToPhiPhi_KKKK", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),

    Sample("RelValH125GGgluonfusion_14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValQCDPt20toInf", midfix="MuEnrichPt15_14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    #Sample("RelValQCDPt20toInf", midfix="MuEnrichPt15_14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    Sample("RelValQCD_Pt-15To7000_Flat", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTauToMuMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValTauGunPt2To150", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    #Sample("RelValTauToMuMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU", version="v2"),
    Sample("RelValZEE_14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZMM_14", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValZTT", midfix="14TeV", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

    Sample("RelValSingleMuPt10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuPt100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValSingleMuPt1000", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

    Sample("RelValDisplacedMuPt30To100", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValDisplacedMuPt2To10", scenario="2026D49", appendGlobalTag="_2026D49noPU"),
    Sample("RelValDisplacedMuPt10To30", scenario="2026D49", appendGlobalTag="_2026D49noPU"),

]

'''
#For faster and convenience I name PU noPU. 
#phase2samples_noPU_D41_pre11 = [
#    Sample("RelValTTbar", midfix="14TeV", scenario="2026D41", putype=putype("25ns"), punum=200, appendGlobalTag="_2026D41PU200", version="v1"),
#    Sample("RelValZMM_14", scenario="2026D41", putype=putype("25ns"), punum=200, appendGlobalTag="_2026D41PU200", version="v1")
#
#]
#D49_pre13
#phase2samples_noPU = [
#    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", putype=putype("25ns"), punum=200, appendGlobalTag="_2026D49PU200", version="v2"),
#    Sample("RelValZMM_14", scenario="2026D49", putype=putype("25ns"), punum=200, appendGlobalTag="_2026D49PU200", version="v2")
#
#]


### Reference and new repository
RefRepository = '/eos/cms/store/group/dpg_hgcal/comm_hgcal/apsallid/RelVals'
NewRepository = '/eos/cms/store/group/dpg_hgcal/comm_hgcal/apsallid/RelVals' 

#HGCal validation plots
val = Validation(
    fullsimSamples = phase2samples_noPU,#phase2samples_Survived
    fastsimSamples = [],
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository
)

if(opt.DOWNLOAD): 
    val.download()

    #Keep them in eos, save afs space. 
    if (not os.path.isdir(RefRepository+'/'+NewRelease)) :
        processCmd('mkdir -p '+RefRepository+'/'+NewRelease)

    for infi in phase2samples_noPU:
        processCmd('mv ' + infi.filename(NewRelease) + ' ' + RefRepository+'/'+NewRelease)

############################################################################################
#This is the hgcalLayerClusters part
if (opt.OBJ == 'hgcalLayerClusters' or opt.OBJ == 'hitCalibration'):
    fragments = []
    #Now  that we have them in eos lets produce plots
    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__CMSSW_10_6_0_pre4",1)[0]
        samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'

        if RefRelease == None:
            cmd = 'python Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample %s ' %(opt.HTMLVALNAME, samplename) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename)+ ' --collection %s' %(opt.HTMLVALNAME)
        else: 
            #print inputpathRef, infi.filename(RefRelease).replace("D49","D41")
            #YOU SHOULD INSPECT EACH TIME THIS COMMAND AND THE REPLACE
            cmd = 'python Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("D49","D41").replace("200-v2","200-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample %s ' %(opt.HTMLVALNAME, samplename) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME) 
            #print cmd

        if(opt.DRYRUN):
            print 'Dry-run: ['+cmd+']'
        else:
            output = processCmd(cmd)
            if opt.OBJ == 'hgcalLayerClusters':
                processCmd('awk \'NR>=6&&NR<=44\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
            if opt.OBJ == 'hitCalibration':
                processCmd('awk \'NR>=6&&NR<=15\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))

    
        fragments.append( 'HGCValid_%s_Plots/index_%s.html'% (opt.HTMLVALNAME, samplename) )
            
    
    #Let's also create the final index xml file. 
    processCmd('mv HGCValid_%s_Plots/index.html HGCValid_%s_Plots/test.html' %(opt.HTMLVALNAME,opt.HTMLVALNAME) )
    index_file = open('HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME),'w')            
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title>HGCal validation %s </title>\n' %(opt.HTMLVALNAME) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
                   
    for frag in fragments:   
        with open(frag,'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                print line
                index_file.write(line + '\n')
                #processCmd( 'cat ' + frag + ' >> HGCalValidationPlots/index.html '   )
                #index_file.write(frag)

        
    #Writing postamble"
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()
    
############################################################################################
#This is the SimHits part
if (opt.OBJ == 'SimHits'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalSimHitStudy")) :
        processCmd('mkdir -p hgcalSimHitStudy')
    #Prepare for www
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalSimHitStudy/.')

    #The input to this is for the moment 100 GeV muon from runnin cmsRun runHGCalSimHitStudy_cfg.py 
    #Input: hgcSimHits.root
    cmd = 'root.exe -b -q validationplots.C\(\\"hgcSimHit.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print 'Dry-run: ['+cmd+']'
    else:
        output = processCmd(cmd)
        processCmd('cp -r hgcalSimHitStudy /eos/user/a/apsallid/www/RelVals/' + NewRelease + '/.')

############################################################################################
if (opt.OBJ == 'hitValidation'):
    fragments = []
    #Now  that we have them in eos lets produce plots
    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__CMSSW_10_6_0_pre4",1)[0]
        samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'

        if RefRelease == None:
            cmd = 'python Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample %s ' %(opt.HTMLVALNAME, samplename) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename)+ ' --collection %s' %(opt.HTMLVALNAME)
        else: 
            cmd = 'python Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("D49","D41").replace("200-v2","200-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample %s ' %(opt.HTMLVALNAME, samplename) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME) 
 

        if(opt.DRYRUN):
            print 'Dry-run: ['+cmd+']'
        else:
            output = processCmd(cmd)
            processCmd('awk \'NR>=6&&NR<=28\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
    
        fragments.append( 'HGCValid_%s_Plots/index_%s.html'% (opt.HTMLVALNAME, samplename) )
            
    
    #Let's also create the final index xml file. 
    processCmd('mv HGCValid_%s_Plots/index.html HGCValid_%s_Plots/test.html' %(opt.HTMLVALNAME,opt.HTMLVALNAME) )
    index_file = open('HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME),'w')            
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title>HGCal validation %s </title>\n' %(opt.HTMLVALNAME) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
                   
    for frag in fragments:   
        with open(frag,'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                print line
                index_file.write(line + '\n')
                #processCmd( 'cat ' + frag + ' >> HGCalValidationPlots/index.html '   )
                #index_file.write(frag)

        
    #Writing postamble"
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()


############################################################################################
#This is the Digis part
if (opt.OBJ == 'Digis'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalDigiStudy")) :
        processCmd('mkdir -p hgcalDigiStudy')
        processCmd('mkdir -p hgcalDigiStudyEE')
        processCmd('mkdir -p hgcalDigiStudyHEF')
        processCmd('mkdir -p hgcalDigiStudyHEB')
    #Prepare for www
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalDigiStudy/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalDigiStudyEE/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalDigiStudyHEF/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalDigiStudyHEB/.')
   #The input here is from running cmsRun runHGCalDigiStudy_cfg.py, to which 
    #we usually give ttbar noPU as input 
    #Input: hgcDigi.root
    cmd = 'root.exe -b -q validationplots.C\(\\"hgcDigi.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print 'Dry-run: ['+cmd+']'
    else:
        output = processCmd(cmd)
        #mv the output under the main directory
        processCmd('mv hgcalDigiStudyEE hgcalDigiStudy/.')
        processCmd('mv hgcalDigiStudyHEF hgcalDigiStudy/.')
        processCmd('mv hgcalDigiStudyHEB hgcalDigiStudy/.')
        processCmd('cp -r hgcalDigiStudy /eos/user/a/apsallid/www/RelVals/' + NewRelease + '/.')

############################################################################################
#This is the RecHits part
if (opt.OBJ == 'RecHits'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalRecHitStudy")) :
        processCmd('mkdir -p hgcalRecHitStudy')
        processCmd('mkdir -p hgcalRecHitStudyEE')
        processCmd('mkdir -p hgcalRecHitStudyHEF')
        processCmd('mkdir -p hgcalRecHitStudyHEB')
    #Prepare for www
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalRecHitStudy/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalRecHitStudyEE/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalRecHitStudyHEF/.')
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php hgcalRecHitStudyHEB/.')
    #The input here is from running cmsRun runHGCalRecHitStudy_cfg.py, to which 
    #we usually give ttbar noPU as input 
    #Input: hgcRecHit.root
    cmd = 'root.exe -b -q validationplots.C\(\\"hgcRecHit.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print 'Dry-run: ['+cmd+']'
    else:
        output = processCmd(cmd)
        #mv the output under the main directory
        processCmd('mv hgcalRecHitStudyEE hgcalRecHitStudy/.')
        processCmd('mv hgcalRecHitStudyHEF hgcalRecHitStudy/.')
        processCmd('mv hgcalRecHitStudyHEB hgcalRecHitStudy/.')
        processCmd('cp -r hgcalRecHitStudy /eos/user/a/apsallid/www/RelVals/' + NewRelease + '/.')

############################################################################################
#This is the RecHits part
if (opt.OBJ == 'Calibrations'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("Calibrations")) :
        processCmd('mkdir -p Calibrations')
    #Prepare for www
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php Calibrations/.')

    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__"+NewRelease,1)[0]
        samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)
        if (not os.path.isdir(samplename)) :
            processCmd('mkdir -p ' + samplename)
            #Prepare for www
            processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php ' + samplename + '/.')

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'
        cmd = 'root.exe -b -q validationplots.C\(\\"'+ inputpathNew + infi.filename(NewRelease) +  '\\",\\"'+ opt.OBJ + '\\",\\"'+ samplename + '\\"\\)'
        if(opt.DRYRUN):
            print 'Dry-run: ['+cmd+']'
        else:
            output = processCmd(cmd)
            #mv the output under the main directory
            processCmd('mv ' +samplename+ ' Calibrations/.' )
    processCmd('cp -r Calibrations /eos/user/a/apsallid/www/RelVals/' + NewRelease + '/.')


############################################################################################
## TODO #This is the CaloParticles part
if (opt.OBJ == 'CaloParticles'):
    particletypes = ["-11","-13","-211","-321","11","111","13","211","22","321"]
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("CaloParticles")) :
        processCmd('mkdir -p CaloParticles')
    #Prepare for www
    processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php CaloParticles/.')

    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__"+NewRelease,1)[0]
        samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)
        if (not os.path.isdir(samplename)) :
            processCmd('mkdir -p ' + samplename )
            processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php ' + samplename + '/.')
            for part in particletypes: 
                processCmd('mkdir -p ' + samplename + '/' +part )
                #Prepare for www
                processCmd('cp /eos/user/a/apsallid/www/RelVals/index.php ' + samplename + '/' +part + '/.')

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'
        cmd = 'root.exe -b -q validationplots.C\(\\"'+ inputpathNew + infi.filename(NewRelease) +  '\\",\\"'+ opt.OBJ + '\\",\\"'+ samplename + '\\"\\)'
        if(opt.DRYRUN):
            print 'Dry-run: ['+cmd+']'
        else:
            output = processCmd(cmd)
            processCmd('mv ' +samplename+ ' CaloParticles/.' )
    processCmd('cp -r CaloParticles /eos/user/a/apsallid/www/RelVals/' + NewRelease + '/.')
