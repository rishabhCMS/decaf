#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from coffea import hist, nanoevents, util
from coffea.util import load, save
import coffea.processor as processor
import awkward as ak
import numpy as np
import glob as glob
import re
import itertools
# import vector as vec
from coffea.nanoevents.methods import vector, candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea.lumi_tools import LumiMask
# for applying JECs
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, JetCorrectionUncertainty, CorrectedMETFactory
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor
# from jmeCorrections import JetMetCorrections
import json


import coffea.processor as processor
from coffea import hist
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False
from optparse import OptionParser
import pickle
np.errstate(invalid='ignore', divide='ignore')

def get_data_era(events, year):
    runNMax = max(events.run)
    runNMin = min(events.run)
    if "2018" in year:

        if (runNMin >=315252) and (runNMax <=316995):
            era = "A"

        elif (runNMin >=316998) and (runNMax <=319312):
            era = "B"

        elif (runNMin >=319313) and (runNMax <=320393):
            era = "C"

        elif (runNMin >=320394) and (runNMax <=325273):
            era = "D"
       
    elif "2017" in year:
        if (runNMin >=297020) and (runNMax <=299329):
            era = "B"

        elif (runNMin >=299337) and (runNMax <=302029):
            era = "C"

        elif (runNMin >=302030) and (runNMax <=303434):
            era = "D"

        elif (runNMin >=303435) and (runNMax <=304826):
            era = "E"

        elif (runNMin >=304911) and (runNMax <=306462):
            era = "F"

        else: era = "F"
    if "2016" in year:

        if (runNMin >=272007) and (runNMax <=275376):
            era = "BCD"

        elif (runNMin >=275656) and (runNMax <=276283):
            era = "BCD"
#                         #print(era)
        elif (runNMin >=276315) and (runNMax <=276811):
            era = "BCD"

        elif (runNMin >=276831) and (runNMax <=277420) :
            era = "EF"

        elif (runNMin >=277772 and runNMax <=278807) or (runNMin in [278770, 278806, 278807]) or (runNMax in [278770, 278806, 278807] ):
            era = "EF"
        elif (runNMin >=278801 and runNMax <=278808) or (runNMax in [278769, 278806, 278807]) or (runNMax in [278769,278806, 278807]):
            era = "EF"

        elif (runNMin >=278820) and (runNMax <=280385):
            era = "G"
        elif (runNMin >=281613) and (runNMax <=284044):
            era = "H"
    return(era)
def METXYCorr_Met_MetPhi(events, year, metpt, metphi):


    #https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18.h
    if 'genWeight' in events.fields:
        isData = False
    else: isData = True
    dataset=events.metadata['dataset'] 
#     dataset = 'WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8____1_'

    originalMet = metpt
    originalMet_phi = metpt

    npv  = events.PV.npvsGood
    mask = np.asarray(npv>100)
    npv = np.asarray(npv)
#     #print(npv)
    npv[mask==True] = 100


    if not isData:
        if (year == '2016') and ("preVFP" in dataset):
            METxcorr = -(-0.188743*npv +0.136539)
            METycorr = -(0.0127927*npv +0.117747)

        if (year == '2016') and ("postVFP" in dataset):
            METxcorr = -(-0.153497*npv -0.231751)
            METycorr = -(0.00731978*npv +0.243323)

        elif year == '2017':
            METxcorr = -(-0.300155*npv +1.90608)
            METycorr = -(0.300213*npv +-2.02232)
        elif year == '2018':
            #print("hi I am in MC metxycorr")
            #print(year)
            #print(-(0.183518*npv +0.546754))
            METxcorr = -(0.183518*npv +0.546754)
            METycorr = -(0.192263*npv +-0.42121)

    if isData:

#                 #print(runNMax, runNMin)
        if "2018" in year:
            runNMax = max(events.run)
            runNMin = min(events.run)
            if (runNMin >=315252) and (runNMax <=316995):
                era = "A"

            elif (runNMin >=316998) and (runNMax <=319312):
                era = "B"

            elif (runNMin >=319313) and (runNMax <=320393):
                era = "C"

            elif (runNMin >=320394) and (runNMax <=325273):
                era = "D"
            if "A" in era:
                METxcorr = -(0.263733*npv +-1.91115)
                METycorr = -(0.0431304*npv +-0.112043)
            elif "B" in era:
                METxcorr = -(0.400466*npv +-3.05914)
                METycorr = -(0.146125*npv +-0.533233)
            elif "C" in era:
                METxcorr = -(0.430911*npv +-1.42865)
                METycorr = -(0.0620083*npv +-1.46021)
            elif "D" in era:
                METxcorr = -(0.457327*npv +-1.56856)
                METycorr = -(0.0684071*npv +-0.928372)


        if "2017" in year:

            runNMax = max(events.run)
            runNMin = min(events.run)
            if (runNMin >=297020) and (runNMax <=299329):
                era = "B"

            elif (runNMin >=299337) and (runNMax <=302029):
                era = "C"

            elif (runNMin >=302030) and (runNMax <=303434):
                era = "D"

            elif (runNMin >=303435) and (runNMax <=304826):
                era = "E"

            elif (runNMin >=304911) and (runNMax <=306462):
                era = "F"

            else: era = "X"
            if "E" in era:
                METxcorr = -(-0.162472*npv +0.176329)
                METycorr = -(0.138076*npv +-0.250239)
            elif "B" in era:
                METxcorr = -(-0.211161*npv +0.419333)
                METycorr = -(0.251789*npv +-1.28089)
            elif "C" in era:
                METxcorr = -(-0.185184*npv +-0.164009)
                METycorr = -(0.200941*npv +-0.56853)
            elif "D" in era:
                METxcorr = -(-0.201606*npv +0.426502)
                METycorr = -(0.188208*npv +-0.58313)
            elif "F" in era:
                METxcorr = -(-0.210639*npv +0.72934)
                METycorr = -(0.198626*npv +1.028)
            elif "X" in era:
                METxcorr = np.zeros(len(events))
                METycorr = np.zeros(len(events))
        if "2016" in year:
            runNMax = max(events.run)
            runNMin = min(events.run)

            if (runNMin >=272007) and (runNMax <=275376):
                era = "B"
                #print(era)
            elif (runNMin >=275656) and (runNMax <=276283):
                era = "C"
                #print('runNMax', runNMax, 'runMin', runNMin, ' era:', era)
#                         #print(era)
            elif (runNMin >=276315) and (runNMax <=276811):
                era = "D"
                #print(era)
            elif (runNMin >=276831) and (runNMax <=277420) :
                era = "E"
                #print(era)
            elif (runNMin >=277772 and runNMax <=278807) or (runNMin in [278770, 278806, 278807]) or (runNMax in [278770, 278806, 278807] ):
                era = "F"
#                  https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVDatasetsUL2016
                #print(era)
            elif (runNMin >=278801 and runNMax <=278808) or (runNMax in [278769, 278806, 278807]) or (runNMax in [278769,278806, 278807]):
                era = "F_late"
                #print(era)
            elif (runNMin >=278820) and (runNMax <=280385):
                era = "G"
                #print(era)
            elif (runNMin >=281613) and (runNMax <=284044):
                era = "H"
                #print(era)

            if "B" in era:
                METxcorr = -(-0.0214894*npv +-0.188255)
                METycorr = -(0.0876624*npv +0.812885)
            elif "C" in era:
                METxcorr = -(-0.032209*npv +0.067288)
                METycorr = -(0.113917*npv +0.743906)
            elif "D" in era:
                METxcorr = -(-0.0293663*npv +0.21106)
                METycorr = -(0.11331*npv +0.815787)
            elif "E" in era:
                METxcorr = -(-0.0132046*npv +0.20073)
                METycorr = -(0.134809*npv +0.679068)
            elif "F" in era:
                METxcorr = -(-0.0543566*npv +0.816597)
                METycorr = -(0.114225*npv +1.17266)
            elif "F_late" in era:
                METxcorr = -(0.134616*npv +-0.89965)
                METycorr = -(0.0397736*npv +1.0385)
            elif "G" in era:
                METxcorr = -(0.121809*npv +-0.584893)
                METycorr = -(0.0558974*npv +0.891234)
            elif "H" in era:
                METxcorr = -(0.0868828*npv +-0.703489)
                METycorr = -(0.0888774*npv +0.902632)

    CorrectedMET_x = np.multiply(originalMet,np.cos(originalMet_phi))+ (np.array(METxcorr))
    CorrectedMET_y = np.multiply(originalMet,np.sin(originalMet_phi))+ (np.array(METycorr))
#             #print(CorrectedMET_x/CorrectedMET_y)
#   double CorrectedMET_x = uncormet *cos( uncormet_phi)+METxcorr;
#   double CorrectedMET_y = uncormet *sin( uncormet_phi)+METycorr;
    CorrectedMET = np.sqrt((CorrectedMET_x)**2+(CorrectedMET_y)**2)
    CorrectedMETPhi = np.zeros(len(events))



    
    
    mask1 = (CorrectedMET_x ==0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask1==True] = np.pi

    mask2 = (CorrectedMET_x ==0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask2==True] = -np.pi

    mask3 = (CorrectedMET_x <0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask3==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) + np.pi )[mask3==True] 

    mask4 = (CorrectedMET_x <0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask4==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) - np.pi )[mask4==True]

    mask5 = (CorrectedMET_x >0 )
    CorrectedMETPhi[mask5==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x))[mask5==True]   

    mask6 = (~mask1 & ~mask2 & ~mask3 & ~mask4 & ~mask5) 
    CorrectedMETPhi[mask6==True] = 0

    return (CorrectedMET, CorrectedMETPhi)

class AnalysisProcessor(processor.ProcessorABC):
    
    
    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
        '2016': {'preVFP':19.52,'postVFP':16.81},
                 #postVFP
        '2017': 40.66,
        '2018': 59.74
    }

    
    met_filter_flags = {

        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
#                  'BadPFMuonDzFilter',
                 'eeBadScFilter',
                 
                 ],

        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter',
                 'eeBadScFilter',
#                  'BadPFMuonDzFilter',
#                  'hfNoisyHitsFilter'
                 ],

        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter',
                 'eeBadScFilter',
                 'BadPFMuonDzFilter',
#                  'hfNoisyHitsFilter'
                 ]
    }

    def __init__(self, year, xsec, corrections, ids, common, vfp):
        ak.behavior.update(nanoaod.behavior)
        self._year = year
        if self._year == "2016":
            self._lumi = 1000.*float(AnalysisProcessor.lumis[year][vfp])
        else:   
            self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec
        
        self._samples = {
            'sre':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD', 'SingleElectron','EGamma',
                   'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
            'srm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon',
                    'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
            'ttbare':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma',
                     'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
            'ttbarm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon',
                     'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
            'wjete':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD', 'SingleElectron','EGamma',
                    'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
            'wjetm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon',
                    'Mphi-1995_Mchi-1000',
                    'Mphi-2000_Mchi-1500',
                    'Mphi-2000_Mchi-500'),
        }
        
        self._singleelectron_triggers = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
#                 'Ele105_CaloIdVT_GsfTrkIdT'
                'Ele115_CaloIdVT_GsfTrkIdT'
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200','Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
            ]
        }
        
        self._singlemuon_triggers = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24',
                'Mu50',
                'TkMu50'

            ],
            '2017':
                [
                'IsoMu27',
                'Mu50',
                'OldMu100',
                'TkMu100'
            ],
            '2018':
                [
                'IsoMu24',
                'Mu50',
                'OldMu100',
                'TkMu100'
            ]
        }
        

        self._jec = {

            '2016':  
                    {
                        'no_apv':
                        ['Summer19UL16_V7_MC_L1FastJet_AK4PFchs',
                        'Summer19UL16_V7_MC_L2L3Residual_AK4PFchs',
                        'Summer19UL16_V7_MC_L2Relative_AK4PFchs',
                        'Summer19UL16_V7_MC_L2Residual_AK4PFchs',
                        'Summer19UL16_V7_MC_L3Absolute_AK4PFchs',
#                         'Summer19UL16_RunFGH_V7_DATA_L1FastJet_AK4PFchs',
#                         'Summer19UL16_RunFGH_V7_DATA_L2L3Residual_AK4PFchs',
#                         'Summer19UL16_RunFGH_V7_DATA_L2Relative_AK4PFchs',
#                         'Summer19UL16_RunFGH_V7_DATA_L3Absolute_AK4PFchs'
                        ],
                    
                        'apv':
                        ['Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs',
                        'Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs',
                        'Summer19UL16APV_V7_MC_L2Relative_AK4PFchs',
                        'Summer19UL16APV_V7_MC_L2Residual_AK4PFchs',
                        'Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs',
#                         'Summer19UL16APV_RunBCD_V7_DATA_L1FastJet_AK4PFchs',
#                         'Summer19UL16APV_RunEF_V7_DATA_L1FastJet_AK4PFchs',
#                         'Summer19UL16APV_RunBCD_V7_DATA_L2L3Residual_AK4PFchs',
#                         'Summer19UL16APV_RunEF_V7_DATA_L2L3Residual_AK4PFchs',
#                         'Summer19UL16APV_RunBCD_V7_DATA_L2Relative_AK4PFchs',
#                         'Summer19UL16APV_RunEF_V7_DATA_L2Relative_AK4PFchs',
#                         'Summer19UL16APV_RunBCD_V7_DATA_L3Absolute_AK4PFchs',
#                         'Summer19UL16APV_RunEF_V7_DATA_L3Absolute_AK4PFchs',
                        ]
                    },


            '2017': [
                'Summer19UL17_V5_MC_L1FastJet_AK4PFchs',
                'Summer19UL17_V5_MC_L2L3Residual_AK4PFchs',
                'Summer19UL17_V5_MC_L2Relative_AK4PFchs',
                'Summer19UL17_V5_MC_L2Residual_AK4PFchs',
                'Summer19UL17_V5_MC_L3Absolute_AK4PFchs',
#                 'Summer19UL17_RunB_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL17_RunC_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL17_RunD_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL17_RunE_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL17_RunF_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL17_RunE_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL17_RunF_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL17_RunB_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL17_RunC_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL17_RunD_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL17_RunE_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_RunF_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_RunB_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_RunC_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_RunD_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_RunF_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL17_RunE_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL17_RunB_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL17_RunC_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL17_RunD_V5_DATA_L3Absolute_AK4PFchs',
                
            ],

            '2018': [
                'Summer19UL18_V5_MC_L1FastJet_AK4PFchs',
                'Summer19UL18_V5_MC_L2Relative_AK4PFchs',
                'Summer19UL18_V5_MC_L2L3Residual_AK4PFchs',
                'Summer19UL18_V5_MC_L3Absolute_AK4PFchs',
                
#                 'Summer19UL18_RunA_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL18_RunB_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL18_RunC_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL18_RunD_V5_DATA_L1FastJet_AK4PFchs',
#                 'Summer19UL18_RunA_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL18_RunB_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL18_RunC_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL18_RunD_V5_DATA_L2Relative_AK4PFchs',
#                 'Summer19UL18_RunA_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL18_RunB_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL18_RunC_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL18_RunD_V5_DATA_L2L3Residual_AK4PFchs',
#                 'Summer19UL18_RunA_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL18_RunB_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL18_RunC_V5_DATA_L3Absolute_AK4PFchs',
#                 'Summer19UL18_RunD_V5_DATA_L3Absolute_AK4PFchs',
            ]
        }
        
        self._junc = {

            '2016': 
            {
            'no_apv': ['Summer19UL16_V7_MC_Uncertainty_AK4PFchs',
#                       'Summer19UL16_RunFGH_V7_DATA_Uncertainty_AK4PFchs',
                      ],
                     'apv': ['Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs',
#                             'Summer19UL16APV_RunBCD_V7_DATA_Uncertainty_AK4PFchs',
#                             'Summer19UL16APV_RunEF_V7_DATA_Uncertainty_AK4PFchs'
                            ]},
#                 'Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs',
            

            '2017': [
                'Summer19UL17_V5_MC_Uncertainty_AK4PFchs',
#                 'Summer19UL17_RunB_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL17_RunC_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL17_RunD_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL17_RunE_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL17_RunF_V5_DATA_Uncertainty_AK4PFchs'
#                 'Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFchs'
            ],

            '2018': [
                'Summer19UL18_V5_MC_Uncertainty_AK4PFchs',
#                 'Summer19UL18_RunA_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL18_RunB_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL18_RunC_V5_DATA_Uncertainty_AK4PFchs',
#                 'Summer19UL18_RunD_V5_DATA_Uncertainty_AK4PFchs',
                
#                 'Autumn18_V19_MC_Uncertainty_AK4PFchs'
            ],
        }
        
        self._jr = {

            '2016': 
            {'apv': ['Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs',
#                      'Summer20UL16APV_JRV3_MC_EtaResolution_AK4PFchs',
#                      'Summer20UL16APV_JRV3_MC_PhiResolution_AK4PFchs',
#                     'Summer20UL16APV_JRV3_DATA_PtResolution_AK4PFchs',
#                     'Summer20UL16APV_JRV3_DATA_EtaResolution_AK4PFchs',
#                      'Summer20UL16APV_JRV3_DATA_PhiResolution_AK4PFchs',
                    ],
                     'no_apv': ['Summer20UL16_JRV3_MC_PtResolution_AK4PFchs',
#                                 'Summer20UL16_JRV3_MC_EtaResolution_AK4PFchs',
#                                 'Summer20UL16_JRV3_MC_PhiResolution_AK4PFchs',
#                                'Summer20UL16_JRV3_DATA_PtResolution_AK4PFchs',
#                                'Summer20UL16_JRV3_DATA_EtaResolution_AK4PFchs',
#                                 'Summer20UL16_JRV3_DATA_PhiResolution_AK4PFchs',
                               ]},
#                 'Summer16_25nsV1b_MC_PtResolution_AK4PFchs',


            '2017': [
                'Summer19UL17_JRV2_MC_PtResolution_AK4PFchs',
#                 'Summer19UL17_JRV2_MC_EtaResolution_AK4PFchs',
#                 'Summer19UL17_JRV2_MC_PhiResolution_AK4PFchs',
#                 'Summer19UL17_JRV2_DATA_PtResolution_AK4PFchs',
#                 'Summer19UL17_JRV2_DATA_EtaResolution_AK4PFchs',
#                 'Summer19UL17_JRV2_DATA_PhiResolution_AK4PFchs',
#                 'Fall17_V3b_MC_PtResolution_AK4PFchs'
            ],

            '2018': [
                'Summer19UL18_JRV2_MC_PtResolution_AK4PFchs',
#                 'Summer19UL18_JRV2_MC_EtaResolution_AK4PFchs',
#                 'Summer19UL18_JRV2_MC_PhiResolution_AK4PFchs',
#                 'Summer19UL18_JRV2_DATA_PtResolution_AK4PFchs',
#                 'Summer19UL18_JRV2_DATA_EtaResolution_AK4PFchs',
#                 'Summer19UL18_JRV2_DATA_PhiResolution_AK4PFchs',
#                 'Autumn18_V7b_MC_PtResolution_AK4PFchs'
            ]
        }
        
        self._jersf = {

            '2016': 
            {'no_apv':['Summer20UL16_JRV3_MC_SF_AK4PFchs',
#                       'Summer20UL16_JRV3_DATA_SF_AK4PFchs'
                      ],
                     'apv':['Summer20UL16APV_JRV3_MC_SF_AK4PFchs',
#                            'Summer20UL16APV_JRV3_DATA_SF_AK4PFchs'
                           ]},
#                 'Summer16_25nsV1b_MC_SF_AK4PFchs',
            

            '2017': [
                'Summer19UL17_JRV2_MC_SF_AK4PFchs',
#                 'Summer19UL17_JRV2_DATA_SF_AK4PFchs'
#                 'Fall17_V3b_MC_SF_AK4PFchs'
            ],

            '2018': [
                'Summer19UL18_JRV2_MC_SF_AK4PFchs',
#                 'Summer19UL18_JRV2_DATA_SF_AK4PFchs'
#                 'Autumn18_V7b_MC_SF_AK4PFchs'
            ]
        }
        
        self._golden_json_path = {
        '2016': 'data/golden_txt/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
        '2017': 'data/golden_txt/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
        '2018': 'data/golden_txt/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}
        self._corrections = corrections
        self._ids = ids
        self._common = common

        self._accumulator = processor.dict_accumulator({
            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])),
            
            'template': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('npvs', 'NPVS', 99, 0, 99),
                hist.Bin('mT_corrected', '$m_{T}$ [GeV] Corrected', [100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500])),
            
             'mT_corrected': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('mT_corrected', '$m_{T} XY \Phi Corrected$ [GeV]', [100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500])),
#             "EventCount": processor.value_accumulator(int),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
#                 hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])),
            'eT_miss_corrected': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('eT_miss_corrected', '$E^T_{miss} Corrected$[GeV]', 20, 0, 600)),
            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),
            'leading_mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('leading_mu_pT', 'Leading Muon $p_{T}$ [GeV]', 10, 0, 200)),
            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('mu_pT', 'Tight Muon $p_{T}$ [GeV]', 10, 0, 200)),
            'j1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('j1pt','AK4 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'j1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('j1eta','AK4 Leading Jet Eta',35,-3.5,3.5)),
            
            'j1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('j1phi','AK4 Leading Jet Phi',35,-3.5,3.5)),

            'dphi_e_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('dphi_e_etmiss', '$\Delta\phi (e, E^T_{miss} )$', 30, 0, 3.5)),
            
            'dphi_mu_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('dphi_mu_etmiss','$\Delta\phi (\mu, E^T_{miss} )$', 30, 0, 3.5)),
            
            'ndflvL': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ndflvL', 'AK4 Number of deepFlavor Loose Jets', 6, -0.5, 5.5)),
            'ndflvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ndflvM', 'AK4 Number of deepFlavor Medium Jets', 6, -0.5, 5.5)),
            'njets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('njets', 'AK4 Number of Jets', 7, -0.5, 6.5)),

            'ndcsvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ndcsvM', 'AK4 Number of deepCSV Medium Jets', 6, -0.5, 5.5)),
            
            'dphi_Corr_Met_LJ': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('dphi_Corr_Met_LJ', '$\Delta \Phi (E^T_{miss corr}, Leading Jet)$', 30, 0, 3.5)),
            'dr_e_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('dr_e_lj', '$\Delta r (Leading e, Leading Jet)$', 30, 0, 5.0)),
            'dr_mu_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('dr_mu_lj', '$\Delta r (Leading \mu, Leading Jet)$', 30, 0, 5.0)),
            'ele_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ele_eta', 'Leading Electron Eta', 48, -2.4, 2.4)),
            'mu_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('mu_eta', 'Leading Muon Eta', 48, -2.4, 2.4)),
            'ele_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ele_phi', 'Leading Electron Phi', 64, -3.2, 3.2)),
            
            'ele_phi_loose': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ele_phi_loose', 'Loose Electron Phi', 64, -3.2, 3.2)),

            'eT_miss_corrected_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('eT_miss_corrected_phi','MET phi Corrected',35,-3.5,3.5)),
            'mu_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('mu_phi', 'Leading Muon Phi', 64, -3.2, 3.2)),
            'npvs': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('npvs', 'NPVS', 99, 0, 99)),
 
        })

    @property
    def accumulator(self):
        return self._accumulator



    def process(self, events):
        ## Here we pre-compute a few common variables that we will re-use later and add them to the events object

        # Temporary patch so we can add photon and lepton four vectors. Not needed for newer versions of NanoAOD
        events["Photon", "charge"] = 0
        # Calculate charged hadron isolation for photons
        events["Photon", "chIso"] = (events.Photon.pfRelIso03_chg) * (events.Photon.pt)
        self.isMC ='genWeight' in events.fields
        if self.isMC:
            output = self.accumulator.identity()
            shift_systs = [None,]
#             shift_systs = [None, "JESUp", "JESDown", "JERUp", "JERDown", "METunclUp", "METunclDown"]
            for _syst in shift_systs:
                output += self.process_shift(events, _syst)
        else:
            # data doesn't need each systematic shift to be processed
            output = self.process_shift(events, "nominal")

        # only add events up once
#         output["EventCount"] += len(events)
        return output
            
    def process_shift(self, events, shift_syst=None):
            
        dataset = events.metadata['dataset']
        isData = 'genWeight' not in events.fields
        isMC = 'genWeight' in events.fields
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()
            
        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)
            
        ###
        # Getting corrections, ids from .coffea files
        ###
        if ("preVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_preVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_preVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_preVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_preVFP_below20'][self._year]
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_preVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            get_mu_loose_err_sf = self._corrections['get_mu_loose_id_err_preVFP'][self._year]            
            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_preVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_preVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_preVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_preVFP'][self._year]            

            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_err_preVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_preVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_preVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_preVFP']['deepflav'][self._year]
            
            get_pu_nom_weight = self._corrections['get_pu_nom_weight_preVFP'][self._year]
            get_pu_up_weight = self._corrections['get_pu_up_weight_preVFP'][self._year]
            get_pu_down_weight = self._corrections['get_pu_down_weight_preVFP'][self._year]
        elif ("postVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_err_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]
            get_pu_nom_weight = self._corrections['get_pu_nom_weight_postVFP'][self._year]
            get_pu_up_weight = self._corrections['get_pu_up_weight_postVFP'][self._year]
            get_pu_down_weight = self._corrections['get_pu_down_weight_postVFP'][self._year]
        else:
#             #print("hi")
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_err_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]
            get_pu_nom_weight = self._corrections['get_pu_nom_weight_postVFP'][self._year]
            get_pu_up_weight = self._corrections['get_pu_up_weight_postVFP'][self._year]
            get_pu_down_weight = self._corrections['get_pu_down_weight_postVFP'][self._year]
            
            
        get_ttbar_weight = self._corrections['get_ttbar_weight']
        get_nnlo_nlo_weight = self._corrections['get_nnlo_nlo_weight'][self._year]
        get_nlo_qcd_weight = self._corrections['get_nlo_qcd_weight'][self._year]
        get_nlo_ewk_weight = self._corrections['get_nlo_ewk_weight'][self._year]
#         get_pu_weight = self._corrections['get_pu_weight'][self._year]
        

        get_ele_trig_weight = self._corrections['get_ele_trig_weight'][self._year]
        get_ele_trig_err    = self._corrections['get_ele_trig_err'][self._year]
        get_pho_tight_id_sf = self._corrections['get_pho_tight_id_sf'][self._year]
        get_pho_csev_sf = self._corrections['get_pho_csev_sf'][self._year]
        get_ecal_bad_calib = self._corrections['get_ecal_bad_calib']

        Jetevaluator = self._corrections['Jetevaluator']
            
        isLooseElectron = self._ids['isLooseElectron']
        isTightElectron = self._ids['isTightElectron']
        isLooseMuon = self._ids['isLooseMuon']
        isTightMuon = self._ids['isTightMuon']
        isLooseTau = self._ids['isLooseTau']
        isLoosePhoton = self._ids['isLoosePhoton']
        isTightPhoton = self._ids['isTightPhoton']
        isGoodJet = self._ids['isGoodJet']
        isHEMJet = self._ids['isHEMJet']

        match = self._common['match']
        # to calculate photon trigger efficiency
        sigmoid = self._common['sigmoid']
        deepflavWPs = self._common['btagWPs']['deepflav'][self._year]
        deepcsvWPs = self._common['btagWPs']['deepcsv'][self._year]
            
        ###
        # Derive jet corrector for JEC/JER
        ### Not taking JEC because they have already been applied
        jec_names = []
        j = events.Jet
        met = events.MET
        if isMC:

            if (self._year == '2016'):
                if ("preVFP" in dataset):
    #                 #print(self._jr[self._year]['apv'][0])
                    list_of_jec = self._jec[self._year]['apv']
                    list_of_jr = self._jr[self._year]['apv']
                    list_of_junc = self._junc[self._year]['apv']
                    list_of_jersf = self._jersf[self._year]['apv']
                else: 
                    list_of_jec = self._jec[self._year]['no_apv']
                    list_of_jr =self ._jr[self._year]['no_apv']
                    list_of_junc =self ._junc[self._year]['no_apv']
                    list_of_jersf = self._jersf[self._year]['no_apv']

                for name in list_of_jec:
                    jec_names.append(name)
                for name in list_of_jr:
                    jec_names.append(name)
                for name in list_of_junc:
                    jec_names.append(name)    
                for name in list_of_jersf:
                    jec_names.append(name)                
            else:
                for name in self._jec[self._year]:
                    jec_names.append(name)                 
                for name in self._jr[self._year]:
                    jec_names.append(name)    
                for name in self._jersf[self._year]:
                    jec_names.append(name)
                for name in self._junc[self._year]:
                    jec_names.append(name)
            #print("afer____")
            #print(jec_names)
            jec_inputs = {name: Jetevaluator[name] for name in jec_names}
            jec_stack = JECStack(jec_inputs)

            jec_name_map = jec_stack.blank_name_map
            jec_name_map = {
                'JetPt': 'pt',
                'JetMass': 'mass',
                'JetEta': 'eta',
                'JetA': 'area',
                'ptGenJet': 'pt_gen',
                'ptRaw': 'pt_raw',
                'massRaw': 'mass_raw',
                'Rho': 'rho',
                'METpt': 'pt',
                'METphi': 'phi',
                'JetPhi': 'phi',
                'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
                'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
            }

            jet_factory = CorrectedJetsFactory(jec_name_map, jec_stack)

            j["pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
            j["mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
            j["pt_gen"] = ak.values_astype(
                ak.fill_none(events.Jet.matched_gen.pt, 0), np.float32
            )
            j["rho"] = ak.broadcast_arrays( events.fixedGridRhoFastjetAll, events.Jet.pt)[0]

            j["pt"] = j.pt_raw
            j["mass"] = j.mass_raw
    #         j["p4","pt"] = j.pt_raw
    #         j["p4","mass"] = j.mass_raw
            events_cache = events.caches[0]
            met_factory = CorrectedMETFactory(jec_name_map)
            corrected_jets = jet_factory.build(j, lazy_cache=events_cache)
            corrected_mets = met_factory.build(events.MET, corrected_jets, {})



            if shift_syst == "JESUp":
                j = corrected_jets['JES_jes'].up
    #                 #print(shift_syst)

            elif shift_syst == "JESDown":
                j = corrected_jets['JES_jes'].down
    #                 #print(shift_syst)

            elif shift_syst == "JERUp":
                j = corrected_jets['JER'].up
    #                 #print(shift_syst)

            elif shift_syst == "JERDown":
                j = corrected_jets['JER'].down
    #                 #print(shift_syst)

            elif shift_syst == "METunclUp":
                met = corrected_mets.MET_UnclusteredEnergy.up
                j   = events.Jet

            elif shift_syst == "METunclDown":
                met = corrected_mets.MET_UnclusteredEnergy.down
                j   = events.Jet
            
            else:
                j   = corrected_jets
    #                 j   = events.Jet
    #             met = events.MET
                met = corrected_mets
#                 #print(shift_syst)
        ###
        # Initialize global quantities (MET ecc.)
        ###
        
        
            met["T"] = ak.zip({"pt": met.pt_orig, "phi": met.phi_orig}, 
                              with_name="PolarTwoVector", 
                              behavior=vector.behavior)
            j["T"] = ak.zip({"pt": j.pt, "phi": j.phi}, 
                      with_name="PolarTwoVector", 
                      behavior=vector.behavior)
            j['p4'] = ak.zip({
            "pt": j.pt,
            "eta": j.eta,
            "phi": j.phi,
            "mass": j.mass},
            with_name="PtEtaPhiMLorentzVector",
            )
        
        if not isMC:
            met["T"] = ak.zip({"pt": met.pt, "phi": met.phi}, 
                              with_name="PolarTwoVector", 
                              behavior=vector.behavior)
            j["T"] = ak.zip({"pt": j.pt, "phi": j.phi}, 
                      with_name="PolarTwoVector", 
                      behavior=vector.behavior)
            j['p4'] = ak.zip({
            "pt": j.pt,
            "eta": j.eta,
            "phi": j.phi,
            "mass": j.mass},
            with_name="PtEtaPhiMLorentzVector",
            )
###################################################################################
    # Met Phi X-Y corrections
#         print(" met.pt_orig", met.pt_orig)
#         print("met.pt ", met.pt)
#         print('met["T"]["pt"]', met["T"]["pt"])

        XY_correctedmet_pt , XY_correctedmet_phi = METXYCorr_Met_MetPhi(events, self._year, met["T"]['pt'], met["T"]['phi'])

        corrected_met = met
        corrected_met["T"] = ak.zip({"pt": XY_correctedmet_pt, "phi": XY_correctedmet_phi}, 
                          with_name="PolarTwoVector", 
                          behavior=vector.behavior)
##################################################################################
        ###
        # Initialize physics objects
        ###


#         print()
#         print("XYcorrected_met.pt_orig",corrected_met.pt_orig)
#         print("XYcorrected_met[T].[pt]", corrected_met["T"]["pt"])
#         print('XY_correctedmet_pt', XY_correctedmet_pt)

#         import sys
#         sys.exit(0)
        mu = events.Muon
        mu['isloose'] = isLooseMuon(mu.pt, mu.eta, mu.pfIsoId, mu.looseId, self._year, mu.isPFcand,
                                   mu.isGlobal,
                                   mu.isTracker)
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.pfIsoId, mu.tightId, self._year, mu.isPFcand,
                                   mu.isGlobal,
                                   mu.isTracker)
        mu["T"] = ak.zip({"pt": mu.pt, "phi": mu.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        mu['p4'] = ak.zip({
                            "pt": mu.pt,
                            "eta": mu.eta,
                            "phi": mu.phi,
                            "mass": mu.mass},
                            with_name="PtEtaPhiMLorentzVector",
        )
    
        
        mu_loose = mu[ak.values_astype(mu.isloose, np.bool)]
        mu_tight = mu[ak.values_astype(mu.istight, np.bool)]
        mu_ntot = ak.num(mu, axis=1)
        mu_nloose = ak.num(mu_loose, axis=1)
        mu_ntight = ak.num(mu_tight, axis=1)
        leading_mu = mu_tight[:,:1]
        #print(ak.max(mu_tight.eta))
        
        e = events.Electron
        event_size = len(events)
        e['isclean'] = ~(ak.any(e.metric_table(mu_loose) < 0.3, axis=2)) 
        e['isloose'] = isLooseElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e["T"] = ak.zip({"pt": e.pt, "phi": e.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        e['p4'] = ak.zip({
                            "pt": e.pt,
                            "eta": e.eta,
                            "phi": e.phi,
                            "mass": e.mass},
                            with_name="PtEtaPhiMLorentzVector",
        )
        e_clean = e[ak.values_astype(e.isclean, np.bool)]
        e_loose = e_clean[ak.values_astype(e_clean.isloose, np.bool)]
        e_tight = e_clean[ak.values_astype(e_clean.istight, np.bool)]
        e_ntot = ak.num(e, axis=1)
        e_nloose = ak.num(e_loose, axis=1)

#         e_nloose = e_loose.counts
        e_ntight = ak.num(e_tight, axis=1)
        leading_e = e_tight[:,:1]
        

        tau = events.Tau
        tau['isclean'] = ~(ak.any(tau.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(tau.metric_table(e_loose) < 0.4, axis=2))
#         ak.all(tau.metric_table(mu_loose) > 0.4, axis=-1)
        try:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayMode,tau.idDeepTau2017v2p1VSjet,self._year)
        except:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeOldDMs,tau.idDeepTau2017v2p1VSjet,self._year)
        else: 
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeNewDMs,tau.idDeepTau2017v2p1VSjet,self._year)

        tau_clean = tau[ak.values_astype(tau.isclean, np.bool)]

        tau_loose = tau_clean[ak.values_astype(tau_clean.isloose, np.bool)]
        tau_ntot = ak.num(tau, axis=1)
        tau_nloose = ak.num(tau_loose, axis=1)

        pho = events.Photon
        pho['isclean'] = ~(ak.any(pho.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(pho.metric_table(e_loose) < 0.4, axis=2))
        _id = 'cutBased'
        if self._year == '2016':
            _id = 'cutBased'
        pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho[_id], self._year) & (pho.electronVeto)  # added electron veto flag
        pho['istight'] = isTightPhoton(pho.pt, pho[_id], self._year) & (pho.isScEtaEB) & (pho.electronVeto)  # tight photons are barrel only

        pho["T"] = ak.zip({"pt": pho.pt, "phi": pho.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
    
        pho_clean = pho[ak.values_astype(pho.isclean, np.bool)]
        pho_loose = pho_clean[ak.values_astype(pho_clean.isloose, np.bool)]
        pho_tight = pho_clean[ak.values_astype(pho_clean.istight, np.bool)]
        pho_ntot = ak.num(pho,axis=1)
        pho_nloose = ak.num(pho_loose, axis=1)
        pho_ntight = ak.num(pho_tight, axis=1)
        leading_pho = pho_tight[:,:1] #new way to define leading photon

#         j = events.Jet
#         JET Corrections
#         JEC Already applied in nanoAOD
#         Junc Already applied in nanoAOD
#         Jer  to be applied
#         JERSF  to be applied
        def jetcleaning(j, mu_loose, e_loose, pho_loose, r):
            NumNeutralParticles = (j.nConstituents-j.nElectrons-j.nMuons)
            NumConst = j.nConstituents
            NHF = j.neHEF
            NEMF = j.neEmEF
            CHF = j.chHEF
            MUF = j.muEF
            CEMF = j.chEmEF
            CHM = j.nElectrons+j.nMuons
            #1 Select good jets
            j_isgood_mask = isGoodJet(j["T"]["pt"], j.eta, j.jetId, j.puId,NHF, NEMF, NumConst, MUF, CHF, CHM, CEMF, year=self._year)
            j_good = j[j_isgood_mask]

            #2 Select good jets which don't have a lepton closeby
            jetMuMask = ak.any(j_good.metric_table(mu_loose) < r, axis=-1)
            jetEleMask = ak.any(j_good.metric_table(e_loose) < r, axis=-1)
            jetPhoMask = ak.any(j_good.metric_table(pho_loose) < r, axis=-1)



            j_clean = j_good[~jetPhoMask & ~jetEleMask & ~jetMuMask]

            return(j_clean)
        
        j['isHEM'] = isHEMJet(j["T"]["pt"], j.eta, j.phi)
        j['isdcsvL'] = (j.btagDeepB>deepcsvWPs['loose'])
        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium']) ## from Rishab
        j['isdcsvM'] = (j.btagDeepB > deepcsvWPs['medium']) ## from Rishabh

        j_clean = jetcleaning(j, mu_loose, e_loose, pho_loose, 0.4)
        j_nclean = ak.num(j_clean, axis=1)
        j_HEM = j[ak.values_astype(j.isHEM, np.bool)] 
        j_nHEM = ak.num(j_HEM, axis=1)

        j_dflvM = j_clean[ak.values_astype(j_clean.isdflvM, np.bool)]
        j_ndflvM = ak.num(j_dflvM, axis=1)
        leading_j = j_clean[:,:1]

        atleast_one_jet_with_pt_grt_50 = (j_nclean>0)&(ak.any(j_clean["T"]["pt"]>70, axis=1))
        

        ###
        # Calculating derivatives
        ###

        # ************ calculate delta phi( leading ak4jet, met) > 1.5***********

#         met_LJ_Pairs = ak.cartesian({"met":met, "lj": leading_j})
#         Delta_Phi_Met_LJ_var = (abs(met.delta_phi(leading_j)))
#         Delta_Phi_Met_LJ = (Delta_Phi_Met_LJ_var > 1.5)

        corrected_met['T'].delta_phi(leading_j['T'])
        Delta_Phi_Corr_Met_LJ_var = (abs(corrected_met['T'].delta_phi(leading_j['T'])))
        Delta_Phi_Corr_Met_LJ = (Delta_Phi_Corr_Met_LJ_var > 1.5)

        # *******calculate deltaR( leading ak4jet, e/mu) < 3.4 *****
        LJ_Ele = ak.cartesian({"leading_j":leading_j, "e_loose": e_loose})
        DeltaR_LJ_Ele = abs(LJ_Ele.leading_j.delta_r(LJ_Ele.e_loose))
        
        DeltaR_LJ_Ele_mask = ak.any(DeltaR_LJ_Ele < 3.4, axis=1)

        LJ_Mu = ak.cartesian({"leading_j":leading_j, "mu_loose": mu_loose})
        DeltaR_LJ_Mu = abs(LJ_Mu.leading_j.delta_r(LJ_Mu.mu_loose))
        
        DeltaR_LJ_Mu_mask = ak.any(DeltaR_LJ_Mu < 3.4, axis=1)


        ###
        # Calculate transverse mass
        ###



#         mT = {
#             'sre': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
#             'srm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
#             'ttbare': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
#             'ttbarm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
#             'wjete': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
#             'wjetm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
#         }
        

#         import sys
#         sys.exit(0)
        
        mT_corrected = {
            'sre': np.sqrt(2*leading_e.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_e))))),
            'srm': np.sqrt(2*leading_mu.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_mu))))),
            'ttbare': np.sqrt(2*leading_e.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_e))))),
            'ttbarm': np.sqrt(2*leading_mu.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_mu))))),
            'wjete': np.sqrt(2*leading_e.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_e))))),
            'wjetm': np.sqrt(2*leading_mu.pt*corrected_met["T"]["pt"]*(1-np.cos(abs(corrected_met["T"].delta_phi(leading_mu)))))
        }

        if isData:
            lumi_mask = np.array(LumiMask(self._golden_json_path[self._year])(events.run, events.luminosityBlock), dtype=bool)
            
        else: lumi_mask = np.ones(len(events))
        ###
        # Calculating weights
        ###
        if not isData:

            ###
            # JEC/JER
            ###

            gen = events.GenPart

            gen['isb'] = (abs(gen.pdgId) == 5) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isc'] = (abs(gen.pdgId) == 4) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isTop'] = (abs(gen.pdgId) == 6) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            ttjet_weights = np.ones(event_size)
            if('TTJets' in dataset):
                ttjet_weights = np.sqrt(get_ttbar_weight(genTops.pt.sum()) * get_ttbar_weight(genTops.pt.sum()))

            gen['isW'] = (abs(gen.pdgId) == 24) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId) == 23) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isA'] = (abs(gen.pdgId) == 22) & gen.hasFlags(['isPrompt', 'fromHardProcess', 'isLastCopy']) & (gen.status == 1)

            ###
            # Calculating gen photon dynamic isolation as in https://arxiv.org/pdf/1705.04664.pdf
            ###
                


            genWs = gen[gen.isW & (gen.pt > 100)]
            genZs = gen[gen.isZ & (gen.pt > 100)]
            genDYs = gen[gen.isZ & (gen.mass > 30)]
            # Based on photon weight distribution
#             genIsoAs = gen[gen.isIsoA & (gen.pt > 100)]

            nnlo_nlo = {}
            nlo_qcd = np.ones(event_size)
            nlo_ewk = np.ones(event_size)
#             if('GJets' in dataset):
#                 if self._year == '2016':
#                     nlo_qcd = get_nlo_qcd_weight['a'](genIsoAs.pt.max())
#                     nlo_ewk  = get_nlo_ewk_weight['a'](genIsoAs.pt.max())
#                 for systematic in get_nnlo_nlo_weight['a']:
#                     nnlo_nlo[systematic] = get_nnlo_nlo_weight['a'][systematic](genIsoAs.pt.max())*ak.values_astype((ak.num(genIsoAs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genIsoAs, axis=1) > 0), np.int)

            if ('WJetsToLNu' in dataset):
                nlo_qcd = get_nlo_qcd_weight['w'](ak.max(genWs.pt,axis=1))
                nlo_ewk = get_nlo_ewk_weight['w'](ak.max(genWs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['w']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['w'][systematic](ak.max(genWs.pt, axis=1))*ak.values_astype((ak.num(genWs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genWs, axis=1) > 0), np.int)

            elif('DY' in dataset):
                nlo_qcd = get_nlo_qcd_weight['dy'](ak.max(genDYs.pt, axis=1))
                nlo_ewk = get_nlo_ewk_weight['dy'](ak.max(genDYs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['dy']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['dy'][systematic](ak.max(genDYs.pt, axis=1))*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)
            elif('ZJets' in dataset):
                nlo_qcd = get_nlo_qcd_weight['z'](ak.max(genZs.pt, axis=1))
                nlo_ewk = get_nlo_ewk_weight['z'](ak.max(genZs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['z']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['z'][systematic](ak.max(genZs.pt, axis=1))*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)

            ###
            # Calculate fewz corrections
            ###
            k_wjets_fewz = np.ones(event_size)
            if re.search("^WJetsToLNu", dataset):
#                 if ('Pt' in dataset):
#                     xs_wjets = 789.2836
#                     print("hi I am applying the fewz for pt binned wjets")
                if re.search("^WJetsToLNu_[012]J_", dataset):
                    xs_wjets = 65582
                
                xs_wjets_fewz = 61526.7 # from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
                kfactor_wjets_fewz = xs_wjets_fewz/xs_wjets
                k_wjets_fewz = k_wjets_fewz*kfactor_wjets_fewz
                
            k_dyjets_fewz = np.ones(event_size)
            if re.search("^DYJetsToLNu", dataset):
#                 if ('Pt' in dataset):
#                     xs_dyjets = 400.29979
                if re.search("^DYJetsToLNu_[012]J_", dataset):
                    xs_dyjets = 6434.7
                
                xs_dyjets_fewz = 6077.22 # from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV                kfactor_wjets_fewz = xs_wjets_fewz/xs_wjets_jetbinned
                kfactor_dyjets = xs_dyjets_fewz/xs_dyjets
                k_dyjets_fewz = k_dyjets_fewz*kfactor_dyjets
            ###
            # Calculate PU weight and systematic variations
            ###

            pu = np.minimum(5, get_pu_nom_weight(events.Pileup.nTrueInt))
            puUp = np.minimum(5, get_pu_up_weight(events.Pileup.nTrueInt))
            puDown = np.minimum(5, get_pu_down_weight(events.Pileup.nTrueInt))
            print("pu nominal", pu)
            print("pu Up", puUp)
            print("pu Down", puDown)
            ###
            # Trigger efficiency weight
            ###


            trig = {
                'sre': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'srm':get_mu_trig_weight(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'ttbarm': get_mu_trig_weight(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'wjetm':get_mu_trig_weight(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
#                 #                 'dilepe' : get_met_trig_weight(leading_e.eta.sum(),leading_e.pt.sum()),
#                 #                 'dilepm' : get_met_trig_weight(leading_mu.eta.sum(),leading_mu.pt.sum()),
            }
            trig_err = {
                'sre': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'srm':get_mu_trig_err(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'ttbarm': get_mu_trig_err(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'wjetm':get_mu_trig_err(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
#                                 'dilepe' : get_met_trig_weight(leading_e.eta.sum(),leading_e.pt.sum()),
#                                 'dilepm' : get_met_trig_weight(leading_mu.eta.sum(),leading_mu.pt.sum()),
            }

            ###
            # Calculating electron and muon ID weights
            ###


            ids = {
                'sre': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'srm': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'ttbarm': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'wjetm': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                # 'dilepe': get_ele_loose_id_sf(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum()) * get_ele_loose_id_sf(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum()),
                # 'dilepm': get_mu_loose_id_sf(mu1eta, leading_mu_pair.i0.pt.sum()) * get_mu_loose_id_sf(mu2eta, leading_mu_pair.i1.pt.sum()),
            }
            ids_err = {
                'sre': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'srm': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'ttbarm': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'wjetm': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                # 'dilepe': get_ele_loose_id_sf(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum()) * get_ele_loose_id_sf(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum()),
                # 'dilepm': get_mu_loose_id_sf(mu1eta, leading_mu_pair.i0.pt.sum()) * get_mu_loose_id_sf(mu2eta, leading_mu_pair.i1.pt.sum()),
            }

            ###
            # Reconstruction weights for electrons
            ###

            # 2017 has separate weights for low/high pT (threshold at 20 GeV)
            def ele_reco_sf(pt, eta): 
                return get_ele_reco_sf(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_sf(eta, pt)*ak.values_astype((~(pt > 20)), np.int)
            
            def ele_reco_err(pt, eta):
                return get_ele_reco_err(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_err(eta, pt)*ak.values_astype((~(pt > 20)), np.int)

            #look at this rRISHABH
            if self._year == '2017' or self._year == '2018' or self._year == '2016':
                sf = ele_reco_sf
            else:
                sf = get_ele_reco_sf

            reco = {
                'sre': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'srm': np.ones(event_size),
                'ttbare': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'ttbarm': np.ones(event_size),
                'wjete': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'wjetm': np.ones(event_size),
                # 'dilepe': sf(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum()) * sf(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum()),
                # 'dilepm': np.ones(event_size)
            }
            reco_err = {
                'sre': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'srm': np.zeros(event_size),
                'ttbare': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'ttbarm': np.zeros(event_size),
                'wjete': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'wjetm': np.zeros(event_size),
                # 'dilepe': sf(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum()) * sf(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum()),
                # 'dilepm': np.ones(event_size)
            }
            ###
            # Isolation weights for muons
            ###

            isolation = {
                'sre': np.ones(event_size),
                'srm': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': np.ones(event_size),
                'ttbarm': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': np.ones(event_size),
                'wjetm': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
#                 'dilepm': get_mu_loose_iso_sf(mu1eta, leading_mu_pair.i0.pt.sum()) * get_mu_loose_iso_sf(mu2eta, leading_mu_pair.i1.pt.sum()),
#                 'dilepe': np.ones(event_size),
            }
            isolation_err = {
                'sre': np.zeros(event_size),
                'srm': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': np.zeros(event_size),
                'ttbarm': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': np.zeros(event_size),
                'wjetm': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
#                 'dilepm': get_mu_loose_iso_sf(mu1eta, leading_mu_pair.i0.pt.sum()) * get_mu_loose_iso_sf(mu2eta, leading_mu_pair.i1.pt.sum()),
#                 'dilepe': np.ones(event_size),
            }
            ###
            # CSEV weight for photons: https://twiki.cern.ch/twiki/bin/view/CMS/EgammaIDRecipesRun2#Electron_Veto_CSEV_or_pixel_seed
            ###

#             if self._year == '2016':
#                 csev_weight = get_pho_csev_sf(abs(ak.sum(leading_pho.eta, axis=-1)), ak.sum(leading_pho.pt))
#             elif self._year == '2017':
#                 csev_sf_index = 0.5*(ak.values_astype(ak.sum(leading_pho.isScEtaEB, axis=-1), np.int))+3.5*(~(ak.values_astype(ak.sum(leading_pho.isScEtaEB, axis=-1), np.int)))+1*(ak.values_astype(ak.sum(leading_pho.r9, axis=-1) > 0.94), np.int)+2*(ak.values_astype(ak.sum(leading_pho.r9, axis=-1) <= 0.94), np.int)
#                 csev_weight = get_pho_csev_sf(csev_sf_index)
#             elif self._year == '2018':
#                 csev_weight = get_pho_csev_sf(ak.sum(leading_pho.pt, axis=-1), abs(ak.sum(leading_pho.eta, axis=-1)))
#             csev_weight[csev_weight == 0] = 1



            ###
            # AK4 b-tagging weights
            ###
            '''
            if in a region you are asking for 0 btags, you have to apply the 0-btag weight
            if in a region you are asking for at least 1 btag, you need to apply the -1-btag weight

            it’s “-1” because we want to reserve “1" to name the weight that should be applied when you ask for exactly 1 b-tag

            that is different from the weight you apply when you ask for at least 1 b-tag
            '''
#             if 'preVFP' in dataset:
#                 VFP_status = 'preVFP'
#             elif 'postVFP' in dataset:
#                 VFP_status = 'postVFP'
#             else:
# #                 VFP_status = False
            btag = {}
            btagUp = {}
            btagDown = {}
            btag['sre'],   btagUp['sre'],   btagDown['sre'] = get_deepflav_weight['medium'](j_clean,j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '+1', )
            btag['srm'],   btagUp['srm'],   btagDown['srm'] = get_deepflav_weight['medium'](j_clean,j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '+1', )
            btag['ttbare'], btagUp['ttbare'], btagDown['ttbare'] = get_deepflav_weight['medium'](j_clean, j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '2', )
            btag['ttbarm'], btagUp['ttbarm'], btagDown['ttbarm'] = get_deepflav_weight['medium'](j_clean, j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '2', )
            btag['wjete'], btagUp['wjete'], btagDown['wjete'] = get_deepflav_weight['medium'](j_clean, j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '0', )
            btag['wjetm'], btagUp['wjetm'], btagDown['wjetm'] = get_deepflav_weight['medium'](j_clean, j_clean["T"]["pt"], j_clean.eta, j_clean.hadronFlavour, '0', ) 
#             This part is for handling the large btag weight exp(138) . this brings in nans in the template.
            
            for region in ['ttbare', 'ttbarm', 'wjete', 'wjetm', 'sre', 'srm']:
                for tag in [btag, btagUp, btagDown]:
                    tag[region] = np.asarray(tag[region])
                    mask = abs(tag[region])>=4
                    tag[region][mask==True] = 1

        ###
        # Selections
        ###

        met_filters = np.ones(event_size, dtype=np.bool)
        # this filter is recommended for data only
        if isData:
            met_filters = met_filters & events.Flag['eeBadScFilter']
        for flag in AnalysisProcessor.met_filter_flags[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters', ak.to_numpy(met_filters, np.bool))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singleelectron_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_electron_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singlemuon_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_muon_triggers', ak.to_numpy(triggers))

        noHEMj = np.ones(event_size, dtype=np.bool)
        if self._year == '2018':
            noHEMj = (j_nHEM == 0)

        noHEMmet = np.ones(event_size, dtype=np.bool)
        if self._year == '2018':
            noHEMmet = (corrected_met["T"]["pt"] > 470) | (corrected_met["T"]["phi"] > -0.62) | (corrected_met["T"]["phi"] < -1.62)

        '''
        what the next 6 lines of code do:

        main object is to exclude events from JetHt sample with W_pT b/w 70-100 GeV

        events.metadata['dataset'] = 'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8____27_'
        dataset = 'WJetsToLNu'

        see if the 'HT' is in the name of the sample
        so, it first goes to genpart,
        figures out if the genlevel process is hardprocess and firstcopy and there are genlevel particle with
        abs(pdgID)= 24

        ad selects only those events for the pT of W was > 100 GeV

        '''


        if ('WJetsToLNu' in dataset) & ('Pt' in dataset):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt > 200)
            selection.add("exclude_wjets_greater_200", ak.to_numpy(ak.all(remove_overlap, axis=1)))
        else:
            selection.add("exclude_wjets_greater_200", np.full(event_size, True))
            
        if ('WJetsToLNu' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt <= 200)
            selection.add("exclude_wjets_less_200", ak.to_numpy(ak.all(remove_overlap, axis=1)))
        else:
            selection.add("exclude_wjets_less_200", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt <= 400)
            selection.add("exclude_dyjets_less_400", ak.to_numpy(ak.all(remove_overlap, axis=1)))
        else:
            selection.add("exclude_dyjets_less_400", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & ('Pt' in dataset):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt > 400)
            selection.add("exclude_dyjets_greater_400", ak.to_numpy(ak.all(remove_overlap, axis=1)))
        else:
            selection.add("exclude_dyjets_greater_400", np.full(event_size, True))

        selection.add('DeltaR_LJ_mask',ak.to_numpy(DeltaR_LJ_Ele_mask | DeltaR_LJ_Mu_mask))
        selection.add('isoneM', ak.to_numpy((e_nloose == 0) & (mu_ntight == 1) & ( mu_nloose == 1)))
        selection.add('isoneE', ak.to_numpy((e_ntight == 1) & (e_nloose == 1) & (mu_nloose == 0)))

        selection.add('exactly_1_medium_btag', ak.to_numpy(j_ndflvM == 1))
        selection.add('atleast_2_medium_btag', ak.to_numpy(j_ndflvM >= 2))
        selection.add('zero_medium_btags', ak.to_numpy(j_ndflvM == 0))

        selection.add('noHEMj', ak.to_numpy(noHEMj))
        selection.add('noHEMmet', ak.to_numpy(noHEMmet))
        selection.add('corr_met100', ak.to_numpy(corrected_met["T"]["pt"] > 100))
        selection.add('Delta_Phi_Corr_Met_LJ', ak.to_numpy(ak.sum(Delta_Phi_Corr_Met_LJ, axis=1)>0))

        selection.add('DeltaR_LJ_Ele_mask', ak.to_numpy((DeltaR_LJ_Ele_mask)>0))

        selection.add('one_muon', ak.to_numpy(ak.num(mu_tight, axis=1) == 1))
        selection.add('zero_loose_electron', ak.to_numpy(ak.num(e_loose, axis=1) == 0))
        selection.add('DeltaR_LJ_Mu_mask', ak.to_numpy((DeltaR_LJ_Mu_mask)>0))
        selection.add('lumi_mask', ak.to_numpy(lumi_mask>0))

        for region in mT_corrected.keys():
            sel_name = 'mtcorr'+'_'+region+'>150'
            select = (mT_corrected[region] >150)
            selection.add(sel_name, ak.to_numpy(ak.sum(select,axis=1)>0))
            
        selection.add('leading_j>70',ak.to_numpy(ak.sum(leading_j["T"]["pt"], axis=1) >70))# from the monotop paper
        selection.add('atleast_one_jet_with_pt_grt_50',ak.to_numpy(atleast_one_jet_with_pt_grt_50))

#         #print(selection.all())
        regions = {
            'sre': ['isoneE', 'exactly_1_medium_btag', 'met_filters', 'single_electron_triggers', 
                    'atleast_one_jet_with_pt_grt_50',
                     'Delta_Phi_Corr_Met_LJ',
                    'mtcorr_sre>150','corr_met100',
                    'noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                    "exclude_wjets_less_200", "exclude_wjets_greater_200",
                    "leading_j>70",
                    'lumi_mask', 'DeltaR_LJ_Ele_mask'
                   ],
            
            'srm': ['isoneM', 'exactly_1_medium_btag', 'met_filters', 'single_muon_triggers', 
                    'atleast_one_jet_with_pt_grt_50',
                    'Delta_Phi_Corr_Met_LJ', 
                    'mtcorr_srm>150',  'corr_met100', 
                    'noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                    "exclude_wjets_less_200", "exclude_wjets_greater_200",
                    "leading_j>70", 
                    "lumi_mask", 'DeltaR_LJ_Mu_mask'
                   ],
            
            'ttbare': ['isoneE', 'atleast_2_medium_btag', 'met_filters', 'single_electron_triggers', 
                       'atleast_one_jet_with_pt_grt_50',
                       'Delta_Phi_Corr_Met_LJ',
                       'mtcorr_ttbare>150', 'corr_met100', 
                       'noHEMj','noHEMmet' ,"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                       "exclude_wjets_less_200", "exclude_wjets_greater_200",
                       "leading_j>70", 
                       'lumi_mask', 'DeltaR_LJ_Ele_mask',
                      ],
            
            'ttbarm': ['isoneM', 'atleast_2_medium_btag', 'met_filters', 'single_muon_triggers', 
                       'atleast_one_jet_with_pt_grt_50',
                        'Delta_Phi_Corr_Met_LJ',
                       'mtcorr_ttbarm>150' , 'corr_met100',
                       'noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                       "exclude_wjets_less_200", "exclude_wjets_greater_200",
                       "leading_j>70", 
                       "lumi_mask", 'DeltaR_LJ_Mu_mask',
                      ],
            
            'wjete': [
                      'corr_met100',
                'atleast_one_jet_with_pt_grt_50',
                      'Delta_Phi_Corr_Met_LJ','zero_medium_btags','isoneE', 'met_filters', 'single_electron_triggers',
                       'mtcorr_wjete>150' ,
                      'noHEMj', 'noHEMmet', "exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                      "exclude_wjets_less_200", "exclude_wjets_greater_200", 
                "leading_j>70",
                      "lumi_mask", 'DeltaR_LJ_Ele_mask',
                     ],
            
            'wjetm': ['atleast_one_jet_with_pt_grt_50',
                'corr_met100',
                      'Delta_Phi_Corr_Met_LJ',
                      'zero_medium_btags','isoneM', 'met_filters', 'single_muon_triggers',
                       'mtcorr_wjetm>150' ,
                      'noHEMj', 'noHEMmet', 
                "exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                      "exclude_wjets_less_200", "exclude_wjets_greater_200", 
                      "leading_j>70",
                      "lumi_mask",'DeltaR_LJ_Mu_mask'
                     ],
        }
#     small code piece for checcking cutflow for after each selection
#         mult = np.ones(len(events))
#         for sel in regions['wjetm']:
#             print(sel,selection.all(sel),"\nSUM:",sum(selection.all(sel)))
#             mult *= selection.all(sel)
#             print("events left",sum(mult))
#         print("MULT", mult)
#         print("SUM_MULT", sum(mult))
#         import sys
#         sys.exit(0)
        isFilled = False
#         #print("mu_ntight->", mu_ntight.sum(),
#               '\n', 'e_ntight->', e_ntight.sum())
        for region, cuts in regions.items():
            if region not in selected_regions: continue
#             if 'wjetm' not in region: continue
#             #print('Considering region:', region)

            ###
            # Adding recoil and minDPhi requirements
            ###

            # selection.add('recoil_'+region, (u[region].mag>250))
            # selection.add('mindphi_'+region, (abs(u[region].delta_phi(j_clean.T)).min()>0.8))
            # regions[region].update({'recoil_'+region,'mindphi_'+region})
            #             #print('Selection:',regions[region])

            variables = {
                'npvs': events.PV.npvsGood,
                'mu_pT':              leading_mu.pt,
#                     'eT_miss':              met.pt,
                'ele_pT':              leading_e.pt,
#                 'jet_pT':              leading_j.pt,
#                     'metphi':                 met.phi,
                'dphi_Corr_Met_LJ':             Delta_Phi_Corr_Met_LJ_var,
#                     'dphi_Met_LJ':Delta_Phi_Met_LJ_var,
                'j1pt':                   leading_j["T"]["pt"],
                'j1eta':                  leading_j.eta,
                'j1phi':                  leading_j.phi,
                # 'njets':                  j_nclean,
                # 'ndflvL':                 j_ndflvL,
                # 'ndcsvL':     j_ndcsvL,
                # 'e1pt'      : leading_e.pt,
                'ele_phi'     : leading_e.phi,
                'ele_eta'     : leading_e.eta,
                # 'dielemass' : leading_diele.mass,
                # 'dielept'   : leading_diele.pt,
                # 'mu1pt' : leading_mu.pt,
                'mu_phi' : leading_mu.phi,
                'mu_eta' : leading_mu.eta,
                # 'dimumass' : leading_dimu.mass,
                'dphi_e_etmiss':          abs(corrected_met.delta_phi(leading_e)),
                'dphi_mu_etmiss':         abs(corrected_met.delta_phi(leading_mu)),
                'dr_e_lj': DeltaR_LJ_Ele,
                'dr_mu_lj': DeltaR_LJ_Mu,
                'njets':                  j_nclean,
                'ndflvM':                 j_ndflvM,
#                 'ndcsvM':     j_ndcsvM,
#                 'scale_factors': np.ones(event_size, dtype=np.bool),
                "eT_miss_corrected":corrected_met["T"]["pt"],
                "eT_miss_corrected_phi":corrected_met["T"]["phi"]

                }

            if region in mT_corrected:
#                 variables['mT'] = mT[region]
                variables['mT_corrected'] = mT_corrected[region]

            def fill(dataset, weight, cut, systematic):

                flat_variables = {k: ak.flatten(v[cut], axis=None) for k, v in variables.items()}
                flat_weight = {k: ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None) for k, v in variables.items()}
                for histname, h in hout.items():
#                     #print("L1410")
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname not in variables:
                        continue
                    elif histname == 'sumw':
                        continue
                    elif histname == 'template':
                        continue

                    else:
                        flat_variable = {histname: flat_variables[histname]}
#                         #print("L1427",flat_variable)
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               systematic=systematic,
                               weight=flat_weight[histname])

            if isData:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True
                cut = selection.all(*regions[region])
                hout['template'].fill(dataset=dataset,
                                      region=region,
                                      systematic='nominal',
                                      mT_corrected = ak.sum(mT_corrected[region], axis=-1),
                                      npvs = events.PV.npvs,
                                      weight=np.ones(event_size)*cut)

                fill(dataset, np.ones(event_size), cut, systematic='nominal')
                        # Cutflow loop
                vcut=np.zeros(event_size, dtype=np.int)
                hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size))
                allcuts = set()
                for i, icut in enumerate(cuts):
                    allcuts.add(icut)
                    jcut = selection.all(*allcuts)
                    vcut = (i+1)*jcut

                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size)*jcut)
            else:
                weights = Weights(len(events))
#                 #print("L1445")
                if 'L1PreFiringWeight' in events.fields:
                    weights.add("L1Prefiring", events.L1PreFiringWeight.Nom,
                                events.L1PreFiringWeight.Up, 
                                events.L1PreFiringWeight.Dn)
                print("events.L1PreFiringWeight.Nom",events.L1PreFiringWeight.Nom)
                print("events.L1PreFiringWeight.Up",events.L1PreFiringWeight.Up)
                print("events.L1PreFiringWeight.Do",events.L1PreFiringWeight.Dn)
                weights.add('genw', events.genWeight)
                weights.add('nlo_qcd', nlo_qcd)
                weights.add('nlo_ewk', nlo_ewk)
                weights.add('k_wjets_fewz', k_wjets_fewz)
                weights.add('k_dyjets_fewz', k_dyjets_fewz)
#                 weights.add('ttjet_weights', ttjet_weights)
                if 'cen' in nnlo_nlo:
                    #print("if 'cen' in nnlo_nlo")
#                     weights.add('nnlo_nlo', nnlo_nlo['cen'])
                    weights.add('qcd1', np.ones(
                        event_size), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
                    weights.add('qcd2', np.ones(
                        event_size), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
                    weights.add('qcd3', np.ones(
                        event_size), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
                    weights.add('ew1', np.ones(
                        event_size), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
                    weights.add('ew2G', np.ones(
                        event_size), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
                    weights.add('ew3G', np.ones(
                        event_size), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
                    weights.add('ew2W', np.ones(
                        event_size), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
                    weights.add('ew3W', np.ones(
                        event_size), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
                    weights.add('ew2Z', np.ones(
                        event_size), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
                    weights.add('ew3Z', np.ones(
                        event_size), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
                    weights.add('mix', np.ones(
                        event_size), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
                    weights.add('muF', np.ones(
                        event_size), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
                    weights.add('muR', np.ones(
                        event_size), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
                weights.add('pileup', pu, puUp, puDown)
                
                if 'e' == region[-1]:
                    trig_name = 'trig_e'
                elif 'm' == region[-1]:
                    trig_name = 'trig_m'
                weights.add(trig_name, trig[region],trig_err[region], trig_err[region], shift=True)
                if 'e' == region[-1]:
                    ids_name = 'id_e'
                elif 'm' == region[-1]:
                    ids_name = 'id_m'                
                weights.add(ids_name, ids[region], ids_err[region], ids_err[region], shift=True)
                
                if 'e' == region[-1]:
                    reco_name = 'reco_e'
                elif 'm' == region[-1]:
                    reco_name = 'reco_m'
                weights.add(reco_name, reco[region], reco_err[region], reco_err[region], shift=True)
                
                if 'e' == region[-1]:
                    isolation_name = 'isolation_e'
                elif 'm' == region[-1]:
                    isolation_name = 'isolation_m'                
                weights.add(isolation_name, isolation[region], isolation_err[region], isolation_err[region], shift=True)
#                 weights.add('csev', csev[region])
                weights.add('btag', btag[region],btagUp[region], btagDown[region])
#                 #print("btag['wjete']",btag['wjete'])
    
    
##################################################################################################################
#####################         The structure of the following if else loop   ######################################
#####################
#####################   if the dataset is Wjets, DY, Zjets                   #####################################
# ###################        then: separate into Heavy and Light Flavor      ####################################
#                                   

# ###################   else: donot separate into HF and LF                  ###########################

                if 'WJets' in dataset or 'DY' in dataset or 'ZJets' in dataset or 'GJets' in dataset:
                    if (not isFilled) and (shift_syst is None):
                        hout['sumw'].fill(dataset='HF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='LF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        isFilled = True
                    whf = ak.values_astype(((ak.num(gen[gen.isb],axis=1) > 0) | (ak.num(gen[gen.isc], axis=1) > 0)), np.int)
                    wlf = ak.values_astype(~(ak.values_astype(whf,np.bool)), np.int)
                    cut = ak.to_numpy(selection.all(*regions[region]))
                    if shift_syst is None:
                        systematics = ['nominal',
                                       'btagUp',
                                       'btagDown',
                                       'qcd1Up',
                                       'qcd1Down',
                                       'qcd2Up',
                                       'qcd2Down',
                                       'qcd3Up',
                                       'qcd3Down',
                                       'muFUp',
                                       'muFDown',
                                       'muRUp',
                                       'muRDown',
                                       'ew1Up',
                                       'ew1Down',
                                       'ew2GUp',
                                       'ew2GDown',
                                       'ew2WUp',
                                       'ew2WDown',
                                       'ew2ZUp',
                                       'ew2ZDown',
                                       'ew3GUp',
                                       'ew3GDown',
                                       'ew3WUp',
                                       'ew3WDown',
                                       'ew3ZUp',
                                       'ew3ZDown',
                                       'mixUp',
                                       'mixDown',
                                       trig_name+'Up', trig_name+'Down',
                                       ids_name+'Up', ids_name+'Down',
                                       reco_name+'Up', reco_name+'Down',
                                       isolation_name+'Up', isolation_name+'Down',
                                       'pileupUp','pileupDown',
                                      ]
                        if 'L1PreFiringWeight' in events.fields:
                            systematics.extend(["L1PrefiringUp", "L1PrefiringDown"])
                    else:
                        systematics = [shift_syst] #[None, "JERUp"]
                        
                    for systematic in systematics:
                        if systematic is None:
                            systematic = 'nominal'
                            
                        if systematic in ["nominal", "JERUp", "JERDown", "JESUp", "JESDown", "METunclUp", "METunclDown"]:
                            sname = systematic
                            systematic = None
                        else:
                            sname = systematic
                        sname = 'nominal' if systematic is None else systematic
                        #print(sname)
                        hout['template'].fill(dataset='HF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              mT_corrected = ak.sum(mT_corrected[region], axis=-1),
                                              npvs = events.PV.npvs,
                                              weight=weights.weight(modifier=systematic)*whf*cut)
    
                        hout['template'].fill(dataset='LF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              mT_corrected = ak.sum(mT_corrected[region], axis=-1),
                                              npvs = events.PV.npvsGood,
                                              weight=weights.weight(modifier=systematic)*wlf*cut)

                    ## Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*whf)
                    hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*wlf)
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*whf)
                        hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*wlf)
                    sname = 'nominal' if shift_syst is None else shift_syst
#                     #print(sname)
                    if shift_syst is None:
                        fill('HF--'+dataset, weights.weight()*whf, cut,systematic=sname)
                        fill('LF--'+dataset, weights.weight()*wlf, cut, systematic=sname)
#                     fill('HF--'+dataset, weights.weight()*whf, cut)
#                     fill('LF--'+dataset, weights.weight()*wlf, cut)
                else: #for bkgs WW, QCD etc
#                     #print("hi")
                    if (not isFilled) and (shift_syst is None):
                        hout['sumw'].fill(dataset=dataset, sumw=1, weight=ak.sum(events.genWeight, axis=-1))
                        isFilled = True
                    cut = selection.all(*regions[region])
#                     for systematic in [None]:
                    if shift_syst is None:
                        systematic =['nominal',
                                       'btagUp',
                                       'btagDown',
                                           trig_name+'Up', trig_name+'Down',
                                           ids_name+'Up', ids_name+'Down',
                                           reco_name+'Up', reco_name+'Down',
                                           isolation_name+'Up', isolation_name+'Down',
                                         'pileupUp', 'pileupDown',
                                          ]
                        if 'L1PreFiringWeight' in events.fields:
                            systematics.extend(["L1PrefiringUp", "L1PrefiringDown"])
                    else:
                        systematic = [shift_syst] 
                    for systematic in systematic:
                        if systematic is None:
                            systematic = "nominal"
                        if systematic in ["nominal", "JERUp", "JERDown", "JESUp", "JESDown","METunclUp", "METunclDown"]:
                            sname = systematic
                            systematic = None
                        else: sname=systematic   
#                         sname = 'nominal' if systematic is None else systematic
#                         #print(sname)
#                         template is the place we keep track of all the uncertainities
                        hout['template'].fill(dataset=dataset,
                                              region=region,
                                              systematic=sname,
                                              mT_corrected = ak.sum(mT_corrected[region], axis=-1),
                                              npvs = events.PV.npvsGood,
                                              weight=weights.weight(modifier=systematic)*cut)
                        # Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight())
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        
                        hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight()*jcut)
                    sname = 'nominal' if shift_syst is None else shift_syst
                    if shift_syst is None:
                        fill(dataset, weights.weight(), cut, systematic=sname)
#                     fill(dataset, weights.weight(), cut)
#         time.sleep(0.5)
        return hout
    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
#             nevents = accumulator['sumw'].integrate('dataset', d).values(overflow='all')[()][1]
#             #print(nevents)

            print('Scaling:', d.name)
            dataset = d.name
            if '--' in dataset:
                dataset = dataset.split('--')[1]
            print('Cross section:', self._xsec[dataset])
            if self._xsec[dataset] != -1:
#                 scale[d.name] = 1
#                 scale[d.name] = (self._lumi*self._xsec[dataset])/nevents
                scale[d.name] = (self._lumi*self._xsec[dataset])

            else:
                scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw':
                continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')

        return accumulator

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-v', '--vfp', help='vfp', dest='vfp', default=None)
    (options, args) = parser.parse_args()
    if options.vfp is not None:
        with open('metadata/UL_'+options.year+'_'+options.vfp+'.json') as fin:
            samplefiles = json.load(fin)
            xsec = {k: v['xs'] for k, v in samplefiles.items()}
    else:
        with open('metadata/UL_'+options.year+'.json') as fin:
            samplefiles = json.load(fin)
            xsec = {k: v['xs'] for k, v in samplefiles.items()}
            
    corrections = util.load('data/corrections.coffea')
    ids = util.load('data/ids.coffea')
    common = util.load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year,
                                           xsec=xsec,
                                           corrections=corrections,
                                           ids=ids,
                                           common=common,
                                           vfp=options.vfp)
    if options.vfp is not None:
        save(processor_instance, 'data/UL_lep_monotop_v1_test_'+options.vfp+'_'+options.year+'.processor')
        print("processor name: UL_lep_monotop_v1_test_{}_{}".format(options.vfp, options.year))

    else:
        save(processor_instance, 'data/UL_lep_monotop_v1_test_'+options.year+'.processor')
        print("processor name: UL_lep_monotop_v1_test_{}".format(options.year))
  
  
  
