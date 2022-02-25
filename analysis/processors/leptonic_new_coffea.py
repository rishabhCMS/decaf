#!/usr/bin/env python
import time
from coffea import hist, nanoevents, util
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
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor
# from jmeCorrections import JetMetCorrections
import json

np.errstate(invalid='ignore', divide='ignore')


ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)
# vec.register_awkward()


from optparse import OptionParser
from coffea.arrays import Initialize
import lz4.frame as lz4f
import cloudpickle

# import pprint

import math
import time

import coffea.processor as processor
from coffea import hist
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False

import pickle
import re



class AnalysisProcessor(processor.ProcessorABC):

    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
        '2016': 19.52, #preVFP
#         '2016': 16.81, #postVFP
        '2017': 40.66,
        '2018': 59.74
    }

    met_filter_flags = {

        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter'
                 ],

        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter'
                 ],

        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter'
                 ]
    }

    def __init__(self, year, xsec, corrections, ids, common):

        self._fields = """
        CaloMET_pt
        CaloMET_phi
        Electron_charge
        Electron_cutBased
        Electron_dxy
        Electron_dz
        Electron_eta
        Electron_mass
        Electron_phi
        Electron_pt
        Flag_BadPFMuonFilter
        Flag_EcalDeadCellTriggerPrimitiveFilter
        Flag_HBHENoiseFilter
        Flag_HBHENoiseIsoFilter
        Flag_globalSuperTightHalo2016Filter
        Flag_goodVertices
        GenPart_eta
        GenPart_genPartIdxMother
        GenPart_pdgIdGenPart_phi
        GenPart_pt
        GenPart_statusFlags
        HLT_Ele115_CaloIdVT_GsfTrkIdT
        HLT_Ele32_WPTight_Gsf
        HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
        HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
        HLT_Photon200
        Jet_btagDeepB
        Jet_btagDeepFlavB
        Jet_chEmEF
        Jet_chHEF
        Jet_eta
        Jet_hadronFlavour
        Jet_jetId
        Jet_mass
        Jet_neEmEF
        Jet_neHEF
        Jet_phi
        Jet_pt
        Jet_rawFactor
        MET_phi
        MET_pt
        Muon_charge
        Muon_eta
        Muon_looseId
        Muon_mass
        Muon_pfRelIso04_all
        Muon_phi
        Muon_pt
        Muon_tightId
        PV_npvs
        Photon_eta
        Photon_phi
        Photon_pt
        Tau_eta
        Tau_idDecayMode
        Tau_idMVAoldDM2017v2
        Tau_phi
        Tau_pt
        fixedGridRhoFastjetAll
        genWeight
        nElectron
        nGenPart
        nJet
        nMuon
        nPhoton
        nTau
        """.split()

        self._year = year

        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec
        
        self._samples = {
            'sre':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','SinglePhoton','EGamma'),
            'srm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon'),
            'ttbare':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','SinglePhoton','EGamma'),
            'ttbarm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon'),
            'wjete':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD', 'SingleElectron','SinglePhoton','EGamma'),
            'wjetm':('WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleMuon'),
#             'dilepe':('DY','TT','ST','WW','WZ','ZZ','SingleElectron','SinglePhoton','EGamma'),
#             'dilepm':('DY','TT','ST','WW','WZ','ZZ','SingleMuon')
        }

        self._gentype_map = {
            'xbb':      1,
            'tbcq':     2,
            'tbqq':     3,
            'zcc':      4,
            'wcq':      5,
            'vqq':      6,
            'bb':       7,
            'bc':       8,
            'b':        9,
            'cc':     10,
            'c':       11,
            'other':   12
            # 'garbage': 13
        }

        self._ZHbbvsQCDwp = {
            '2016': 0.53,
            '2017': 0.61,
            '2018': 0.65
        }

        self._met_triggers = {
            '2016': [
                'PFMETNoMu90_PFMHTNoMu90_IDTight',
                'PFMETNoMu100_PFMHTNoMu100_IDTight',
                'PFMETNoMu110_PFMHTNoMu110_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ],
            '2017': [
                'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ],
            '2018': [
                'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ]
        }

        self._singlephoton_triggers = {
            '2016': [
                'Photon175',
                'Photon165_HE10'
            ],
            '2017': [
                'Photon200'
            ],
            '2018': [
                'Photon200'
            ]
        }

        self._singleelectron_triggers = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
#                 'Ele105_CaloIdVT_GsfTrkIdT'
                'Ele115_CaloIdVT_GsfTrkIdT'
#                 'Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
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

            '2016': [ 
#                     {
#                         'no_apv':
#                         ['Summer19UL16_V7_MC_L1FastJet_AK4PFchs',
#                         'Summer19UL16_V7_MC_L2L3Residual_AK4PFchs',
#                         'Summer19UL16_V7_MC_L2Relative_AK4PFchs',
#                         'Summer19UL16_V7_MC_L2Residual_AK4PFchs',
#                         'Summer19UL16_V7_MC_L3Absolute_AK4PFchs'],
                    
#                         'apv':
#                         ['Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs',
#                         'Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs',
#                         'Summer19UL16APV_V7_MC_L2Relative_AK4PFchs',
#                         'Summer19UL16APV_V7_MC_L2Residual_AK4PFchs',
#                         'Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs']
#                     }
                
                'Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs',
                'Summer16_07Aug2017_V11_MC_L2L3Residual_AK4PFchs',
                'Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs',
                'Summer16_07Aug2017_V11_MC_L2Residual_AK4PFchs',
                'Summer16_07Aug2017_V11_MC_L3Absolute_AK4PFchs'
            ],

            '2017': [
#                 'Summer19UL17_V5_MC_L1FastJet_AK4PFchs',
#                 'Summer19UL17_V5_MC_L2L3Residual_AK4PFchs',
#                 'Summer19UL17_V5_MC_L2Relative_AK4PFchs,
#                 'Summer19UL17_V5_MC_L2Residual_AK4PFchs',
#                 'Summer19UL17_V5_MC_L3Absolute_AK4PFchs
                
                'Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFchs',
                'Fall17_17Nov2017_V32_MC_L2L3Residual_AK4PFchs',
                'Fall17_17Nov2017_V32_MC_L2Relative_AK4PFchs',
                'Fall17_17Nov2017_V32_MC_L2Residual_AK4PFchs',
                'Fall17_17Nov2017_V32_MC_L3Absolute_AK4PFchs'
            ],

            '2018': [
#                 'Summer19UL18_V5_MC_L1FastJet_AK4PFchs',
#                 'Summer19UL18_V5_MC_L2L3Residual_AK4PFchs',
#                 'Summer19UL18_V5_MC_L3Absolute_AK4PFchs',
#                 'Summer19UL18_V5_MC_L2Relative_AK4PFchs',
#                 'Summer19UL18_V5_MC_L2Residual_AK4PFchs',
#                 'Summer19UL18_V5_MC_L3Absolute_AK4PFchs'
                     
                'Autumn18_V19_MC_L1FastJet_AK4PFchs',
                'Autumn18_V19_MC_L2L3Residual_AK4PFchs',
                'Autumn18_V19_MC_L2Relative_AK4PFchs',  # currently broken
                'Autumn18_V19_MC_L2Residual_AK4PFchs',
                'Autumn18_V19_MC_L3Absolute_AK4PFchs'
            ]
        }
# not updated JUNC
        self._junc = {

            '2016': 
#             {
#             'apv': 'Summer19UL16_V7_MC_Uncertainty_AK4PFchs',
#                      'no_apv': 'Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs'}
                'Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs',
            

            '2017': [
#                 'Summer19UL17_V5_MC_Uncertainty_AK4PFchs'
                'Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFchs'
            ],

            '2018': [
#                 'Summer19UL18_V5_MC_Uncertainty_AK4PFchs'
                'Autumn18_V19_MC_Uncertainty_AK4PFchs'
            ]
        }

        self._jr = {

            '2016': 
#             {'apv': 'Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs',
#                      'no_apv': 'Summer19UL18_JRV2_DATA_PtResolution_AK4PFchs'}
                'Summer16_25nsV1b_MC_PtResolution_AK4PFchs',


            '2017': [
#                 'Summer19UL17_JRV2_MC_PtResolution_AK4PFchs'
                'Fall17_V3b_MC_PtResolution_AK4PFchs'
            ],

            '2018': [
#                 'Summer19UL18_JRV2_MC_PtResolution_AK4PFchs'
                'Autumn18_V7b_MC_PtResolution_AK4PFchs'
            ]
        }

        self._jersf = {

            '2016': 
#             {'no_apv':'Summer20UL16_JRV3_MC_SF_AK8PFchs',
#                      'apv':'Summer20UL16APV_JRV3_MC_SF_AK4PFchs'},
                'Summer16_25nsV1b_MC_SF_AK4PFchs',
            

            '2017': [
#                 'Summer19UL17_JRV2_MC_SF_AK4PFchs'
                'Fall17_V3b_MC_SF_AK4PFchs'
            ],

            '2018': [
#                 'Summer19UL18_JRV2_MC_SF_AK4PFchs'
                'Autumn18_V7b_MC_SF_AK4PFchs'
            ]
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
                hist.Bin('mT', '$m_{T}$ [GeV]', 20, 0, 600)),
            
            'mT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mT', '$m_{T}$ [GeV]', 20, 0, 600)),
            
            'eT_miss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('eT_miss', '$E^T_{miss}$[GeV]', 20, 0, 600)),

            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),

            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_pT', 'Tight Muon $p_{T}$ [GeV]', 10, 0, 200)),
            
            'j1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1pt','AK4 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'j1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1eta','AK4 Leading Jet Eta',35,-3.5,3.5)),
            
            'j1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('j1phi','AK4 Leading Jet Phi',35,-3.5,3.5)),

            'dphi_e_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_e_etmiss', '$\Delta\phi (e, E^T_{miss} )$', 30, 0, 3.5)),
            
            'dphi_mu_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_mu_etmiss','$\Delta\phi (\mu, E^T_{miss} )$', 30, 0, 3.5)),
            
            'ndflvL': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvL', 'AK4 Number of deepFlavor Loose Jets', 6, -0.5, 5.5)),
            'ndflvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvM', 'AK4 Number of deepFlavor Medium Jets', 6, -0.5, 5.5)),
            'njets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('njets', 'AK4 Number of Jets', 7, -0.5, 6.5)),

            'ndcsvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndcsvM', 'AK4 Number of deepCSV Medium Jets', 6, -0.5, 5.5)),
            
            'dphi_Met_LJ': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_Met_LJ', '$\Delta \Phi (E^T_{miss}, Leading Jet)$', 30, 0, 3.5)),
            'dr_e_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_e_lj', '$\Delta r (Leading e, Leading Jet)$', 30, 0, 5.0)),
            'dr_mu_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_mu_lj', '$\Delta r (Leading \mu, Leading Jet)$', 30, 0, 5.0)),
            'ele_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_eta', 'Leading Electron Eta', 48, -2.4, 2.4)),
            'mu_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_eta', 'Leading Muon Eta', 48, -2.4, 2.4)),
            'ele_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_phi', 'Leading Electron Phi', 64, -3.2, 3.2)),
            'metphi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('metphi','MET phi',35,-3.5,3.5)),
            'mu_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_phi', 'Leading Muon Phi', 64, -3.2, 3.2)),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                #hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])),
 
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def fields(self):
        return self._fields

    def process(self, events):

        dataset = events.metadata['dataset']
            
        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)

        isData = 'genWeight' not in events.fields
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

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
            get_mu_trig_err = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_preVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_preVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_preVFP']['deepflav'][self._year]
            
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
            get_mu_trig_err = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        else:
#             print("hi")
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
            get_mu_trig_err = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        get_msd_weight = self._corrections['get_msd_weight']
        get_ttbar_weight = self._corrections['get_ttbar_weight']
        get_nnlo_nlo_weight = self._corrections['get_nnlo_nlo_weight'][self._year]
        get_nlo_qcd_weight = self._corrections['get_nlo_qcd_weight'][self._year]
        get_nlo_ewk_weight = self._corrections['get_nlo_ewk_weight'][self._year]
        get_pu_weight = self._corrections['get_pu_weight'][self._year]
        get_met_trig_weight = self._corrections['get_met_trig_weight'][self._year]
#         get_met_zmm_trig_weight = self._corrections['get_met_zmm_trig_weight'][self._year]
        get_ele_trig_weight = self._corrections['get_ele_trig_weight'][self._year]
        get_ele_trig_err    = self._corrections['get_ele_trig_err'][self._year]
#         get_mu_trig_weight = self._corrections['get_mu_trig_weight'][self._year]
        get_pho_trig_weight = self._corrections['get_pho_trig_weight'][self._year]
#         get_ele_loose_id_sf = self._corrections['get_ele_loose_id_sf'][self._year]
#         get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf'][self._year]
        get_pho_tight_id_sf = self._corrections['get_pho_tight_id_sf'][self._year]
        get_pho_csev_sf = self._corrections['get_pho_csev_sf'][self._year]
#         get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf'][self._year]
#         get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf'][self._year]
#         get_ele_reco_sf = self._corrections['get_ele_reco_sf'][self._year]
#         get_ele_reco_lowet_sf = self._corrections['get_ele_reco_lowet_sf']
#         get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf'][self._year]
#         get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf'][self._year]
        get_ecal_bad_calib = self._corrections['get_ecal_bad_calib']
#         get_deepflav_weight = self._corrections['get_btag_weight']['deepflav'][self._year]
#         get_deepcsv_weight = self._corrections['get_btag_weight']['deepcsv'][self._year]
#         Jetevaluator = self._corrections['Jetevaluator']

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
        ###

#         JECcorrector = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in self._jec[self._year]})
#         JECuncertainties = JetCorrectionUncertainty(**{name: Jetevaluator[name] for name in self._junc[self._year]})
#         JER = JetResolution(**{name: Jetevaluator[name] for name in self._jr[self._year]})
#         JERsf = JetResolutionScaleFactor(**{name: Jetevaluator[name] for name in self._jersf[self._year]})
#         Jet_transformer = JetTransformer(jec=JECcorrector, junc=JECuncertainties, jer=JER, jersf=JERsf)

        ###
        # Initialize global quantities (MET ecc.)
        ###

        met = events.MET
        met["T"] = ak.zip({"pt": met.pt, "phi": met.phi}, 
                          with_name="PolarTwoVector", 
                          behavior=vector.behavior)
    
        calomet = events.CaloMET
        puppimet = events.PuppiMET
        puppimet["T"] = ak.zip({"pt": puppimet.pt, "phi": puppimet.phi}, 
                          with_name="PolarTwoVector", 
                          behavior=vector.behavior)

        ###
        # Initialize physics objects
        ###

        mu = events.Muon
        mu['isloose'] = isLooseMuon(mu.pt, mu.eta, mu.pfRelIso04_all, mu.looseId, self._year)
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.pfRelIso04_all, mu.tightId, self._year)
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
        ak.num(mu, axis=1)
        mu_ntot = ak.num(mu, axis=1)
        mu_nloose = ak.num(mu_loose, axis=1)
        mu_ntight = ak.num(mu_tight, axis=1)
        leading_mu = mu_tight[:,:1]
        
        
        e = events.Electron
        event_size = len(e)
        
        
        e['isclean'] = ak.all(e.metric_table(mu_loose) > 0.3, axis=-1)
        e['isloose'] = isLooseElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
#         e['T'] = TVector2Array.from_polar(e.pt, e.phi)
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
        tau['isclean'] = ak.all(tau.metric_table(mu_loose) > 0.4, axis=-1) & ak.all(tau.metric_table(e_loose) > 0.4, axis=-1)
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
        pho['isclean'] = ak.all(pho.metric_table(mu_loose) > 0.4, axis=-1) & ak.all(pho.metric_table(e_loose) > 0.4, axis=-1)
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
        leading_pho = pho[:,:1] #new way to define leading photon
        leading_pho = leading_pho[ak.values_astype(leading_pho.isclean, np.bool)]
        
        leading_pho = leading_pho[ak.values_astype(leading_pho.istight, np.bool)]
        leading_pho = leading_pho[ak.values_astype(leading_pho.istight, np.bool)]

        j = events.Jet
        j['isgood'] = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j['isdcsvL'] = (j.btagDeepB>deepcsvWPs['loose'])
        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium']) ## from Rishab
        j['isdcsvM'] = (j.btagDeepB > deepcsvWPs['medium']) ## from Rishabh

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
        
#         https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.vector.html#
#         j['ptRaw'] =j.pt * (1-j.rawFactor)
#         j['massRaw'] = j.mass * (1-j.rawFactor)
#         j['rho'] = j.pt.ones_like()*events.fixedGridRhoFastjetAll.array

#         j_dflvL = j_clean[j_clean.isdflvL.astype(np.bool)]
        jetMuMask = ak.all(j.metric_table(mu_loose) > 0.4, axis=-1)
        jetEleMask = ak.all(j.metric_table(e_loose) > 0.4, axis=-1)
        jetPhoMask = ak.all(j.metric_table(pho_loose) > 0.4, axis=-1)

        j_isclean_mask = (jetMuMask & jetEleMask & jetPhoMask)
        j_isgood_mask = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j_good_clean = j[j_isclean_mask & j_isgood_mask]
        j_ngood_clean = ak.num(j_good_clean)
        j_good_clean_dflvB = j_good_clean.isdflvM
        j_ndflvM = ak.num(j[j_good_clean_dflvB])
        leading_j = j_good_clean[:,:1] # new way to define leading jet
        j_HEM = j[ak.values_astype(j.isHEM, np.bool)]       
        j_nHEM = ak.num(j_HEM, axis=1)
        atleast_one_jet_with_pt_grt_50 = ((ak.num(j_good_clean)>=1) & ak.any(j_good_clean.pt>=50, axis=-1))
        # *****btag
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X#Supported_Algorithms_and_Operati
        # medium     0.4184
#         btagWP_medium = 0.4184
#         Jet_btag_medium = j_clean[j_clean['btagDeepB'] > btagWP_medium]
        ###
        # Calculating derivatives
        ###

        # ************ calculate delta phi( leading ak4jet, met) > 1.5***********

        met_LJ_Pairs = ak.cartesian({"met":met, "lj": leading_j})
        Delta_Phi_Met_LJ_var = (abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj)))
        Delta_Phi_Met_LJ = (abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj))>1.5)

        # *******calculate deltaR( leading ak4jet, e/mu) < 3.4 *****
        LJ_Ele = ak.cartesian({"leading_j":leading_j, "e_loose": e_loose})
        DeltaR_LJ_Ele = abs(LJ_Ele.leading_j.delta_r(LJ_Ele.e_loose))
        
        DeltaR_LJ_Ele_mask = ak.any(DeltaR_LJ_Ele < 3.4, axis=-1)

        LJ_Mu = ak.cartesian({"leading_j":leading_j, "mu_loose": mu_loose})
        DeltaR_LJ_Mu = abs(LJ_Mu.leading_j.delta_r(LJ_Mu.mu_loose))
        
        DeltaR_LJ_Mu_mask = ak.any(DeltaR_LJ_Mu < 3.4, axis=-1)
#         ele_pairs = e_loose.distincts()
#         diele = ele_pairs.i0+ele_pairs.i1
#         diele['T'] = TVector2Array.from_polar(diele.pt, diele.phi)
#         leading_ele_pair = ele_pairs[diele.pt.argmax()]
#         leading_diele = diele[diele.pt.argmax()]

#         mu_pairs = mu_loose.distincts()
#         dimu = mu_pairs.i0+mu_pairs.i1
# #         dimu['T'] = TVector2Array.from_polar(dimu.pt, dimu.phi)
#         leading_mu_pair = mu_pairs[dimu.pt.argmax()]
#         leading_dimu = dimu[dimu.pt.argmax()]

        ###
        # Calculate transverse mass
        ###



        mT = {
            'sre': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'srm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'ttbare': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'ttbarm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'wjete': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e.T)))),
            'wjetm': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu.T))))
        }


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
                
#             epsilon_0_dyn = 0.1
#             n_dyn = 1
#             gen['R_dyn'] = (91.1876/(gen.pt * np.sqrt(epsilon_0_dyn))) * ak.values_astype((gen.isA), np.int) + (-999)*ak.values_astype((~gen.isA), np.int)
#             gen['R_0_dyn'] = gen.R_dyn * \
#                 ak.values_astype((gen.R_dyn < 1.0), np.int) + ak.values_astype((gen.R_dyn >= 1.0), np.int)   

#             def isolation(R):
#                 hadrons = gen[  # Stable hadrons not in NanoAOD, using quarks/glouns instead
#                     ((abs(gen.pdgId) <= 5) | (abs(gen.pdgId) == 21)) & gen.hasFlags(['fromHardProcess', 'isFirstCopy'])
#                 ]
#                 genhadrons = ak.cartesian({"gen":gen, "hadrons": hadrons})
#                 genhadrons = gen.cross(hadrons, nested=True)
#                 print("genhadrons.gen.pt", genhadrons.gen.pt)
#                 print("gen.pt", gen.pt)
#                 print("genhadrons.hadrons.pt", genhadrons.hadrons.pt)
#                 print("hadrons.pt", hadrons.pt)
#                 print(genhadrons.gen.delta_r(genhadrons.hadrons))
#                 print("R", R)
#                 import sys
#                 sys.exit(0)
#                 mask_gen_had = abs(genhadrons.gen.delta_r(genhadrons.hadrons)) <=R
#                 hadronic_et = (genhadrons.hadrons[mask_gen_had].pt)
# #                 hadronic_et = genhadrons.i1[(genhadrons.i0.delta_r(genhadrons.i1) <= R)].pt.sum()
#                 return (hadronic_et <= (epsilon_0_dyn * gen.pt * np.power((1 - np.cos(R)) / (1 - np.cos(gen.R_0_dyn)), n_dyn))) | (ak.num(hadrons, axis=1) == 0)

#             isIsoA = gen.isA
#             iterations = 5.
#             for i in range(1, int(iterations) + 1):
#                 isIsoA = isIsoA & isolation(gen.R_0_dyn*i/iterations)
#             gen['isIsoA'] = isIsoA

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
#                     nlo_ewk = get_nlo_ewk_weight['a'](genIsoAs.pt.max())
#                 for systematic in get_nnlo_nlo_weight['a']:
#                     nnlo_nlo[systematic] = get_nnlo_nlo_weight['a'][systematic](genIsoAs.pt.max())*ak.values_astype((ak.num(genIsoAs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genIsoAs, axis=1) > 0), np.int)

            if ('WJetsToLNu' in dataset) & (events.metadata['dataset'].split('-')[0].split('_')[1] == 'HT'):
                nlo_qcd = get_nlo_qcd_weight['w'](genWs.pt.max())
                nlo_ewk = get_nlo_ewk_weight['w'](genWs.pt.max())
                for systematic in get_nnlo_nlo_weight['w']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['w'][systematic](genWs.pt.max())*ak.values_astype((ak.num(genWs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genWs, axis=1) > 0), np.int)

            elif('DY' in dataset):
                nlo_qcd = get_nlo_qcd_weight['dy'](genDYs.pt.max())
                nlo_ewk = get_nlo_ewk_weight['dy'](genDYs.pt.max())
                for systematic in get_nnlo_nlo_weight['dy']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['dy'][systematic](genZs.pt.max())*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)
            elif('ZJets' in dataset):
                nlo_qcd = get_nlo_qcd_weight['z'](genZs.pt.max())
                nlo_ewk = get_nlo_ewk_weight['z'](genZs.pt.max())
                for systematic in get_nnlo_nlo_weight['z']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['z'][systematic](genZs.pt.max())*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)

            ###
            # Calculate PU weight and systematic variations
            ###

            pu = get_pu_weight(events.Pileup.nTrueInt)

            ###
            # Trigger efficiency weight
            ###

#             e1sf = get_ele_trig_weight(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum())*(leading_ele_pair.i0.pt.sum() > 40).astype(np.int)
#             e2sf = get_ele_trig_weight(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum())*(leading_ele_pair.i1.pt.sum() > 40).astype(np.int)

#             if self._year == '2016':
#                 sf = get_pho_trig_weight(leading_pho.pt.sum())
#             elif self._year == '2017':  # Sigmoid used for 2017 and 2018, values from monojet
#                 sf = sigmoid(leading_pho.pt.sum(), 0.335, 217.91, 0.065, 0.996) / sigmoid(leading_pho.pt.sum(), 0.244, 212.34, 0.050, 1.000)
#                 sf[np.isnan(sf) | np.isinf(sf)] == 1
#             elif self._year == '2018':
#                 sf = sigmoid(leading_pho.pt.sum(), 1.022, 218.39, 0.086, 0.999) / sigmoid(leading_pho.pt.sum(), 0.301, 212.83, 0.062, 1.000)
#                 sf[np.isnan(sf) | np.isinf(sf)] == 1

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

#             mueta = abs(leading_mu.eta.sum())
#             mu1eta = abs(leading_mu_pair.i0.eta.sum())
#             mu2eta = abs(leading_mu_pair.i1.eta.sum())
#             if self._year == '2016':
#                 mueta = leading_mu.eta.sum()
#                 mu1eta = leading_mu_pair.i0.eta.sum()
#                 mu2eta = leading_mu_pair.i1.eta.sum()
#             if self._year == '2016':
#                 sf = get_pho_tight_id_sf(
#                     leading_pho.eta.sum(), leading_pho.pt.sum())
#             else:  # 2017/2018 monojet measurement depends only on abs(eta)
#                 sf = get_pho_tight_id_sf(abs(leading_pho.eta.sum()))

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
                'srm': np.ones(event_size),
                'ttbare': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'ttbarm': np.ones(event_size),
                'wjete': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'wjetm': np.ones(event_size),
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
                'sre': np.ones(event_size),
                'srm': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'ttbare': np.ones(event_size),
                'ttbarm': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wjete': np.ones(event_size),
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
#                 VFP_status = False
            btag = {}
            btagUp = {}
            btagDown = {}
            btag['sre'],   btagDown['sre'],   btagUp['sre'] = get_deepflav_weight['medium'](j_good_clean,j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '+1', )
            btag['srm'],   btagDown['srm'],   btagUp['srm'] = get_deepflav_weight['medium'](j_good_clean,j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '+1', )
            btag['ttbare'], btagDown['ttbare'], btagUp['ttbare'] = get_deepflav_weight['medium'](j_good_clean, j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '2', )
            btag['ttbarm'], btagDown['ttbarm'], btagUp['ttbarm'] = get_deepflav_weight['medium'](j_good_clean, j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '2', )
            btag['wjete'], btagDown['wjete'], btagUp['wjete'] = get_deepflav_weight['medium'](j_good_clean, j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '0', )
            btag['wjetm'], btagDown['wjetm'], btagUp['wjetm'] = get_deepflav_weight['medium'](j_good_clean, j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '0', )  


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
        for path in self._met_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('met_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singleelectron_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_electron_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singlephoton_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_photon_triggers', ak.to_numpy(triggers))

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
            noHEMmet = (met.pt > 470) | (met.phi > -0.62) | (met.phi < -1.62)

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

        # predeclration just in cas I don't want the filter
        # selection.add("exclude_low_WpT_JetHT", np.full(len(events), True))
#         if ('WJetsToLNu' in dataset) & (events.metadata['dataset'].split('-')[0].split('_')[1] == 'HT'):

#             GenPart = events.GenPart
#             remove_overlap = (GenPart[GenPart.hasFlags(['fromHardProcess', 'isFirstCopy', 'isPrompt']) &
#                                       ((abs(GenPart.pdgId) == 24))].pt > 100).all()
#             selection.add("exclude_low_WpT_JetHT", remove_overlap)

#         else:
#             selection.add("exclude_low_WpT_JetHT", np.full(event_size, True))
        if ('WJetsToLNu' in dataset) & ('Pt' in dataset):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt >= 200)
            selection.add("exclude_wjets_greater_200", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_wjets_greater_200", np.full(event_size, True))
            
        if ('WJetsToLNu' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = ak.all(gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt < 200)
            selection.add("exclude_wjets_less_200", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_wjets_less_200", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = ak.all(gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt < 400)
            selection.add("exclude_dyjets_less_400", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_dyjets_less_400", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & ('Pt' in dataset):
            remove_overlap = ak.all(gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt >= 400)
            selection.add("exclude_dyjets_greater_400", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
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
        selection.add('met80', ak.to_numpy(met.pt < 80))
        selection.add('met100', ak.to_numpy(met.pt > 100))
        selection.add('Delta_Phi_Met_LJ', ak.to_numpy(ak.sum(abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj))>1.5, axis=1)>0))
        selection.add('DeltaR_LJ_Ele_mask', ak.to_numpy((DeltaR_LJ_Ele_mask)>0))

        selection.add('one_muon', ak.to_numpy(ak.num(mu_tight, axis=1) == 1))
        selection.add('zero_loose_electron', ak.to_numpy(ak.num(e_loose, axis=1) == 0))
        selection.add('DeltaR_LJ_Mu_mask', ak.to_numpy((DeltaR_LJ_Mu_mask)>0))

        for region in mT.keys():
            sel_name = 'mt'+'_'+region+'>50'
            select = (mT[region] > 50)
            selection.add(sel_name, ak.to_numpy(ak.sum(select,axis=1)>0))
        selection.add('leading_j>70',ak.to_numpy(ak.sum(leading_j.pt, axis=1) >70))# from the monotop paper
        selection.add('atleast_one_jet_with_pt_grt_50',ak.to_numpy(atleast_one_jet_with_pt_grt_50))

#         print(selection.all())
        regions = {
            'sre': ['isoneE', 'exactly_1_medium_btag', 'met_filters', 'single_electron_triggers', 'atleast_one_jet_with_pt_grt_50',
                     'Delta_Phi_Met_LJ', 'mt_sre>50','met100','noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                    "exclude_wjets_less_200", "exclude_wjets_greater_200","leading_j>70"],
            
            'srm': ['isoneM', 'exactly_1_medium_btag', 'met_filters', 'single_muon_triggers', 'atleast_one_jet_with_pt_grt_50',
                    'Delta_Phi_Met_LJ', 'mt_srm>50',  'met100', 'noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                    "exclude_wjets_less_200", "exclude_wjets_greater_200","leading_j>70"],
            
            'ttbare': ['isoneE', 'atleast_2_medium_btag', 'met_filters', 'single_electron_triggers', 'atleast_one_jet_with_pt_grt_50',
                       'Delta_Phi_Met_LJ', 'mt_ttbare>50', 'met100', 'noHEMj','noHEMmet' ,"exclude_dyjets_less_400", "exclude_dyjets_greater_400", 
                       "exclude_wjets_less_200", "exclude_wjets_greater_200","leading_j>70"],
            
            'ttbarm': ['isoneM', 'atleast_2_medium_btag', 'met_filters', 'single_muon_triggers', 'atleast_one_jet_with_pt_grt_50',
                        'Delta_Phi_Met_LJ', 'mt_ttbarm>50' , 'met100','noHEMj','noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                       "exclude_wjets_less_200", "exclude_wjets_greater_200","leading_j>70"],
            
            'wjete': [  'atleast_one_jet_with_pt_grt_50','met100','Delta_Phi_Met_LJ','zero_medium_btags','isoneE', 'met_filters', 'single_electron_triggers',
                       'mt_wjete>50' ,'noHEMj', 'noHEMmet', "exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                      "exclude_wjets_less_200", "exclude_wjets_greater_200", "leading_j>70"],
            
#             'test': [ 'atleast_one_jet_with_pt_grt_50', 
#                      'met100',
#                      'Delta_Phi_Met_LJ',
#                      'zero_medium_btags',
#                         'isoneE', 
                      
#                      'met_filters',
#                      'single_electron_triggers', 
#                      'mt_wjete>50',
#                         'noHEMj',
#                      'noHEMmet',
#                     'exclude_dyjets_less_400', 
#                      'exclude_dyjets_greater_400', 
#                     'exclude_wjets_less_200',
#                      'exclude_wjets_greater_200'],
            'wjetm': ['isoneM', 'zero_medium_btags', 'met_filters', 'single_muon_triggers', 'atleast_one_jet_with_pt_grt_50',
                      'Delta_Phi_Met_LJ', 'mt_wjetm>50', 'met100', 'noHEMj' ,'noHEMmet',"exclude_dyjets_less_400", "exclude_dyjets_greater_400",
                      "exclude_wjets_less_200", "exclude_wjets_greater_200", "leading_j>70"]
        }
#     small code piece for checcking cutflow for after each selection
#         mult = np.ones(len(events))
#         for sel in regions['test']:
#             print(sel,selection.all(sel),"\nSUM:",sum(selection.all(sel)))
#             mult *= selection.all(sel)
#             print("events left",sum(mult))
#         print("MULT", mult)
#         print("SUM_MULT", sum(mult))
        isFilled = False
#         print("mu_ntight->", mu_ntight.sum(),
#               '\n', 'e_ntight->', e_ntight.sum())
        for region, cuts in regions.items():
            if region not in selected_regions: continue
            print('Considering region:', region)

            ###
            # Adding recoil and minDPhi requirements
            ###

            # selection.add('recoil_'+region, (u[region].mag>250))
            # selection.add('mindphi_'+region, (abs(u[region].delta_phi(j_clean.T)).min()>0.8))
            # regions[region].update({'recoil_'+region,'mindphi_'+region})
            #             print('Selection:',regions[region])
            variables = {

                'mu_pT':              mu_tight.pt,
                # 'recoil':                 u[region].mag,
                # 'mindphirecoil':          abs(u[region].delta_phi(j_clean.T)).min(),
                # 'CaloMinusPfOverRecoil':  abs(calomet.pt - met.pt) / u[region].mag,
                'eT_miss':              met.pt,
                'ele_pT':              e_tight.pt,
#                 'jet_pT':              leading_j.pt,
                'metphi':                 met.phi,
                'dphi_Met_LJ':             Delta_Phi_Met_LJ_var,
                'j1pt':                   leading_j.pt,
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
                'dphi_e_etmiss':          abs(met.delta_phi(leading_e)),
                'dphi_mu_etmiss':         abs(met.delta_phi(leading_mu)),
                'dr_e_lj': DeltaR_LJ_Ele,
                'dr_mu_lj': DeltaR_LJ_Mu,
                'njets':                  j_ngood_clean,
                'ndflvM':                 j_ndflvM,
#                 'ndcsvM':     j_ndcsvM,
#                 'scale_factors': np.ones(event_size, dtype=np.bool)
                }
            if region in mT:
                variables['mT'] = mT[region]
#                 if 'e' in region[-1]:
#                     WRF = leading_e.T.sum()-met.T
#                 else:
#                     pass
#                     WRF = leading_mu.T.sum()-met.T
#                 variables['recoilphiWRF'] = abs(u[region].delta_phi(WRF))
#             print('Variables:', variables.keys())

            def fill(dataset, weight, cut):

                flat_variables = {k: ak.flatten(v[cut], axis=None) for k, v in variables.items()}
                flat_weight = {k: ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None) for k, v in variables.items()}
#                 for k, v in variables.items():
#                     print(k)
#                     print(v)
#                     print(cut)
#                     print("v[cut]:",v[cut])
#                     print("ak.flatten(v[cut], axis=None):",ak.flatten(v[cut], axis=None))
#                     print("ak.flatten(v[cut]):",ak.flatten(v[cut]))
#                     print("~np.isnan(v[cut]",~np.isnan(v[cut]))   
#                     print("weight[cut]:",weight[cut])
#                     print("~np.isnan(v[cut]",~np.isnan(v[cut]))
#                     print("ak.flatten(~np.isnan(v[cut])*weight[cut]))", ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None))
# #                     print("len(v*cut)",len(v*cut))
# #                     print("len(weight[cut])",len(weight[cut]))
# #                     print("len(~np.isnan(v[cut])*weight[cut])",len(~np.isnan(v[cut])*weight[cut]))
# #                     print((~np.isnan(v[cut])*weight[cut]))
#                     import sys
#                     sys.exit(0)
                for histname, h in hout.items():
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname not in variables:
                        continue
                    elif histname == 'sumw':
                        continue
                    elif histname == 'template':
                        continue
#                     elif histname == 'scale_factors':
#                         flat_variable = {histname: flat_weight[histname]}
#                         h.fill(dataset=dataset,
#                                region=region,
#                                **flat_variable)

                    else:
                        flat_variable = {histname: flat_variables[histname]}
#                         print(flat_variable)
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               weight=flat_weight[histname])

            if isData:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True
                cut = selection.all(*regions[region])
                hout['template'].fill(dataset=dataset,
                                      region=region,
                                      systematic='nominal',
                                      mT = mT[region],
                                      weight=np.ones(event_size)*cut)
                fill(dataset, np.ones(event_size), cut)
            else:
                weights = Weights(len(events))
                if 'L1PreFiringWeight' in events.fields:
                    weights.add('prefiring', events.L1PreFiringWeight.Nom)
                weights.add('genw', events.genWeight)
                weights.add('nlo_qcd', nlo_qcd)
                weights.add('nlo_ewk', nlo_ewk)
                weights.add('ttjet_weights', ttjet_weights)
                if 'cen' in nnlo_nlo:
                    print('hi')
                    weights.add('nnlo_nlo', nnlo_nlo['cen'])
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
                weights.add('pileup', pu)
                
                if 'e' == region[-1]:
                    trig_name = 'trig_e'
                elif 'm' == region[-1]:
                    trig_name = 'trig_m'
                weights.add(trig_name, trig[region],trig[region]+trig_err[region], trig[region]-trig_err[region])
                if 'e' == region[-1]:
                    ids_name = 'id_e'
                elif 'm' == region[-1]:
                    ids_name = 'id_m'                
                weights.add(ids_name, ids[region], ids[region]+ids_err[region], ids[region]-ids_err[region])
                
                if 'e' == region[-1]:
                    reco_name = 'reco_e'
                elif 'm' == region[-1]:
                    reco_name = 'reco_m'
                weights.add(reco_name, reco[region], reco[region]+reco_err[region], reco[region]-reco_err[region])
                
                if 'e' == region[-1]:
                    isolation_name = 'isolation_e'
                elif 'm' == region[-1]:
                    isolation_name = 'isolation_m'                
                weights.add(isolation_name, isolation[region], isolation[region]+isolation_err[region], isolation[region]-isolation_err[region])
#                 weights.add('csev', csev[region])
                weights.add('btag', btag[region],btagUp[region], btagDown[region])

                if 'WJets' in dataset or 'DY' in dataset or 'ZJets' in dataset or 'GJets' in dataset:
                    if not isFilled:
#                         print(events.genWeight)
#                         print(len(events.genWeight))
                        hout['sumw'].fill(dataset='HF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='LF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        isFilled = True
                    whf = ak.values_astype(((ak.num(gen[gen.isb],axis=1) > 0) | (ak.num(gen[gen.isc], axis=1) > 0)), np.int)
                    wlf = ak.values_astype(~(ak.values_astype(whf,np.bool)), np.int)
                    cut = ak.to_numpy(selection.all(*regions[region]))
#                     import sys
#                     print(weights._modifiers.keys())
#                     sys.exit(0)
                    if 'WJets' in dataset:
                        systematics = [None,
                                   'btagUp',
                                   'btagDown',
                                       trig_name+'Up', trig_name+'Down',
                                       ids_name+'Up', ids_name+'Down',
                                       reco_name+'Up', reco_name+'Down',
                                       isolation_name+'Up', isolation_name+'Down',
                                      ]
                    else:
                        systematics = [None,
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
                                      ]
                    for systematic in systematics:
                        sname = 'nominal' if systematic is None else systematic
#                         import sys
#                         print('weights.weight(modifier=systematic)', weights.weight(modifier=systematic))
#                         print('whf', whf)
#                         print("cut", cut)
#                         print("len ->weights.weight(modifier=systematic)", len(weights.weight(modifier=systematic)))
#                         print("len ->whf", len(whf))
#                         print("len ->cut", len(cut))
#                         x = weights.weight(modifier=systematic)*whf
#                         print('weights.weight(modifier=systematic)*whf', weights.weight(modifier=systematic)*whf)
#                         print('x[cut]', x[cut])
#                         print(len(cut))
#                         print(sum(ak.sum(mT[region], axis=-1)*x*cut))
#                         sys.exit(0)
                        hout['template'].fill(dataset='HF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              mT = ak.sum(mT[region], axis=-1),
                                              weight=weights.weight(modifier=systematic)*whf*cut)
    
                        hout['template'].fill(dataset='LF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              mT = ak.sum(mT[region], axis=-1),
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
                    fill('HF--'+dataset, weights.weight()*whf, cut)
                    fill('LF--'+dataset, weights.weight()*wlf, cut)

                else:
                    if not isFilled:
                        hout['sumw'].fill(
                            dataset=dataset, sumw=1, weight=events.genWeight)
                        isFilled = True
                    cut = selection.all(*regions[region])
#                     for systematic in [None]:
                    for systematic in [None,
                                   'btagUp',
                                   'btagDown',
                                       trig_name+'Up', trig_name+'Down',
                                       ids_name+'Up', ids_name+'Down',
                                       reco_name+'Up', reco_name+'Down',
                                       isolation_name+'Up', isolation_name+'Down',
                                      ]:
                        sname = 'nominal' if systematic is None else systematic
                        hout['template'].fill(dataset=dataset,
                                              region=region,
                                              systematic=sname,
                                              mT = mT[region],
                                              weight=weights.weight(modifier=systematic)*cut)
                        ## Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight())
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight()*jcut)
                    fill(dataset, weights.weight(), cut)
#         time.sleep(0.5)
        return hout

    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            print('Scaling:', d.name)
            dataset = d.name
            if '--' in dataset:
                dataset = dataset.split('--')[1]
            print('Cross section:', self._xsec[dataset])
            if self._xsec[dataset] != -1:
#                 scale[d.name] = 1
                scale[d.name] = self._lumi*self._xsec[dataset]
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
    (options, args) = parser.parse_args()

    with open('metadata/UL_'+options.year+'.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k, v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids = load('data/ids.coffea')
    common = load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year,
                                           xsec=xsec,
                                           corrections=corrections,
                                           ids=ids,
                                           common=common)

    save(processor_instance, 'data/UL_lep_monotop_v1_test_'+options.year+'.processor')
    print("processor name: UL_lep_monotop_v1_test_{}".format(options.year))
