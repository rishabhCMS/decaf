#!/usr/bin/env python
from uproot_methods import TVector2Array, TLorentzVectorArray
from optparse import OptionParser
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty, JetTransformer, JetResolution, JetResolutionScaleFactor
from coffea.util import load, save
from coffea import hist, processor
from coffea.arrays import Initialize
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import numpy as np
import math
import awkward
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import time

class AnalysisProcessor(processor.ProcessorABC):
    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
        '2016': 35.92,
        '2017': 40.66,
        '2018': 59.74
    }

    def __init__(self, year, xsec, corrections, ids, common):

        self._year = year

        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec

        self._samples = {
            'srIsoMu': ("Mphi-2000_Mchi-500","Mphi-1495_Mchi-750","Mphi-2000_Mchi-150"),
            'srMu50': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'srNoSel': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'IsoMu|Mu50': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500")
        }







        self._singlemuon_triggers_isomu = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24',
#                 'Mu50',
#                 'TkMu50'

            ],
            '2017':
                [
                'IsoMu27',
                #                 'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ],
            '2018':
                [
                'IsoMu24',
                #                 'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ]
        }

        self._singlemuon_triggers_mu50 = {
            '2016': [
#                 'IsoMu24',
#                 'IsoTkMu24',
                'Mu50',
                'TkMu50'

            ],
            '2017':
                [
#                 'IsoMu27',
                                'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ],
            '2018':
                [
#                 'IsoMu24',
                                'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ]
        }
        self._singlemuon_triggers_IsoMu24_or_Mu50 = {
            '2016': [
#                 'IsoMu24',
#                 'IsoTkMu24',
                'Mu50',
                'TkMu50'

            ],
            '2017':
                [
#                 'IsoMu27',
                                'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ],
            '2018':
                [
                'IsoMu24',
                'Mu50',
                #                 'OldMu100',
                #                 'TkMu100'
            ]
        }




        self._corrections = corrections
        self._ids = ids
        self._common = common

        self._accumulator = processor.dict_accumulator({
            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])
            ),
            'template': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
            ),
#             'mT': hist.Hist(
#                 'Events',
#                 hist.Cat('dataset', 'Dataset'),
#                 hist.Cat('region', 'Region'),
#                 hist.Bin('mT', '$m_{T}$ [GeV]', 20, 0, 600)),
            'recoil': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
#                 hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
                hist.Bin('recoil','Hadronic Recoil',[250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0, 3000])),
            
#             'ele_pT': hist.Hist(
#                 'Events',
#                 hist.Cat('dataset', 'dataset'),
#                 hist.Cat('region', 'Region'),
#                 hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),

        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, events):

        dataset = events.metadata['dataset']
        print("dataset:",dataset)
        selected_regions = []
        for region, samples in self._samples.items():
#             print("sample:", samples)
#             print("region:", region)
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)
        print(selected_regions)
        
        isData = 'genWeight' not in events.columns
        isSignal = True
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

        ###
        # Getting corrections, ids from .coffea files
        ###
        isLooseElectron = self._ids['isLooseElectron'] 
        isTightElectron = self._ids['isTightElectron'] 
        isLooseMuon     = self._ids['isLooseMuon']     
        isTightMuon     = self._ids['isTightMuon']     
        isLooseTau      = self._ids['isLooseTau']      
        isLoosePhoton   = self._ids['isLoosePhoton']   
        isTightPhoton   = self._ids['isTightPhoton']   
        isGoodJet       = self._ids['isGoodJet']       
        isGoodFatJet    = self._ids['isGoodFatJet']    
        isHEMJet        = self._ids['isHEMJet']  

        mu = events.Muon
        mu['isloose'] = isLooseMuon(mu.pt,mu.eta,mu.pfRelIso04_all,mu.looseId,self._year)
        mu['istight'] = isTightMuon(mu.pt,mu.eta,mu.pfRelIso04_all,mu.tightId,self._year)
        mu['T'] = TVector2Array.from_polar(mu.pt, mu.phi)
        mu_loose=mu[mu.isloose.astype(np.bool)]
        mu_tight=mu[mu.istight.astype(np.bool)]
        mu_ntot = mu.counts
        mu_nloose = mu_loose.counts
        mu_ntight = mu_tight.counts
        leading_mu = mu[mu.pt.argmax()]
        leading_mu = leading_mu[leading_mu.istight.astype(np.bool)]
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
        # Initialize global quantities (MET ecc.)
        ###

        met = events.MET
        if self._year == '2017':
            events.METFixEE2017  # Recommended for 2017
        met['T'] = TVector2Array.from_polar(met.pt, met.phi)
        calomet = events.CaloMET
        puppimet = events.PuppiMET

        ###
        # Initialize physics objects
        ###



        e = events.Electron
        e['isclean'] = ~match(e, mu_loose, 0.3)
        e['isloose'] = isLooseElectron(
            e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(
            e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['T'] = TVector2Array.from_polar(e.pt, e.phi)
        e['p4'] = TLorentzVectorArray.from_ptetaphim(
            e.pt, e.eta, e.phi, e.mass)
        e_clean = e[e.isclean.astype(np.bool)]
        e_loose = e_clean[e_clean.isloose.astype(np.bool)]
        e_tight = e_clean[e_clean.istight.astype(np.bool)]
        e_ntot = e.counts
        e_nloose = e_loose.counts
        e_ntight = e_tight.counts
        leading_e = e[e.pt.argmax()]
        leading_e = leading_e[leading_e.isclean.astype(np.bool)]
        leading_e = leading_e[leading_e.istight.astype(np.bool)]

        





        ###
        # Calculate recoil and transverse mass
        ###

        u = {
            'srIsoMu': met.T+leading_mu.T.sum(),
            'srMu50': met.T+leading_mu.T.sum(),
            'srNoSel': met.T+leading_mu.T.sum(),
            'IsoMu|Mu50':met.T+leading_mu.T.sum()
        }





        

        ###
        # Selections
        ###

        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singlemuon_triggers_IsoMu24_or_Mu50[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singlemuon_triggers_IsoMu24_or_Mu50', triggers)
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singlemuon_triggers_isomu[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_muon_triggers_isomu', triggers)
 
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singlemuon_triggers_mu50[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
            print(triggers)
        selection.add('single_muon_triggers_mu50', triggers)
        
        triggers = np.ones(events.size, dtype=np.bool)
        selection.add('no_trigger', triggers)
        
        selTWMu = ((abs(mu.matched_gen.pdgId) == 13)& ((abs(mu.matched_gen.parent.pdgId) == 24) |(abs(mu.matched_gen.parent.parent.pdgId) == 6))).any()
        selection.add('selTWMu', selTWMu)
        
        selection.add('mu_pt>20', (mu.pt>20).any())
        selection.add('mu_eta<2.4', (abs(mu.eta)<2.4).any())




        regions = {

            'srIsoMu': {'selTWMu','single_muon_triggers_isomu', 'mu_pt>20', 'mu_eta<2.4'},
            'srMu50': {'selTWMu','single_muon_triggers_mu50', 'mu_pt>20', 'mu_eta<2.4'},
            'srNoSel': {'selTWMu', 'mu_pt>20', 'mu_eta<2.4' },
            'IsoMu|Mu50':{'selTWMu','singlemuon_triggers_IsoMu24_or_Mu50', 'mu_pt>20', 'mu_eta<2.4'}

            # 'dilepe' : {'istwoE','onebjet','noHEMj','met_filters','single_electron_triggers', 'met100', 'exclude_low_WpT_JetHT',
            #             'Delta_Phi_Met_LJ', 'DeltaR_LJ_Ele'},
            # 'dilepm' : {'istwoM','onebjet','noHEMj','met_filters','single_mu_triggers', 'met100', 'exclude_low_WpT_JetHT',
            #             'Delta_Phi_Met_LJ', 'DeltaR_LJ_Mu'},

            # 'gcr': {'isoneA','fatjet','noHEMj','met_filters','singlephoton_triggers'}
        }
        isFilled = False
#         print("mu_ntight->", mu_ntight.sum(),
#               '\n', 'e_ntight->', e_ntight.sum())
        for region in selected_regions:
            #             print('Considering region:', region)

            ###
            # Adding recoil and minDPhi requirements
            ###

            # selection.add('recoil_'+region, (u[region].mag>250))
            # selection.add('mindphi_'+region, (abs(u[region].delta_phi(j_clean.T)).min()>0.8))
            # regions[region].update({'recoil_'+region,'mindphi_'+region})
            #             print('Selection:',regions[region])
            variables = {

#                 'mu_pT':              mu_tight.pt,
                'recoil':                 u[region].mag,
                # 'mindphirecoil':          abs(u[region].delta_phi(j_clean.T)).min(),
                # 'CaloMinusPfOverRecoil':  abs(calomet.pt - met.pt) / u[region].mag,
#                 'eT_miss':              met.pt,
#                 'ele_pT':              e_tight.pt,
#                 'jet_pT':              leading_j.pt,
#                 'metphi':                 met.phi,
                # 'mindphimet':             abs(met.T.delta_phi(j_clean.T)).min(),
                # 'j1pt':                   leading_j.pt,
                # 'j1eta':                  leading_j.eta,
                # 'j1phi':                  leading_j.phi,
                # 'njets':                  j_nclean,
                # 'ndflvL':                 j_ndflvL,
                # 'ndcsvL':     j_ndcsvL,
                # 'e1pt'      : leading_e.pt,
#                 'ele_phi'     : leading_e.phi,
#                 'ele_eta'     : leading_e.eta,
                # 'dielemass' : leading_diele.mass,
                # 'dielept'   : leading_diele.pt,
                # 'mu1pt' : leading_mu.pt,
#                 'mu_phi' : leading_mu.phi,
#                 'mu_eta' : leading_mu.eta,
                # 'dimumass' : leading_dimu.mass,
#                 'dphi_e_etmiss':          abs(met['T'].delta_phi(leading_e['T'].sum())),
#                 'dphi_mu_etmiss':         abs(met['T'].delta_phi(leading_mu['T'].sum())),
#                 'dr_e_lj': DeltaR_LJ_Ele,
#                 'dr_mu_lj': DeltaR_LJ_Mu,
#                 'njets':                  j_nclean,
#                 'ndflvM':                 j_ndflvM,
#                 'ndcsvM':     j_ndcsvM,
#                 'eff': np.ones(events.size, dtype=np.bool)
            }

            def fill(dataset, weight, cut):

                flat_variables = {k: v[cut].flatten()
                                  for k, v in variables.items()}
                flat_weight = {
                    k: (~np.isnan(v[cut])*weight[cut]).flatten() for k, v in variables.items()}

                for histname, h in hout.items():
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname not in variables:
                        continue
                    elif histname == 'sumw':
                        continue
                    elif histname == 'template':
                        continue
                    elif histname == 'scale_factors':
                        flat_variable = {histname: flat_weight[histname]}
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable)

                    else:
                        flat_variable = {histname: flat_variables[histname]}
#                         print(flat_variable)
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               weight=flat_weight[histname])

            if isSignal:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True
                cut = selection.all(*regions[region])
                hout['template'].fill(dataset=dataset,
                                      region=region,
                                      systematic='nominal',
                                      weight=np.ones(events.size)*cut)
                fill(dataset, np.ones(events.size), cut)
            


        time.sleep(0.5)
        return hout
    
    def postprocess(self, accumulator):


        return accumulator
        
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    (options, args) = parser.parse_args()


    with open('metadata/signal2018.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k,v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids         = load('data/ids.coffea')
    common      = load('data/common.coffea')

    processor_instance=AnalysisProcessor(year=options.year,
                                         xsec=xsec,
                                         corrections=corrections,
                                         ids=ids,
                                         common=common)

    save(processor_instance, 'data/signal_eff'+options.year+'.processor')
    print("processor name: signal_eff{}".format(options.year))
