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
            'srEle32': ("Mphi-2000_Mchi-500","Mphi-1495_Mchi-750","Mphi-2000_Mchi-150"),
            'srEle115': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'srNoSel': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'Ele32|Ele115': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'Ele32|Ele115|Pho200': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'Ele32|Ele115|Pho200|Ele50PFj165': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500"),
            'Ele32|Pho200|Ele50PFj165': ("Mphi-1495_Mchi-750","Mphi-2000_Mchi-150","Mphi-2000_Mchi-500")
        }

        self._singleelectron_triggers_ele32_or_pho200_or_ele50 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                #'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200',
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
                
            ]
        }
        
        self._singleelectron_triggers_ele32 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                #'Ele115_CaloIdVT_GsfTrkIdT',
                #'Photon200'
            ]
        }

        self._singleelectron_triggers_ele115 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                #'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                #'Photon200'
            ]
        }

        self._singleelectron_triggers_ele115_or_ele32 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                #'Photon200'
            ]
        }

        self._singleelectron_triggers_ele115_or_ele32_or_pho200 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
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
        self._singleelectron_triggers_ele115_or_ele32_or_pho200_or_ele50pfjet165 = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele105_CaloIdVT_GsfTrkIdT'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200',
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
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
            'leptonic_recoil': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
#                 hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
                hist.Bin('leptonic_recoil','Leptonic Recoil',[250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0, 3000])),
            
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
        mu['p4'] = TLorentzVectorArray.from_ptetaphim(
            mu.pt, mu.eta, mu.phi, mu.mass)
        mu_loose=mu[mu.isloose.astype(np.bool)]
        mu_tight=mu[mu.istight.astype(np.bool)]
        mu_ntot = mu.counts
        mu_nloose = mu_loose.counts
        mu_ntight = mu_tight.counts
        leading_mu = mu[mu.pt.argmax()]
        leading_mu = leading_mu[leading_mu.istight.astype(np.bool)]


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

        pho = events.Photon
        pho['isclean'] = ~match(pho, mu_loose, 0.5) & ~match(pho, e_loose, 0.5)
        _id = 'cutBasedBitmap'
        if self._year == '2016':
            _id = 'cutBased'
        pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho[_id], self._year) & (
            pho.electronVeto)  # added electron veto flag
        pho['istight'] = isTightPhoton(pho.pt, pho[_id], self._year) & (
            pho.isScEtaEB) & (pho.electronVeto)  # tight photons are barrel only
        pho['T'] = TVector2Array.from_polar(pho.pt, pho.phi)
        pho_clean = pho[pho.isclean.astype(np.bool)]
        pho_loose = pho_clean[pho_clean.isloose.astype(np.bool)]
        pho_tight = pho_clean[pho_clean.istight.astype(np.bool)]
        pho_ntot = pho.counts
        pho_nloose = pho_loose.counts
        pho_ntight = pho_tight.counts
        leading_pho = pho[pho.pt.argmax()]
        leading_pho = leading_pho[leading_pho.isclean.astype(np.bool)]
        leading_pho = leading_pho[leading_pho.istight.astype(np.bool)]
        
        j = events.Jet
        j['isgood'] = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF)
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j['isclean'] = ~match(j, e_loose, 0.4) & ~match(
            j, mu_loose, 0.4) & ~match(j, pho_loose, 0.4)
#         print(j.isclean)

#         j['isiso'] = ~match(j,j[j.pt.argmax()],0.4)
        j['isdcsvL'] = (j.btagDeepB > deepcsvWPs['loose'])
        j['isdflvL'] = (j.btagDeepFlavB > deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium'])
        j['isdcsvM'] = (j.btagDeepB > deepcsvWPs['medium'])
        j['T'] = TVector2Array.from_polar(j.pt, j.phi)
        j['p4'] = TLorentzVectorArray.from_ptetaphim(j.pt, j.eta, j.phi, j.mass)
        j['ptRaw'] = j.pt * (1-j.rawFactor)
        j['massRaw'] = j.mass * (1-j.rawFactor)
        j['rho'] = j.pt.ones_like()*events.fixedGridRhoFastjetAll.array
        j_good = j[j.isgood.astype(np.bool)]
        j_clean = j[j.isclean.astype(np.bool)]
#         print(j_c)
#         j_iso = j_clean[j_clean.isiso.astype(np.bool)]
#         j_iso=j_clean[j_clean.astype(np.bool)]  # Sunil changed
        j_dcsvL = j_clean[j_clean.isdcsvL.astype(np.bool)]
        j_dflvL = j_clean[j_clean.isdflvL.astype(np.bool)]
        j_dflvM = j_clean[j_clean.isdflvM.astype(np.bool)]
        j_dcsvM = j_clean[j_clean.isdcsvM.astype(np.bool)]
        j_HEM = j[j.isHEM.astype(np.bool)]
        j_ntot = j.counts
        j_ngood = j_good.counts
        j_nclean = j_clean.counts
#         j_niso=j_iso.counts
        j_ndcsvL = j_dcsvL.counts
        j_ndflvL = j_dflvL.counts
        j_ndflvM = j_dflvM.counts
        j_ndcsvM = j_dcsvM.counts
        j_nHEM = j_HEM.counts
        leading_j = j[j.pt.argmax()]
        
        leading_j = leading_j[leading_j.isgood.astype(np.bool)]
        leading_j = leading_j[leading_j.isclean.astype(np.bool)]
        leading_bjet_dflvM = j_dflvM[j_dflvM.pt.argmax()]
        


        ###
        # Calculate hadronic recoil equivalent in leptonic channel, I will call it leptonic recoil,
        # The Idea is to find something highly correlated with top quark pT
        # (bjet_p4 + leading_lepton_p4).pT()
        ###

        leptonic_recoil = {
            'srEle32': (leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'srEle115': (leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'srNoSel': (leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'Ele32|Ele115':(leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'Ele32|Ele115|Pho200':(leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'Ele32|Ele115|Pho200|Ele50PFj165': (leading_bjet_dflvM.p4.sum()+leading_e.p4.sum()),
            'Ele32|Pho200|Ele50PFj165': (leading_bjet_dflvM.p4.sum()+leading_e.p4.sum())
        }





        

        ###
        # Selections
        ###
        
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele32_or_pho200_or_ele50[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele32_or_pho200_or_ele50', triggers)
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele115_or_ele32_or_pho200[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele115_or_ele32_or_pho200', triggers)

        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele115_or_ele32[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele115_or_ele32', triggers)
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele115[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele115', triggers)
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele32[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele32', triggers)
        
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._singleelectron_triggers_ele115_or_ele32_or_pho200_or_ele50pfjet165[self._year]:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers_ele115_or_ele32_or_pho200_or_ele50pfjet165', triggers)    
        
        triggers = np.ones(events.size, dtype=np.bool)
        selection.add('no_trigger', triggers)
        
        selTWEle = ((abs(e.matched_gen.pdgId) == 11)& ((abs(e.matched_gen.parent.pdgId) == 24) |(abs(e.matched_gen.parent.parent.pdgId) == 6))).any()
        selection.add('selTWEle', selTWEle)
        
        selection.add('e_pt>30', (e.pt>30).any())
        selection.add('e_eta<2.5', (abs(e.eta)<2.5).any())




        regions = {

            'srEle32': {'selTWEle','singleelectron_triggers_ele32', 'e_pt>30', 'e_eta<2.5'},
            'srEle115': {'selTWEle','singleelectron_triggers_ele115', 'e_pt>30', 'e_eta<2.5'},
            'srNoSel': {'selTWEle', 'e_pt>30', 'e_eta<2.5' },
            'Ele32|Ele115':{'selTWEle','singleelectron_triggers_ele115_or_ele32', 'e_pt>30', 'e_eta<2.5'},
            'Ele32|Ele115|Pho200':{'selTWEle','singleelectron_triggers_ele115_or_ele32_or_pho200', 'e_pt>30', 'e_eta<2.5'},
            'Ele32|Ele115|Pho200|Ele50PFj165': {'selTWEle','singleelectron_triggers_ele115_or_ele32_or_pho200_or_ele50pfjet165', 'e_pt>30', 'e_eta<2.5'},
            'Ele32|Pho200|Ele50PFj165':{'singleelectron_triggers_ele32_or_pho200_or_ele50', 'e_pt>30', 'e_eta<2.5'}
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
                'leptonic_recoil':     leptonic_recoil[region].pt,
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
