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
import awkward
np.seterr(divide='ignore', invalid='ignore', over='ignore')


class AnalysisProcessor(processor.ProcessorABC):
    lumis = { #Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable                                                      
        '2016': 35.92,
        '2017': 41.53,
        '2018': 59.74
    }
    
    def __init__(self, year, xsec):
        self._year = year
        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])
        self._xsec = xsec

        self._accumulator = processor.dict_accumulator({

            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])
            ),

            'mT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mT', '$m_{T}$ [GeV]', 20, 0, 600)),
            'eT_miss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('eT', '$E^T_{miss}$[GeV]', 20, 0, 600)),


            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),

            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('pT', 'Tight Muon $p_{T}$ [GeV]', 10, 0, 200)),

            'jet_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('pT', 'Leading $AK4 Jet p_{T}$ [GeV]',
                         [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])),
            'dphi_e_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi', '$\Delta\phi (e, E^T_{miss} )$', 30, 0, 3.5)),
            'dphi_mu_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi', '$\Delta\phi (\mu, E^T_{miss} )$', 30, 0, 3.5)),

            'cutflow': processor.defaultdict_accumulator(int)
        }
        )

        self._samples = {
            'secr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleElectron'],
            'smcr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleMuon'],
            'tecr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleElectron'],
            'tmucr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleMuon'],
            'wecr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleElectron'],
            'wmucr': ['WJets', 'DY', 'TT', 'ST', 'WW', 'WZ', 'ZZ', 'QCD', 'SingleMuon'],
            # 'dilepe':'DY','TT','ST','WW','WZ','ZZ','SingleElectron'),
            # 'dilepm':'DY','TT','ST','WW','WZ','ZZ','SingleMuon')

        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        # This gets us the accumulator dictionary we defined in init
        output = self.accumulator.identity()

        dataset_name = events.metadata['dataset'].split('_')[0]
        dataset = events.metadata['dataset']
        isFilled=False

#                 print("a JetHT dataset was found and the processor has excluded events with W_pT <100 GeV because we are trying to inspect the W+Jets Low pT sample")
        Electron = events.Electron
        Muon = events.Muon
        Met = events.MET
        Jet = events.Jet
#         else: pass
      # **********step 1 Object Selection***************
        '''
        1. Electrons: 
          a. Tight Electron
          b. Loose Electron
          c. Ele['T'] = ( pt*cos(phi), pt*sin(phi))
          d. Ele['p4']
        '''
        Electron['T'] = TVector2Array.from_polar(Electron.pt, Electron.phi)
        Electron['p4'] = TLorentzVectorArray.from_ptetaphim(
            Electron.pt, Electron.eta, Electron.phi, Electron.mass)

        TightEleSel = ~(Electron.pt == np.nan)
        TightEleSel = (((Electron.pt > 40) &
                        (abs(Electron.eta) < 1.4442) &
                        (abs(Electron.dxy) < 0.05) &
                        (abs(Electron.dz) < 0.1) &
                        (Electron.cutBased == 4)) |
                       ((Electron.pt > 40) &
                        (abs(Electron.eta) > 1.5660) &
                        (abs(Electron.eta) < 2.5) &
                        (abs(Electron.dxy) < 0.1) &
                        (abs(Electron.dz) < 0.2) &
                        (Electron.cutBased == 4)
                        ))  # Trigger: HLT_Ele32_WPTight_Gsf_v

        LooseEleSel = ~(Electron.pt == np.nan)
        LooseEleSel = (((Electron.pt > 10) &
                        (abs(Electron.eta) < 1.4442) &
                        (abs(Electron.dxy) < 0.05) &
                        (abs(Electron.dz) < 0.1) &
                        (Electron.cutBased >= 1)) |
                       ((Electron.pt > 10) &
                        (abs(Electron.eta) > 1.5660) &
                        (abs(Electron.eta) < 2.5) &
                        (abs(Electron.dxy) < 0.1) &
                        (abs(Electron.dz) < 0.2) &
                        (Electron.cutBased >= 1)))

        TightElectron = Electron[TightEleSel]
        LooseElectron = Electron[LooseEleSel]
        LeadingEle = TightElectron[TightElectron.pt.argmax()]
        #         output['ele_pt'].fill(dataset=dataset,
        #                      region='everthing',
        #                      pT = Electron.pt[LooseEleSel].flatten())

        '''
        2. Muons:
          a. Tight Muon
          b. Loose Muon
          c. Mu['T'] = ( pt*cos(phi), pt*sin(phi))
          d. Mu['p4']
        '''
        Muon['T'] = TVector2Array.from_polar(Muon.pt, Muon.phi)
        Muon['p4'] = TLorentzVectorArray.from_ptetaphim(
            Muon.pt, Muon.eta, Muon.phi, Muon.mass)

        TightMuSel = ~(Muon.pt == np.nan)
        TightMuSel = (
            (Muon.pt > 30) &
            (abs(Muon.eta) < 2.4) &
            (Muon.tightId) &
            (Muon.pfRelIso04_all < 0.15)
        )

        LooseMuonSel = ~(Muon.pt == np.nan)
        LooseMuonSel = (Muon.pt > 15) & (abs(Muon.eta) < 2.4) & (
            Muon.looseId > 0) & (Muon.pfRelIso04_all < 0.25)

        LooseMuon = Muon[LooseMuonSel]
        TightMuon = Muon[TightMuSel]
        LeadingMu = TightMuon[TightMuon.pt.argmax()]
        '''
        3. Photons:
          a. Loose photon
          b. Tight Photon

        '''
        Photon = events.Photon
        # just a complicated way to initialize a jagged array with the needed shape to True
        LoosePhoSel = ~(Photon.pt == np.nan)
        LoosePhoSel = (
            (Photon.pt > 15) &
            ~((abs(Photon.eta) > 1.4442) & (abs(Photon.eta) < 1.566)) &
            (abs(Photon.eta) < 2.5) &
            (abs(Photon.cutBasedBitmap & 1) == 1)
        )

        TightPhoSel = ~(Photon.pt == np.nan)
        TightPhoSel = ((Photon.pt > 230) & ((Photon.cutBasedBitmap & 2) == 2))

        LoosePhoton = Photon[LoosePhoSel]
        TightPhoton = Photon[TightPhoSel]

        #         output['ele_pt'].fill(dataset=dataset,
        #              region='everthing',
        #              pT = TightPhoton.pt.flatten())

        '''
        4. MET:
        Met['T']  = TVector2Array.from_polar(Met.pt, Met.phi)
        Met['p4'] = TLorentzVectorArray.from_ptetaphim(Met.pt, 0., Met.phi, 0.)
        TightMet =  ~(Met.pt==np.nan)
        '''

        Met['T'] = TVector2Array.from_polar(Met.pt, Met.phi)
        Met['p4'] = TLorentzVectorArray.from_ptetaphim(Met.pt, 0., Met.phi, 0.)
        TightMet = ~(Met.pt == np.nan)

        '''
        5. Jets:
        '''

        Jet['T'] = TVector2Array.from_polar(Jet.pt, Jet.phi)
        Jet['p4'] = TLorentzVectorArray.from_ptetaphim(
            Jet.pt, Jet.eta, Jet.phi, Jet.mass)
        LeadingJet = Jet[Jet.pt.argmax()]

        # ******** step 2: triggers*******

        # single e triggers for e events
        EleTrigger = ['Ele32_WPTight_Gsf',
                      'Ele115_CaloIdVT_GsfTrkIdT', 'Photon200']

        # single mu triggers for µ events
        MuTrigger = ['IsoMu24',
                     'Mu50',
                     'OldMu100',
                     'TkMu100']
        # Photon trigger
        PhoTrigger = ['Photon200']

        # met trigger
        MetTrigger = ['PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                      'PFMETNoMu120_PFMHTNoMu120_IDTight']

        # ************ calculate delta phi( leading ak4jet, met) > 1.5***********

        Jet['T'] = TVector2Array.from_polar(Jet.pt, Jet.phi)
        Jet['p4'] = TLorentzVectorArray.from_ptetaphim(
            Jet.pt, Jet.eta, Jet.phi, Jet.mass)
        LeadingJet = Jet[Jet.pt.argmax()]

        Delta_Phi_Met_LJ = (Met['T'].delta_phi(LeadingJet['T'].sum()) > 1.5)

        # *******calculate deltaR( leading ak4jet, e/mu) < 3.4 *****
        LJ_Ele = LeadingJet['p4'].cross(TightElectron['p4'])
        DeltaR_LJ_Ele = LJ_Ele.i0.delta_r(LJ_Ele.i1)
        DeltaR_LJ_Ele_mask = (DeltaR_LJ_Ele < 3.4).any()

        LJ_Mu = LeadingJet['p4'].cross(TightMuon['p4'])
        DeltaR_LJ_Mu = LJ_Mu.i0.delta_r(LJ_Mu.i1)
        DeltaR_LJ_Mu_mask = (DeltaR_LJ_Mu < 3.4).any()

        # *****btag
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X#Supported_Algorithms_and_Operati
        # medium     0.4184
        btagWP_medium = 0.4184
        Jet_btag_medium = Jet[Jet['btagDeepB'] > btagWP_medium]

        # ****** to add event selection in coffea ********

        selection = processor.PackedSelection()

        triggers = np.zeros(events.size, dtype=np.bool)
        for path in MetTrigger:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('met_triggers', triggers)
        selection.add('Met100', (Met.pt >= 100))

        triggers = np.zeros(events.size, dtype=np.bool)
        for path in EleTrigger:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('ele_triggers', triggers)

        triggers = np.zeros(events.size, dtype=np.bool)
        for path in MuTrigger:
            if path not in events.HLT.columns:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('mu_triggers', triggers)
        selection.add('one_electron', (TightElectron.counts == 1))
        selection.add('zero_loose_muon', (LooseMuon.counts == 0))
        selection.add('zero_loose_photon', (LoosePhoton.counts == 0))
        selection.add('zero_medium_btags',
                      (Jet[Jet['btagDeepB'] > btagWP_medium].counts == 0))
        selection.add('Delta_Phi_Met_LJ', (Delta_Phi_Met_LJ))
        selection.add('DeltaR_LJ_Ele', (DeltaR_LJ_Ele_mask))

        selection.add('one_muon', (TightMuon.counts == 1))
        selection.add('zero_loose_electron', (LooseElectron.counts == 0))
        selection.add('DeltaR_LJ_Mu', (DeltaR_LJ_Mu_mask))

        selection.add('atleast_2_medium_btag',
                      (Jet[Jet['btagDeepB'] > btagWP_medium].counts >= 2))

        selection.add('exactly_1_medium_btag',
                      (Jet[Jet['btagDeepB'] > btagWP_medium].counts == 1))

        '''
        what the next 6 lines of code do:

        main object is to exclude events from JetHt sample with W_pT b/w 70-100 GeV

        events.metadata['dataset'] = 'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8____27_'
        dataset = 'WJetsToLNu'

        see if the 'HT' is in the name of the sample
        so, it first goes to genpart,
        figures out if the genlevel process is hardprocess and firstcopy and there are genlevel particle with 
        abs(pdgID)= 24

        and selects only those events for the pT of W was > 50 GeV

        '''

        # predeclration just in cas I don't want the filter
        selection.add("exclude_low_WpT_JetHT", np.full(events.size, True, dtype=np.bool))
        if dataset_name == 'WJetsToLNu':
            if events.metadata['dataset'].split('-')[0].split('_')[1] == 'HT':
                GenPart = events.GenPart
                remove_overlap = (GenPart[GenPart.hasFlags(['fromHardProcess', 'isFirstCopy', 'isPrompt']) &
                                          ((abs(GenPart.pdgId) == 24))].pt > 50).all()
                selection.add("exclude_low_WpT_JetHT", remove_overlap)

        else:
            selection.add("exclude_low_WpT_JetHT", np.full(len(events), True))
        # i dont think I need a photon trigger
        #         triggers = np.zeros(events.size, dtype=np.bool)
        #         for path in PhoTrigger:
        #             if path not in events.HLT.columns: continue
        #             triggers = triggers | events.HLT[path]
        #         selection.add('pho_triggers', triggers)

        selection.add('DeltaR_LJ_mask',
                      (DeltaR_LJ_Ele_mask | DeltaR_LJ_Mu_mask))

        #
        region = {
            'wecr': (selection.all(*('ele_triggers',
                                     'Met100',
                                     'one_electron',
                                     'zero_loose_muon',
                                     'zero_loose_photon',
                                     'zero_medium_btags',
                                     'Delta_Phi_Met_LJ',
                                     'DeltaR_LJ_Ele',
                                     'exclude_low_WpT_JetHT'
                                     )),
                     np.sqrt(2*LeadingEle.pt.sum()*Met.pt *
                             (1-np.cos(Met.T.delta_phi(LeadingEle.T.sum()))))
                     ),
            'wmucr': (selection.all(*('mu_triggers',
                                      'Met100',
                                      'one_muon',
                                      'zero_loose_electron',
                                      'zero_loose_photon',
                                      'zero_medium_btags',
                                      'Delta_Phi_Met_LJ',
                                      'DeltaR_LJ_Mu',
                                      'exclude_low_WpT_JetHT'
                                      )),
                      np.sqrt(2*LeadingMu.pt.sum()*Met.pt *
                              (1-np.cos(Met.T.delta_phi(LeadingMu.T.sum()))))
                      ),
            'tecr': (selection.all(*('ele_triggers',
                                     'Met100',
                                     'one_electron',
                                     'zero_loose_muon',
                                     'zero_loose_photon',
                                     'atleast_2_medium_btag',
                                     'Delta_Phi_Met_LJ',
                                     'DeltaR_LJ_Ele',
                                     'exclude_low_WpT_JetHT'
                                     )),
                     np.sqrt(2*LeadingEle.pt.sum()*Met.pt *
                             (1-np.cos(Met.T.delta_phi(LeadingEle.T.sum()))))
                     ),
            'tmucr': (selection.all(*('mu_triggers',
                                      'Met100',
                                      'one_muon',
                                      'zero_loose_electron',
                                      'zero_loose_photon',
                                      'atleast_2_medium_btag',
                                      'Delta_Phi_Met_LJ',
                                      'DeltaR_LJ_Mu',
                                      'exclude_low_WpT_JetHT'
                                      )),
                      np.sqrt(2*LeadingMu.pt.sum()*Met.pt *
                              (1-np.cos(Met.T.delta_phi(LeadingMu.T.sum()))))
                      ),
            'secr': (selection.all(*('ele_triggers',
                                     'Met100',
                                     'one_electron',
                                     'zero_loose_muon',
                                     'zero_loose_photon',
                                     'exactly_1_medium_btag',
                                     'Delta_Phi_Met_LJ',
                                     'DeltaR_LJ_Ele',
                                     'exclude_low_WpT_JetHT'
                                     )),
                     np.sqrt(2*LeadingEle.pt.sum()*Met.pt *
                             (1-np.cos(Met.T.delta_phi(LeadingEle.T.sum()))))
                     ),
            'smucr': (selection.all(*('mu_triggers',
                                      'Met100',
                                      'one_muon',
                                      'zero_loose_electron',
                                      'zero_loose_photon',
                                      'exactly_1_medium_btag',
                                      'Delta_Phi_Met_LJ',
                                      'DeltaR_LJ_Mu',
                                      'exclude_low_WpT_JetHT'
                                      )),
                      np.sqrt(2*LeadingMu.pt.sum()*Met.pt *
                              (1-np.cos(Met.T.delta_phi(LeadingMu.T.sum()))))
                      )


        }

# ******************* CUTFLOW **************************

#         all_true = np.full(events.size, True, dtype=np.bool)
#         selection.add('all_true', all_true)

#         output['cutflow']['all events'] += events.size
#         output['cutflow']['tight_e'] += TightEleSel.any().sum()
#         output['cutflow']['tight_mu'] += TightMuSel.any().sum()
#         output['cutflow']['loose_e'] += LooseEleSel.any().sum()
#         output['cutflow']['loose_mu'] += LooseMuonSel.any().sum()
#         output['cutflow']['met>100'] += TightMet.any().sum()

#         output['cutflow']['met_triggers'] += events[selection.all(
#             *('all_true', 'met_triggers'))].size
#         output['cutflow']['ele_triggers'] += events[selection.all(
#             *('all_true', 'ele_triggers'))].size
#         output['cutflow']['mu_triggers'] += events[selection.all(
#             *('all_true', 'mu_triggers'))].size
#         output['cutflow']['Delta_Phi_Met_LJ'] += events[selection.all(
#             *('all_true', 'Delta_Phi_Met_LJ'))].size
#         output['cutflow']['DeltaR_LJ_Mu'] += events[selection.all(
#             *('all_true', 'DeltaR_LJ_Mu'))].size
#         output['cutflow']['exactly_1_medium_btag'] += events[selection.all(
#             *('all_true', 'exactly_1_medium_btag'))].size
#         output['cutflow']['DeltaR_LJ_Ele'] += events[selection.all(
#             *('all_true', 'DeltaR_LJ_Ele'))].size

        for reg, sel_mt in region.items():
            output['mT'].fill(dataset=dataset,
                              region=reg,
                              mT=sel_mt[1][sel_mt[0]].flatten())
            output['eT_miss'].fill(dataset=dataset,
                                   region=reg,
                                   eT=Met[sel_mt[0]].pt.flatten())
            output['ele_pT'].fill(dataset=dataset,
                                  region=reg,
                                  pT=TightElectron[sel_mt[0]].pt.flatten())
            output['mu_pT'].fill(dataset=dataset,
                                 region=reg,
                                 pT=TightMuon[sel_mt[0]].pt.flatten()),
            # data condition
            if 'genWeight' in events.columns:
              if not isFilled:
                output['sumw'].fill(dataset=dataset, sumw=1, weight=events.genWeight.sum())
                isFilled=True
    #               print(reg,'->events.genWeight.sum()->',events.genWeight.sum())
    #               print(reg,'->events.size->',events.size,'\n')

    #               print("lumi:", self._lumi,
    #                    "sxec:", self._xsec[dataset],
    #                    "lumi*xs", )
            else:
              if not isFilled:
                output['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                isFilled=True

        return output

    
    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            print('Scaling:',d.name)
            dataset = d.name
            if '--' in dataset: dataset = dataset.split('--')[1]
            print('Cross section:',self._xsec[dataset])
            if self._xsec[dataset]!= -1: scale[d.name] = self._lumi*self._xsec[dataset]
            else: scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw': continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')
        print(scale)
        return accumulator


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    (options, args) = parser.parse_args()

    with open('metadata/'+options.year+'.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k, v in samplefiles.items()}

#    corrections = load('data/corrections.coffea')
#    ids         = load('data/ids.coffea')
#    common      = load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year, xsec=xsec)
#    processor_instance=AnalysisProcessor(year=options.year,
#                                         xsec=xsec,
#                                         corrections=corrections,
#                                         ids=ids,
#                                         common=common)

    save(processor_instance, 'data/lep_lowWpT'+options.year+'.processor')
    print("processor have been cretaed inside folder data, the name of the processor is lep_lowWpT{}.processor".format(options.year))
