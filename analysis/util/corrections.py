#!/usr/bin/env python
import uproot3
import numpy as np
import os
from coffea import hist, lookup_tools
# from coffea.lookup_tools import extractor, dense_lookup
from coffea import util
from coffea.lookup_tools import dense_lookup
from coffea.util import save, load
from coffea.btag_tools import BTagScaleFactor
import awkward as ak
from coffea.lookup_tools import extractor
###
# Pile-up weight
###

pu_files = {
    '2018': uproot.open("data/pileup/PileupHistograms_2018_69mb_pm5.root"),
    '2017': uproot.open("data/pileup/PileupHistograms_2017_69mb_pm5.root"),
    '2016': uproot.open("data/pileup/PileupHistograms_2016_69mb_pm5.root")
}
get_pu_weight = {}
for year in ['2016','2017','2018']:
    pu_hist=pu_files[year]['pu_weights_central']
    get_pu_weight[year] = lookup_tools.dense_lookup.dense_lookup(pu_hist.values, pu_hist.edges)
# https://github.com/nsmith-/coffeandbacon/blob/master/analysis/compile_corrections.py#L194-L236

# pu_files = {
#     '2018': uproot3.open("data/pileup/UL/PileupHistogram-goldenJSON-13tev-2016-69200ub-99bins.root"),
#     '2017': uproot3.open("data/pileup/UL/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root"),
#     '2016': uproot3.open("data/pileup/UL/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root")
# }
# get_pu_weight = {}
# for year in ['2016','2017','2018']:
#     pu_hist=pu_files[year]['pileup']
# # with uproot3.open("new_coffea_nbs/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root") as fin_pileup:
#     norm = lambda x: x / x.sum()
#     data_pu = norm(pu_hist.values)
# #     data_pu_puUp = norm(fin_pileup["pileup_plus"].values)
# #     data_pu_puDown = norm(fin_pileup["pileup_minus"].values)

# # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2016_25ns_UltraLegacy_PoissonOOTPU_cfi.py
# # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py
# # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py

#     if year == '2018':
#         mc_pu = np.array([
#         8.89374611122e-07, 1.1777062868e-05, 3.99725585118e-05, 0.000129888015252, 0.000265224848687,
#         0.000313088635109, 0.000353781668514, 0.000508787237162, 0.000873670065767, 0.00147166880932,
#         0.00228230649018, 0.00330375581273, 0.00466047608406, 0.00624959203029, 0.00810375867901,
#         0.010306521821, 0.0129512453978, 0.0160303925502, 0.0192913204592, 0.0223108613632,
#         0.0249798930986, 0.0273973789867, 0.0294402350483, 0.031029854302, 0.0324583524255,
#         0.0338264469857, 0.0351267479019, 0.0360320204259, 0.0367489568401, 0.0374133183052,
#         0.0380352633799, 0.0386200967002, 0.039124376968, 0.0394201612616, 0.0394673457109,
#         0.0391705388069, 0.0384758587461, 0.0372984548399, 0.0356497876549, 0.0334655175178,
#         0.030823567063, 0.0278340752408, 0.0246009685048, 0.0212676009273, 0.0180250593982,
#         0.0149129830776, 0.0120582333486, 0.00953400069415, 0.00738546929512, 0.00563442079939,
#         0.00422052915668, 0.00312446316347, 0.00228717533955, 0.00164064894334, 0.00118425084792,
#         0.000847785826565, 0.000603466454784, 0.000419347268964, 0.000291768785963, 0.000199761337863,
#         0.000136624574661, 9.46855200945e-05, 6.80243180179e-05, 4.94806013765e-05, 3.53122628249e-05,
#         2.556765786e-05, 1.75845711623e-05, 1.23828210848e-05, 9.31669724108e-06, 6.0713272037e-06,
#         3.95387384933e-06, 2.02760874107e-06, 1.22535149516e-06, 9.79612472109e-07, 7.61730246474e-07,
#         4.2748847738e-07, 2.41170461205e-07, 1.38701083552e-07, 3.37678010922e-08, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0
#         ])
#     elif year=='2017':
#         mc_pu = np.array([
#         1.1840841518e-05, 3.46661037703e-05, 8.98772521472e-05, 7.47400487733e-05, 0.000123005176624,
#         0.000156501700614, 0.000154660478659, 0.000177496185603, 0.000324149805611, 0.000737524009713,
#         0.00140432980253, 0.00244424508696, 0.00380027898037, 0.00541093042612, 0.00768803501793,
#         0.010828224552, 0.0146608623707, 0.01887739113, 0.0228418813823, 0.0264817796874,
#         0.0294637401336, 0.0317960986171, 0.0336645950831, 0.0352638818387, 0.036869429333,
#         0.0382797316998, 0.039386705577, 0.0398389681346, 0.039646211131, 0.0388392805703,
#         0.0374195678161, 0.0355377892706, 0.0333383902828, 0.0308286549265, 0.0282914440969,
#         0.0257860718304, 0.02341635055, 0.0213126338243, 0.0195035612803, 0.0181079838989,
#         0.0171991315458, 0.0166377598339, 0.0166445341361, 0.0171943735369, 0.0181980997278,
#         0.0191339792146, 0.0198518804356, 0.0199714909193, 0.0194616474094, 0.0178626975229,
#         0.0153296785464, 0.0126789254325, 0.0100766041988, 0.00773867100481, 0.00592386091874,
#         0.00434706240169, 0.00310217013427, 0.00213213401899, 0.0013996000761, 0.000879148859271,
#         0.000540866009427, 0.000326115560156, 0.000193965828516, 0.000114607606623, 6.74262828734e-05,
#         3.97805301078e-05, 2.19948704638e-05, 9.72007976207e-06, 4.26179259146e-06, 2.80015581327e-06,
#         1.14675436465e-06, 2.52452411995e-07, 9.08394910044e-08, 1.14291987912e-08, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0
#         ])
#     else:
#         mc_pu = np.array([
#         1.00402360149e-05, 5.76498797172e-05, 7.37891400294e-05, 0.000110932895295, 0.000158857714773,
#         0.000368637432599, 0.000893114107873, 0.00189700774575, 0.00358880167437, 0.00636052573486,
#         0.0104173961179, 0.0158122597405, 0.0223785660712, 0.0299186888073, 0.0380275944896,
#         0.0454313901624, 0.0511181088317, 0.0547434577348, 0.0567906239028, 0.0577145461461,
#         0.0578176902735, 0.0571251566494, 0.0555456541498, 0.053134383488, 0.0501519041462,
#         0.0466815838899, 0.0429244592524, 0.0389566776898, 0.0348507152776, 0.0307356862528,
#         0.0267712092206, 0.0229720184534, 0.0193388653099, 0.0159602510813, 0.0129310510552,
#         0.0102888654183, 0.00798782770975, 0.00606651703058, 0.00447820948367, 0.00321589786478,
#         0.0022450422045, 0.00151447388514, 0.000981183695515, 0.000609670479759, 0.000362193408119,
#         0.000211572646801, 0.000119152364744, 6.49133515399e-05, 3.57795801581e-05, 1.99043569043e-05,
#         1.13639319832e-05, 6.49624103579e-06, 3.96626216416e-06, 2.37910222874e-06, 1.50997403362e-06,
#         1.09816650247e-06, 7.31298519122e-07, 6.10398791529e-07, 3.74845774388e-07, 2.65177281359e-07,
#         2.01923536742e-07, 1.39347583555e-07, 8.32600052913e-08, 6.04932421298e-08, 6.52536630583e-08,
#         5.90574603808e-08, 2.29162474068e-08, 1.97294602668e-08, 1.7731096903e-08, 3.57547932012e-09,
#         1.35039815662e-09, 8.50071242076e-09, 5.0279187473e-09, 4.93736669066e-10, 8.13919708923e-10,
#         5.62778926097e-09, 5.15140589469e-10, 8.21676746568e-10, 0.0, 1.49166873577e-09,
#         8.43517992503e-09, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0
#         ])
        
#     mask = mc_pu > 0.
#     corr = data_pu.copy()
# #     corr_puUp = data_pu_puUp.copy()
# #     corr_puDown = data_pu_puDown.copy()
#     corr[mask] /= mc_pu[mask]
# #     corr_puUp[mask] /= mc_pu[mask]
# #     corr_puDown[mask] /= mc_pu[mask]
#     pileup_corr = lookup_tools.dense_lookup.dense_lookup(corr, pu_hist.edges)
#     get_pu_weight[year] = pileup_corr
###
# MET trigger efficiency SFs, 2017/18 from monojet. Depends on recoil.
###

met_trig_hists = {
    '2016': uproot3.open("data/trigger_eff/metTriggerEfficiency_recoil_monojet_TH1F.root")['hden_monojet_recoil_clone_passed'],
    '2017': uproot3.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2017'],
    '2018': uproot3.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2018']
}
get_met_trig_weight = {}
for year in ['2016','2017','2018']:
    met_trig_hist=met_trig_hists[year]
    get_met_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(met_trig_hist.values, met_trig_hist.edges)

###
# MET z->mumu efficiency SF. 2017/18 using 1m as done in monojet, 2m used only for systematics. Depends on recoil.
###

# zmm_trig_hists ={
#     '2016': uproot.open("data/trigger_eff/metTriggerEfficiency_zmm_recoil_monojet_TH1F.root")['hden_monojet_recoil_clone_passed'],
#     '2017': uproot.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2017'],
#     '2018': uproot.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2018']
# }
# get_met_zmm_trig_weight = {}
# for year in ['2016','2017','2018']:
#     zmm_trig_hist = zmm_trig_hists[year]
#     get_met_zmm_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(zmm_trig_hist.values, zmm_trig_hist.edges)

###
# Electron trigger efficiency SFs. depends on supercluster eta and pt:
###

ele_trig_hists = {
    '2016': uproot3.open("data/trigger_eff/eleTrig.root")['hEffEtaPt'],
    '2017': uproot3.open("data/trigger_eff/electron_trigger_sf_2017.root")['EGamma_SF2D'],#monojet measurement for the combined trigger path
    '2018': uproot3.open("data/trigger_eff/electron_trigger_sf_2018.root")['EGamma_SF2D'] #approved by egamma group: https://indico.cern.ch/event/924522/
}
get_ele_trig_weight = {}
for year in ['2016','2017','2018']:
    ele_trig_hist = ele_trig_hists[year]
    get_ele_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(ele_trig_hist.values, ele_trig_hist.edges)
    

get_ele_trig_err = {}
for year in ['2016','2017','2018']:
    ele_trig_hist = ele_trig_hists[year]
    get_ele_trig_err[year] = lookup_tools.dense_lookup.dense_lookup(ele_trig_hist.variances ** 0.5, ele_trig_hist.edges)
    

###
# Muon trigger efficiency SFs. depends on supercluster eta and pt:
###

# mu_trig_hists = {
#     '2016': uproot.open("data/trigger_eff/SingleMuTriggerEfficienciesAndSF_2016_RunBtoH.root")['IsoMu24_OR_IsoTkMu24_PtEtaBins']['pt_abseta_ratio'],
#     '2017': uproot.open("data/trigger_eff/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root")['IsoMu27_PtEtaBins']['pt_abseta_ratio'],
#     '2018': uproot.open("data/trigger_eff/SingleMuTriggerEfficienciesAndSF_2018_RunAtoD_kr.root")['IsoMu24_PtEtaBins']['pt_abseta_ratio']
# }
# get_mu_trig_weight = {}
# for year in ['2016','2017','2018']:
#     mu_trig_hist = mu_trig_hists[year]
#     get_mu_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(mu_trig_hist.values, mu_trig_hist.edges)
    
    
mu_trig_hists_preVFP = {
    '2016': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2016_preVFP/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.root")['NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt'],
    '2017': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.root")['NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt'],
    '2018': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root")['NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt']
}
get_mu_trig_weight_preVFP = {}
for year in ['2018', '2017', '2016']:
    mu_trig_hist = mu_trig_hists_preVFP[year]
    get_mu_trig_weight_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_trig_hist.values, mu_trig_hist.edges)
    
get_mu_trig_err_preVFP = {}
for year in ['2018', '2017', '2016']:
    mu_trig_hist = mu_trig_hists_preVFP[year]
    get_mu_trig_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_trig_hist.variances  ** 0.5, mu_trig_hist.edges)
    
mu_trig_hists_postVFP = {
    '2016': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2016_postVFP/Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.root")['NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt'],
    '2017': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.root")['NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt'],
    '2018': uproot3.open("data/trigger_eff_UL/UL_SingleMuon/2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root")['NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt']
}
get_mu_trig_weight_postVFP = {}
for year in ['2018', '2017', '2016']:
    mu_trig_hist = mu_trig_hists_preVFP[year]
    get_mu_trig_weight_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_trig_hist.values, mu_trig_hist.edges)
    
get_mu_trig_err_postVFP = {}
for year in ['2018', '2017', '2016']:
    mu_trig_hist = mu_trig_hists_postVFP[year]
    get_mu_trig_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_trig_hist.variances  ** 0.5, mu_trig_hist.edges)
###
# Photon trigger efficiency SFs. 2017/18 not actually used, sigmoid is used instead.
###

pho_trig_files = {
    '2016': uproot3.open("data/trigger_eff/photonTriggerEfficiency_photon_TH1F.root"),
    "2017": uproot3.open("data/trigger_eff/photonTriggerEfficiency_photon_TH1F.root"),
    "2018": uproot3.open("data/trigger_eff/photonTriggerEfficiency_photon_TH1F.root")
}
get_pho_trig_weight = {}
for year in ['2016','2017','2018']:
    pho_trig_hist = pho_trig_files[year]["hden_photonpt_clone_passed"]
    get_pho_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(pho_trig_hist.values, pho_trig_hist.edges)

###
# Electron id SFs. 2017/18 used dedicated weights from monojet. depends on supercluster eta and pt.
###

# ele_loose_files = {
#     '2016': uproot.open("data/ScaleFactor/2016_ElectronWPVeto_Fall17V2.root"),
#     '2017': uproot.open("data/ScaleFactor/2017_ElectronWPVeto_Fall17V2_BU.root"),
#     '2018': uproot.open("data/ScaleFactor/2018_ElectronWPVeto_Fall17V2_BU.root")
# }
# ele_tight_files = {
#     '2016': uproot.open("data/ScaleFactor/2016LegacyReReco_ElectronTight_Fall17V2.root"),
#     '2017': uproot.open("data/ScaleFactor/2017_ElectronTight_Fall17V2_BU.root"),
#     '2018': uproot.open("data/ScaleFactor/2018_ElectronTight_Fall17V2_BU.root")
# }
# get_ele_loose_id_sf = {}
# get_ele_tight_id_sf = {}
# for year in ['2016','2017','2018']:
#     ele_loose_sf_hist = ele_loose_files[year]["EGamma_SF2D"]
#     get_ele_loose_id_sf[year]  = lookup_tools.dense_lookup.dense_lookup(ele_loose_sf_hist.values, ele_loose_sf_hist.edges)
#     ele_tight_sf_hist =ele_tight_files[year]["EGamma_SF2D"]
#     get_ele_tight_id_sf[year]  = lookup_tools.dense_lookup.dense_lookup(ele_tight_sf_hist.values, ele_tight_sf_hist.edges)

###
# Electron id SFs. 16/17/18 . depends on supercluster eta and pt.
# https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018#SFs_for_Electrons_UL_2017
###
#preVFP
ele_loose_files_preVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_pVFP_id_loose_egammaEffi.txt_Ele_Loose_preVFP_EGM2D.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_loose_egammaEffi.txt_EGM2D_Loose_UL17.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_loose_egammaEffi.txt_Ele_Loose_EGM2D.root")
}
ele_tight_files_preVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_pVFP_id_tight_egammaEffi.txt_Ele_Tight_preVFP_EGM2D.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_tight_egammaEffi.txt_EGM2D_Tight_UL17.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_tight_egammaEffi.txt_Ele_Tight_EGM2D.root")
}
get_ele_loose_id_sf_preVFP = {}
get_ele_tight_id_sf_preVFP = {}
for year in ['2016','2017','2018']:
    ele_loose_sf_hist = ele_loose_files_preVFP[year]["EGamma_SF2D"]
    get_ele_loose_id_sf_preVFP[year]  = lookup_tools.dense_lookup.dense_lookup(ele_loose_sf_hist.values, ele_loose_sf_hist.edges)
    
    ele_tight_sf_hist =ele_tight_files_preVFP[year]["EGamma_SF2D"]
    get_ele_tight_id_sf_preVFP[year]  = lookup_tools.dense_lookup.dense_lookup(ele_tight_sf_hist.values, ele_tight_sf_hist.edges)
    
get_ele_loose_id_err_preVFP = {}
for year in ['2016','2017','2018']:
    ele_id_hist = ele_loose_files_preVFP[year]["EGamma_SF2D"]
    get_ele_loose_id_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(ele_id_hist.variances ** 0.5, ele_trig_hist.edges)
    
get_ele_tight_id_err_preVFP = {}
for year in ['2016','2017','2018']:
    ele_id_hist = ele_tight_files_preVFP[year]["EGamma_SF2D"]
    get_ele_tight_id_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(ele_id_hist.variances ** 0.5, ele_trig_hist.edges)
     
#postVFP
ele_loose_files_postVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_id_loose_egammaEffi.txt_Ele_Loose_postVFP_EGM2D.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_loose_egammaEffi.txt_EGM2D_Loose_UL17.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_loose_egammaEffi.txt_Ele_Loose_EGM2D.root")
}
ele_tight_files_postVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_id_tight_egammaEffi.txt_Ele_Tight_postVFP_EGM2D.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_tight_egammaEffi.txt_EGM2D_Tight_UL17.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_tight_egammaEffi.txt_Ele_Tight_EGM2D.root")
}
get_ele_loose_id_sf_postVFP = {}
get_ele_tight_id_sf_postVFP = {}
for year in ['2016','2017','2018']:
    ele_loose_sf_hist = ele_loose_files_postVFP[year]["EGamma_SF2D"]
    get_ele_loose_id_sf_postVFP[year]  = lookup_tools.dense_lookup.dense_lookup(ele_loose_sf_hist.values, ele_loose_sf_hist.edges)
    
    ele_tight_sf_hist =ele_tight_files_postVFP[year]["EGamma_SF2D"]
    get_ele_tight_id_sf_postVFP[year]  = lookup_tools.dense_lookup.dense_lookup(ele_tight_sf_hist.values, ele_tight_sf_hist.edges)
    
get_ele_loose_id_err_postVFP = {}
for year in ['2016','2017','2018']:
    ele_id_hist = ele_loose_files_postVFP[year]["EGamma_SF2D"]
    get_ele_loose_id_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(ele_id_hist.variances ** 0.5, ele_trig_hist.edges)
    
get_ele_tight_id_err_postVFP = {}
for year in ['2016','2017','2018']:
    ele_id_hist = ele_tight_files_postVFP[year]["EGamma_SF2D"]
    get_ele_tight_id_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(ele_id_hist.variances ** 0.5, ele_trig_hist.edges)

###
# Electron reconstruction SFs. Depends on supercluster eta and pt.    
###

# ele_reco_files = {
#     '2016': uproot.open("data/ScaleFactor/EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root"),
#     '2017': uproot.open("data/ScaleFactor/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO.root"),
#     '2018': uproot.open("data/ScaleFactor/2018_egammaEffi_txt_EGM2D_updatedAll.root")
# }
# get_ele_reco_sf = {}
# for year in ['2016','2017','2018']:
#     ele_reco_hist = ele_reco_files[year]["EGamma_SF2D"]
#     get_ele_reco_sf[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)
# #2017 has a separate set of weights for low pt electrons (pt<20).
# ele_reco_lowet_hist = uproot.open("data/ScaleFactor/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO_lowEt.root")['EGamma_SF2D']
# get_ele_reco_lowet_sf=lookup_tools.dense_lookup.dense_lookup(ele_reco_lowet_hist.values, ele_reco_lowet_hist.edges)


###
# Electron reconstruction SFs. Depends on supercluster eta and pt.  UL
#https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018#SFs_for_Electrons_UL_2017
###
ele_reco_files_preVFP_below20 = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_pVFP_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2016preVFP.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2017.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2018.root")
}
get_ele_reco_sf_preVFP_below20 = {}
get_ele_reco_err_preVFP_below20 = {}
for year in ['2016','2017','2018']:
    ele_reco_hist = ele_reco_files_preVFP_below20[year]["EGamma_SF2D"]
    get_ele_reco_sf_preVFP_below20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)
    get_ele_reco_err_preVFP_below20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.variances ** 0.5, ele_reco_hist.edges)


ele_reco_files_postVFP_below20 = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2017.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2018.root")
}
get_ele_reco_sf_postVFP_below20 = {}
get_ele_reco_err_postVFP_below20 = {}
for year in ['2016','2017','2018']:
    ele_reco_hist = ele_reco_files_postVFP_below20[year]["EGamma_SF2D"]
    get_ele_reco_sf_postVFP_below20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)
    get_ele_reco_err_postVFP_below20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.variances ** 0.5, ele_reco_hist.edges)

ele_reco_files_postVFP_above20 = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_reco_egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_reco_egammaEffi_ptAbove20.txt_EGM2D_UL2017.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_reco_egammaEffi_ptAbove20.txt_EGM2D_UL2018.root")
}
get_ele_reco_sf_postVFP_above20 = {}
get_ele_reco_err_postVFP_above20 = {}
for year in ['2016','2017','2018']:
    ele_reco_hist = ele_reco_files_postVFP_above20[year]["EGamma_SF2D"]
    get_ele_reco_sf_postVFP_above20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)
    get_ele_reco_err_postVFP_above20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.variances ** 0.5, ele_reco_hist.edges)


ele_reco_files_preVFP_above20 = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_pVFP_reco_egammaEffi_ptAbove20.txt_EGM2D_UL2016preVFP.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_reco_egammaEffi_ptAbove20.txt_EGM2D_UL2017.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_reco_egammaEffi_ptAbove20.txt_EGM2D_UL2018.root")
}
get_ele_reco_sf_preVFP_above20 = {}
get_ele_reco_err_preVFP_above20 = {}
for year in ['2016','2017','2018']:
    ele_reco_hist = ele_reco_files_preVFP_above20[year]["EGamma_SF2D"]
    get_ele_reco_sf_preVFP_above20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)
    get_ele_reco_err_preVFP_above20[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.variances ** 0.5, ele_reco_hist.edges)
    


###
# Photon ID SFs. Tight photons use medium id. 2017/18 use dedicated measurement from monojet, depends only on abs(eta): https://indico.cern.ch/event/879924/
###

pho_tight_hists = {
    '2016': uproot3.open("data/ScaleFactor/Fall17V2_2016_Medium_photons.root")['EGamma_SF2D'],
    '2017': uproot3.open("data/ScaleFactor/photon_medium_id_sf_v0.root")['photon_medium_id_sf_2017'],
    '2018': uproot3.open("data/ScaleFactor/photon_medium_id_sf_v0.root")['photon_medium_id_sf_2018']
}
get_pho_tight_id_sf = {}
for year in ['2016','2017','2018']:
    pho_tight_hist=pho_tight_hists[year]
    get_pho_tight_id_sf[year] = lookup_tools.dense_lookup.dense_lookup(pho_tight_hist.values, pho_tight_hist.edges)

###
# Photon CSEV weight: https://twiki.cern.ch/twiki/bin/view/CMS/EgammaIDRecipesRun2#Electron_Veto_CSEV_or_pixel_seed
###

pho_csev_hists = {
    '2016': uproot3.open("data/ScaleFactor/ScalingFactors_80X_Summer16_rename.root")['Scaling_Factors_CSEV_R9_Inclusive'],
    '2017': uproot3.open("data/ScaleFactor/CSEV_ScaleFactors_2017.root")['Medium_ID'],
    '2018': uproot3.open("data/ScaleFactor/CSEV_2018.root")['eleVeto_SF'],
}
get_pho_csev_sf = {}
for year in ['2016','2017','2018']:
    pho_csev_hist=pho_csev_hists[year]
    get_pho_csev_sf[year] = lookup_tools.dense_lookup.dense_lookup(pho_csev_hist.values, pho_csev_hist.edges)

###
# Muon ID SFs
###

# mu_files = {
#     '2016': uproot.open("data/ScaleFactor/2016LegacyReReco_Muon_SF_ID.root"),
#     '2017': uproot.open("data/ScaleFactor/2017_Muon_RunBCDEF_SF_ID.root"),
#     '2018': uproot.open("data/ScaleFactor/2018_Muon_RunABCD_SF_ID.root")
# }
# mu_tight_hist = {
#     '2016': mu_files['2016']["NUM_TightID_DEN_genTracks_eta_pt"],
#     '2017': mu_files['2017']["NUM_TightID_DEN_genTracks_pt_abseta"],
#     '2018': mu_files['2018']["NUM_TightID_DEN_TrackerMuons_pt_abseta"]
# }
# mu_loose_hist = {
#     '2016': mu_files['2016']["NUM_LooseID_DEN_genTracks_eta_pt"],
#     '2017': mu_files['2017']["NUM_LooseID_DEN_genTracks_pt_abseta"],
#     '2018': mu_files['2018']["NUM_LooseID_DEN_TrackerMuons_pt_abseta"]
# }
# get_mu_tight_id_sf = {}
# get_mu_loose_id_sf = {}
# for year in ['2016','2017','2018']:
#     get_mu_tight_id_sf[year] = lookup_tools.dense_lookup.dense_lookup(mu_tight_hist[year].values, mu_tight_hist[year].edges)
#     get_mu_loose_id_sf[year] = lookup_tools.dense_lookup.dense_lookup(mu_loose_hist[year].values, mu_loose_hist[year].edges)

###
# Muon ID SFs
#https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018#ID_efficiencies_AN1
###

mu_files_preVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_preVFP_id_mu_Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_mu_Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_mu_Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root")
}
mu_tight_hist_preVFP = {
    '2016': mu_files_preVFP['2016']["NUM_TightID_DEN_TrackerMuons_abseta_pt"],
    '2017': mu_files_preVFP['2017']["NUM_TightID_DEN_TrackerMuons_abseta_pt"],
    '2018': mu_files_preVFP['2018']["NUM_TightID_DEN_TrackerMuons_abseta_pt"]
}
mu_loose_hist_preVFP = {
    '2016': mu_files_preVFP['2016']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"],
    '2017': mu_files_preVFP['2017']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"],
    '2018': mu_files_preVFP['2018']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"]
}
get_mu_tight_id_sf_preVFP = {}
get_mu_loose_id_sf_preVFP = {}
for year in ['2016','2018', '2017']:
    get_mu_tight_id_sf_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_tight_hist_preVFP[year].values, mu_tight_hist_preVFP[year].edges)
    get_mu_loose_id_sf_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_loose_hist_preVFP[year].values, mu_loose_hist_preVFP[year].edges)
    
get_mu_tight_id_err_preVFP = {}
get_mu_loose_id_err_preVFP = {}
for year in ['2016','2018', '2017']:
    get_mu_tight_id_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_tight_hist_preVFP[year].variances ** 0.5, mu_tight_hist_preVFP[year].edges)
    get_mu_loose_id_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_loose_hist_preVFP[year].variances ** 0.5, mu_loose_hist_preVFP[year].edges)

mu_files_postVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_id_mu_Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_id_mu_Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_id_mu_Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root")
}
mu_tight_hist_postVFP = {
    '2016': mu_files_postVFP['2016']["NUM_TightID_DEN_TrackerMuons_abseta_pt"],
    '2017': mu_files_postVFP['2017']["NUM_TightID_DEN_TrackerMuons_abseta_pt"],
    '2018': mu_files_postVFP['2018']["NUM_TightID_DEN_TrackerMuons_abseta_pt"]
}
mu_loose_hist_postVFP = {
    '2016': mu_files_postVFP['2016']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"],
    '2017': mu_files_postVFP['2017']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"],
    '2018': mu_files_postVFP['2018']["NUM_LooseID_DEN_TrackerMuons_abseta_pt"]
}
get_mu_tight_id_sf_postVFP = {}
get_mu_loose_id_sf_postVFP = {}
for year in ['2016','2018', '2017']:
    get_mu_tight_id_sf_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_tight_hist_postVFP[year].values, mu_tight_hist_postVFP[year].edges)
    get_mu_loose_id_sf_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_loose_hist_postVFP[year].values, mu_loose_hist_postVFP[year].edges) 

get_mu_tight_id_err_postVFP = {}
get_mu_loose_id_err_postVFP = {}
for year in ['2016','2018', '2017']:
    get_mu_tight_id_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_tight_hist_postVFP[year].variances, mu_tight_hist_preVFP[year].edges)
    get_mu_loose_id_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_loose_hist_postVFP[year].variances, mu_loose_hist_preVFP[year].edges)
###
# Muon isolation SFs
###

# mu_iso_files = {
#     '2016': uproot.open("data/ScaleFactor/Merged_SF_ISO.root"),
#     '2017': uproot.open("data/ScaleFactor/RunBCDEF_SF_ISO_syst.root"),
#     '2018': uproot.open("data/ScaleFactor/RunABCD_SF_ISO.root")
# }
# mu_iso_tight_hist = {
#     '2016': mu_iso_files['2016']["NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"],
#     '2017': mu_iso_files['2017']["NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"],
#     '2018': mu_iso_files['2018']["NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"]
# }
# mu_iso_loose_hist = {
#     '2016': mu_iso_files['2016']["NUM_LooseRelIso_DEN_LooseID_eta_pt"],
#     '2017': mu_iso_files['2017']["NUM_LooseRelIso_DEN_LooseID_pt_abseta"],
#     '2018': mu_iso_files['2018']["NUM_LooseRelIso_DEN_LooseID_pt_abseta"]
# }
# get_mu_tight_iso_sf = {}
# get_mu_loose_iso_sf = {}
# for year in ['2016','2017','2018']:
#     get_mu_tight_iso_sf[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_tight_hist[year].values, mu_iso_tight_hist[year].edges)
#     get_mu_loose_iso_sf[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_loose_hist[year].values, mu_iso_loose_hist[year].edges)

###
# Muon isolation SFs
#https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018#ID_efficiencies_AN1
###
#iso
mu_iso_files_preVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_preVFP_iso_mu_Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_iso_mu_Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_iso_mu_Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root")
}
mu_iso_tight_hist_preVFP = {
    '2016': mu_iso_files_preVFP['2016']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"],
    '2017': mu_iso_files_preVFP['2017']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"],
    '2018': mu_iso_files_preVFP['2018']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
}
mu_iso_loose_hist_preVFP = {
    '2016': mu_iso_files_preVFP['2016']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"],
    '2017': mu_iso_files_preVFP['2017']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"],
    '2018': mu_iso_files_preVFP['2018']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
}
get_mu_tight_iso_sf_preVFP = {}
get_mu_loose_iso_sf_preVFP = {}

get_mu_tight_iso_err_preVFP = {}
get_mu_loose_iso_err_preVFP = {}
for year in ['2016','2017','2018']:
    get_mu_tight_iso_sf_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_tight_hist_preVFP[year].values, mu_iso_tight_hist_preVFP[year].edges)
    get_mu_loose_iso_sf_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_loose_hist_preVFP[year].values, mu_iso_loose_hist_preVFP[year].edges)

    get_mu_tight_iso_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_tight_hist_preVFP[year].variances ** 0.5, mu_iso_tight_hist_preVFP[year].edges)
    get_mu_loose_iso_err_preVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_loose_hist_preVFP[year].variances ** 0.5, mu_iso_loose_hist_preVFP[year].edges)

#iso
mu_iso_files_postVFP = {
    '2016': uproot3.open("data/ScaleFactor_UL/2016_postVFP_iso_mu_Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root"),
    '2017': uproot3.open("data/ScaleFactor_UL/2017_iso_mu_Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root"),
    '2018': uproot3.open("data/ScaleFactor_UL/2018_iso_mu_Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root")
}
mu_iso_tight_hist_postVFP = {
    '2016': mu_iso_files_postVFP['2016']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"],
    '2017': mu_iso_files_postVFP['2017']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"],
    '2018': mu_iso_files_postVFP['2018']["NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
}
mu_iso_loose_hist_postVFP = {
    '2016': mu_iso_files_postVFP['2016']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"],
    '2017': mu_iso_files_postVFP['2017']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"],
    '2018': mu_iso_files_postVFP['2018']["NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
}
get_mu_tight_iso_sf_postVFP = {}
get_mu_loose_iso_sf_postVFP = {}

get_mu_tight_iso_err_postVFP = {}
get_mu_loose_iso_err_postVFP = {}

for year in ['2016','2017','2018']:
    get_mu_tight_iso_sf_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_tight_hist_postVFP[year].values, mu_iso_tight_hist_postVFP[year].edges)
    get_mu_loose_iso_sf_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_loose_hist_postVFP[year].values, mu_iso_loose_hist_postVFP[year].edges)

    get_mu_tight_iso_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_tight_hist_postVFP[year].variances ** 0.5, mu_iso_tight_hist_postVFP[year].edges)
    get_mu_loose_iso_err_postVFP[year] = lookup_tools.dense_lookup.dense_lookup(mu_iso_loose_hist_postVFP[year].variances ** 0.5, mu_iso_loose_hist_postVFP[year].edges)

###
# V+jets NLO k-factors
###

nlo_qcd_hists = {
    '2016':{
        'dy': uproot3.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_qcd"],
        'w': uproot3.open("data/vjets_SFs/merged_kfactors_wjets.root")["kfactor_monojet_qcd"],
        'z': uproot3.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_qcd"],
        'a': uproot3.open("data/vjets_SFs/merged_kfactors_gjets.root")["kfactor_monojet_qcd"]
    },
    '2017':{
        'z': uproot3.open("data/vjets_SFs/SF_QCD_NLO_ZJetsToNuNu.root")["kfac_znn_filter"],
        'w': uproot3.open("data/vjets_SFs/SF_QCD_NLO_WJetsToLNu.root")["wjet_dress_monojet"],
        'dy': uproot3.open("data/vjets_SFs/SF_QCD_NLO_DYJetsToLL.root")["kfac_dy_filter"],
        'a': uproot3.open("data/vjets_SFs/SF_QCD_NLO_GJets.root")["gjets_stat1_monojet"]
    },
    '2018':{
        'z': uproot3.open("data/vjets_SFs/SF_QCD_NLO_ZJetsToNuNu.root")["kfac_znn_filter"],
        'w': uproot3.open("data/vjets_SFs/SF_QCD_NLO_WJetsToLNu.root")["wjet_dress_monojet"],
        'dy': uproot3.open("data/vjets_SFs/SF_QCD_NLO_DYJetsToLL.root")["kfac_dy_filter"],
        'a': uproot3.open("data/vjets_SFs/SF_QCD_NLO_GJets.root")["gjets_stat1_monojet"]
    }
}
nlo_ewk_hists = {
    'dy': uproot3.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_ewk"],
    'w': uproot3.open("data/vjets_SFs/merged_kfactors_wjets.root")["kfactor_monojet_ewk"],
    'z': uproot3.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_ewk"],
    'a': uproot3.open("data/vjets_SFs/merged_kfactors_gjets.root")["kfactor_monojet_ewk"]
}    
get_nlo_qcd_weight = {}
get_nlo_ewk_weight = {}
for year in ['2016','2017','2018']:
    get_nlo_qcd_weight[year] = {}
    get_nlo_ewk_weight[year] = {}
    for p in ['dy','w','z','a']:
        get_nlo_qcd_weight[year][p] = lookup_tools.dense_lookup.dense_lookup(nlo_qcd_hists[year][p].values, nlo_qcd_hists[year][p].edges)
        get_nlo_ewk_weight[year][p] = lookup_tools.dense_lookup.dense_lookup(nlo_ewk_hists[p].values, nlo_ewk_hists[p].edges)

###
# V+jets NNLO weights
# The schema is process_NNLO_NLO_QCD1QCD2QCD3_EW1EW2EW3_MIX, where 'n' stands for 'nominal', 'u' for 'up', and 'd' for 'down'
###

histname={
    'dy': 'eej_NNLO_NLO_',
    'w':  'evj_NNLO_NLO_',
    'z': 'vvj_NNLO_NLO_',
    'a': 'aj_NNLO_NLO_'
}
correlated_variations = {
    'cen':    'nnn_nnn_n',
    'qcd1up': 'unn_nnn_n',
    'qcd1do': 'dnn_nnn_n',
    'qcd2up': 'nun_nnn_n',
    'qcd2do': 'ndn_nnn_n',
    'qcd3up': 'nnu_nnn_n',
    'qcd3do': 'nnd_nnn_n',
    'ew1up' : 'nnn_unn_n',
    'ew1do' : 'nnn_dnn_n',
    'mixup' : 'nnn_nnn_u',
    'mixdo' : 'nnn_nnn_d',
    'muFup' : 'nnn_nnn_n_Weight_scale_variation_muR_1p0_muF_2p0',
    'muFdo' : 'nnn_nnn_n_Weight_scale_variation_muR_1p0_muF_0p5',
    'muRup' : 'nnn_nnn_n_Weight_scale_variation_muR_2p0_muF_1p0',
    'muRdo' : 'nnn_nnn_n_Weight_scale_variation_muR_0p5_muF_1p0'
}
uncorrelated_variations = {
    'dy': {
        'ew2Gup': 'nnn_nnn_n',
        'ew2Gdo': 'nnn_nnn_n',
        'ew2Wup': 'nnn_nnn_n',
        'ew2Wdo': 'nnn_nnn_n',
        'ew2Zup': 'nnn_nun_n',
        'ew2Zdo': 'nnn_ndn_n',
        'ew3Gup': 'nnn_nnn_n',
        'ew3Gdo': 'nnn_nnn_n',
        'ew3Wup': 'nnn_nnn_n',
        'ew3Wdo': 'nnn_nnn_n',
        'ew3Zup': 'nnn_nnu_n',
        'ew3Zdo': 'nnn_nnd_n'
    },
    'w': {
        'ew2Gup': 'nnn_nnn_n',
        'ew2Gdo': 'nnn_nnn_n',
        'ew2Wup': 'nnn_nun_n',
        'ew2Wdo': 'nnn_ndn_n',
        'ew2Zup': 'nnn_nnn_n',
        'ew2Zdo': 'nnn_nnn_n',
        'ew3Gup': 'nnn_nnn_n',
        'ew3Gdo': 'nnn_nnn_n',
        'ew3Wup': 'nnn_nnu_n',
        'ew3Wdo': 'nnn_nnd_n',
        'ew3Zup': 'nnn_nnn_n',
        'ew3Zdo': 'nnn_nnn_n'
    },
    'z': {
        'ew2Gup': 'nnn_nnn_n',
        'ew2Gdo': 'nnn_nnn_n',
        'ew2Wup': 'nnn_nnn_n',
        'ew2Wdo': 'nnn_nnn_n',
        'ew2Zup': 'nnn_nun_n',
        'ew2Zdo': 'nnn_ndn_n',
        'ew3Gup': 'nnn_nnn_n',
        'ew3Gdo': 'nnn_nnn_n',
        'ew3Wup': 'nnn_nnn_n',
        'ew3Wdo': 'nnn_nnn_n',
        'ew3Zup': 'nnn_nnu_n',
        'ew3Zdo': 'nnn_nnd_n'
    },
    'a': {
        'ew2Gup': 'nnn_nun_n',
        'ew2Gdo': 'nnn_ndn_n',
        'ew2Wup': 'nnn_nnn_n',
        'ew2Wdo': 'nnn_nnn_n',
        'ew2Zup': 'nnn_nnn_n',
        'ew2Zdo': 'nnn_nnn_n',
        'ew3Gup': 'nnn_nnu_n',
        'ew3Gdo': 'nnn_nnd_n',
        'ew3Wup': 'nnn_nnn_n',
        'ew3Wdo': 'nnn_nnn_n',
        'ew3Zup': 'nnn_nnn_n',
        'ew3Zdo': 'nnn_nnn_n'
    }
}
get_nnlo_nlo_weight = {}
for year in ['2016','2017','2018']:
    get_nnlo_nlo_weight[year] = {}
    nnlo_file = {
        'dy': uproot3.open("data/Vboson_Pt_Reweighting/"+year+"/TheoryXS_eej_madgraph_"+year+".root"),
        'w': uproot3.open("data/Vboson_Pt_Reweighting/"+year+"/TheoryXS_evj_madgraph_"+year+".root"),
        'z': uproot3.open("data/Vboson_Pt_Reweighting/"+year+"/TheoryXS_vvj_madgraph_"+year+".root"),
        'a': uproot3.open("data/Vboson_Pt_Reweighting/"+year+"/TheoryXS_aj_madgraph_"+year+".root")
    }
    for p in ['dy','w','z','a']:
        get_nnlo_nlo_weight[year][p] = {}
        for cv in correlated_variations:
            hist=nnlo_file[p][histname[p]+correlated_variations[cv]]
            get_nnlo_nlo_weight[year][p][cv]=lookup_tools.dense_lookup.dense_lookup(hist.values, hist.edges)
        for uv in uncorrelated_variations[p]:
            hist=nnlo_file[p][histname[p]+uncorrelated_variations[p][uv]]
            get_nnlo_nlo_weight[year][p][uv]=lookup_tools.dense_lookup.dense_lookup(hist.values, hist.edges)

def get_ttbar_weight(pt):
    return np.exp(0.0615 - 0.0005 * np.clip(pt, 0, 800))

def get_msd_weight(pt, eta):
    gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
    cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
    fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
    genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
    ptpow = np.power.outer(pt, np.arange(cpar.size))
    cenweight = np.dot(ptpow, cpar)
    forweight = np.dot(ptpow, fpar)
    weight = np.where(np.abs(eta)<1.3, cenweight, forweight)
    return genw*weight


def get_ecal_bad_calib(run_number, lumi_number, event_number, year, dataset):
    bad = {}
    bad["2016"] = {}
    bad["2017"] = {}
    bad["2018"] = {}
    bad["2016"]["MET"]            = "ecalBadCalib/Run2016_MET.root"
    bad["2016"]["SinglePhoton"]   = "ecalBadCalib/Run2016_SinglePhoton.root"
    bad["2016"]["SingleElectron"] = "ecalBadCalib/Run2016_SingleElectron.root"
    bad["2017"]["MET"]            = "ecalBadCalib/Run2017_MET.root"
    bad["2017"]["SinglePhoton"]   = "ecalBadCalib/Run2017_SinglePhoton.root"
    bad["2017"]["SingleElectron"] = "ecalBadCalib/Run2017_SingleElectron.root"
    bad["2018"]["MET"]            = "ecalBadCalib/Run2018_MET.root"
    bad["2018"]["EGamma"]         = "ecalBadCalib/Run2018_EGamma.root"
    
    regular_dataset = ""
    regular_dataset = [name for name in ["MET","SinglePhoton","SingleElectron","EGamma"] if (name in dataset)]
    fbad = uproot3.open(bad[year][regular_dataset[0]])
    bad_tree = fbad["vetoEvents"]
    runs_to_veto = bad_tree.array("Run")
    lumis_to_veto = bad_tree.array("LS")
    events_to_veto = bad_tree.array("Event")

    # We want events that do NOT have (a vetoed run AND a vetoed LS and a vetoed event number)
    return np.logical_not(np.isin(run_number, runs_to_veto) * np.isin(lumi_number, lumis_to_veto) * np.isin(event_number, events_to_veto))

class BTagCorrectorPreVFP:

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        files = {
           'deepflav': {
               '2016': 'wp_deepJet_106XUL16preVFP_v2.csv',
               '2017': 'wp_deepJet_106XUL17_v3.csv',
               '2018': 'wp_deepJet_106XUL18_v2.csv'
           },
           'deepcsv': {
                '2016': 'wp_deepCSV_106XUL16preVFP_v2.csv',
                '2017': 'wp_deepCSV_106XUL17_v3.csv',
                '2018': 'wp_deepCSV_106XUL18_v2.csv'
           }
        }

#            files = {
#                'deepflav': {
#                    '2016': 'wp_deepJet_106XUL16postVFP_v2.csv',
#                    '2017': 'wp_deepJet_106XUL17_v3.csv',
#                    '2018': 'wp_deepJet_106XUL18_v2.csv'
#                },
#                'deepcsv': {
#                     '2016': 'wp_deepCSV_106XUL16postVFP_v2.csv',
#                     '2017': 'wp_deepCSV_106XUL17_v3.csv',
#                     '2018': 'wp_deepCSV_106XUL18_v2.csv'
#                }
#            }
        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        filename = 'data/btag_SF/UL/'+files[tagger][year]
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2016': 'btag2016.merged',
            '2017': 'btag2017.merged',
            '2018': 'btag2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        nom = bpass / np.maximum(ball, 1.)
        self.eff = dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, events, pt, eta, flavor, tag):
        abseta = abs(eta)
        tightJet = events
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def zerotag(eff):
            return ak.prod(1.0 - eff[tightJet.isdflvM], axis=-1)
        
        def onetag(eff):
            output = ak.prod(eff[tightJet.isdflvM], axis=-1) * ak.prod((1.0 - eff[np.invert(tightJet.isdflvM)]), axis=-1)
            return(output)
        
        eff = self.eff(flavor, pt, abseta)
        sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_up = self.sf.eval('up', flavor, abseta, pt)
        sf_down = self.sf.eval('down', flavor, abseta, pt)

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        eff_data_up   = np.minimum(1., sf_up*eff)
        eff_data_down = np.minimum(1., sf_down*eff)


        if '-1' in tag: 
            nom = (1 - zerotag(eff_data_nom)) / (1 - zerotag(eff))
            up = (1 - zerotag(eff_data_up)) / (1 - zerotag(eff))
            down = (1 - zerotag(eff_data_down)) / (1 - zerotag(eff))
        elif '2' in tag:
            nom = (1- zerotag(eff_data_nom) - onetag(eff_data_nom))/(1- zerotag(eff) - onetag(eff))
            up =(1- zerotag(eff_data_up) - onetag(eff_data_up))/(1- zerotag(eff) - onetag(eff))
            down =(1- zerotag(eff_data_down) - onetag(eff_data_down))/(1- zerotag(eff) - onetag(eff))
        elif '+1' in tag:
            nom = onetag(eff_data_nom)/onetag(eff)
            up= onetag(eff_data_up)/onetag(eff)
            down = onetag(eff_data_down)/onetag(eff)
        else:
            nom = zerotag(eff_data_nom)/zerotag(eff)
            up = zerotag(eff_data_up)/zerotag(eff)
            down = zerotag(eff_data_down)/zerotag(eff)

        return np.nan_to_num(nom), np.nan_to_num(up), np.nan_to_num(down)

get_btag_weight_preVFP = {
    'deepflav': {
        '2016': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2016','loose').btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2016','medium').btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2016','tight').btag_weight,

        },
        '2017': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2018','tight' ).btag_weight
        }
    },
    'deepcsv' : {
        '2016': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2016','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2016','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2016','tight' ).btag_weight
        },
        '2017': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2018','tight').btag_weight
        }
    }
}

class BTagCorrectorPostVFP:

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        files = {
           'deepflav': {
               '2016': 'wp_deepJet_106XUL16postVFP_v3.csv',
               '2017': 'wp_deepJet_106XUL17_v3.csv',
               '2018': 'wp_deepJet_106XUL18_v2.csv'
           },
           'deepcsv': {
                '2016': 'wp_deepCSV_106XUL16postVFP_v3.csv',
                '2017': 'wp_deepCSV_106XUL17_v3.csv',
                '2018': 'wp_deepCSV_106XUL18_v2.csv'
           }
        }

        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        filename = 'data/btag_SF/UL/'+files[tagger][year]
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2016': 'btag2016.merged',
            '2017': 'btag2017.merged',
            '2018': 'btag2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = util.load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        nom = bpass / np.maximum(ball, 1.)
        self.eff = dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, events, pt, eta, flavor, tag):
        abseta = abs(eta)
        tightJet = events
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def zerotag(eff):
            return ak.prod(1.0 - eff[tightJet.isdflvM], axis=-1)
        
        def onetag(eff):
            output = ak.prod(eff[tightJet.isdflvM], axis=-1) * ak.prod((1.0 - eff[np.invert(tightJet.isdflvM)]), axis=-1)
            return(output)
        
        eff = self.eff(flavor, pt, abseta)
        sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_up = self.sf.eval('up', flavor, abseta, pt)
        sf_down = self.sf.eval('down', flavor, abseta, pt)

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        eff_data_up   = np.minimum(1., sf_up*eff)
        eff_data_down = np.minimum(1., sf_down*eff)


        if '-1' in tag: 
            nom = (1 - zerotag(eff_data_nom)) / (1 - zerotag(eff))
            up = (1 - zerotag(eff_data_up)) / (1 - zerotag(eff))
            down = (1 - zerotag(eff_data_down)) / (1 - zerotag(eff))
        elif '2' in tag:
            nom = (1- zerotag(eff_data_nom) - onetag(eff_data_nom))/(1- zerotag(eff) - onetag(eff))
            up =(1- zerotag(eff_data_up) - onetag(eff_data_up))/(1- zerotag(eff) - onetag(eff))
            down =(1- zerotag(eff_data_down) - onetag(eff_data_down))/(1- zerotag(eff) - onetag(eff))
        elif '+1' in tag:
            nom = onetag(eff_data_nom)/onetag(eff)
            up= onetag(eff_data_up)/onetag(eff)
            down = onetag(eff_data_down)/onetag(eff)
        else:
            nom = zerotag(eff_data_nom)/zerotag(eff)
            up = zerotag(eff_data_up)/zerotag(eff)
            down = zerotag(eff_data_down)/zerotag(eff)

        return np.nan_to_num(nom), np.nan_to_num(up), np.nan_to_num(down)

get_btag_weight_postVFP = {
    'deepflav': {
        '2016': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2016','loose').btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2016','medium').btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2016','tight').btag_weight,

        },
        '2017': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2018','tight' ).btag_weight
        }
    },
    'deepcsv' : {
        '2016': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2016','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2016','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2016','tight' ).btag_weight
        },
        '2017': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2018','tight').btag_weight
        }
    }
}


Jetext = extractor()
for directory in ['jec_UL', 'jersf_UL', 'jr_UL', 'junc_UL']:
    directory='data/'+directory
    print('Loading files in:',directory)
    for filename in os.listdir(directory):
        if '~' in filename: continue
        if 'AK4PFchs' not in filename: continue
        if 'DATA' in filename: continue
        if "Regrouped" in filename: continue
        if "UncertaintySources" in filename: continue
        filename=directory+'/'+filename
        print('Loading file:',filename)
        Jetext.add_weight_sets(['* * '+filename])
    print('All files in',directory,'loaded')
Jetext.finalize()
Jetevaluator = Jetext.make_evaluator()

corrections = {
    'get_msd_weight':           get_msd_weight,
    'get_ttbar_weight':         get_ttbar_weight,
    'get_nnlo_nlo_weight':      get_nnlo_nlo_weight,
    'get_nlo_qcd_weight':       get_nlo_qcd_weight,
    'get_nlo_ewk_weight':       get_nlo_ewk_weight,
    'get_pu_weight':            get_pu_weight,
    'get_met_trig_weight':      get_met_trig_weight,
#     'get_met_zmm_trig_weight':  get_met_zmm_trig_weight,
    'get_ele_trig_weight':      get_ele_trig_weight,
#     'get_mu_trig_weight':      get_mu_trig_weight,    
    'get_pho_trig_weight':      get_pho_trig_weight,
#     'get_ele_loose_id_sf':      get_ele_loose_id_sf,
#     'get_ele_tight_id_sf':      get_ele_tight_id_sf,

    'get_pho_tight_id_sf':      get_pho_tight_id_sf,
    'get_pho_csev_sf':          get_pho_csev_sf,
#     'get_mu_tight_id_sf':       get_mu_tight_id_sf,
#     'get_mu_loose_id_sf':       get_mu_loose_id_sf,
#     'get_ele_reco_sf':          get_ele_reco_sf,
#     'get_ele_reco_lowet_sf':    get_ele_reco_lowet_sf,
#     'get_mu_tight_iso_sf':      get_mu_tight_iso_sf,
#     'get_mu_loose_iso_sf':      get_mu_loose_iso_sf,
    'get_ecal_bad_calib':       get_ecal_bad_calib,
#     'get_btag_weight':          get_btag_weight,
    'Jetevaluator':             Jetevaluator,
    
    'get_ele_tight_id_sf_postVFP': get_ele_tight_id_sf_postVFP,
    'get_ele_tight_id_sf_preVFP': get_ele_tight_id_sf_preVFP,
    'get_ele_loose_id_sf_postVFP': get_ele_loose_id_sf_postVFP,
    'get_ele_loose_id_sf_preVFP': get_ele_loose_id_sf_preVFP,
    
    'get_ele_reco_sf_preVFP_below20': get_ele_reco_sf_preVFP_below20,
    'get_ele_reco_sf_postVFP_below20':get_ele_reco_sf_postVFP_below20,
    'get_ele_reco_sf_postVFP_above20':get_ele_reco_sf_postVFP_above20,
    'get_ele_reco_sf_preVFP_above20':get_ele_reco_sf_preVFP_above20,
    
    'get_mu_tight_id_sf_preVFP':get_mu_tight_id_sf_preVFP,
    'get_mu_loose_id_sf_preVFP':get_mu_loose_id_sf_preVFP,
    'get_mu_tight_id_sf_postVFP':get_mu_tight_id_sf_postVFP,
    'get_mu_loose_id_sf_postVFP':get_mu_loose_id_sf_postVFP,
    'get_mu_tight_iso_sf_preVFP':get_mu_tight_iso_sf_preVFP,
    'get_mu_loose_iso_sf_preVFP':get_mu_loose_iso_sf_preVFP,
    'get_mu_tight_iso_sf_postVFP':get_mu_tight_iso_sf_postVFP,
    'get_mu_loose_iso_sf_postVFP':get_mu_loose_iso_sf_postVFP,
    
    'get_mu_trig_weight_preVFP':get_mu_trig_weight_preVFP,
    'get_mu_trig_weight_postVFP':get_mu_trig_weight_postVFP,
    
#     err files
    'get_ele_trig_err':get_ele_trig_err,
    'get_mu_trig_err_postVFP':get_mu_trig_err_postVFP,
    'get_mu_trig_err_postVFP':get_mu_trig_err_postVFP,
    'get_ele_loose_id_err_preVFP':get_ele_loose_id_err_preVFP,
    'get_ele_tight_id_err_preVFP':get_ele_tight_id_err_preVFP,
    'get_ele_loose_id_err_postVFP':get_ele_loose_id_err_postVFP,
    'get_ele_tight_id_err_postVFP':get_ele_tight_id_err_postVFP,
    
    'get_mu_loose_id_err_preVFP':get_mu_loose_id_err_preVFP,
    'get_mu_tight_id_err_preVFP':get_mu_tight_id_err_preVFP,
    'get_mu_loose_id_err_postVFP':get_mu_loose_id_err_postVFP,
    'get_mu_tight_id_err_postVFP':get_mu_tight_id_err_postVFP,
    
    'get_ele_reco_err_preVFP_above20':get_ele_reco_err_preVFP_above20,
    'get_ele_reco_err_postVFP_above20':get_ele_reco_err_postVFP_above20,
    'get_ele_reco_err_postVFP_below20':get_ele_reco_err_postVFP_below20,
    'get_ele_reco_err_preVFP_below20':get_ele_reco_err_preVFP_below20,
    
    'get_mu_loose_iso_err_postVFP':get_mu_loose_iso_err_postVFP,
    'get_mu_tight_iso_err_postVFP':get_mu_tight_iso_err_postVFP,
    'get_mu_loose_iso_err_preVFP':get_mu_loose_iso_err_preVFP,
    'get_mu_tight_iso_err_preVFP':get_mu_tight_iso_err_preVFP,
    
    'get_btag_weight_preVFP':get_btag_weight_preVFP,
    'get_btag_weight_postVFP':get_btag_weight_postVFP,
    
}
save(corrections, 'data/corrections.coffea')



 
