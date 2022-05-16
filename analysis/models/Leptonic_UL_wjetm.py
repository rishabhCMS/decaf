from __future__ import print_function, division
from optparse import OptionParser
from collections import defaultdict, OrderedDict
import concurrent.futures
import sys
import os
import rhalphalib as rl
import numpy as np
import pickle
import gzip
import json
from coffea import hist, processor
from coffea.util import load, save
from scipy import stats
import ROOT

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

mT_corrected_binning_dict = {
    "2018": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500]),
    "2017": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500]),
    "2016": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500])
}



def remap_histograms(hists):
    data_hists = {}
    bkg_hists = {}
    signal_hists = {}

    process = hist.Cat("process", "Process", sorting="placement")
    cats = ("process",)
    sig_map = OrderedDict()
    bkg_map = OrderedDict()
    data_map = OrderedDict()
    bkg_map["Hbb"] = ("Hbb*",)
    bkg_map["DY+jets"] = ("DY+*",)
    bkg_map["VV"] = (["WW", "WZ", "ZZ"],)
    bkg_map["ST"] = ("ST*",)
    bkg_map["TT"] = ("TT*",)
    bkg_map["W+jets"] = ("W+*",)
#     bkg_map["W+HF"] = ("W+HF",)
#     bkg_map["W+LF"] = ("W+LF",)
    bkg_map["Z_jets"] = ("Z+*",)
#     bkg_map["Z+HF"] = ("Z+HF",)
#     bkg_map["Z+LF"] = ("Z+LF",)
#     bkg_map["G+jets"] = ("G+*",)
    bkg_map["QCD"] = ("QCD*",)
    data_map["SingleElectron"] = ("SingleElectron",)
    data_map["SingleMuon"] = ("SingleMuon",)
    data_map["EGamma"] = ("EGamma",)
    sig_map["Mphi1995_Mchi1000"] = ("Mphi-1995_Mchi-1000")
    sig_map["Mphi2000_Mchi1500"] = ("Mphi-2000_Mchi-1500")
    sig_map["Mphi2000_Mchi500"] = ("Mphi-2000_Mchi-500")
#     haven't processed signal yet
    for signal in hists['sig']['template'].identifiers('process'):
        if 'mchi' not in str(signal): continue
        sig_map[str(signal)] = (str(signal),)  ## signals

    for key in hists["data"].keys():
        bkg_hists[key] = hists["bkg"][key].group(cats, process, bkg_map)
        signal_hists[key] = hists["sig"][key].group(cats, process, sig_map)
        data_hists[key] = hists["data"][key].group(cats, process, data_map)

    print('initial mT_corrected binning',bkg_hists["template"].axis("mT_corrected").edges())


    bkg_hists["template"] = bkg_hists["template"].rebin(
        "mT_corrected", hist.Bin("mT_corrected", "mT_corrected", mT_corrected_binning)
    )
    signal_hists["template"] = signal_hists["template"].rebin(
        "mT_corrected", hist.Bin("mT_corrected", "mT_corrected", mT_corrected_binning)
    )
    data_hists["template"] = data_hists["template"].rebin(
        "mT_corrected", hist.Bin("mT_corrected", "mT_corrected", mT_corrected_binning)
    )

    hists = {"bkg": bkg_hists, "data": data_hists, "sig":signal_hists}

    return hists


def addBtagSyst(dictionary, process, region, templ):
    btagUp = template(dictionary, process, "btagUp", region)[0]
    btagDown = template(dictionary, process, "btagDown", region)[0]
    templ.setParamEffect(btag, btagUp, btagDown)

def addVJetsSyst(dictionary, process, region, templ):
    def addSyst(dictionary, process, region, templ, syst, string):
        histogram = dictionary[region].integrate("process", process)
#         print(histogram.integrate("systematic", "nominal").sum('npvs').values()[()])
        nominal=histogram.integrate("systematic", "nominal").sum('npvs').values()[()]
        up=histogram.integrate("systematic", string+"Up").sum('npvs').values()[()]
        down=histogram.integrate("systematic",string+"Down").sum('npvs').values()[()]
        systUp = np.array( up.sum() / nominal.sum() )
        systUp[np.isnan(systUp)] = 1.
        systUp = systUp.sum()
        templ.setParamEffect(syst, systUp)
#     addSyst(dictionary, process, region, templ, ew1, "ew1")
#     addSyst(dictionary, process, region, templ, ew2W, "ew2W")
#     addSyst(dictionary, process, region, templ, ew2Z, "ew2Z")
#     addSyst(dictionary, process, region, templ, ew3W, "ew3W")
#     addSyst(dictionary, process, region, templ, ew3Z, "ew3Z")
#     addSyst(dictionary, process, region, templ, mix, "mix")
#     addSyst(dictionary, process, region, templ, qcd1, "qcd1")
#     addSyst(dictionary, process, region, templ, qcd2, "qcd2")
#     addSyst(dictionary, process, region, templ, qcd3, "qcd3")
    addSyst(dictionary, process, region, templ, muR, "muR")
    addSyst(dictionary, process, region, templ, muF, "muF")
    
def template(dictionary, process, systematic, region, read_sumw2=False):
    histogram = dictionary[region].integrate("process", process).sum('npvs')
    nominal, sumw2 = histogram.integrate("systematic", "nominal").values(sumw2=True)[()]
    zerobins = nominal <= 0.
    output = nominal
    
    if "nominal" not in systematic and "data" not in systematic:
    #print('Normalizing',systematic,'histogram of',process,'in region',region)
        output = histogram.integrate("systematic", systematic).values()[()]
        output[zerobins] = 1.
        output[~zerobins] /= nominal[~zerobins]
        output[np.isnan(output)] = 1.
    binning = (
        dictionary[region]
        .integrate("process", process)
        .integrate("systematic", systematic)
        .axis("mT_corrected")
        .edges()
    )
    if read_sumw2:
        return (output, binning, "mT_corrected", sumw2)
    return (output, binning, "mT_corrected")

def model(year):

#     model_id = year + category + "recoil" + str(recoil)
#     print(model_id)
#     model = rl.Model(str(s) + model_id)



    model_id = 'w'
    model = rl.Model(model_id)
    ch_name = "wjetm" 
    # Create a channel/Control region
    wjetm = rl.Channel(ch_name)
    # add the Channel/CR to the model
    model.addChannel(wjetm)
    
    ###
    # Add data distribution to the channel
    ###

    if year == "2018":
        wjetm.setObservation(template(data, "SingleMuon", "data", "wjetm"))
    else:
        wjetm.setObservation(template(data, "SingleMuon", "data","wjetm"))
        
    #template for signal
    for s in signal['wjetm'].identifiers('process'):
        print(s)
        if 'Mphi1995' not in str(s): continue
        wjetm_signalTemplate = template(signal,s,'nominal','wjetm')
        wjetm_signal=rl.TemplateSample(ch_name+'_'+str(s), rl.Sample.SIGNAL, wjetm_signalTemplate)
        wjetm_signal.setParamEffect(lumi, 1.015)
    #     sr_signal.setParamEffect(trig_met, 1.01)
    #     sr_signal.setParamEffect(veto_tau, 1.03)
    #     sr_signal.setParamEffect(jec, 1.05)
        #btag
        btagUp=template(signal, s,'btagUp','wjetm')[0]
        btagDown=template(signal, s,'btagDown','wjetm')[0]
        wjetm_signal.setParamEffect(btag, btagUp, btagDown)
        #trigger Muon
        trig_muUp=template(signal, s, 'trig_mUp', 'wjetm')[0] 
        trig_muDown=template(signal, s, 'trig_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(trig_m, trig_muUp, trig_muDown)
        #id Muon
        id_mUp=template(signal, s, 'id_mUp', 'wjetm')[0] 
        id_mDown=template(signal, s, 'id_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(id_m, id_mUp, id_mDown)
        #pileup
        pileup_Up=template(signal, s,'pileupUp','wjetm')[0] 
        pileup_Down=template(signal, s,'pileupDown','wjetm')[0] 
        wjetm_signal.setParamEffect(pileup, pileup_Up, pileup_Down)
        #iso Muon
        iso_mUp=template(signal, s, 'isolation_mUp', 'wjetm')[0] 
        iso_mDown=template(signal, s, 'isolation_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(isolation_m, iso_mUp, iso_mDown)
        wjetm.addSample(wjetm_signal)
    ###
    # W(->lnu)+jets data-driven model
    ###

    wjetm_wjetsTemplate = template(background, "W+jets", "nominal", "wjetm", read_sumw2=True)
    wjetm_wjetsMC = rl.TemplateSample(
        ch_name + "_wjetsMC",
        rl.Sample.BACKGROUND,
        wjetm_wjetsTemplate
    )
    wjetm_wjetsMC.setParamEffect(lumi, 1.015)
#     ttbarm_wjetsMC.setParamEffect(trig_e, 1.01)
#     ttbarm_wjetsMC.setParamEffect(veto_tau, 1.03)
#     ttbarm_wjetsMC.setParamEffect(wjets_norm, 1.4)
#     ttbarm_wjetsMC.setParamEffect(jec, 1.05)
#     ttbarm_wjetsMC.setParamEffect(id_e, 1.02)
#     ttbarm_wjetsMC.setParamEffect(reco_e, 1.02)
    #ttbarm_wjetsMC.autoMCStats()
    addBtagSyst(background, "W+jets", "wjetm", wjetm_wjetsMC)
    #trigger Muon
    trig_muUp=template(background, 'W+jets', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'W+jets', 'trig_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "W+jets", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "W+jets", 'id_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'W+jets','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'W+jets','pileupDown','wjetm')[0] 
    wjetm_wjetsMC.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'W+jets', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'W+jets', 'isolation_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    # addVJetsSyst(background, "W+jets", "ttbarm", ttbarm_wjetsMC)
    wjetm.addSample(wjetm_wjetsMC)

    # ###
    # # top-antitop model
    # ###

    wjetm_ttTemplate = template(background, "TT", "nominal","wjetm", read_sumw2=True)
    wjetm_ttMC = rl.TemplateSample(
        ch_name + "_ttMC",
        rl.Sample.BACKGROUND,
        wjetm_ttTemplate
    )
    wjetm_ttMC.setParamEffect(lumi, 1.015)
#     ttbarm_ttMC.setParamEffect(trig_e, 1.01)
#     ttbarm_ttMC.setParamEffect(veto_tau, 1.03)
#     ttbarm_ttMC.setParamEffect(jec, 1.05)
#     ttbarm_ttMC.setParamEffect(id_e, 1.02)
#     ttbarm_ttMC.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "TT", "ttbarm", wjetm_ttMC)
    #trigger Muon
    trig_muUp=template(background, 'TT', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'TT', 'trig_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "TT", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "TT", 'id_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'TT','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'TT','pileupDown','wjetm')[0] 
    wjetm_ttMC.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'TT', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'TT', 'isolation_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    # ttbarm_ttMC.setParamEffect(ttMC_norm, 1.2)
    wjetm.addSample(wjetm_ttMC)

    # ###
    # # Other MC-driven processes
    # ###
    # ST Background
    wjetm_stTemplate = template(background, "ST", "nominal", "wjetm")
    wjetm_st = rl.TemplateSample(
        ch_name + "_stMC", rl.Sample.BACKGROUND, wjetm_stTemplate
    )
    wjetm_st.setParamEffect(lumi, 1.015)
#     ttbarm_st.setParamEffect(trig_e, 1.01)
#     ttbarm_st.setParamEffect(veto_tau, 1.03)
#     ttbarm_st.setParamEffect(st_norm, 1.2) something related to signal
#     ttbarm_st.setParamEffect(jec, 1.05)
#     ttbarm_st.setParamEffect(id_e, 1.02)
#     ttbarm_st.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "ST", "ttbarm", wjetm_st)
    #trigger Muon
    trig_muUp=template(background, 'ST', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'ST', 'trig_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "ST", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "ST", 'id_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'ST','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'ST','pileupDown','wjetm')[0] 
    wjetm_st.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'ST', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'ST', 'isolation_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    wjetm.addSample(wjetm_st)

    # DY Jet Background
    wjetm_dyjetsTemplate = template(background, "DY+jets", "nominal","ttbarm")
    wjetm_dyjets = rl.TemplateSample(
        ch_name + "_dyjetsMC", rl.Sample.BACKGROUND, wjetm_dyjetsTemplate
    )
    wjetm_dyjets.setParamEffect(lumi, 1.015)
#     ttbarm_dyjets.setParamEffect(trig_e, 1.01)
#     ttbarm_dyjets.setParamEffect(veto_tau, 1.03)
#     ttbarm_dyjets.setParamEffect(zjetsMC_norm, 1.4)
#     ttbarm_dyjets.setParamEffect(jec, 1.05)
#     ttbarm_dyjets.setParamEffect(id_e, 1.02)
#     ttbarm_dyjets.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "DY+jets", "wjetm", wjetm_dyjets)
    addVJetsSyst(background,"DY+jets", "wjetm", wjetm_dyjets)
    #trigger Muon
    trig_muUp=template(background, 'DY+jets', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'DY+jets', 'trig_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "DY+jets", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "DY+jets", 'id_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'DY+jets','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'DY+jets','pileupDown','wjetm')[0] 
    wjetm_dyjets.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'DY+jets', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'DY+jets', 'isolation_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    wjetm.addSample(wjetm_dyjets)


    # V Jets Background
    wjetm_vvTemplate = template(background, "VV", "nominal", "wjetm")
    wjetm_vv = rl.TemplateSample(
        ch_name + "_vvMC", rl.Sample.BACKGROUND, wjetm_vvTemplate
    )
    wjetm_vv.setParamEffect(lumi, 1.015)
#     ttbarm_vv.setParamEffect(trig_e, 1.01)
#     ttbarm_vv.setParamEffect(veto_tau, 1.03)
#     ttbarm_vv.setParamEffect(vv_norm, 1.2)
#     ttbarm_vv.setParamEffect(jec, 1.05)
#     ttbarm_vv.setParamEffect(id_e, 1.02)
#     ttbarm_vv.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "VV", "ttbarm", wjetm_vv)
    #trigger Muon
    trig_muUp=template(background, 'VV', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'VV', 'trig_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "VV", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "VV", 'id_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'VV','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'VV','pileupDown','wjetm')[0] 
    wjetm_vv.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'VV', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'VV', 'isolation_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    
    wjetm.addSample(wjetm_vv)

    # QCD background
    wjetm_qcdTemplate = template(background, "QCD", "nominal", "wjetm")
    wjetm_qcd = rl.TemplateSample(
        ch_name + "_qcdMC", rl.Sample.BACKGROUND, wjetm_qcdTemplate
    )
    wjetm_qcd.setParamEffect(lumi, 1.015)
#     ttbarm_qcd.setParamEffect(trig_e, 1.01)
#     ttbarm_qcd.setParamEffect(veto_tau, 1.03)
#     ttbarm_qcd.setParamEffect(qcde_norm, 2.0)
#     ttbarm_qcd.setParamEffect(jec, 1.05)
#     ttbarm_qcd.setParamEffect(id_e, 1.02)
#     ttbarm_qcd.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "QCD", "wjetm", wjetm_qcd)
    #trigger Muon
    trig_muUp=template(background, 'QCD', 'trig_mUp', 'wjetm')[0] 
    trig_muDown=template(background, 'QCD', 'trig_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(trig_m, trig_muUp, trig_muDown)
    #id Muon
    id_mUp=template(background, "QCD", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "QCD", 'id_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileup_Up=template(background, 'QCD','pileupUp','wjetm')[0] 
    pileup_Down=template(background, 'QCD','pileupDown','wjetm')[0] 
    wjetm_qcd.setParamEffect(pileup, pileup_Up, pileup_Down)
    #iso Muon
    iso_mUp=template(background, 'QCD', 'isolation_mUp', 'wjetm')[0] 
    iso_mDown=template(background, 'QCD', 'isolation_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(isolation_m, iso_mUp, iso_mDown)
    
    wjetm.addSample(wjetm_qcd)
    return model

if __name__ == "__main__":
    if not os.path.exists("datacards"):
        os.mkdir("datacards")
    parser = OptionParser()
    parser.add_option("-y", "--year", help="year", dest="year", default="")
    (options, args) = parser.parse_args()
    year = options.year
    year = '2018'

    mT_corrected_binning_dict = {
        "2018": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500]),
        "2017": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500]),
        "2016": np.array([100.0,125.0,150.0, 165.0, 180.0, 200.0, 220.0, 240.0, 270.0, 310.0, 350.0, 500])
    }
    mT_corrected_binning = mT_corrected_binning_dict[year]
    mT_correctedbins = np.array(mT_corrected_binning)
    nmT_corrected = len(mT_correctedbins) - 1

    ###
    # Extract histograms from input file
    ###

    print("Grouping histograms")
    hists = load("hists/UL_lep_monotop_v1_test_" + year + ".scaled")
    hists = remap_histograms(hists)
    data_hists = hists["data"]
    bkg_hists = hists["bkg"]
    signal_hists = hists["sig"]

    ###
    # Preparing histograms for fit
    ##

    data = {}
    for r in data_hists["template"].identifiers("region"):
        data[str(r)] = data_hists["template"].integrate("region", r)

    background = {}
    for r in bkg_hists["template"].identifiers("region"):
        background[str(r)] = bkg_hists["template"].integrate("region", r)

    signal = {}
    for r in signal_hists["template"].identifiers("region"):
        signal[str(r)] = signal_hists["template"].integrate("region", r)
    ###
    ###
    # Setting up systematics
    ###
    ###
    lumi = rl.NuisanceParameter("lumi" + year, "lnN")
#     zjets_norm = rl.NuisanceParameter("zjets_norm", "lnN")
#     wjets_norm = rl.NuisanceParameter("wjets_norm", "lnN")
#     tt_norm = rl.NuisanceParameter("tt_norm", "lnN")
    id_e = rl.NuisanceParameter("id_e" + year, "shape")
    id_m = rl.NuisanceParameter("id_m" + year, "shape")
#     id_pho = rl.NuisanceParameter("id_pho" + year, "lnN")
    reco_e = rl.NuisanceParameter("reco_e" + year, "shape")
    isolation_m = rl.NuisanceParameter("isolation_m" + year, "shape")
    trig_e = rl.NuisanceParameter("trig_e" + year, "shape")
    trig_m = rl.NuisanceParameter("trig_m" + year, "shape")

#     veto_tau = rl.NuisanceParameter("veto_tau" + year, "lnN")
    jec = rl.NuisanceParameter("jec" + year, "lnN")
    #btag = rl.NuisanceParameter("btag" + year, "shapeN")  # AK4 btag
    btag = rl.NuisanceParameter("btag" + year, "shape")  # AK4 btag
    ew1 = rl.NuisanceParameter("ew1", "lnN")
    #ew2G = rl.NuisanceParameter("ew2G", "lnN")
    ew2W = rl.NuisanceParameter("ew2W", "lnN")
    ew2Z = rl.NuisanceParameter("ew2Z", "lnN")
    #ew3G = rl.NuisanceParameter("ew3G", "lnN")
    ew3W = rl.NuisanceParameter("ew3W", "lnN")
    ew3Z = rl.NuisanceParameter("ew3Z", "lnN")
    mix = rl.NuisanceParameter("mix", "lnN")
    muF = rl.NuisanceParameter("muF", "lnN")
    muR = rl.NuisanceParameter("muR", "lnN")
    qcd1 = rl.NuisanceParameter("qcd1", "lnN")
    qcd2 = rl.NuisanceParameter("qcd2", "lnN")
    qcd3 = rl.NuisanceParameter("qcd3", "lnN")
    whf_fraction = rl.NuisanceParameter("whf_fraction", "lnN")
    zhf_fraction = rl.NuisanceParameter("zhf_fraction", "lnN")
    #ghf_fraction = rl.NuisanceParameter("ghf_fraction", "shapeN")
    pileup = rl.NuisanceParameter("pileup"+year, "shape")
    
    with open(
        "data/"+"test_wjetm"+year+".model",
        "wb",) as fout:
        pickle.dump(model(year), fout, protocol=2)
