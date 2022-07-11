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
#     "2018":np.linspace(0, 600, 21),
#     "2017":np.linspace(0, 600, 21),
#     "2016":np.linspace(0, 600, 21)
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
        if 'Mchi' not in str(signal): continue
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
    addSyst(dictionary, process, region, templ, ew1, "ew1")
    addSyst(dictionary, process, region, templ, ew2W, "ew2W")
    addSyst(dictionary, process, region, templ, ew2Z, "ew2Z")
    addSyst(dictionary, process, region, templ, ew3W, "ew3W")
    addSyst(dictionary, process, region, templ, ew3Z, "ew3Z")
    addSyst(dictionary, process, region, templ, mix, "mix")
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
    if "data" not in systematic:
        #output[zerobins] = 1.
        #sumw2[zerobins] = 0.
        if "ST" in str(process) :
            output[zerobins] = 1.
            sumw2[zerobins] = 0.
        elif "W+" in str(process):
            output[zerobins] = 1.
            sumw2[zerobins] = 0.
        elif "W+" in str(process):
            output[zerobins] = 1.
            sumw2[zerobins] = 0.
        elif "TT" in str(process):
            output[zerobins] = 1.
            sumw2[zerobins] = 0.
            
        else:
            output[zerobins] = 1e-5
            sumw2[zerobins] = 0.

            
    if "nominal" not in systematic and "data" not in systematic:
    #print('Normalizing',systematic,'histogram of',process,'in region',region)
        output = histogram.integrate("systematic", systematic).values()[()]
        zerobins1 = output <= 0.
        output[zerobins1]  = 1.
        output[zerobins]   = 1.
        output[~zerobins]  /= nominal[~zerobins]
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
    if year == '2018':
        lumi_unc = 1.025
    if year == '2017':
        lumi_unc = 1.023
    if '2016' in year:
        lumi_unc = 1.012


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
        wjetm_signal.setParamEffect(lumi, lumi_unc)
    #     sr_signal.setParamEffect(trig_met, 1.01)
    #     sr_signal.setParamEffect(veto_tau, 1.03)
    #     sr_signal.setParamEffect(jec, 1.05)
        #btag
        btagUp=template(signal, s,'btagUp','wjetm')[0]
        btagDown=template(signal, s,'btagDown','wjetm')[0]
        wjetm_signal.setParamEffect(btag, btagUp, btagDown)
    #         #trigger Electron
        trigmUp=template(signal, s, 'trig_mUp', 'wjetm')[0] 
        trigmDown=template(signal, s, 'trig_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(trig_m, trigmUp, trigmDown)

        #pileup
        pileupUp=template(signal, s,'puUp','wjetm')[0] 
        pileupDown=template(signal, s,'puDown','wjetm')[0] 
        wjetm_signal.setParamEffect(pileup, pileupUp, pileupDown)
        #iso Muon
        isolation_mUp=template(signal, s, 'isolation_mUp', 'wjetm')[0] 
        isolation_mDown=template(signal, s, 'isolation_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
        #id Muon
        id_mUp=template(signal, s, 'id_mUp', 'wjetm')[0] 
        id_mDown=template(signal, s, 'id_mDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(id_m, id_mUp, id_mDown)        


        #JEC_JER
#         jerUp=template(signal, s, 'JERUp', 'wjetm')[0] 
#         jerDown=template(signal, s, 'JERDown', 'wjetm')[0]
#         wjetm_signal.setParamEffect(jec_jer, jerUp, jerDown)

        #JEC_JEs
        jesUp=template(signal, s, 'JESUp', 'wjetm')[0] 
        jesDown=template(signal, s, 'JESDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(jec_jes, jesUp, jesDown)

        #met_uncl
        metUnclUp=template(signal, s, 'METunclUp', 'wjetm')[0] 
        metUnclDown=template(signal, s, 'METunclDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(metUncl, metUnclUp, metUnclDown)


        #L1 prefire
        L1prefireUp=template(signal, s, 'L1PrefireUp', 'wjetm')[0] 
        L1prefireDown=template(signal, s, 'L1PrefireDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #L1 ISR
        ISRUp=template(signal, s, 'ISRUp', 'wjetm')[0] 
        ISRDown=template(signal, s, 'ISRDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(isr, ISRUp, ISRDown)

        #L1 FSR
        FSRUp=template(signal, s, 'FSRUp', 'wjetm')[0] 
        FSRDown=template(signal, s, 'FSRDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(fsr, FSRUp, FSRDown)

        #pdf 
        pdfUp=template(signal, s, 'PDFUp', 'wjetm')[0] 
        pdfDown=template(signal, s, 'PDFDown', 'wjetm')[0]
        wjetm_signal.setParamEffect(pdf, pdfUp, pdfDown) 

    #         wjetm_signal.autoMCStats()
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
    wjetm_wjetsMC.setParamEffect(lumi, lumi_unc)
    #     wjetm_wjetsMC.setParamEffect(trig_m, 1.01)
    #     wjetm_wjetsMC.setParamEffect(veto_tau, 1.03)
    #     wjetm_wjetsMC.setParamEffect(wjets_norm, 1.4)
    #     wjetm_wjetsMC.setParamEffect(jec, 1.05)
    #     wjetm_wjetsMC.setParamEffect(id_m, 1.02)
    #     wjetm_wjetsMC.setParamEffect(reco_e, 1.02)
    #     wjetm_wjetsMC.autoMCStats()
    addBtagSyst(background, "W+jets", "wjetm", wjetm_wjetsMC)
    addVJetsSyst(background, "W+jets", "wjetm", wjetm_wjetsMC)
    #trigger Muon
    trigmUp=template(background, 'W+jets', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'W+jets', 'trig_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(trig_m, trigmUp, trigmDown)

    #pileup
    pileupUp=template(background, 'W+jets','puUp','wjetm')[0] 
    pileupDown=template(background, 'W+jets','puDown','wjetm')[0] 
    wjetm_wjetsMC.setParamEffect(pileup, pileupUp, pileupDown)
    #iso Muon
    isolation_mUp=template(background, 'W+jets', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'W+jets', 'isolation_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "W+jets", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "W+jets", 'id_mDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(id_m, id_mUp, id_mDown)


    #JEC_JER
#     jerUp=template(background, 'W+jets', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'W+jets', 'JERDown', 'wjetm')[0]
#     wjetm_wjetsMC.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'W+jets', 'JESUp', 'wjetm')[0] 
    jesDown=template(background, 'W+jets', 'JESDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
    metUnclUp=template(background, 'W+jets', 'METunclUp', 'wjetm')[0] 
    metUnclDown=template(background, 'W+jets', 'METunclDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'W+jets', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'W+jets', 'L1PrefireDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'W+jets', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'W+jets', 'ISRDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'W+jets', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'W+jets', 'FSRDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'W+jets', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'W+jets', 'PDFDown', 'wjetm')[0]
    wjetm_wjetsMC.setParamEffect(pdf, pdfUp, pdfDown)    
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
    wjetm_ttMC.setParamEffect(lumi, lumi_unc)
    #     wjetm_ttMC.setParamEffect(trig_m, 1.01)
    #     wjetm_ttMC.setParamEffect(veto_tau, 1.03)
    #     wjetm_ttMC.setParamEffect(jec, 1.05)
    #     wjetm_ttMC.setParamEffect(id_m, 1.02)
    #     wjetm_ttMC.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "TT", "wjetm", wjetm_ttMC)
    #     wjetm_ttMC.autoMCStats()
    #trigger Muon
    trigmUp=template(background, 'TT', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'TT', 'trig_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'TT', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'TT', 'isolation_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "TT", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "TT", 'id_mDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'TT','puUp','wjetm')[0] 
    pileupDown=template(background, 'TT','puDown','wjetm')[0] 
    wjetm_ttMC.setParamEffect(pileup, pileupUp, pileupDown)

    # wjetm_ttMC.setParamEffect(ttMC_norm, 1.2)

    #JEC_JER
#     jerUp=template(background, 'TT', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'TT', 'JERDown', 'wjetm')[0]
#     wjetm_ttMC.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'TT', 'JESUp', 'wjetm')[0] 
    jesDown=template(background, 'TT', 'JESDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
    metUnclUp=template(background, 'TT', 'METunclUp', 'wjetm')[0] 
    metUnclDown=template(background, 'TT', 'METunclDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'TT', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'TT', 'L1PrefireDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #ISR
    ISRUp=template(background, 'TT', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'TT', 'ISRDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(isr, ISRUp, ISRDown)

    #FSR
    FSRUp=template(background, 'TT', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'TT', 'FSRDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'TT', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'TT', 'PDFDown', 'wjetm')[0]
    wjetm_ttMC.setParamEffect(pdf, pdfUp, pdfDown)    

    wjetm.addSample(wjetm_ttMC)

    # ###
    # # Other MC-driven processes
    # ###
    # ST Background
    wjetm_stTemplate = template(background, "ST", "nominal", "wjetm")
    wjetm_st = rl.TemplateSample(
        ch_name + "_stMC", rl.Sample.BACKGROUND, wjetm_stTemplate
    )
    wjetm_st.setParamEffect(lumi, lumi_unc)
    #     wjetm_st.setParamEffect(trig_m, 1.01)
    #     wjetm_st.setParamEffect(veto_tau, 1.03)
    #     wjetm_st.setParamEffect(st_norm, 1.2) something related to signal
    #     wjetm_st.setParamEffect(jec, 1.05)
    #     wjetm_st.setParamEffect(id_m, 1.02)
    #     wjetm_st.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "ST", "wjetm", wjetm_st)
    #     wjetm_st.autoMCStats()
    #trigger Muon
    trigmUp=template(background, 'ST', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'ST', 'trig_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'ST', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'ST', 'isolation_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "ST", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "ST", 'id_mDown', 'wjetm')[0]
    wjetm_st.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'ST','puUp','wjetm')[0] 
    pileupDown=template(background, 'ST','puDown','wjetm')[0] 
    wjetm_st.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
#     jerUp=template(background, 'ST', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'ST', 'JERDown', 'wjetm')[0]
#     wjetm_st.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'ST', 'JESUp', 'wjetm')[0] 
    jesDown=template(background, 'ST', 'JESDown', 'wjetm')[0]
    wjetm_st.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
    metUnclUp=template(background, 'ST', 'METunclUp', 'wjetm')[0] 
    metUnclDown=template(background, 'ST', 'METunclDown', 'wjetm')[0]
    wjetm_st.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'ST', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'ST', 'L1PrefireDown', 'wjetm')[0]
    wjetm_st.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'ST', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'ST', 'ISRDown', 'wjetm')[0]
    wjetm_st.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'ST', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'ST', 'FSRDown', 'wjetm')[0]
    wjetm_st.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'ST', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'ST', 'PDFDown', 'wjetm')[0]
    wjetm_st.setParamEffect(pdf, pdfUp, pdfDown)  

    wjetm.addSample(wjetm_st)

    # DY Jet Background
    wjetm_dyjetsTemplate = template(background, "DY+jets", "nominal","wjetm")
    wjetm_dyjets = rl.TemplateSample(
        ch_name + "_dyjetsMC", rl.Sample.BACKGROUND, wjetm_dyjetsTemplate
    )
    wjetm_dyjets.setParamEffect(lumi, lumi_unc)
    #     wjetm_dyjets.setParamEffect(trig_m, 1.01)
    #     wjetm_dyjets.setParamEffect(veto_tau, 1.03)
    #     wjetm_dyjets.setParamEffect(zjetsMC_norm, 1.4)
    #     wjetm_dyjets.setParamEffect(jec, 1.05)
    #     wjetm_dyjets.setParamEffect(id_m, 1.02)
    #     wjetm_dyjets.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "DY+jets", "wjetm", wjetm_dyjets)
    addVJetsSyst(background,"DY+jets", "wjetm", wjetm_dyjets)
    #     wjetm_dyjets.autoMCStats()
    #trigger Muon
    trigmUp=template(background, 'DY+jets', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'DY+jets', 'trig_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'DY+jets', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'DY+jets', 'isolation_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "DY+jets", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "DY+jets", 'id_mDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'DY+jets','puUp','wjetm')[0] 
    pileupDown=template(background, 'DY+jets','puDown','wjetm')[0] 
    wjetm_dyjets.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
#     jerUp=template(background, 'DY+jets', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'DY+jets', 'JERDown', 'wjetm')[0]
#     wjetm_dyjets.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'DY+jets', 'JESUp', 'wjetm')[0] 
    jesDown=template(background, 'DY+jets', 'JESDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
    metUnclUp=template(background, 'DY+jets', 'METunclUp', 'wjetm')[0] 
    metUnclDown=template(background, 'DY+jets', 'METunclDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'DY+jets', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'DY+jets', 'L1PrefireDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'DY+jets', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'DY+jets', 'ISRDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'DY+jets', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'DY+jets', 'FSRDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'DY+jets', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'DY+jets', 'PDFDown', 'wjetm')[0]
    wjetm_dyjets.setParamEffect(pdf, pdfUp, pdfDown) 

    wjetm.addSample(wjetm_dyjets)


    # V Jets Background
    wjetm_vvTemplate = template(background, "VV", "nominal", "wjetm")
    wjetm_vv = rl.TemplateSample(
        ch_name + "_vvMC", rl.Sample.BACKGROUND, wjetm_vvTemplate
    )
    wjetm_vv.setParamEffect(lumi, lumi_unc)
    #     wjetm_vv.setParamEffect(trig_m, 1.01)
    #     wjetm_vv.setParamEffect(veto_tau, 1.03)
    #     wjetm_vv.setParamEffect(vv_norm, 1.2)
    #     wjetm_vv.setParamEffect(jec, 1.05)
    #     wjetm_vv.setParamEffect(id_m, 1.02)
    #     wjetm_vv.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "VV", "wjetm", wjetm_vv)
    #     wjetm_vv.autoMCStats()
    #     #trigger Electron
    trigmUp=template(background, 'VV', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'VV', 'trig_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'VV', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'VV', 'isolation_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "VV", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "VV", 'id_mDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'VV','puUp','wjetm')[0] 
    pileupDown=template(background, 'VV','puDown','wjetm')[0] 
    wjetm_vv.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
#     jerUp=template(background, 'VV', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'VV', 'JERDown', 'wjetm')[0]
#     wjetm_vv.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'VV', 'JESUp', 'wjetm')[0] 
    jesDown=template(background, 'VV', 'JESDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
    metUnclUp=template(background, 'VV', 'METunclUp', 'wjetm')[0] 
    metUnclDown=template(background, 'VV', 'METunclDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'VV', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'VV', 'L1PrefireDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'VV', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'VV', 'ISRDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'VV', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'VV', 'FSRDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'VV', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'VV', 'PDFDown', 'wjetm')[0]
    wjetm_vv.setParamEffect(pdf, pdfUp, pdfDown) 

    wjetm.addSample(wjetm_vv)

    # QCD background
    wjetm_qcdTemplate = template(background, "QCD", "nominal", "wjetm")
    wjetm_qcd = rl.TemplateSample(
        ch_name + "_qcdMC", rl.Sample.BACKGROUND, wjetm_qcdTemplate
    )
    wjetm_qcd.setParamEffect(lumi, lumi_unc)
    #     wjetm_qcd.setParamEffect(trig_m, 1.01)
    #     wjetm_qcd.setParamEffect(veto_tau, 1.03)
    #     wjetm_qcd.setParamEffect(qcde_norm, 2.0)
    #     wjetm_qcd.setParamEffect(jec, 1.05)
    #     wjetm_qcd.setParamEffect(id_m, 1.02)
    #     wjetm_qcd.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "QCD", "wjetm", wjetm_qcd)
    #     wjetm_qcd.autoMCStats()
    #trigger Muon
    trigmUp=template(background, 'QCD', 'trig_mUp', 'wjetm')[0] 
    trigmDown=template(background, 'QCD', 'trig_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'QCD', 'isolation_mUp', 'wjetm')[0] 
    isolation_mDown=template(background, 'QCD', 'isolation_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "QCD", 'id_mUp', 'wjetm')[0] 
    id_mDown=template(background, "QCD", 'id_mDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'QCD','puUp','wjetm')[0] 
    pileupDown=template(background, 'QCD','puDown','wjetm')[0] 
    wjetm_qcd.setParamEffect(pileup, pileupUp, pileupDown)

    #JEC_JER
#     jerUp=template(background, 'QCD', 'JERUp', 'wjetm')[0] 
#     jerDown=template(background, 'QCD', 'JERDown', 'wjetm')[0]
#     wjetm_qcd.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
#     jesUp=template(background, 'QCD', 'JESUp', 'wjetm')[0] 
#     jesDown=template(background, 'QCD', 'JESDown', 'wjetm')[0]
#     wjetm_qcd.setParamEffect(jec_jes, jesUp, jesDown)

    #met_uncl
#     metUnclUp=template(background, 'QCD', 'METunclUp', 'wjetm')[0] 
#     metUnclDown=template(background, 'QCD', 'METunclDown', 'wjetm')[0]
#     wjetm_qcd.setParamEffect(metUncl, metUnclUp, metUnclDown)


    #L1 prefire
    L1prefireUp=template(background, 'QCD', 'L1PrefireUp', 'wjetm')[0] 
    L1prefireDown=template(background, 'QCD', 'L1PrefireDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'QCD', 'ISRUp', 'wjetm')[0] 
    ISRDown=template(background, 'QCD', 'ISRDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'QCD', 'FSRUp', 'wjetm')[0] 
    FSRDown=template(background, 'QCD', 'FSRDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'QCD', 'PDFUp', 'wjetm')[0] 
    pdfDown=template(background, 'QCD', 'PDFDown', 'wjetm')[0]
    wjetm_qcd.setParamEffect(pdf, pdfUp, pdfDown) 

    wjetm.addSample(wjetm_qcd)

    return model

if __name__ == "__main__":
    if not os.path.exists("datacards"):
        os.mkdir("datacards")
    parser = OptionParser()
    parser.add_option("-y", "--year", help="year", dest="year", default="")
    parser.add_option("-f", "--file", help="file", dest="file", default="")
    (options, args) = parser.parse_args()
    year = options.year
    hist_scaled_file = options.file
    
    mT_corrected_binning_dict = {
#         "2018":np.linspace(0, 600, 21),
#         "2017":np.linspace(0, 600, 21),
#         "2016":np.linspace(0, 600, 21)
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
    hists = load("hists/" + hist_scaled_file + ".scaled")
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
    zjets_norm = rl.NuisanceParameter("zjets_norm", "lnN")
    wjets_norm = rl.NuisanceParameter("wjets_norm", "lnN")
    tt_norm = rl.NuisanceParameter("tt_norm", "lnN")
    id_m = rl.NuisanceParameter("id_m" + year, "shape")
    id_pho = rl.NuisanceParameter("id_pho" + year, "shape")
    reco_e = rl.NuisanceParameter("reco_e" + year, "shape")
    iso_mu = rl.NuisanceParameter("iso_mu" + year, "shape")
    trig_m = rl.NuisanceParameter("trig_m" + year, "shape")
    trig_met = rl.NuisanceParameter("trig_met" + year, "lnN")
    trig_pho = rl.NuisanceParameter("trig_pho" + year, "lnN")
    veto_tau = rl.NuisanceParameter("veto_tau" + year, "lnN")
    jec_jer = rl.NuisanceParameter("jec_jer" + year, "shape")
    jec_jes = rl.NuisanceParameter("jec_jes" + year, "shape")
    #btag = rl.NuisanceParameter("btag" + year, "shapeN")  # AK4 btag
    btag = rl.NuisanceParameter("btag" + year, "shape")  # AK4 btag
    ew1 = rl.NuisanceParameter("ew1", "lnN")
    ew2G = rl.NuisanceParameter("ew2G", "lnN")
    ew2W = rl.NuisanceParameter("ew2W", "lnN")
    ew2Z = rl.NuisanceParameter("ew2Z", "lnN")
    ew3G = rl.NuisanceParameter("ew3G", "lnN")
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
    L1prefire = rl.NuisanceParameter("L1prefire"+year, "shape")
    metUncl = rl.NuisanceParameter("metUncl"+year, "shape")
    isr = rl.NuisanceParameter("isr"+year, "shape")
    fsr = rl.NuisanceParameter("fsr"+year, "shape")
    pdf = rl.NuisanceParameter("pdf"+year, "shape")
    if "preVFP" in hist_scaled_file:
        year = year+"preVFP"
    elif "postVFP" in hist_scaled_file:
        year = year+"postVFP"
    with open(
        "data/"+"test_wjetm"+year+".model",
        "wb",) as fout:
        pickle.dump(model(year), fout, protocol=2)
