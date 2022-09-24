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
    "2018":np.array([120., 150., 180., 210.,  270., 390.,
        510.]),
    "2017":np.array([120., 150., 180., 210.,  270., 390.,
        510.]),
    "2016":np.array([120., 150., 180., 210.,  270., 390.,
        510.])

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
    sig_map['Mphi-1000_Mchi-1000'] = ('*Mphi-1000_Mchi-1000*')
    sig_map['Mphi-1000_Mchi-150'] = ('*Mphi-1000_Mchi-150*')
    sig_map['Mphi-1250_Mchi-150'] = ('*Mphi-1250_Mchi-150*')
    sig_map['Mphi-1495_Mchi-750'] = ('*Mphi-1495_Mchi-750*')
    sig_map['Mphi-1500_Mchi-1000'] = ('*Mphi-1500_Mchi-1000*')
    sig_map['Mphi-1500_Mchi-150'] = ('*Mphi-1500_Mchi-150*')
    sig_map['Mphi-1700_Mchi-800'] = ('*Mphi-1700_Mchi-800*')
    sig_map['Mphi-1750_Mchi-150'] = ('*Mphi-1750_Mchi-150*')
    sig_map['Mphi-1750_Mchi-700'] = ('*Mphi-1750_Mchi-700*')
    sig_map['Mphi-195_Mchi-100'] = ('*Mphi-195_Mchi-100*')
    sig_map['Mphi-1995_Mchi-1000'] = ('*Mphi-1995_Mchi-1000*')
    sig_map['Mphi-2000_Mchi-1500'] = ('*Mphi-2000_Mchi-1500*')
    sig_map['Mphi-2000_Mchi-150'] = ('*Mphi-2000_Mchi-150*')
    sig_map['Mphi-2000_Mchi-500'] = ('*Mphi-2000_Mchi-500*')
    sig_map['Mphi-200_Mchi-150'] = ('*Mphi-200_Mchi-150*')
    sig_map['Mphi-200_Mchi-50'] = ('*Mphi-200_Mchi-50*')
    sig_map['Mphi-2495_Mchi-1250'] = ('*Mphi-2495_Mchi-1250*')
    sig_map['Mphi-2500_Mchi-2000'] = ('*Mphi-2500_Mchi-2000*')
    sig_map['Mphi-2500_Mchi-750'] = ('*Mphi-2500_Mchi-750*')
    sig_map['Mphi-295_Mchi-150'] = ('*Mphi-295_Mchi-150*')
    sig_map['Mphi-2995_Mchi-1500'] = ('*Mphi-2995_Mchi-1500*')
    sig_map['Mphi-3000_Mchi-1000'] = ('*Mphi-3000_Mchi-1000*')
    sig_map['Mphi-3000_Mchi-2000'] = ('*Mphi-3000_Mchi-2000*')
    sig_map['Mphi-300_Mchi-100'] = ('*Mphi-300_Mchi-100*')
    sig_map['Mphi-300_Mchi-300'] = ('*Mphi-300_Mchi-300*')
    sig_map['Mphi-495_Mchi-250'] = ('*Mphi-495_Mchi-250*')
    sig_map['Mphi-500_Mchi-150'] = ('*Mphi-500_Mchi-150*')
    sig_map['Mphi-500_Mchi-500'] = ('*Mphi-500_Mchi-500*')
    sig_map['Mphi-750_Mchi-150'] = ('*Mphi-750_Mchi-150*')
    sig_map['Mphi-995_Mchi-500'] = ('*Mphi-995_Mchi-500*')
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
#     addSyst(dictionary, process, region, templ, ew2W, "ew2W")
    addSyst(dictionary, process, region, templ, ew2Z, "ew2Z")
    addSyst(dictionary, process, region, templ, ew3W, "ew3W")
    addSyst(dictionary, process, region, templ, ew3Z, "ew3Z")
#     addSyst(dictionary, process, region, templ, mix, "mix")
#     addSyst(dictionary, process, region, templ, qcd1, "qcd1")
#     addSyst(dictionary, process, region, templ, qcd2, "qcd2")
#     addSyst(dictionary, process, region, templ, qcd3, "qcd3")
#     addSyst(dictionary, process, region, templ, muR, "muR")
#     addSyst(dictionary, process, region, templ, muF, "muF")
    
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
#         output[zerobins1]  = 1.
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
    ch_name = "ttbarm" + str(year)
    # Create a channel/Control region
    ttbarm = rl.Channel(ch_name)
    # add the Channel/CR to the model
    model.addChannel(ttbarm)
    


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
        ttbarm.setObservation(template(data, "SingleMuon", "data", "ttbarm"))
    else:
        ttbarm.setObservation(template(data, "SingleMuon", "data","ttbarm"))

    #template for signal
    for s in signal['ttbarm'].identifiers('process'):
        print(s)
        if 'Mphi-200' not in str(s): continue
        ttbarm_signalTemplate = template(signal,s,'nominal','ttbarm')
        ttbarm_signal=rl.TemplateSample(ch_name+'_'+str(s), rl.Sample.SIGNAL, ttbarm_signalTemplate)
#         ttbarm_signal.setParamEffect(lumi, lumi_unc)
    #     sr_signal.setParamEffect(trig_met, 1.01)
    #     sr_signal.setParamEffect(veto_tau, 1.03)
    #     sr_signal.setParamEffect(jec, 1.05)
        #btag
        btagUp=template(signal, s,'btagUp','ttbarm')[0]
        btagDown=template(signal, s,'btagDown','ttbarm')[0]
#         ttbarm_signal.setParamEffect(btag, btagUp, btagDown)
    #         #trigger Electron
        trigmUp=template(signal, s, 'trig_mUp', 'ttbarm')[0] 
        trigmDown=template(signal, s, 'trig_mDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(trig_m, trigmUp, trigmDown)

        #pileup
        pileupUp=template(signal, s,'puUp','ttbarm')[0] 
        pileupDown=template(signal, s,'puDown','ttbarm')[0] 
#         ttbarm_signal.setParamEffect(pileup, pileupUp, pileupDown)
        #iso Muon
        isolation_mUp=template(signal, s, 'isolation_mUp', 'ttbarm')[0] 
        isolation_mDown=template(signal, s, 'isolation_mDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
        #id Muon
        id_mUp=template(signal, s, 'id_mUp', 'ttbarm')[0] 
        id_mDown=template(signal, s, 'id_mDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(id_m, id_mUp, id_mDown)        


        #JEC_JER
        jerUp=template(signal, s, 'JERUp', 'ttbarm')[0] 
        jerDown=template(signal, s, 'JERDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(jec_jer, jerUp, jerDown)

        #JEC_JEs
        jesUp=template(signal, s, 'JESUp', 'ttbarm')[0] 
        jesDown=template(signal, s, 'JESDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(jec_jes, jesUp, jesDown)


        #L1 prefire
        L1prefireUp=template(signal, s, 'L1PrefireUp', 'ttbarm')[0] 
        L1prefireDown=template(signal, s, 'L1PrefireDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #L1 ISR
        ISRUp=template(signal, s, 'ISRUp', 'ttbarm')[0] 
        ISRDown=template(signal, s, 'ISRDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(isr, ISRUp, ISRDown)

        #L1 FSR
        FSRUp=template(signal, s, 'FSRUp', 'ttbarm')[0] 
        FSRDown=template(signal, s, 'FSRDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(fsr, FSRUp, FSRDown)

        #pdf 
        pdfUp=template(signal, s, 'PDFUp', 'ttbarm')[0] 
        pdfDown=template(signal, s, 'PDFDown', 'ttbarm')[0]
#         ttbarm_signal.setParamEffect(pdf, pdfUp, pdfDown) 

    #         ttbarm_signal.autoMCStats()
        ttbarm.addSample(ttbarm_signal)
    ###
    # W(->lnu)+jets data-driven model
    ###

    ttbarm_wjetsTemplate = template(background, "W+jets", "nominal", "ttbarm", read_sumw2=True)
    ttbarm_wjetsMC = rl.TemplateSample(
        ch_name + "_wjetsMC",
        rl.Sample.BACKGROUND,
        ttbarm_wjetsTemplate
    )
    ttbarm_wjetsMC.setParamEffect(lumi, lumi_unc)
    #     ttbarm_wjetsMC.setParamEffect(trig_m, 1.01)
    #     ttbarm_wjetsMC.setParamEffect(veto_tau, 1.03)
    #     ttbarm_wjetsMC.setParamEffect(wjets_norm, 1.4)
    #     ttbarm_wjetsMC.setParamEffect(jec, 1.05)
    #     ttbarm_wjetsMC.setParamEffect(id_m, 1.02)
    #     ttbarm_wjetsMC.setParamEffect(reco_e, 1.02)
    ttbarm_wjetsMC.autoMCStats(epsilon=1e-5)
    addBtagSyst(background, "W+jets", "ttbarm", ttbarm_wjetsMC)
#     addVJetsSyst(background, "W+jets", "ttbarm", ttbarm_wjetsMC)
    #trigger Muon
    trigmUp=template(background, 'W+jets', 'trig_mUp', 'ttbarm')[0] 
    trigmDown=template(background, 'W+jets', 'trig_mDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(trig_m, trigmUp, trigmDown)

    #pileup
    pileupUp=template(background, 'W+jets','puUp','ttbarm')[0] 
    pileupDown=template(background, 'W+jets','puDown','ttbarm')[0] 
    ttbarm_wjetsMC.setParamEffect(pileup, pileupUp, pileupDown)
    print("#### W+Jets###")
    print('pu_up ', pileupUp)
    print('pu_do ', pileupDown)
    #iso Muon
    isolation_mUp=template(background, 'W+jets', 'isolation_mUp', 'ttbarm')[0] 
    isolation_mDown=template(background, 'W+jets', 'isolation_mDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "W+jets", 'id_mUp', 'ttbarm')[0] 
    id_mDown=template(background, "W+jets", 'id_mDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(id_m, id_mUp, id_mDown)


    #JEC_JER
    jerUp=template(background, 'W+jets', 'JERUp', 'ttbarm')[0] 
    jerDown=template(background, 'W+jets', 'JERDown', 'ttbarm')[0]
#     ttbarm_wjetsMC.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'W+jets', 'JESUp', 'ttbarm')[0] 
    jesDown=template(background, 'W+jets', 'JESDown', 'ttbarm')[0]
#     ttbarm_wjetsMC.setParamEffect(jec_jes, jesUp, jesDown)
# 

    #L1 prefire
    L1prefireUp=template(background, 'W+jets', 'L1PrefireUp', 'ttbarm')[0] 
    L1prefireDown=template(background, 'W+jets', 'L1PrefireDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'W+jets', 'ISRUp', 'ttbarm')[0] 
    ISRDown=template(background, 'W+jets', 'ISRDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'W+jets', 'FSRUp', 'ttbarm')[0] 
    FSRDown=template(background, 'W+jets', 'FSRDown', 'ttbarm')[0]
#     ttbarm_wjetsMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'W+jets', 'PDFUp', 'ttbarm')[0] 
    pdfDown=template(background, 'W+jets', 'PDFDown', 'ttbarm')[0]
    ttbarm_wjetsMC.setParamEffect(pdf, pdfUp, pdfDown)    
    ttbarm.addSample(ttbarm_wjetsMC)

    # ###
    # # top-antitop model
    # ###

    ttbarm_ttTemplate = template(background, "TT", "nominal","ttbarm", read_sumw2=True)
    ttbarm_ttMC = rl.TemplateSample(
        ch_name + "_ttMC",
        rl.Sample.BACKGROUND,
        ttbarm_ttTemplate
    )
    ttbarm_ttMC.setParamEffect(lumi, lumi_unc)
    #     ttbarm_ttMC.setParamEffect(trig_m, 1.01)
    #     ttbarm_ttMC.setParamEffect(veto_tau, 1.03)
    #     ttbarm_ttMC.setParamEffect(jec, 1.05)
    #     ttbarm_ttMC.setParamEffect(id_m, 1.02)
    #     ttbarm_ttMC.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "TT", "ttbarm", ttbarm_ttMC)
#     ttbarm_ttMC.autoMCStats(epsilon=1e-5)
    #trigger Muon
    trigmUp=template(background, 'TT', 'trig_mUp', 'ttbarm')[0] 
    trigmDown=template(background, 'TT', 'trig_mDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'TT', 'isolation_mUp', 'ttbarm')[0] 
    isolation_mDown=template(background, 'TT', 'isolation_mDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "TT", 'id_mUp', 'ttbarm')[0] 
    id_mDown=template(background, "TT", 'id_mDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'TT','puUp','ttbarm')[0] 
    pileupDown=template(background, 'TT','puDown','ttbarm')[0] 
    ttbarm_ttMC.setParamEffect(pileup, pileupUp, pileupDown)
    print("#### TT ###")
    print('pu_up ', pileupUp)
    print('pu_do ', pileupDown)
    # ttbarm_ttMC.setParamEffect(ttMC_norm, 1.2)

    #JEC_JER
    jerUp=template(background, 'TT', 'JERUp', 'ttbarm')[0] 
    jerDown=template(background, 'TT', 'JERDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'TT', 'JESUp', 'ttbarm')[0] 
    jesDown=template(background, 'TT', 'JESDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(jec_jes, jesUp, jesDown)


    #L1 prefire
    L1prefireUp=template(background, 'TT', 'L1PrefireUp', 'ttbarm')[0] 
    L1prefireDown=template(background, 'TT', 'L1PrefireDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #ISR
    ISRUp=template(background, 'TT', 'ISRUp', 'ttbarm')[0] 
    ISRDown=template(background, 'TT', 'ISRDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(isr, ISRUp, ISRDown)

    #FSR
    FSRUp=template(background, 'TT', 'FSRUp', 'ttbarm')[0] 
    FSRDown=template(background, 'TT', 'FSRDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'TT', 'PDFUp', 'ttbarm')[0] 
    pdfDown=template(background, 'TT', 'PDFDown', 'ttbarm')[0]
    ttbarm_ttMC.setParamEffect(pdf, pdfUp, pdfDown)    

    ttbarm.addSample(ttbarm_ttMC)

    # ###
    # # Other MC-driven processes
    # ###
    # ST Background
    ttbarm_stTemplate = template(background, "ST", "nominal", "ttbarm", read_sumw2=True)
    ttbarm_st = rl.TemplateSample(
        ch_name + "_stMC", rl.Sample.BACKGROUND, ttbarm_stTemplate
    )
    ttbarm_st.setParamEffect(lumi, lumi_unc)
    #     ttbarm_st.setParamEffect(trig_m, 1.01)
    #     ttbarm_st.setParamEffect(veto_tau, 1.03)
    #     ttbarm_st.setParamEffect(st_norm, 1.2) something related to signal
    #     ttbarm_st.setParamEffect(jec, 1.05)
    #     ttbarm_st.setParamEffect(id_m, 1.02)
    #     ttbarm_st.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "ST", "ttbarm", ttbarm_st)
#     ttbarm_st.autoMCStats(epsilon=1e-5)
    #trigger Muon
    trigmUp=template(background, 'ST', 'trig_mUp', 'ttbarm')[0] 
    trigmDown=template(background, 'ST', 'trig_mDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'ST', 'isolation_mUp', 'ttbarm')[0] 
    isolation_mDown=template(background, 'ST', 'isolation_mDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "ST", 'id_mUp', 'ttbarm')[0] 
    id_mDown=template(background, "ST", 'id_mDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'ST','puUp','ttbarm')[0] 
    pileupDown=template(background, 'ST','puDown','ttbarm')[0] 
    ttbarm_st.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
    jerUp=template(background, 'ST', 'JERUp', 'ttbarm')[0] 
    jerDown=template(background, 'ST', 'JERDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'ST', 'JESUp', 'ttbarm')[0] 
    jesDown=template(background, 'ST', 'JESDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(jec_jes, jesUp, jesDown)


    #L1 prefire
    L1prefireUp=template(background, 'ST', 'L1PrefireUp', 'ttbarm')[0] 
    L1prefireDown=template(background, 'ST', 'L1PrefireDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'ST', 'ISRUp', 'ttbarm')[0] 
    ISRDown=template(background, 'ST', 'ISRDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'ST', 'FSRUp', 'ttbarm')[0] 
    FSRDown=template(background, 'ST', 'FSRDown', 'ttbarm')[0]
    ttbarm_st.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'ST', 'PDFUp', 'ttbarm')[0] 
    pdfDown=template(background, 'ST', 'PDFDown', 'ttbarm')[0]
#     ttbarm_st.setParamEffect(pdf, pdfUp, pdfDown)  
    print("#### ST ###")
    print('pdf_up ', pdfUp)
    print('pdf_do ', pdfDown)
    ttbarm.addSample(ttbarm_st)

    # DY Jet Background
    
    ttbarm_dyjetsTemplate = template(background, "DY+jets", "nominal","ttbarm")
    ttbarm_dyjets = rl.TemplateSample(
        ch_name + "_dyjetsMC", rl.Sample.BACKGROUND, ttbarm_dyjetsTemplate
    )
    ttbarm_dyjets.setParamEffect(lumi, lumi_unc)
    #     ttbarm_dyjets.setParamEffect(trig_m, 1.01)
    #     ttbarm_dyjets.setParamEffect(veto_tau, 1.03)
    #     ttbarm_dyjets.setParamEffect(zjetsMC_norm, 1.4)
    #     ttbarm_dyjets.setParamEffect(jec, 1.05)
    #     ttbarm_dyjets.setParamEffect(id_m, 1.02)
    #     ttbarm_dyjets.setParamEffect(reco_e, 1.02)
#     addBtagSyst(background, "DY+jets", "ttbarm", ttbarm_dyjets)
#     addVJetsSyst(background,"DY+jets", "ttbarm", ttbarm_dyjets)
#     ttbarm_dyjets.autoMCStats()
    #trigger Muon
    trigmUp=template(background, 'DY+jets', 'trig_mUp', 'ttbarm')[0] 
    trigmDown=template(background, 'DY+jets', 'trig_mDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'DY+jets', 'isolation_mUp', 'ttbarm')[0] 
    isolation_mDown=template(background, 'DY+jets', 'isolation_mDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "DY+jets", 'id_mUp', 'ttbarm')[0] 
    id_mDown=template(background, "DY+jets", 'id_mDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'DY+jets','puUp','ttbarm')[0] 
    pileupDown=template(background, 'DY+jets','puDown','ttbarm')[0] 
#     ttbarm_dyjets.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
    jerUp=template(background, 'DY+jets', 'JERUp', 'ttbarm')[0] 
    jerDown=template(background, 'DY+jets', 'JERDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'DY+jets', 'JESUp', 'ttbarm')[0] 
    jesDown=template(background, 'DY+jets', 'JESDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(jec_jes, jesUp, jesDown)

    #L1 prefire
    L1prefireUp=template(background, 'DY+jets', 'L1PrefireUp', 'ttbarm')[0] 
    L1prefireDown=template(background, 'DY+jets', 'L1PrefireDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'DY+jets', 'ISRUp', 'ttbarm')[0] 
    ISRDown=template(background, 'DY+jets', 'ISRDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'DY+jets', 'FSRUp', 'ttbarm')[0] 
    FSRDown=template(background, 'DY+jets', 'FSRDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'DY+jets', 'PDFUp', 'ttbarm')[0] 
    pdfDown=template(background, 'DY+jets', 'PDFDown', 'ttbarm')[0]
#     ttbarm_dyjets.setParamEffect(pdf, pdfUp, pdfDown) 

    ttbarm.addSample(ttbarm_dyjets)


    # V Jets Background
    ttbarm_vvTemplate = template(background, "VV", "nominal", "ttbarm", read_sumw2=True)
    ttbarm_vv = rl.TemplateSample(
        ch_name + "_vvMC", rl.Sample.BACKGROUND, ttbarm_vvTemplate
    )
    ttbarm_vv.setParamEffect(lumi, lumi_unc)
    #     ttbarm_vv.setParamEffect(trig_m, 1.01)
    #     ttbarm_vv.setParamEffect(veto_tau, 1.03)
    #     ttbarm_vv.setParamEffect(vv_norm, 1.2)
    #     ttbarm_vv.setParamEffect(jec, 1.05)
    #     ttbarm_vv.setParamEffect(id_m, 1.02)
    #     ttbarm_vv.setParamEffect(reco_e, 1.02)
#     addBtagSyst(background, "VV", "ttbarm", ttbarm_vv)
#     ttbarm_vv.autoMCStats()
    #     #trigger Electron
    trigmUp=template(background, 'VV', 'trig_mUp', 'ttbarm')[0] 
    trigmDown=template(background, 'VV', 'trig_mDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(trig_m, trigmUp, trigmDown)
    #iso Muon
    isolation_mUp=template(background, 'VV', 'isolation_mUp', 'ttbarm')[0] 
    isolation_mDown=template(background, 'VV', 'isolation_mDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(iso_mu, isolation_mUp, isolation_mDown)
    #id Muon
    id_mUp=template(background, "VV", 'id_mUp', 'ttbarm')[0] 
    id_mDown=template(background, "VV", 'id_mDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(id_m, id_mUp, id_mDown)
    #pileup
    pileupUp=template(background, 'VV','puUp','ttbarm')[0] 
    pileupDown=template(background, 'VV','puDown','ttbarm')[0] 
#     ttbarm_vv.setParamEffect(pileup, pileupUp, pileupDown)


    #JEC_JER
    jerUp=template(background, 'VV', 'JERUp', 'ttbarm')[0] 
    jerDown=template(background, 'VV', 'JERDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'VV', 'JESUp', 'ttbarm')[0] 
    jesDown=template(background, 'VV', 'JESDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(jec_jes, jesUp, jesDown)



    #L1 prefire
    L1prefireUp=template(background, 'VV', 'L1PrefireUp', 'ttbarm')[0] 
    L1prefireDown=template(background, 'VV', 'L1PrefireDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'VV', 'ISRUp', 'ttbarm')[0] 
    ISRDown=template(background, 'VV', 'ISRDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'VV', 'FSRUp', 'ttbarm')[0] 
    FSRDown=template(background, 'VV', 'FSRDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'VV', 'PDFUp', 'ttbarm')[0] 
    pdfDown=template(background, 'VV', 'PDFDown', 'ttbarm')[0]
#     ttbarm_vv.setParamEffect(pdf, pdfUp, pdfDown) 

    ttbarm.addSample(ttbarm_vv)
    


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
        "2018":np.array([120., 150., 180., 210.,  270., 390.,
            510.]),
        "2017":np.array([120., 150., 180., 210.,  270., 390.,
            510.]),
        "2016":np.array([120., 150., 180., 210.,  270., 390.,
            510.])

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
#     trig_met = rl.NuisanceParameter("trig_met" + year, "lnN")
#     trig_pho = rl.NuisanceParameter("trig_pho" + year, "lnN")
#     veto_tau = rl.NuisanceParameter("veto_tau" + year, "lnN")
    jec_jer = rl.NuisanceParameter("jec_jer" + year, "shape")
    jec_jes = rl.NuisanceParameter("jec_jes" + year, "shape")
    #btag = rl.NuisanceParameter("btag" + year, "shapeN")  # AK4 btag
    btag = rl.NuisanceParameter("btag" + year, "shape")  # AK4 btag
    ew1 = rl.NuisanceParameter("ew1", "shape")
    ew2G = rl.NuisanceParameter("ew2G", "shape")
    ew2W = rl.NuisanceParameter("ew2W", "shape")
    ew2Z = rl.NuisanceParameter("ew2Z", "shape")
    ew3G = rl.NuisanceParameter("ew3G", "shape")
    ew3W = rl.NuisanceParameter("ew3W", "shape")
    ew3Z = rl.NuisanceParameter("ew3Z", "shape")
    mix = rl.NuisanceParameter("mix", "shape")
    muF = rl.NuisanceParameter("muF", "shape")
    muR = rl.NuisanceParameter("muR", "shape")
    qcd1 = rl.NuisanceParameter("qcd1", "shape")
    qcd2 = rl.NuisanceParameter("qcd2", "shape")
    qcd3 = rl.NuisanceParameter("qcd3", "shape")
    whf_fraction = rl.NuisanceParameter("whf_fraction", "lnN")
    zhf_fraction = rl.NuisanceParameter("zhf_fraction", "lnN")
    #ghf_fraction = rl.NuisanceParameter("ghf_fraction", "shapeN")
    if "2018" not in year:
        L1prefire = rl.NuisanceParameter("L1prefire", "shape")
    else: 
        L1prefire = rl.NuisanceParameter("L1prefire"+year, "shape")
    pileup = rl.NuisanceParameter("pileup"+year, "lnN")
    L1prefire = rl.NuisanceParameter("L1prefire"+year, "shape")
    metUncl = rl.NuisanceParameter("metUncl"+year, "shape")
    isr = rl.NuisanceParameter("isr", "shape")
    fsr = rl.NuisanceParameter("fsr", "shape")
    pdf = rl.NuisanceParameter("pdf", "shape")
    if "preVFP" in hist_scaled_file:
        year = year+"preVFP"
    elif "postVFP" in hist_scaled_file:
        year = year+"postVFP"
    with open(
        "data/"+"test_ttbarm"+'.'+year+".model",
        "wb",) as fout:
        pickle.dump(model(year), fout, protocol=2)
