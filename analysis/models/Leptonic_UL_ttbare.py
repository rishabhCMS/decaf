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
    addSyst(dictionary, process, region, templ, ew2W, "ew2W")
    addSyst(dictionary, process, region, templ, ew2Z, "ew2Z")
    addSyst(dictionary, process, region, templ, ew3W, "ew3W")
    addSyst(dictionary, process, region, templ, ew3Z, "ew3Z")
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
    ch_name = "ttbare"+str(year) 
    # Create a channel/Control region
    ttbare = rl.Channel(ch_name)
    # add the Channel/CR to the model
    model.addChannel(ttbare)
    
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
        ttbare.setObservation(template(data, "EGamma", "data", "ttbare"))
    else:
        ttbare.setObservation(template(data, "SingleElectron", "data","ttbare"))
        
    #template for signal
    for s in signal['ttbare'].identifiers('process'):
        print(s)
        if 'Mphi-200' not in str(s): continue
        ttbare_signalTemplate = template(signal,s,'nominal','ttbare')
        ttbare_signal=rl.TemplateSample(ch_name+'_'+str(s), rl.Sample.SIGNAL, ttbare_signalTemplate)
#         ttbare_signal.setParamEffect(lumi, lumi_unc)
    #     sr_signal.setParamEffect(trig_met, 1.01)
    #     sr_signal.setParamEffect(veto_tau, 1.03)
    #     sr_signal.setParamEffect(jec, 1.05)
        #btag
        btagUp=template(signal, s,'btagUp','ttbare')[0]
        btagDown=template(signal, s,'btagDown','ttbare')[0]
#         ttbare_signal.setParamEffect(btag, btagUp, btagDown)
    #         #trigger Electron
        trigeUp=template(signal, s, 'trig_eUp', 'ttbare')[0] 
        trigeDown=template(signal, s, 'trig_eDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(trig_e, trigeUp, trigeDown)
    #         #id Electron
        id_eUp=template(signal, s, 'id_eUp', 'ttbare')[0] 
        id_eDown=template(signal, s, 'id_eDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(id_e, id_eUp, id_eDown)
    #pileup
        pileupUp=template(signal, s,'puUp','ttbare')[0] 
        pileupDown=template(signal, s,'puDown','ttbare')[0] 
#         ttbare_signal.setParamEffect(pileup, pileupUp, pileupDown)

        #reco Electron
        recoeUp=template(signal, s, 'reco_eUp', 'ttbare')[0] 
        recoDown=template(signal, s, 'reco_eDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(reco_e, recoeUp, recoDown)

        #JEC_JER
        jerUp=template(signal, s, 'JERUp', 'ttbare')[0] 
        jerDown=template(signal, s, 'JERDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(jec_jer, jerUp, jerDown)

        #JEC_JEs
        jesUp=template(signal, s, 'JESUp', 'ttbare')[0] 
        jesDown=template(signal, s, 'JESDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(jec_jes, jesUp, jesDown)

        #L1 prefire
        L1prefireUp=template(signal, s, 'L1PrefireUp', 'ttbare')[0] 
        L1prefireDown=template(signal, s, 'L1PrefireDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #L1 ISR
        ISRUp=template(signal, s, 'ISRUp', 'ttbare')[0] 
        ISRDown=template(signal, s, 'ISRDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(isr, ISRUp, ISRDown)

        #L1 FSR
        FSRUp=template(signal, s, 'FSRUp', 'ttbare')[0] 
        FSRDown=template(signal, s, 'FSRDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(fsr, FSRUp, FSRDown)

        #pdf 
        pdfUp=template(signal, s, 'PDFUp', 'ttbare')[0] 
        pdfDown=template(signal, s, 'PDFDown', 'ttbare')[0]
#         ttbare_signal.setParamEffect(pdf, pdfUp, pdfDown) 

    #         ttbare_signal.autoMCStats()
        ttbare.addSample(ttbare_signal)
    ###
    # W(->lnu)+jets data-driven model
    ###

    ttbare_wjetsTemplate = template(background, "W+jets", "nominal", "ttbare", read_sumw2=True)
    ttbare_wjetsMC = rl.TemplateSample(
        ch_name + "_wjetsMC",
        rl.Sample.BACKGROUND,
        ttbare_wjetsTemplate
    )
    ttbare_wjetsMC.setParamEffect(lumi, lumi_unc)
    #     ttbare_wjetsMC.setParamEffect(trig_e, 1.01)
    #     ttbare_wjetsMC.setParamEffect(veto_tau, 1.03)
    #     ttbare_wjetsMC.setParamEffect(wjets_norm, 1.4)
    #     ttbare_wjetsMC.setParamEffect(jec, 1.05)
    #     ttbare_wjetsMC.setParamEffect(id_e, 1.02)
    #     ttbare_wjetsMC.setParamEffect(reco_e, 1.02)
    #     ttbare_wjetsMC.autoMCStats()
    addBtagSyst(background, "W+jets", "ttbare", ttbare_wjetsMC)
        #trigger Electron
    trigeUp=template(background, 'W+jets', 'trig_eUp', 'ttbare')[0] 
    trigeDown=template(background, 'W+jets', 'trig_eDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(trig_e, trigeUp, trigeDown)
    #     #id Electron
    id_eUp=template(background, "W+jets", 'id_eUp', 'ttbare')[0] 
    id_eDown=template(background, "W+jets", 'id_eDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(id_e, id_eUp, id_eDown)
    #pileup
    pileupUp=template(background, 'W+jets','puUp','ttbare')[0] 
    pileupDown=template(background, 'W+jets','puDown','ttbare')[0] 
    ttbare_wjetsMC.setParamEffect(pileup, pileupUp, pileupDown)
#     reco Electron
    recoeUp=template(background, 'W+jets', 'reco_eUp', 'ttbare')[0] 
    recoDown=template(background, 'W+jets', 'reco_eDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(reco_e, recoeUp, recoDown)

    addVJetsSyst(background, "W+jets", "ttbare", ttbare_wjetsMC)

    #     JEC_JER
    jerUp=template(background, 'W+jets', 'JERUp', 'ttbare')[0] 
    jerDown=template(background, 'W+jets', 'JERDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(jec_jer, jerUp, jerDown)

    #     JEC_JEs
    jesUp=template(background, 'W+jets', 'JESUp', 'ttbare')[0] 
    jesDown=template(background, 'W+jets', 'JESDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(jec_jes, jesUp, jesDown)

        #L1 prefire
    L1prefireUp=template(background, 'W+jets', 'L1PrefireUp', 'ttbare')[0] 
    L1prefireDown=template(background, 'W+jets', 'L1PrefireDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #L1 ISR
    ISRUp=template(background, 'W+jets', 'ISRUp', 'ttbare')[0] 
    ISRDown=template(background, 'W+jets', 'ISRDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'W+jets', 'FSRUp', 'ttbare')[0] 
    FSRDown=template(background, 'W+jets', 'FSRDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'W+jets', 'PDFUp', 'ttbare')[0] 
    pdfDown=template(background, 'W+jets', 'PDFDown', 'ttbare')[0]
    ttbare_wjetsMC.setParamEffect(pdf, pdfUp, pdfDown)    
    ttbare.addSample(ttbare_wjetsMC)

        # ###
        # # top-antitop model
        # ###

    ttbare_ttTemplate = template(background, "TT", "nominal","ttbare", read_sumw2=True)
    ttbare_ttMC = rl.TemplateSample(
        ch_name + "_ttMC",
        rl.Sample.BACKGROUND,
        ttbare_ttTemplate
        )
    ttbare_ttMC.setParamEffect(lumi, lumi_unc)
    #     ttbare_ttMC.setParamEffect(trig_e, 1.01)
    #     ttbare_ttMC.setParamEffect(veto_tau, 1.03)
    #     ttbare_ttMC.setParamEffect(jec, 1.05)
    #     ttbare_ttMC.setParamEffect(id_e, 1.02)
    #     ttbare_ttMC.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "TT", "ttbare", ttbare_ttMC)
    #     ttbare_ttMC.autoMCStats()
    #trigger Electron
    trigeUp=template(background, 'TT', 'trig_eUp', 'ttbare')[0] 
    trigeDown=template(background, 'TT', 'trig_eDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(trig_e, trigeUp, trigeDown)
    #id Electron
    id_eUp=template(background, "TT", 'id_eUp', 'ttbare')[0] 
    id_eDown=template(background, "TT", 'id_eDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(id_e, id_eUp, id_eDown)
    #pileup
    pileupUp=template(background, 'TT','puUp','ttbare')[0] 
    pileupDown=template(background, 'TT','puDown','ttbare')[0] 
    ttbare_ttMC.setParamEffect(pileup, pileupUp, pileupDown)
    #reco Electron
    recoeUp=template(background, 'TT', 'reco_eUp', 'ttbare')[0] 
    recoDown=template(background, 'TT', 'reco_eDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(reco_e, recoeUp, recoDown)
#     ttbare_ttMC.setParamEffect(ttMC_norm, 1.2)

    #JEC_JER
    jerUp=template(background, 'TT', 'JERUp', 'ttbare')[0] 
    jerDown=template(background, 'TT', 'JERDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(jec_jer, jerUp, jerDown)

        #JEC_JEs
    jesUp=template(background, 'TT', 'JESUp', 'ttbare')[0] 
    jesDown=template(background, 'TT', 'JESDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(jec_jes, jesUp, jesDown)

    #L1 prefire
    L1prefireUp=template(background, 'TT', 'L1PrefireUp', 'ttbare')[0] 
    L1prefireDown=template(background, 'TT', 'L1PrefireDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #ISR
    ISRUp=template(background, 'TT', 'ISRUp', 'ttbare')[0] 
    ISRDown=template(background, 'TT', 'ISRDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(isr, ISRUp, ISRDown)

    #FSR
    FSRUp=template(background, 'TT', 'FSRUp', 'ttbare')[0] 
    FSRDown=template(background, 'TT', 'FSRDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'TT', 'PDFUp', 'ttbare')[0] 
    pdfDown=template(background, 'TT', 'PDFDown', 'ttbare')[0]
    ttbare_ttMC.setParamEffect(pdf, pdfUp, pdfDown)    

    ttbare.addSample(ttbare_ttMC)

        # ###
        # # Other MC-driven processes
        # ###
        # ST Background
    ttbare_stTemplate = template(background, "ST", "nominal", "ttbare")
    ttbare_st = rl.TemplateSample(
        ch_name + "_stMC", rl.Sample.BACKGROUND, ttbare_stTemplate
    )
    ttbare_st.setParamEffect(lumi, lumi_unc)
    #     ttbare_st.setParamEffect(trig_e, 1.01)
    #     ttbare_st.setParamEffect(veto_tau, 1.03)
    #     ttbare_st.setParamEffect(st_norm, 1.2) something related to signal
    #     ttbare_st.setParamEffect(jec, 1.05)
    #     ttbare_st.setParamEffect(id_e, 1.02)
    #     ttbare_st.setParamEffect(reco_e, 1.02)
    addBtagSyst(background, "ST", "ttbare", ttbare_st)
    #     ttbare_st.autoMCStats()
    #trigger Electron
    trigeUp=template(background, 'ST', 'trig_eUp', 'ttbare')[0] 
    trigeDown=template(background, 'ST', 'trig_eDown', 'ttbare')[0]
    ttbare_st.setParamEffect(trig_e, trigeUp, trigeDown)
    #id Electron
    id_eUp=template(background, "ST", 'id_eUp', 'ttbare')[0] 
    id_eDown=template(background, "ST", 'id_eDown', 'ttbare')[0]
    ttbare_st.setParamEffect(id_e, id_eUp, id_eDown)
    #pileup
    pileupUp=template(background, 'ST','puUp','ttbare')[0] 
    pileupDown=template(background, 'ST','puDown','ttbare')[0] 
    ttbare_st.setParamEffect(pileup, pileupUp, pileupDown)
    #reco Electron
    recoeUp=template(background, 'ST', 'reco_eUp', 'ttbare')[0] 
    recoDown=template(background, 'ST', 'reco_eDown', 'ttbare')[0]
    ttbare_st.setParamEffect(reco_e, recoeUp, recoDown)

        #JEC_JER
    jerUp=template(background, 'ST', 'JERUp', 'ttbare')[0] 
    jerDown=template(background, 'ST', 'JERDown', 'ttbare')[0]
    ttbare_st.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'ST', 'JESUp', 'ttbare')[0] 
    jesDown=template(background, 'ST', 'JESDown', 'ttbare')[0]
    ttbare_st.setParamEffect(jec_jes, jesUp, jesDown)


    #L1 prefire
    L1prefireUp=template(background, 'ST', 'L1PrefireUp', 'ttbare')[0] 
    L1prefireDown=template(background, 'ST', 'L1PrefireDown', 'ttbare')[0]
    ttbare_st.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'ST', 'ISRUp', 'ttbare')[0] 
    ISRDown=template(background, 'ST', 'ISRDown', 'ttbare')[0]
    ttbare_st.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'ST', 'FSRUp', 'ttbare')[0] 
    FSRDown=template(background, 'ST', 'FSRDown', 'ttbare')[0]
    ttbare_st.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'ST', 'PDFUp', 'ttbare')[0] 
    pdfDown=template(background, 'ST', 'PDFDown', 'ttbare')[0]
    ttbare_st.setParamEffect(pdf, pdfUp, pdfDown)  

    ttbare.addSample(ttbare_st)

        # DY Jet Background
    ttbare_dyjetsTemplate = template(background, "DY+jets", "nominal","ttbare")
    ttbare_dyjets = rl.TemplateSample(
        ch_name + "_dyjetsMC", rl.Sample.BACKGROUND, ttbare_dyjetsTemplate
    )
#     ttbare_dyjets.setParamEffect(lumi, lumi_unc)
    #     ttbare_dyjets.setParamEffect(trig_e, 1.01)
    #     ttbare_dyjets.setParamEffect(veto_tau, 1.03)
    #     ttbare_dyjets.setParamEffect(zjetsMC_norm, 1.4)
    #     ttbare_dyjets.setParamEffect(jec, 1.05)
    #     ttbare_dyjets.setParamEffect(id_e, 1.02)
    #     ttbare_dyjets.setParamEffect(reco_e, 1.02)
#     addBtagSyst(background, "DY+jets", "ttbare", ttbare_dyjets)
#     addVJetsSyst(background,"DY+jets", "ttbare", ttbare_dyjets)
    #     ttbare_dyjets.autoMCStats()
    #trigger Electron
    trigeUp=template(background, 'DY+jets', 'trig_eUp', 'ttbare')[0] 
    trigeDown=template(background, 'DY+jets', 'trig_eDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(trig_e, trigeUp, trigeDown)
    #id Electron
    id_eUp=template(background, "DY+jets", 'id_eUp', 'ttbare')[0] 
    id_eDown=template(background, "DY+jets", 'id_eDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(id_e, id_eUp, id_eDown)
        #pileup
    pileupUp=template(background, 'DY+jets','puUp','ttbare')[0] 
    pileupDown=template(background, 'DY+jets','puDown','ttbare')[0] 
#     ttbare_dyjets.setParamEffect(pileup, pileupUp, pileupDown)
    #reco Electron
    recoeUp=template(background, 'DY+jets', 'reco_eUp', 'ttbare')[0] 
    recoDown=template(background, 'DY+jets', 'reco_eDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(reco_e, recoeUp, recoDown)

    #JEC_JER
    jerUp=template(background, 'DY+jets', 'JERUp', 'ttbare')[0] 
    jerDown=template(background, 'DY+jets', 'JERDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'DY+jets', 'JESUp', 'ttbare')[0] 
    jesDown=template(background, 'DY+jets', 'JESDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(jec_jes, jesUp, jesDown)


    #L1 prefire
    L1prefireUp=template(background, 'DY+jets', 'L1PrefireUp', 'ttbare')[0] 
    L1prefireDown=template(background, 'DY+jets', 'L1PrefireDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

    #L1 ISR
    ISRUp=template(background, 'DY+jets', 'ISRUp', 'ttbare')[0] 
    ISRDown=template(background, 'DY+jets', 'ISRDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'DY+jets', 'FSRUp', 'ttbare')[0] 
    FSRDown=template(background, 'DY+jets', 'FSRDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'DY+jets', 'PDFUp', 'ttbare')[0] 
    pdfDown=template(background, 'DY+jets', 'PDFDown', 'ttbare')[0]
#     ttbare_dyjets.setParamEffect(pdf, pdfUp, pdfDown) 

    ttbare.addSample(ttbare_dyjets)


        # V Jets Background
    ttbare_vvTemplate = template(background, "VV", "nominal", "ttbare")
    ttbare_vv = rl.TemplateSample(
        ch_name + "_vvMC", rl.Sample.BACKGROUND, ttbare_vvTemplate
    )
#     ttbare_vv.setParamEffect(lumi, lumi_unc)
    #     ttbare_vv.setParamEffect(trig_e, 1.01)
    #     ttbare_vv.setParamEffect(veto_tau, 1.03)
    #     ttbare_vv.setParamEffect(vv_norm, 1.2)
    #     ttbare_vv.setParamEffect(jec, 1.05)
    #     ttbare_vv.setParamEffect(id_e, 1.02)
    #     ttbare_vv.setParamEffect(reco_e, 1.02)
#     addBtagSyst(background, "VV", "ttbare", ttbare_vv)
    #     ttbare_vv.autoMCStats()
    #     #trigger Electron
    trigeUp=template(background, 'VV', 'trig_eUp', 'ttbare')[0] 
    trigeDown=template(background, 'VV', 'trig_eDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(trig_e, trigeUp, trigeDown)
    #     #id Electron
    id_eUp=template(background, "VV", 'id_eUp', 'ttbare')[0] 
    id_eDown=template(background, "VV", 'id_eDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(id_e, id_eUp, id_eDown)
    #pileup
    pileupUp=template(background, 'VV','puUp','ttbare')[0] 
    pileupDown=template(background, 'VV','puDown','ttbare')[0] 
#     ttbare_vv.setParamEffect(pileup, pileupUp, pileupDown)
        #reco Electron
    recoeUp=template(background, 'VV', 'reco_eUp', 'ttbare')[0] 
    recoDown=template(background, 'VV', 'reco_eDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(reco_e, recoeUp, recoDown)

    #JEC_JER
    jerUp=template(background, 'VV', 'JERUp', 'ttbare')[0] 
    jerDown=template(background, 'VV', 'JERDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(jec_jer, jerUp, jerDown)

    #JEC_JEs
    jesUp=template(background, 'VV', 'JESUp', 'ttbare')[0] 
    jesDown=template(background, 'VV', 'JESDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(jec_jes, jesUp, jesDown)

    #L1 prefire
    L1prefireUp=template(background, 'VV', 'L1PrefireUp', 'ttbare')[0] 
    L1prefireDown=template(background, 'VV', 'L1PrefireDown', 'ttbare')[0]
    ttbare_vv.setParamEffect(L1prefire, L1prefireUp, L1prefireDown)

        #L1 ISR
    ISRUp=template(background, 'VV', 'ISRUp', 'ttbare')[0] 
    ISRDown=template(background, 'VV', 'ISRDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(isr, ISRUp, ISRDown)

    #L1 FSR
    FSRUp=template(background, 'VV', 'FSRUp', 'ttbare')[0] 
    FSRDown=template(background, 'VV', 'FSRDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(fsr, FSRUp, FSRDown)

    #pdf 
    pdfUp=template(background, 'VV', 'PDFUp', 'ttbare')[0] 
    pdfDown=template(background, 'VV', 'PDFDown', 'ttbare')[0]
#     ttbare_vv.setParamEffect(pdf, pdfUp, pdfDown) 

    ttbare.addSample(ttbare_vv)



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
    id_e = rl.NuisanceParameter("id_e" + year, "shape")
    id_mu = rl.NuisanceParameter("id_mu" + year, "shape")
    id_pho = rl.NuisanceParameter("id_pho" + year, "shape")
    reco_e = rl.NuisanceParameter("reco_e" + year, "shape")
    iso_mu = rl.NuisanceParameter("iso_mu" + year, "shape")
    trig_e = rl.NuisanceParameter("trig_e" + year, "shape")
    trig_met = rl.NuisanceParameter("trig_met" + year, "lnN")
    trig_pho = rl.NuisanceParameter("trig_pho" + year, "lnN")
    veto_tau = rl.NuisanceParameter("veto_tau" + year, "lnN")
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
    pileup = rl.NuisanceParameter("pileup"+year, "shape")
    L1prefire = rl.NuisanceParameter("L1prefire"+year, "shape")
    metUncl = rl.NuisanceParameter("metUncl"+year, "shape")
    isr = rl.NuisanceParameter("isr", "lnN")
    fsr = rl.NuisanceParameter("fsr", "lnN")
    pdf = rl.NuisanceParameter("pdf"+year, "shape")
    if "preVFP" in hist_scaled_file:
        year = year+"preVFP"
    elif "postVFP" in hist_scaled_file:
        year = year+"postVFP"
    with open(
        "data/"+"test_ttbare"+'.'+year+".model",
        "wb",) as fout:
        pickle.dump(model(year), fout, protocol=2)
