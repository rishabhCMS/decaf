def METXYCorr_Met_MetPhi(events, isData, isUL=False, dataset=dataset):

    #https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18.h
    met = events.MET
    originalMet = met.pt
    originalMet_phi = met.phi
    npvs = events.PV.npvs
    dataset = dataset
    if not isData:
        if self._year == '2016':
            METxcorr = [-(-0.195191*npvs -0.170948)]*len(events)
            METycorr = [-(-0.0311891*npvs +0.787627)]*len(events)
        elif self._year == '2017':
            METxcorr = [-(-0.217714*npvs +0.493361)]*len(events)
            METycorr = [-(0.177058*npvs -0.336648)*len(events)]
        elif self._year == '2018':
            METxcorr = [-(0.296713*npvs -0.141506)]*len(events)
            METycorr = [-(0.115685*npvs +0.0128193)]*len(events)

    if isData:
        if "Run2018" in dataset:
            if "A" in dataset:
                METxcorr = [-(0.362865*npvs -1.94505)]*len(events)
                METycorr = [-(0.0709085*npvs -0.307365)]*len(events)
            elif "B" in dataset:
                METxcorr = [-(0.492083*npvs -2.93552)]*len(events)
                METycorr = [-(0.17874*npvs -0.786844)]*len(events)
            elif "C" in dataset:
                METxcorr = [-(0.521349*npvs -1.44544)]*len(events)
                METycorr = [-(0.118956*npvs -1.96434)]*len(events)
            elif "D" in dataset:
                METxcorr = [-(0.531151*npvs -1.37568)]*len(events)
                METycorr = [-(0.0884639*npvs -1.57089)]*len(events)



    CorrectedMET_x = np.add(np.multiply(originalMet,np.cos(originalMet_phi)), METxcorr)
    CorrectedMET_y = np.add(np.multiply(originalMet,np.cos(originalMet_phi)), METycorr)
    # CorrectedMET_x = originalMet *np.cos( originalMet_phi)+METxcorr
    # CorrectedMET_y = originalMet *np.sin( originalMet_phi)+METycorr

    CorrectedMET = np.sqrt(np.square(CorrectedMET_x)+np.square(CorrectedMET_y**2))
    CorrectedMETPhi = np.zeros(events.size, dtype=np.bool)

    for indx, (x,y) in zip(CorrectedMET_x,CorrectedMET_y):
        if (x==0) and (y>0):
            CorrectedMETPhi[indx] = np.pi

        elif(x==0 and y<0 ):
            CorrectedMETPhi[indx] = -np.pi

        elif(x >0):
            CorrectedMETPhi[indx] = np.arctan(y/x)

        elif(x <0 and y>0):
            CorrectedMETPhi[indx] = np.arctan(y/x) + np.pi

        elif(x <0 and y<0):
            CorrectedMETPhi[indx] = np.arctan(y/x) - np.pi
        else:
            CorrectedMETPhi[indx] = 0

    return (CorrectedMET, CorrectedMETPhi)
