def METXYCorr_Met_MetPhi(events, year):
    import sys
    
    #https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18.h
    if 'genWeights' not in events.fields:
        isData = True
    else: isData = True
#     dataset=events.metadata['dataset']   
    met = events.MET
    originalMet = met.pt
    originalMet_phi = met.phi
    
    npv = events.PV.npvs
    mask = np.asarray(npv>100)
    npv = np.asarray(npv)
    npv[mask==True] = 100

    year = year
    dataset = 'WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8____1_'
    if not isData:
        if (year == '2016') and ("preVFP" in dataset):
            METxcorr = [-(-0.188743*npv +0.136539)]
            METycorr = [-(0.0127927*npv +0.117747)]
        if (year == '2016') and ("postVFP" in dataset):
            METxcorr = [-(-0.153497*npv +0.231751)]
            METycorr = [-(0.00731978*npv +0.243323)]
        elif year == '2017':
            METxcorr = [-(-0.300155*npv +1.90608)]
            METycorr = [-(0.300213*npv +2.02232)]
        elif year == '2018':
            METxcorr = [-(0.183518*npv +0.546754)]
            METycorr = [-(0.192263*npv +0.42121)]

    if isData:
        runNMax = max(events.run)
        runNMin = min(events.run)
        if "2018" in year:
            if (runNMin >=315252) and r(runNMax <=316995):
                era = "A"

            elif (runNMin >=316998) and (runNMax <=319312):
                era = "B"

            elif (runNMin >=319313) and (runNMax <=320393):
                era = "C"

            elif (runNMin >=320394) and (runNMax <=325273):
                era = "D"
            if "A" in era:
                METxcorr = [-(0.263733*npv -1.91115)]
                METycorr = [-(0.0431304*npv -0.112043)]
            elif "B" in era:
                METxcorr = [-(0.400466*npv -3.05914)]
                METycorr = [-(0.146125*npv -0.533233)]
            elif "C" in era:
                METxcorr = [-(0.430911*npv -1.42865)]
                METycorr = [-(0.0620083*npv -1.46021)]
            elif "D" in era:
                METxcorr = [-(0.457327*npv -1.56856)]
                METycorr = [-(0.0684071*npv -0.928372)]
                
            
        if "2017" in year:
            if (runNMin >=297020) and (runNMax <=299329):
                era = "B"
            
            elif (runNMin >=299337) and (runNMax <=302029):
                era = "C"
            
            elif (runNMin >=302030) and (runNMax <=303434):
                era = "D"
            
            elif (runNMin >=303435) and (runNMax <=304826):
                era = "E"
                
            elif (runNMin >=304911) and (runNMax <=306462):
                era = "F"
            if "E" in era:
                METxcorr = [-(-0.162472*npv +0.176329)]
                METycorr = [-(0.138076*npv -0.250239)]
            elif "B" in era:
                METxcorr = [-(-0.211161*npv +0.419333)]
                METycorr = [-(0.251789*npv -1.28089)]
            elif "C" in era:
                METxcorr = [-(-0.185184*npv -0.164009)]
                METycorr = [-(0.200941*npv -0.56853)]
            elif "D" in era:
                METxcorr = [-(-0.201606*npv +0.426502)]
                METycorr = [-(0.188208*npv -0.58313)]
            elif "F" in era:
                METxcorr = [-(-0.210639*npv +0.72934)]
                METycorr = [-(0.198626*npv +1.028)]
        if "2016" in year in dataset:
            
            if (runNMin >=272007) and (runNMax <=275376):
                era = "B"
            
            elif (runNMin >=275657) and (runNMax <=276283):
                era = "C"
            
            elif (runNMin >=276315) and (runNMax <=276811):
                era = "D"
            
            elif (runNMin >=276831) and (runNMax <=277420) :
                era = "E"
                
            elif (runNMin >=277772 and runNMax <=278768) or (runNMin==278770) or (runNMax==278770):
                era = "F"
                
            elif (runNMin >=278801 and runNMax <=278808) or (runNMax==278769) or (runNMax==278769):
                era = "F_late"
                
            elif (runNMin >=278820) and (runNMax <=280385):
                era = "G"
                
            elif (runNMin >=280919) and (runNMax <=284044):
                era = "H"
                
            if "B" in era:
                METxcorr = [-(-0.0214894*npv -0.188255)]
                METycorr = [-(0.0876624*npv +0.812885)]
            elif "C" in era:
                METxcorr = [-(-0.032209*npv +0.067288)]
                METycorr = [-(0.113917*npv +0.743906)]
            elif "D" in era:
                METxcorr = [-(-0.0293663*npv +0.21106)]
                METycorr = [-(0.11331*npv +0.815787)]
            elif "E" in era:
                METxcorr = [-(-0.0132046*npv +0.20073)]
                METycorr = [-(0.134809*npv +0.679068)] 
            elif "F" in era:
                METxcorr = [-(-0.0543566*npv +0.816597)]
                METycorr = [-(0.114225*npv +1.17266)] 
            elif "F_late" in era:
                METxcorr = [-(0.134616*npv -0.89965)]
                METycorr = [-(0.0397736*npv +1.0385)] 
            elif "G" in era:
                METxcorr = [-(0.121809*npv -0.584893)]
                METycorr = [-(0.0558974*npv +0.891234)]
            elif "H" in era:
                METxcorr = [-(0.0868828*npv -0.703489)]
                METycorr = [-(0.0888774*npv +0.902632)]
#     print((np.array(METxcorr)).flatten())
#     print(np.multiply(originalMet,np.cos(originalMet_phi)))
#     print(np.multiply(originalMet,np.cos(originalMet_phi))+ (np.array(METxcorr)).flatten())
    
#     sys.exit(0)
    CorrectedMET_x = np.multiply(originalMet,np.cos(originalMet_phi))+ (np.array(METxcorr)).flatten()
    CorrectedMET_y = np.multiply(originalMet,np.cos(originalMet_phi))+ (np.array(METycorr)).flatten()
#     print(CorrectedMET_x/CorrectedMET_y)

    CorrectedMET = np.sqrt(np.square(CorrectedMET_x)+np.square(CorrectedMET_y))
    CorrectedMETPhi = np.zeros(len(events))

    print(CorrectedMET)
    print(CorrectedMETPhi)

    mask1 = (CorrectedMET_x ==0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask1==True] = np.pi
    
    mask2 = (CorrectedMET_x ==0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask2==True] = -np.pi
    
    mask3 = (CorrectedMET_x <0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask3==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) + np.pi )[mask3==True] 
 
    mask4 = (CorrectedMET_x <0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask4==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) - np.pi )[mask4==True]
    
    mask5 = (CorrectedMET_x >0 )
    CorrectedMETPhi[mask5==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x))[mask5==True]   
    
    return (CorrectedMET, CorrectedMETPhi)
