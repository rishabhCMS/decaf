#!/usr/bin/env python

from coffea import hist, lookup_tools
from coffea.util import load, save
from optparse import OptionParser
import json
import uproot

def validate(file):
    '''
    this function takes in a root file and return True if the file is valid and
    False is the file is corrupt
    :param file:
    :return: BOOL (True for valid root file and False for invalid file
    '''
    try:
        fin = uproot.open(file)
        print(f"file :{file} valid")
        return True
    except:
        print("Corrupted file: {}".format(file))
        return False

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-f', '--filename', help='Json Filename', dest='filename')
    (options, args) = parser.parse_args()

    files_json = "metadata/"+options.filename
    f = open(files_json)
    data = json.load(f)
    print(f"{files_json} has been loaded")
    f.close()
    datasets_total = len(data.keys())

    
#     for k in data.keys():
#         print(f"processing {k}")
#         file_list = []
# #         2018 data/MC json
#         if ("SingleMuon" in k):
#             num_list= ['_103_','_126_', '_127_', '_157_', '_165_', '_172_', '_253_', '_267_', '_372_', '_374_','_385_', '_405_', '_446_', '_461_', '_57_']
#             for num in num_list:
#                 if num in k:
#                     for file in data[k]['files']:
#                         isValid = validate(file)
#                         if isValid:
#                             file_list.append(file)
#                     data[k]['files'] = file_list

    for k in data.keys():
        print(f"processing {k}")
        file_list = []
        #2016 data/mc
        if ("QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8" in k):
            num_list= ['_4_','_6_']
            for num in num_list:
                if num in k:
                    for file in data[k]['files']:
                        isValid = validate(file)
                        if isValid:
                            file_list.append(file)
                    data[k]['files'] = file_list
                    
        elif ("SingleElectron" in k):
            num_list= ['_125_','_138_']
            for num in num_list:
                if num in k:
                    for file in data[k]['files']:
                        isValid = validate(file)
                        if isValid:
                            file_list.append(file)
                    data[k]['files'] = file_list
                    
        else:continue
    folder = "metadata/"+"_"+options.filename
    with open(folder, "w") as fout:
        json.dump(data, fout, indent=4)
