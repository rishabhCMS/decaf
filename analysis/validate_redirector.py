#!/usr/bin/env python
# this code parses the json file and finds the appropriate redirector for the file
#usage: python validate_redirector.py -f UL_2018


from coffea import hist, lookup_tools
from coffea.util import load, save
from optparse import OptionParser
import json
import uproot

def validate_file(file):
    redirect_infn = 'root://xrootd-cms.infn.it//'
    redirect_cern = 'root://cms-xrd-global.cern.ch//'
    redirect_fnal = 'root://cmsxrootd.fnal.gov//'
    
    final_redirect = ''
    file = file.split('//')[2]
    try: 
        uproot.open(redirect_cern+file)
        final_redirect = redirect_cern
    except:
        try:
            uproot.open(redirect_infn+file)
            final_redirect = redirect_infn
        except:
            try:
                uproot.open(redirect_fnal+file)
                final_redirect = redirect_fnal
            except:
                final_redirect = redirect_cern
    return (final_redirect+file)

if __name__ == "__main__":
  
    parser = OptionParser()
    parser.add_option('-f', '--filename', help='Json Filename', dest='filename')
    (options, args) = parser.parse_args()

    file_json = "metadata/"+options.filename

    with open(file_json) as f:
        data = json.load(f)
    datasets_total = len(data.keys())
    
    for i, sample in enumerate(data.keys()):
        lst = data[sample]['files']
        empty_list = []
        print("Processing sample:{} out of {} datasets".format(i+1, datasets_total))
        for file in lst:
            empty_list.append(validate_file(file))
#         print("Processing sample:{} out of {} datasets".format(i+1, datasets_total))
        data[sample]['files'] = empty_list

    folder = "metadata/"+"_"+options.filename
    with open(folder, "w") as fout:
        json.dump(data, fout, indent=4)
