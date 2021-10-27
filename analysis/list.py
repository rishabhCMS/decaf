#!/usr/bin/env python
import os
from optparse import OptionParser
import yaml
parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
(options, args) = parser.parse_args()

globalredirect = "root://xrootd-cms.infn.it/"

campaigns = {}
campaigns['2016'] = 'RunIISummer16MiniAODv3'
campaigns['2017'] = 'RunIIFall17MiniAODv2'
campaigns['2018'] = 'RunIISummer20UL18NanoAODv9'

yml_file = {}
yml_file['2016'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2016.yml'
yml_file['2017'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2017.yml'
yml_file['2018'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2018.yml'
with open(yml_file[options.year], 'r') as file:
    _data = yaml.safe_load(file)
    data = _data[options.year]
    
slist = list(data.keys())
print('slist:',slist)

# slist = ['DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8']
datasets = []
for sample in slist:
    query="dasgoclient --query=\"dataset dataset=/"+sample+"*/"+campaigns[options.year]+"*/NANOAODSIM\""
    print(query)
    os.system(query+" >"+sample+".txt")
    datasets.append(sample+".txt")
print("datasets:", datasets)
import sys
sys.exit(0)
filenames = []
for dname in datasets:                                                                                                                                                                 
    with open(dname) as infile:
        for line in infile:
            dataset = line.strip()
            query="dasgoclient --query=\"file dataset="+dataset+"\""
            print(query)
            os.system(query+" > "+dataset.split('/')[1]+"____"+dataset.split('/')[2]+".txt")
            filenames.append(dataset.split('/')[1]+"____"+dataset.split('/')[2]+".txt")
    os.system('rm '+dname)

dir_ = 'metadata/'+'UL_'+options.year+'/'
if not os.path.isdir(dir_):
    os.mkdir(dir_)
    
for sample in slist:
    with open(dir_+sample+'.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(globalredirect+line)
for fname in filenames:
    os.system('rm '+fname)
