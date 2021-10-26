#!/usr/bin/env python

#usage: python convert_txt_to_json.py -y [enter the year you want to process] -p [enter the pack size]
import uproot
import json
import pyxrootd.client
import fnmatch
import numpy as np
import numexpr
import subprocess
import concurrent.futures
import warnings
import os
import difflib
from optparse import OptionParser
import glob
import yaml

parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()

# a dict for dataset and cross section reference
yml_file = {}
yml_file['2016'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2016.yml'
yml_file['2017'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2017.yml'
yml_file['2018'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2018.yml'
with open(yml_file[options.year], 'r') as file:
    _data = yaml.safe_load(file)
lst = list(_data[options.year].keys())

xs_reference = {}
for dataset in lst:
#     print(dataset,':',data['2018'][dataset]['xs'])
    if 'EGamma' in dataset or 'SingleMuon' in dataset or 'MET' in dataset or 'SingleElectron' in dataset or 'SinglePhoton' in dataset:
        xs_reference[dataset] = -1
    else:    
        xs_reference[dataset] = data[options.year][dataset]['xs']

flist = glob.glob('metadata/UL_'+options.year+'/*.txt')
datadict_obj = {}
for txt_file in flist:
    if 'MET' in txt_file or 'SinglePhoton' in txt_file:
        continue

#     print(txt_file)
    
    lstobj = []
    n = options.pack
    with open(txt_file) as f:
        data = f.readlines()
        for filename in data:
            file = filename.split('\n')[0]
            lstobj.append(file)
            
    if not len(lstobj) == len(set(lstobj)):
        print("your dataset has duplicate root files")
        
    output=[lstobj[i:i + n] for i in range(0, len(lstobj), n)] 

    for i, lst in enumerate(output):
        key_name = txt_file.split('/')[1].split('.')[0] + '____'+str(i+1)+'_'
        dataset_name = txt_file.split('/')[1].split('.')[0]
        datadict_obj[key_name] = {'files': lst, 'xs': xs_reference[dataset_name]}
        
import json
folder = "metadata/UL_"+options.year+'.json'
with open(folder, "w") as fout:
    json.dump(datadict_obj, fout, indent=4)
    
