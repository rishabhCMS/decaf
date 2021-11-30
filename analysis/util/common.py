from coffea.util import save
import awkward
import uproot, uproot_methods
import numpy as np

def match(a, b, val):
    combinations = a.cross(b, nested=True)
    return (combinations.i0.delta_r(combinations.i1)<val).any()

def sigmoid(x,a,b,c,d):
    """
    Sigmoid function for trigger turn-on fits.
    f(x) = c + (d-c) / (1 + np.exp(-a * (x-b)))
    """
    return c + (d-c) / (1 + np.exp(-a * (x-b)))

deepflavWPs = {
    '2016':{
        'preVFP':  {
            'loose' :  0.0508,
            'medium':  0.2598,
            'tight' :  0.6502 
                    },
        'postVFP': {
            'loose' : 0.0480,
            'medium':  0.2489,
            'tight' : 0.6377 
                   }
    },
    '2017': {
        'loose' : 0.0532,
        'medium': 0.3040,
        'tight' : 0.7476
    },
    '2018': {
        'loose' : 0.0490,
        'medium': 0.2783,
        'tight' : 0.7100       
    },
}
deepcsvWPs = {
    '2016': {
        'loose' : 0.2217,
        'medium': 0.6321,
        'tight' : 0.8953
    },
    '2017': {
        'loose' : 0.1522,
        'medium': 0.4941,
        'tight' : 0.8001
    },
    '2018': {
        'loose' : 0.1241,
        'medium': 0.4184,
        'tight' : 0.7527
    },
}

btagWPs = {
    'deepflav': deepflavWPs,
    'deepcsv' : deepcsvWPs
}
    
common = {}
common['match'] = match
common['sigmoid'] = sigmoid
common['btagWPs'] = btagWPs
save(common, 'data/common.coffea')
