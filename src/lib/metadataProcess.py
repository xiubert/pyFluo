from scipy.io import loadmat
import re
import numpy as np

"""
Functions for extracting metadata.
"""

def getPulseNames(xsgPath: str):
    xsg = loadmat(xsgPath)
    arr = xsg['header']['stimulator'][0,0]['stimulator'][0,0]['pulseNameArray'][0,0][:,0]
    return np.concatenate(arr).tolist()
    
def getPulseSets(xsgPath: str):
    xsg = loadmat(xsgPath)
    arr = xsg['header']['stimulator'][0,0]['stimulator'][0,0]['pulseSetNameArray'][0,0][:,0]
    return np.concatenate(arr).tolist()

def getPulseDB(pulse: str):
    dBre = re.compile(r'_(\d{2,3})dB_\d{2,5}msTotal_')
    try:
        return re.search(dBre,pulse).group(1)
    except AttributeError:
        return None