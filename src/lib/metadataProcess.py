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

def getPulseDB(pulse: str, format: str = 'MAK'):
    if format=='MAK':
        dBre = re.compile(r'_(\d{2,3})dB_\d{2,5}msTotal_')
    elif format=='PAC':
        dBre = re.compile(r'Hz_(\d{2,3})dB_TestTone_\d{2,5}msPulse_')
    else:
        return None
    try:
        return int(re.search(dBre,pulse).group(1))
    except AttributeError:
        return None