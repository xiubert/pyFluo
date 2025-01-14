from scipy.io import loadmat
import re
import numpy as np
import os
from glob import glob

"""
Functions for extracting metadata.
"""

def getPulseNames(xsgPath: str):
    """
    Extracts pulse names from an XSG file.

    Args:
        xsgPath (str): Path to the XSG file (MATLAB .mat format) containing stimulation data.

    Returns:
        list: A list of pulse names extracted from the 'pulseNameArray' field of the XSG file.

    Notes:
        - This function assumes the XSG file has a specific structure with fields:
          `header -> stimulator -> stimulator -> pulseNameArray`.
    """
    xsg = loadmat(xsgPath)
    arr = xsg['header']['stimulator'][0,0]['stimulator'][0,0]['pulseNameArray'][0,0][:,0]

    return np.concatenate(arr).tolist()
    

def getPulseSets(xsgPath: str):
    """
    Extracts pulse set names from an XSG file.

    Args:
        xsgPath (str): Path to the XSG file (MATLAB .mat format) containing stimulation data.

    Returns:
        list: A list of pulse set names extracted from the 'pulseSetNameArray' field of the XSG file.

    Notes:
        - This function assumes the XSG file has a specific structure with fields:
          `header -> stimulator -> stimulator -> pulseSetNameArray`.
    """
    xsg = loadmat(xsgPath)
    arr = xsg['header']['stimulator'][0,0]['stimulator'][0,0]['pulseSetNameArray'][0,0][:,0]

    return np.concatenate(arr).tolist()


def getPulseDB(pulse: str, format: str = 'MAK'):
    """
    Extracts the decibel (dB) value from a pulse string based on the specified format.

    Args:
        pulse (str): The pulse string containing decibel information.
        format (str, optional): The format of the pulse string. Default is 'MAK'.
                                - 'MAK': Matches patterns like "_XXdB_YYYmsTotal_"
                                - 'PAC': Matches patterns like "Hz_XXdB_TestTone_YYYmsPulse_"
                                - Other formats return None.

    Returns:
        int or None: The decibel (dB) value as an integer if found; otherwise, None.

    Raises:
        AttributeError: If the pulse string does not contain a match for the given format.

    Notes:
        - For 'MAK' format, the regex pattern looks for "_XXdB_YYYmsTotal_".
        - For 'PAC' format, the regex pattern looks for "Hz_XXdB_TestTone_YYYmsPulse_".
        - Returns `None` if the format is not recognized or no match is found.
    """
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
    

def getInjectionCond(df):
    """
    Labels files in the DataFrame under a specific experimental condition.
    Adds a column 'INJECTION' with values: 'CTRL', 'pre[DRUG]', or 'post[DRUG]', eg. preZX1, postZX1.

    Args:
        df (pd.DataFrame): DataFrame containing file information, with columns:
                           'dir' (experiment directory) and 'qcam' (file name).

    Returns:
        list: list where each element is the injection condition for that qcamraw file.
    """
    injection_labels = []

    for exp_dir, group in df.groupby('dir'):
        # Check for ZXXX qcam files
        if group['qcam'].str.contains(r'[A-Z]{2}\d{4}ZXXX\d{4}', regex=True).any():
            for _, row in group.iterrows():
                if re.search(r'[A-Z]{2}\d{4}ZXXX\d{4}', row['qcam']):
                    injection_labels.append('postZX1')
                else:
                    injection_labels.append('preZX1')

        # Check for INJECTION_[DRUG]_START files
        elif len(glob(os.path.join(exp_dir, 'INJECTION_*_START*'))) == 1:
            fstart = glob(os.path.join(exp_dir, 'INJECTION_*_START*'))[0]
            match = re.search(r'_([A-Z0-9]+)_START_(\d+)', fstart)
            if match:
                drug = match.group(1)
                start_number = int(match.group(2))  # Start number for postZX1
                for _, row in group.iterrows():
                    qcam_number = int(re.search(r'(\d{4})\.qcamraw$', row['qcam']).group(1))  # Extract qcam number
                    if qcam_number >= start_number:
                        injection_labels.append(f'post{drug}')
                    else:
                        injection_labels.append(f'pre{drug}')
            else:
                raise ValueError('Unable to parse injection start file.')

        # No injection condition
        else:
            injection_labels.extend(['CTRL'] * len(group))

    # Add the INJECTION column to the DataFrame
    # df['INJECTION'] = injection_labels
    # return df
    return injection_labels