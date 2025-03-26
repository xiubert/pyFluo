from scipy.io import loadmat
import re
import pandas as pd
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
        dBre = re.compile(r'_(\d{1,3})dB_\d{2,5}msTotal_')
    elif format=='PAC':
        dBre = re.compile(r'Hz_(\d{2,3})dB_TestTone_\d{2,5}msPulse_')
    else:
        return None
    try:
        return int(re.search(dBre,pulse).group(1))
    except AttributeError:
        return None
    

def getInjectionCond(df: pd.DataFrame) -> list:
    """
    Returns treatment label for files (rows) in the DataFrame under a specific experimental condition.
    No treatment is considered 'CTRL'. Injection treatments are lebeled as: 'pre[DRUG]', or 'post[DRUG]', eg. preZX1, postZX1.

    Args:
        df (pd.DataFrame): DataFrame containing file information, with columns:
                           'dir' (experiment directory) and 'qcam' (file name).

    Returns:
        list: list where each element is the treatment condition for that qcamraw file.
    """
    treatment_labels = []
    ZX1fileNameRegex = r'[A-Z]{2}\d{4}(?=.*[ZX])[A-Z]{4}\d{4}'
    
    for exp_dir, group in df.groupby('dir'):
        # Check for ZXXX qcam files indicating a ZX1 injection treatment
        if group['qcam'].str.contains(ZX1fileNameRegex, regex=True).any():
            for _, row in group.iterrows():
                if re.search(ZX1fileNameRegex, row['qcam']):
                    treatment_labels.append('postZX1')
                else:
                    treatment_labels.append('preZX1')

        # Otherwise, check for INJECTION_[DRUG]_START files in the experiment directory indicating treatment
        elif len(glob(os.path.join(exp_dir, 'INJECTION_*_START*'))) == 1:
            fstart = glob(os.path.join(exp_dir, 'INJECTION_*_START*'))[0]
            match = re.search(r'_([A-Z0-9]+)_START_(\d+)', fstart)
            if match:
                drug = match.group(1)
                start_number = int(match.group(2))  # Start number for post treatment (eg. postZX1)
                for _, row in group.iterrows():
                    qcam_number = int(re.search(r'(\d{4})\.qcamraw$', row['qcam']).group(1))  # Extract qcam number
                    if qcam_number >= start_number:
                        treatment_labels.append(f'post{drug}')
                    else:
                        treatment_labels.append(f'pre{drug}')
            else:
                raise ValueError('Unable to parse injection start file.')

        # No treatment condition
        else:
            treatment_labels.extend(['CTRL'] * len(group))

    return treatment_labels


def getBaseRespWindow(df: pd.DataFrame, t_base: tuple = (2.0, 3.0), t_resp: tuple = (3.3, 4.0), stimStart: float = 3.0) -> dict:
    """
    Returns a dictionary containing two lists: 'baseWindow' and 'respWindow' for files (rows) in the DataFrame.
    Adjust time windows automatically based on file 'STIMULUS_START_*_sec*' in the corresponding directory.

    Args:
        df (pd.DataFrame): DataFrame containing file information, with column: 'dir' (experiment directory).
        t_base (tuple, optional): Baseline time window.
        t_resp (tuple, optional): Response time window.
        stimStart (float, optional): Stimulus start time (in seconds) by default. Defaults to 3.0.

    Returns:
        time_windows (dict): A dictionary with keys 'baseWindow' and 'respWindow', each mapping to a list of tuples.
    """

    # Initialize lists for baseline and response time windows
    base_windows = []
    resp_windows = []

    for _, row in df.iterrows():
        # For each row, search for file indicating stimulus start time in corresponding experiment directory
        # Filename example: 'STIMULUS_START_2_sec.txt'
        stimulus_file = glob(os.path.join(row['dir'], 'STIMULUS_START_*_sec*'))
        
        if stimulus_file:
            # If file exists, take the first matching file
            fstart = stimulus_file[0]
            match = re.search(r'START_([0-9]+)_sec', fstart)
            if match:
                # Extract stimulus start time
                start_time = int(match.group(1))
                # Adjust baseline and response time windows accordingly
                base_window = tuple(x + (start_time-stimStart) for x in t_base)
                resp_window = tuple(x + (start_time-stimStart) for x in t_resp)
            else:
                raise ValueError(f'Unable to parse stimulus start file: {fstart}')
        else:
            # If file not found, use default baseline and response time windows
            base_window = t_base
            resp_window = t_resp

        # Append time windows to their respective lists
        base_windows.append(base_window)
        resp_windows.append(resp_window)

    # Creating a dictionary containing keys 'baseWindow' and 'respWindow'
    time_windows = {'baseWindow': base_windows, 'respWindow': resp_windows}

    return time_windows