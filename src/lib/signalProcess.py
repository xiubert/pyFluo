import numpy as np
import pandas as pd
import os
import warnings
import joblib

from scipy.signal import butter, filtfilt
from typing import Tuple
from glob import glob
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import lib.metadataProcess as metadataProcess


def getTimeVec(nFrames: int, 
               frameRate: int = 20, 
               zeroStart: bool = True,
               delayAdjust: float = 0.025,
               **kwargs):
    """
    Generate time vector from frame count and rate.

    Args:
        nFrames (int): number of frames
        frameRate (int): number of frames acquired per second
        zeroStart (bool): whether first frame acquired at time 0.
        delayAdjust (float): adjustment in time (s) for frame data acquisition.
        **kwargs: Optional arguments that will override default.

    Returns:
        t (numpy array): vector of time values
    """
    # Optionally override parameters using kwargs
    frameRate = kwargs.get('frameRate', frameRate)
    zeroStart = kwargs.get('zeroStart', zeroStart)
    delayAdjust = kwargs.get('delayAdjust', delayAdjust)

    # first frame acquired at start (starts at 0)
    t = (np.arange(nFrames) / frameRate) + delayAdjust
    
    if not zeroStart:   # first frame acquired (1/fr) s after start
        t += 1/frameRate
    return t


def computeFFT(signal: np.ndarray, sample_freq: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple helper function to compute the Fast Fourier Transform (FFT) of a signal.
    Returns positive frequencies and their magnitudes.

    Args:
        signal (numpy.ndarray): 1D signal array of shape [frame].
        sample_freq (int, optional): Sampling frequency of the signal in Hz.
    
    Returns:
        A tuple containing two arrays:
            - freqs (numpy.ndarray): Array of positive frequency bins in Hz.
            - fft_vals (numpy.ndarray): Magnitude spectrum of the FFT (absolute values).
    
    Notes:
        - For real-valued input signals, the FFT is symmetric so negative frequencies are discarded.
        - Only returns the positive frequency components (first half of the FFT).
        - Nyquist frequency (maximum detectable frequency) is sample_freq/2.
    """
    n = signal.shape[-1]
    fft_vals = np.fft.fft(signal)  # Compute FFT (returns complex values)
    freqs = np.fft.fftfreq(n, d=1/sample_freq)  # Generate frequency bins
    return freqs[:n//2], np.abs(fft_vals[:n//2])  # Return only positive frequencies


def butterFilter(signal: np.ndarray, 
                 sample_freq: int = 20, 
                 cutoff_freq: int = 5, 
                 order: int = 4,
                 **kwargs) -> np.ndarray:
    """
    Simple helper function for a lowpass butterworth filter.

    Args:
        signal (numpy array): 1D or 2D signal array to be filtered of shape [frame] or [traceNumber, frame]
        sample_freq (int): sampling frequency of the signal in Hz
        cutoff_freq (float): filter cutoff frequency in Hz
        order (int): filter order ('steepness' of signal drop-off at cutoff_freq)
        **kwargs: Optional arguments that will override default.

    Returns:
        filtered_signal (numpy array): lowpass filtered signal (same shape as input signal)

    """
    # Optionally override parameters using kwargs
    sample_freq = kwargs.get('sample_freq', sample_freq)
    cutoff_freq = kwargs.get('cutoff_freq', cutoff_freq)
    order = kwargs.get('order', order)

    b, a = butter(order, cutoff_freq/(sample_freq/2), 'lowpass') 

    if signal.ndim == 1:
        # If the signal is 1D, treat it as a single trace
        filtered_signal = filtfilt(b, a, signal)
    elif signal.ndim == 2:
        # If the signal is 2D, process each trace (row) independently
        # Initialize the output array
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered_signal[i, :] = filtfilt(b, a, signal[i, :])
    elif signal.ndim == 3:
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                filtered_signal[i,j,:] = filtfilt(b, a, signal[i,j,:])
    else:
        raise ValueError("Signal array must be 1D or 2D.")
    
    return filtered_signal


def subtractLinFit(t, signal: np.ndarray, offset: bool = True, **kwargs) -> np.ndarray:
    """
    Subtracts linear fit of signal from signal. 
    Useful to remove consistent signal drift in one direction.

    Args:
        t (list or array): time vector (in seconds).
        signal (numpy array): 1D, 2D, or 3D signal array of shape [frame], [traceNumber, frame], or [traceNumber, maskNumber, frame].
        offset (bool, optional): whether to add baseline fluorescence (f0) back to the corrected signal as the offset.
                                 Defaults to 'True'.
    
    Returns:
        corrected_signal (numpy array): signal array after removal of linear fit (same shape as input signal).
        slope (numpy array): array of slopes for each trace (one lower dimension than input signal).
        intercept (numpy array): array of intercepts for each trace (one lower dimension than input signal).
    """

    # Optionally override parameters using kwargs
    offset = kwargs.get('offset',offset)
    
    # Ensure t is a numpy array
    t = np.asarray(t)

    # Prepare the design matrix for linear regression
    X = np.vstack([t, np.ones(len(t))]).T

    if signal.ndim == 1:
        # If the signal is 1D, treat it as a single trace
        slope, intercept = np.linalg.lstsq(X, signal, rcond=None)[0]
        
        if offset:
            # baseline fluorescence is added to bring corrected y-values back to approximately the same level of uncorrected ones
            # required if dFF is calculated using the corrected signal after linear fit subtraction
            f0 = getBaseResp(signal, t, **kwargs)[0]
            corrected_signal = signal - (t*slope + intercept) + f0
        else:
            # output (signal - linear fit) directly
            # used to display how linear fit works
            corrected_signal = signal - (t*slope + intercept)
    
    elif signal.ndim == 2:
        # Assume signal shape is [traceNumber, frame]. Process each trace (row) independently
        # Initialize the output array
        corrected_signal = np.zeros_like(signal)
        slope = np.zeros(signal.shape[0])
        intercept = np.zeros(signal.shape[0])
        
        for i in range(signal.shape[0]):
            slope[i], intercept[i] = np.linalg.lstsq(X, signal[i], rcond=None)[0]
            
            if offset:
                f0 = getBaseResp(signal[i], t, **kwargs)[0]
                corrected_signal[i] = signal[i] - (t*slope[i] + intercept[i]) + f0
            else:
                corrected_signal[i] = signal[i] - (t*slope[i] + intercept[i])
    
    elif signal.ndim == 3:
        # Assume signal shape is [traceNumber, maskNumber, frame]. Process each trace (the third dimension) independently
        # Initialize the output array
        corrected_signal = np.zeros_like(signal)
        slope = np.zeros((signal.shape[0], signal.shape[1]))
        intercept = np.zeros((signal.shape[0], signal.shape[1]))
        
        for i in range(signal.shape[0]):
            for j in range (signal.shape[1]):
                slope[i, j], intercept[i, j] = np.linalg.lstsq(X, signal[i, j], rcond=None)[0]
                
                if offset:
                    f0 = getBaseResp(signal[i, j], t, **kwargs)[0]
                    corrected_signal[i, j] = signal[i, j] - (t*slope[i, j] + intercept[i, j]) + f0
                else:
                    corrected_signal[i, j] = signal[i, j] - (t*slope[i, j] + intercept[i, j])

    else:
        raise ValueError("Signal array must be 1D, 2D, or 3D.")
    
    return corrected_signal, slope, intercept


def getBaseResp(signal: np.ndarray, t: np.ndarray, 
                t_base: tuple[float,float] = (2.2,2.9),
                t_resp: tuple[float,float] = (3.0,3.15),
                negResp: bool = False,
                **kwargs) -> tuple[float,float]:
    """
    Extract average signal at t_base and max signal between t_resp.

    Args:
        signal (numpy array): signal array of shape [traceNumber, frame] or [frame].
        t (list or array): time vector (in seconds).
        t_base: time window (in seconds) for baseline.
        t_resp: time window (in seconds) for response.
        negResp (bool, optional): whether to extract max signal between t_resp in either direction.
                                - 'True': Response with max absolute value is returned, whether positive or negative.
                                          Orginal sign of the response is preserved.
                                - 'False': Only max positive response is returned.
                                Defaults to 'False'.
        **kwargs: Optional arguments that will override default.

    Returns:
        base (numpy array): average signal between t_base for each trace.
        resp (numpy array): max signal between t_resp for each trace.

    Notes:
        If negative response is calculated, 'negResp = True' only works for dFF response but not raw F.
    """

    # Optionally override parameters using kwargs
    t_base = kwargs.get('t_base',t_base)
    t_resp = kwargs.get('t_resp',t_resp)
    negResp = kwargs.get('negResp',negResp)

    base_indices = np.where((t >= t_base[0]) & (t <= t_base[1]))[0]
    resp_indices = np.where((t >= t_resp[0]) & (t <= t_resp[1]))[0]

    # To do: threshold (Avg ± 2 SD) should be set to exclude spontaneous activities
    if signal.ndim == 1:
        # If the signal is 1D, treat it as a single trace
        base = signal[base_indices].mean()
        if negResp:
            # find the response with the max absolute value and keep its original sign
            resp_values = signal[resp_indices]
            resp = resp_values[np.argmax(np.abs(resp_values))]
        else:
            # find the response with the max numeric value
            # may ignore negative response
            resp = signal[resp_indices].max()
    elif signal.ndim == 2:
        # If the signal is 2D, process each trace
        base = np.mean(signal[:, base_indices], axis=1)
        if negResp:
            # in each trace, find the responses with the max absolute values and keep their original signs
            resp_values = signal[:, resp_indices]
            resp = resp_values[np.arange(resp_values.shape[0]), np.argmax(np.abs(resp_values), axis=1)]
        else:
            # in each trace, find the responses with the max numeric values
            # may ignore negative responses
            resp = np.max(signal[:, resp_indices], axis=1)
    else:
        raise ValueError("Signal array must be 1D or 2D.")
        
    return base, resp


def dFFcalc(signal, **kwargs):
    """
    Calculates dFF for a signal such as average fluorescence over time.

    Args:
        signal (numpy array): 1D or 2D signal array (e.g., raw fluorescence).
                              Shape can be [frame] or [traceNumber, frame].
        **kwargs: Optional arguments that will override default.
            Ror example:  t_base: time window (in seconds) for baseline

    Returns:
        dFF (numpy array): deltaF/F of input signal (same shape as input signal).
        dF (numpy array): deltaF of input signal (same shape as input signal).
        f0 (float or numpy array): baseline signal (scalar for 1D, array for 2D).
    """

    t = kwargs.get('t', getTimeVec(signal.shape[-1], **kwargs))

    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, **kwargs)[0]
    
    # Calculate dF and dFF
    dF = signal - f0[:, np.newaxis] if signal.ndim == 2 else signal - f0
    dFF = dF / f0[:, np.newaxis] if signal.ndim == 2 else dF / f0

    return dFF,dF,f0
     

def is_valid_resp(imgSeries: np.ndarray, subLinFit: bool = True, dFResp: bool = False, 
                  t_base: tuple[float,float] = (2,3), t_resp_excl: tuple[float,float] = (3.3,4), **kwargs) -> bool:
    """
    Checks whether the response is a negative outlier.
    Negative outliers refer to traces whose Avg response is 3 SDs below Avg baseline or peak response is below 0.

    Args:
        imgSeries (array): 3D array of shape (Y, X, frame)
        subLinFit (bool): whether to subtract fitted line
        dFResp (bool): if true, calculate dF response rather than dFF
        t_base (tuple): time window (in seconds) for baseline
        t_resp_excl (tuple): time window (in seconds) to exclude outliers (negative response)
        **kwargs: Optional arguments that will override default
            example: ROImask (np.ndarray): 2D binary mask array specifying the region of interest

    Returns:
        is_valid (bool): `True` for positive response (non-outlier), `False` for negative response (outlier)

    Notes:
        - Usually `t_resp_excl` time window should be no longer than response window to avoid removing any non-outlier traces.
    """
    
    # optionally override parameters using kwargs
    subLinFit = kwargs.get('subLinFit',subLinFit)
    dFResp = kwargs.get('dFResp',dFResp)
    t_base = kwargs.get('t_base',t_base)
    t_resp_excl = kwargs.get('t_resp_excl',t_resp_excl)

    # get time vector
    t = getTimeVec(imgSeries.shape[-1], **kwargs)

    # calculate response within ROI if provided
    ROImask = kwargs.get('ROImask', np.ones(imgSeries.shape[:2]))
    signal = imgSeries[ROImask==1, :].mean(axis=0)
    
    # whether to subtract fitted line
    if subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]
    else:
        # photo-bleaching may cause unnecessary exclusion
        warnings.warn("Linear fit subtraction is suggested before excluding outliers.")

    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, t_base=t_base, **kwargs)[0]
    
    # calculate dFF or dF response
    resp = (signal - f0) if dFResp else (signal - f0) / f0

    # get time windows for baseline and response
    base_indices = np.where((t >= t_base[0]) & (t <= t_base[1]))[0]
    resp_indices = np.where((t >= t_resp_excl[0]) & (t <= t_resp_excl[1]))[0]
    
    # equivalent to comparing by raw F (`signal`), as baseline F (f0) is consistently positive
    avgResp = resp[resp_indices].mean()
    maxResp = resp[resp_indices].max()
    meanBase = resp[base_indices].mean()
    baseSD = resp[base_indices].std()

    # traces whose Avg response is 3 SDs below Avg baseline are extreme outliers even if no sound is played
    # if maxResp < meanBase, peak dFF response is negative -> makes no sense
    is_valid = (avgResp >= meanBase - 3*baseSD) and (maxResp >= meanBase)

    return is_valid


def is_significant_resp(imgSeries: np.ndarray, subLinFit: bool = True, dFResp: bool = False, 
                        t_base: tuple[float,float] = (2,3), t_resp: tuple[float,float] = (3.3,4), 
                        butterFilt: bool = True, bidirect: bool = True, thres_2SD: bool = False, **kwargs) -> bool:
    """
    Checks whether the response is significant.
    Insigificant response refers to traces whose max response (and min response) is within 3 SDs (2 SDs) range of Avg baseline.

    Args:
        imgSeries (array): 3D array of shape (Y, X, frame)
        subLinFit (bool): whether to subtract fitted line
        dFResp (bool): if true, calculate dF response rather than dFF
        t_base (tuple): time window (in seconds) for baseline
        t_resp (tuple): time window (in seconds) for response
        butterFilt (bool): whether to apply low pass filter
        bidirect (bool): whether to check response significance in both directions
                         if false, assume positive response and test by the positive threshold only
        thres_2SD (bool): if true, thresholds are set 2 SDs from Avg baseline rather than 3 SDs
        **kwargs: Optional arguments that will override default
            example: ROImask (np.ndarray): 2D binary mask array specifying the region of interest
                     cutoff_freq (float): low-pass filter cutoff frequency

    Returns:
        is_significant (bool): `True` for significant response, `False` for insignificant response
    """
    
    # optionally override parameters using kwargs
    subLinFit = kwargs.get('subLinFit',subLinFit)
    dFResp = kwargs.get('dFResp',dFResp)
    t_base = kwargs.get('t_base',t_base)
    t_resp = kwargs.get('t_resp',t_resp)
    butterFilt = kwargs.get('butterFilt',butterFilt)
    bidirect = kwargs.get('bidirect',bidirect)
    thres_2SD = kwargs.get('thres_2SD',thres_2SD)

    # get time vector
    t = getTimeVec(imgSeries.shape[-1], **kwargs)

    # calculate response within ROI if provided
    ROImask = kwargs.get('ROImask', np.ones(imgSeries.shape[:2]))
    signal = imgSeries[ROImask==1, :].mean(axis=0)

    # whether to subtract fitted line
    if subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]
    else:
        # photo-bleaching may cause bias
        warnings.warn("Linear fit subtraction is suggested before testing insignificant responses.")

    # whether to apply low pass filter
    if butterFilt:
        # Default cutoff_freq = 2
        # cutoff_freq = kwargs.get('cutoff_freq', 2)
        # signal = butterFilter(signal, cutoff_freq=cutoff_freq)
        signal = butterFilter(signal, **kwargs)

    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, t_base=t_base, t_resp=t_resp, **kwargs)[0]
    
    # calculate dFF or dF response
    resp = (signal - f0) if dFResp else (signal - f0) / f0

    # get time windows for baseline and response
    base_indices = (t >= t_base[0]) & (t <= t_base[1])
    resp_indices = (t >= t_resp[0]) & (t <= t_resp[1])

    # equivalent to comparing by raw F (`signal`) as baseline F (f0) is consistently positive
    maxResp = resp[resp_indices].max()
    minResp = resp[resp_indices].min()
    meanBase = resp[base_indices].mean()
    baseSD = resp[base_indices].std()

    # set threshold
    thres = 2*baseSD if thres_2SD else 3*baseSD

    # compare max response to upper threshold (Avg+3SD) and min dFF response to lower threshold (Avg-3SD)
    if bidirect:
        # test significance in both directions
        is_significant = (maxResp > meanBase + thres) or (minResp < meanBase - thres)
    else:
        # only consider positive response
        is_significant = maxResp > meanBase + thres

    return is_significant


def pkDFFimg(imgSeries: np.ndarray,
                subLinFit: bool = True, 
                butterFilt: bool = True, 
                dFResp: bool = False, 
                negExcl: bool = True, 
                insigExcl: bool = False, 
                sponCorrect: bool = False, 
                t_base: tuple[float,float] = (2,3), 
                **kwargs) -> float | None:
    """
    Calculates peak dFF response from image series array.
    
    Args:
        imgSeries (array): 3D array of shape (Y, X, frame)
        subLinFit (bool): whether to subtract fitted line
        butterFilt (bool): whether to apply low pass filter
        dFResp (bool): if true, calculate dF response rather than dFF
        negExcl (bool): if true, exclude outliers whose Avg responses (within response time window) are 3 SDs below Avg baseline, 
                        or whose max responses are below Avg baseline
        insigExcl (bool): if true, convert insignificant traces whose max and min responses are within ±3 SDs of Avg baseline to 0
        sponCorrect (bool): Used to correct for spontaneous activities or noise
                            if true, substract max spontaneous response (within baseline time window) from peak dFF response (within response time window)
        t_base (tuple): time window (in seconds) for baseline
        **kwargs: Optional arguments that will override default
            example: ROImask (np.ndarray): 2D binary mask array specifying the region of interest
                     cutoff_freq (float): low-pass filter cutoff frequency

    Returns:
        pk (float or `None`): peak dFF or dF response
    """
    
    # add explicit arguments to kwargs
    kwargs['subLinFit'] = subLinFit
    kwargs['butterFilt'] = butterFilt
    kwargs['dFResp'] = dFResp
    kwargs['t_base'] = t_base

    # check for negative response
    # if `negExcl` is true, return `None`
    if negExcl and not is_valid_resp(imgSeries, **kwargs):
        return None

    # check for insignificant response
    # if `insigExcl` is true, return `0`
    if insigExcl and not is_significant_resp(imgSeries, **kwargs):
        return 0

    t = getTimeVec(imgSeries.shape[-1], **kwargs)
    ROImask = kwargs.get('ROImask', np.ones(imgSeries.shape[:2]))
    signal = imgSeries[ROImask==1, :].mean(axis=0)
    
    # whether to subtract fitted line
    if subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]
    
    # whether to apply low pass filter
    if butterFilt:
        # Default cutoff_freq = 2
        # cutoff_freq = kwargs.get('cutoff_freq', 2)
        # signal = butterFilter(signal, cutoff_freq=cutoff_freq)
        signal = butterFilter(signal, **kwargs)

    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, **kwargs)[0]
    
    # calculate dFF or dF response
    resp = (signal - f0) if dFResp else (signal - f0) / f0

    # get baseline and peak from dFF or dF
    pkBase_output, pkResp_output = getBaseResp(resp, t, **kwargs)
    
    # calculate peak dFF reponse
    pk = pkResp_output - pkBase_output
    
    # correct for spontaneous activities or noise if `sponCorrect` is true
    if sponCorrect:
        # substract the max amplitude within baseline time window from the peak dFF response
        base_indices = np.where((t >= t_base[0]) & (t <= t_base[1]))[0]
        maxSpon = resp[base_indices].max() - pkBase_output
        pk -= maxSpon

    return pk


def meanPlusMinusSem(traceXtimeArray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mean, mean plus standard error of the mean (SEM), 
    and mean minus SEM along the first dimension of a 2D array.

    can use in plot like so:        
    
    u,upsem,umsem = meanPMstd(np.array(b[F].tolist()))
    ax.plot(t, u, '-', color = colors[i], label=a)
    ax.fill_between(t, umsem, upsem, alpha=0.2)

    Parameters:
    -----------
    traceXtimeArray : np.ndarray
        A 2D NumPy array where rows correspond to individual traces 
        and columns correspond to time points.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three 1D arrays:
        - Mean values across traces for each time point.
        - Mean values plus SEM across traces for each time point.
        - Mean values minus SEM across traces for each time point.
    """
    u = traceXtimeArray.mean(axis=0)
    std = traceXtimeArray.std(axis=0)
    sem = std / np.sqrt(traceXtimeArray.shape[0])

    return u, u + sem, u - sem


def updateTable_signal(df: pd.DataFrame, qcam2img: dict, mask_name: str = 'response_mask', 
                       t_base: tuple = (2.0, 3.0), t_resp: tuple = (3.3, 4.0), cutoff_freq: float = 2, 
                       add_filtered_data: bool = True, test_negative: bool = True, **kwargs):
    """
    Update metadata dataframe with raw and processed signals within ROI.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'qcam', 'dir', and 'treatment'.
        qcam2img (dict): Dictionary mapping each qcam file path to its corresponding image data.
        mask_name (str, optional): Filename (case-sensitive) of the binary mask to search for. Defaults to 'response_mask'.
        t_base (tuple, optional): Baseline time window. Defaults to (2.0, 3.0).
        t_resp (tuple, optional): Response time window. Defaults to (3.3, 4.0).
        cutoff_freq (float, optional): Low-pass filter cutoff frequency. Defaults to 2.
        add_filtered_data (bool, optional): Whether to add columns including processed data after applying low-pass filter: 
                                            'F_ROI_linFilt_butterFilt', 'f0_ROI_linFilt_butterFilt', 
                                            'dF_ROI_linFilt_butterFilt', 'dF_ROI_linFilt_butterFilt_peak', 
                                            'dFF_ROI_linFilt_butterFilt', 'dFF_ROI_linFilt_butterFilt_peak'.
        test_negative (bool, optional): Whether to add column 'valid' to indicate negative traces.
        **kwargs: Optional arguments that will override default.
            example: byXSG (bool, optional): Whether to adjust baseline and response time windows according to XSG files.
                                             Defaults to 'False'.
                     stimStart (float, optional): Stimulus start time (in seconds) by default. Defaults to 3.

    Returns:
        df_updated (pd.DataFrame): Updated metadata dataframe including new columns:
                                   'ROImask', 'time', 'baseWindow', 'respWindow', 
                                   'F_ROI_raw', 'F_ROI_linFilt', 'F_ROI_linFilt_butterFilt', 
                                   'f0_ROI_raw', 'f0_ROI_linFilt', 'f0_ROI_linFilt_butterFilt', 
                                   'dF_ROI_raw', 'dF_ROI_raw_peak', 
                                   'dF_ROI_linFilt', 'dF_ROI_linFilt_peak', 
                                   'dF_ROI_linFilt_butterFilt', 'dF_ROI_linFilt_butterFilt_peak', 
                                   'dFF_ROI_raw', 'dFF_ROI_raw_peak', 
                                   'dFF_ROI_linFilt', 'dFF_ROI_linFilt_peak', 
                                   'dFF_ROI_linFilt_butterFilt', 'dFF_ROI_linFilt_butterFilt_peak', 
                                   'valid'.
    
    Notes:
        - The function grabs 'response_mask.joblib' file in the experiment folder ('dir') as ROI masks.
        - The function grabs 'STIMULUS_START_*_sec*' file in the experiment folder ('dir') 
          to adjust baseline and response time windows based on when stimuli start.
        - ROI mask selection priority:
          1. Files containing both '{mask_name}' (case-sensitive) and treatment-specific 'pre'/'post' (case-insensitive).
             - Assume that treatments use different ROIs (due to animal/platform movements when inserting the pipette tip).
          2. Files containing only '{mask_name}' (case-sensitive).
             - Use the same ROI for all treatments.
    """
    
    # Check whether required columns exist
    required_col = ['qcam', 'dir', 'treatment']
    if not all(col in df.columns for col in required_col):
        raise ValueError(f"DataFrame must contain the following columns: {required_col}")
    
    df_updated = df.copy()
    
    # Add binary mask of ROI by searching for 'joblib' file in the same directory
    masks = []
    for dir, treatment in zip(df_updated['dir'], df_updated['treatment']):
        # Determine whether to search for 'pre' or 'post' in filenames (case-insensitive)
        treatment_key = 'post' if 'post' in treatment.lower() else 'pre'
        
        # Find all joblib files in the directory
        all_masks = glob(os.path.join(dir, '*response_mask*.joblib'))
        all_masks = [f for f in all_masks if 'contour' not in f]
        
        # Find treatment-specific filenames which contain both '{mask_name}' and '{treatment_key}'
        treatment_masks = [
            f for f in all_masks 
            if (mask_name in os.path.basename(f) 
            and treatment_key in os.path.basename(f).lower())
        ]
        
        if len(treatment_masks) > 1:
            warnings.warn(f"Multiple {treatment_key} masks found in {dir}: {treatment_masks}, using {treatment_masks[0]}")
            masks.append(joblib.load(sorted(treatment_masks)[0]))
        elif len(treatment_masks) == 1:
            # Choose files with treatment-specific names first
            masks.append(joblib.load(treatment_masks[0]))
        else:
            # Fallback to general filenames (only needs '{mask_name}') if treatment-specific filenames are not found
            general_masks = [f for f in all_masks if mask_name in os.path.basename(f)]
            if not general_masks:
                raise FileNotFoundError(f"No suitable ROI mask found in {dir}. Need file containing '{mask_name}'.")
            # Sort all files and select the first one
            masks.append(joblib.load(sorted(general_masks)[0]))

    df_updated['ROImask'] = masks

    # Add time vector
    df_updated['time'] = df_updated['qcam'].apply(lambda x: getTimeVec(qcam2img[x].shape[-1], **kwargs))

    # Check whether all traces are of the same frame counts
    if df_updated['time'].apply(lambda x: x.shape[0]).nunique() > 1:
        warnings.warn("Traces have more than one frame count. Ensure time windows are adjusted.")

    # Add baseline and response time windows
    # Adjust windows automatically according to XSG files or 'STIMULUS_START_*_sec*' files
    df_updated['baseWindow'] = metadataProcess.getBaseRespWindow(df_updated, t_base=t_base, t_resp=t_resp, **kwargs)['baseWindow']
    df_updated['respWindow'] = metadataProcess.getBaseRespWindow(df_updated, t_base=t_base, t_resp=t_resp, **kwargs)['respWindow']

    # Add fluorescence trace (F) within ROI
    # Raw data
    df_updated['F_ROI_raw'] = df_updated.apply(lambda x: qcam2img[x['qcam']][x['ROImask']==1, :].mean(axis=0), axis=1)
    # Subtracted linear fit
    df_updated['F_ROI_linFilt'] = df_updated.apply(lambda x: subtractLinFit(x['time'], x['F_ROI_raw'], t_base=x['baseWindow'], **kwargs)[0], axis=1)
    if add_filtered_data:
        # Processed data (subtracted linear fit and added low-pass filter)
        df_updated['F_ROI_linFilt_butterFilt'] = df_updated['F_ROI_linFilt'].apply(lambda x: butterFilter(x, cutoff_freq=cutoff_freq, **kwargs))

    # Add baseline fluorescence (f0) within ROI
    # Raw data
    df_updated['f0_ROI_raw'] = df_updated.apply(
        lambda x: getBaseResp(x['F_ROI_raw'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
    )
    # Subtracted linear fit
    df_updated['f0_ROI_linFilt'] = df_updated.apply(
        lambda x: getBaseResp(x['F_ROI_linFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
    )
    if add_filtered_data:
        # Processed data (subtracted linear fit and added low-pass filter)
        df_updated['f0_ROI_linFilt_butterFilt'] = df_updated.apply(
            lambda x: getBaseResp(x['F_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
        )

    # Add dF response within ROI
    # Raw data
    df_updated['dF_ROI_raw'] = df_updated.apply(lambda x: x['F_ROI_raw'] - x['f0_ROI_raw'], axis=1)
    df_updated['dF_ROI_raw_peak'] = df_updated.apply(
        lambda x: pkDFFimg(
            qcam2img[x['qcam']], subLinFit=False, ROImask=x['ROImask'], butterFilt=False, 
            dFResp=True, negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
        ), 
        axis=1
    )
    # Subtracted linear fit
    df_updated['dF_ROI_linFilt'] = df_updated.apply(lambda x: x['F_ROI_linFilt'] - x['f0_ROI_linFilt'], axis=1)
    df_updated['dF_ROI_linFilt_peak'] = df_updated.apply(
        lambda x: pkDFFimg(
            qcam2img[x['qcam']], ROImask=x['ROImask'], butterFilt=False, 
            dFResp=True, negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
        ), 
        axis=1
    )
    if add_filtered_data:
        # Processed data (subtracted linear fit and added low-pass filter)
        df_updated['dF_ROI_linFilt_butterFilt'] = df_updated.apply(
            lambda x: x['F_ROI_linFilt_butterFilt'] - x['f0_ROI_linFilt_butterFilt'], axis=1
        )
        df_updated['dF_ROI_linFilt_butterFilt_peak'] = df_updated.apply(
            lambda x: pkDFFimg(
                qcam2img[x['qcam']], ROImask=x['ROImask'], cutoff_freq=cutoff_freq, 
                dFResp=True, negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
            ), 
            axis=1
        )

    # Add dFF response within ROI
    # Raw data
    df_updated['dFF_ROI_raw'] = df_updated.apply(lambda x: x['dF_ROI_raw'] / x['f0_ROI_raw'], axis=1)
    df_updated['dFF_ROI_raw_peak'] = df_updated.apply(
        lambda x: pkDFFimg(
            qcam2img[x['qcam']], subLinFit=False, ROImask=x['ROImask'], butterFilt=False, 
            negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
        ), 
        axis=1
    )
    # Subtracted linear fit
    df_updated['dFF_ROI_linFilt'] = df_updated.apply(lambda x: x['dF_ROI_linFilt'] / x['f0_ROI_linFilt'], axis=1)
    df_updated['dFF_ROI_linFilt_peak'] = df_updated.apply(
        lambda x: pkDFFimg(
            qcam2img[x['qcam']], ROImask=x['ROImask'], butterFilt=False, 
            negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
        ), 
        axis=1
    )
    if add_filtered_data:
        # Processed data (subtracted linear fit and added low-pass filter)
        df_updated['dFF_ROI_linFilt_butterFilt'] = df_updated.apply(
            lambda x: x['dF_ROI_linFilt_butterFilt'] / x['f0_ROI_linFilt_butterFilt'], axis=1
        )
        df_updated['dFF_ROI_linFilt_butterFilt_peak'] = df_updated.apply(
            lambda x: pkDFFimg(
                qcam2img[x['qcam']], ROImask=x['ROImask'], cutoff_freq=cutoff_freq, 
                negExcl=False, t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs
            ), 
            axis=1
        )

    if test_negative:
        # Identify whether each trace is an outlier with negative response (`False`)
        # Test sign of peak dFF response only, as the result is the same for peak dF response
        # Time window for exclusive thresholds is suggested overlapping with response time window
        df_updated['valid'] = df_updated.apply(
            lambda x: is_valid_resp(
                qcam2img[x['qcam']], ROImask=x['ROImask'], t_base=x['baseWindow'], 
                t_resp=x['respWindow'], t_resp_excl=x['respWindow'], **kwargs
            ), 
            axis=1
        )

    return df_updated


def get_avg_in_rois(img_series: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Computes the average fluorescence within a series of ROIs sweeping across the entire image.

    For each binary mask in 'masks', this function extracts the corresponding region from 
    each frame in 'img_series' and computes the average fluorescence intensity within that ROI.

    Args:
        img_series (np.ndarray): 3D or 4D image array of shape [Y, X, frame] or [traceNumber, Y, X, frame].
        masks (np.ndarray): 3D array of binary masks (ROIs) of shape [maskNumber, Y, X].

    Returns:
        roi_avg (np.ndarray): 2D or 3D array of average fluorescence traces within ROIs.
                              Shape will be [maskNumber, frame] or [traceNumber, maskNumber, frame].
    """
    
    print(f"Image array: {img_series.shape}")

    # Check the shape of image and mask arrays
    if img_series.ndim not in (3, 4):
        raise ValueError("Image array must be 3D or 4D.")
    if masks.ndim != 3:
        raise ValueError("Mask array must be 3D.")
    
    # Initialize an array to store the average values
    roi_avg = np.zeros((len(masks), img_series.shape[-1])) if img_series.ndim == 3 else \
              np.zeros((img_series.shape[0], len(masks), img_series.shape[-1]))

    # Compute the average value within each ROI for each frame
    for i, mask in enumerate(masks):
        # Use broadcasting to apply the mask across all frames (and all traces)
        if img_series.ndim == 3:
            masked_data = img_series[mask, :]  # Shape will be (num_masked_pixels, num_frames)
            roi_avg[i, :] = np.mean(masked_data, axis=0)  # Average across pixels
        else:
            # Image array is 4D
            masked_data = img_series[:, mask, :]  # Shape will be (num_traces, num_masked_pixels, num_frames)
            roi_avg[:, i, :] = np.mean(masked_data, axis=1)

    print(f"ROI fluorescence array: {roi_avg.shape}")

    return roi_avg


def cluster_roi(roi_trace: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    Perform hierarchical clustering on ROI fluorescence responses.

    This function takes the temporal fluorescence traces within a series of sweeping ROIs 
    and clusters them based on their similarity using hierarchical clustering.

    Args:
        roi_trace (np.ndarray): 2D or 3D array of ROI fluorescence traces.
                                Shape should be [maskNumber, frame] or [traceNumber, maskNumber, frame].
        method (str, optional): Linkage method for hierarchical clustering.
                                Defaults to 'ward', using the Ward variance minimization algorithm.
    
    Returns:
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix of shape [maskNumber-1, 4].
                                     Each row contains [cluster1, cluster2, distance, cluster_size].
    """

    # Check the shape of input array
    if roi_trace.ndim not in (2, 3):
        raise ValueError("Trace array must be 2D or 3D.")
    
    # Average across different trials (repetitions) for 3D array
    roi_trace = np.mean(roi_trace, axis=0) if roi_trace.ndim == 3 else roi_trace
    
    # Flatten the 2D ROI data for clustering
    X = roi_trace.reshape(roi_trace.shape[0], -1)
    print(f"Input data shape for clustering: {X.shape}")

    # Compute pairwise distances using Euclidean metric
    distance_matrix = pdist(X, metric='euclidean')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)
    
    return linkage_matrix
