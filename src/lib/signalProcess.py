import numpy as np
import pandas as pd
import os
import warnings
import joblib

from scipy.signal import butter, filtfilt
from typing import Tuple
from glob import glob
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from functools import reduce

import lib.metadataProcess as metadataProcess


def getTimeVec(nFrames: int, 
               frameRate: float = 20, 
               zeroStart: bool = True,
               delayAdjust: float = 0.025,
               **kwargs) -> np.ndarray:
    """
    Generate time vector from frame count and rate.

    Args:
        nFrames (int): number of frames
        frameRate (float): number of frames acquired per second
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

    # first frame acquired (1/fr) s after start
    t = (np.arange(1, nFrames + 1) * (1 / frameRate)) + delayAdjust
    # first frame acquired at start (starts at 0)
    if zeroStart:
        return t-(1/frameRate)
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
        signal (numpy array): 1D, 2D, 3D or 4D signal array to be filtered of shape [frame] or [traceNumber, frame]
                              or [traceNumber, neuronNumber, frame] (2-photon)
                              or [neuronNumber, soundLevel, traceNumber, frame] (2-photon).
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
        # If the signal is 3D, process each trace (the last dimension) independently
        # Initialize the output array
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                filtered_signal[i, j, :] = filtfilt(b, a, signal[i, j, :])
    elif signal.ndim == 4:
        # If the signal is 4D, process each trace (the last dimension) independently
        # Initialize the output array
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                for k in range(signal.shape[2]):
                    filtered_signal[i, j, k, :] = filtfilt(b, a, signal[i, j, k, :])
    else:
        raise ValueError("Signal array must be 1D, 2D, 3D, or 4D.")
    
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
        **kwargs: Optional arguments that will override default.
                For example: t_base (tuple): time window (in seconds) for baseline calculation when `offset = True`.
    
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
        # Assume signal shape is [traceNumber, maskNumber, frame]. Process each trace (the last dimension) independently
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


def subtractLogFit(t, signal: np.ndarray, offset: bool = True, **kwargs) -> np.ndarray:
    """
    Subtracts logarithmic fit (a*ln(t+b) + c) of signal from signal.
    Useful to remove logarithmic-like signal drift.

    Args:
        t (list or array): time vector (in seconds).
        signal (numpy array): 1D, 2D, or 3D signal array of shape 
                              [frame], [traceNumber, frame], or [traceNumber, maskNumber, frame].
        offset (bool, optional): whether to add baseline fluorescence (f0) back to the corrected signal as the offset.
                                 Defaults to 'True'.
        **kwargs: Optional arguments that will override default.
    
    Returns:
        corrected_signal (numpy array): signal array after removal of log fit (same shape as input signal).
        coef (numpy array): array of [a, b, c] coefficients for each trace 
                            (one lower dimension than input signal).
    """

    offset = kwargs.get('offset', offset)
    t = np.asarray(t)

    def log_model(x, a, b, c):
        return a * np.log(x + b) + c

    # 1D signal case
    if signal.ndim == 1:
        # Set maxfev to avoid RuntimeError: Optimal parameters not found (maxfev=800 by default)
        coef, _ = curve_fit(log_model, t, signal, p0=[-1, 1, np.mean(signal)], maxfev=1000000)
        a, b, c = coef

        if offset:
            f0 = getBaseResp(signal, t, **kwargs)[0]
            corrected_signal = signal - log_model(t, a, b, c) + f0
        else:
            corrected_signal = signal - log_model(t, a, b, c)

    # 2D signal case
    elif signal.ndim == 2:
        corrected_signal = np.zeros_like(signal)
        coef = np.zeros((signal.shape[0], 3))

        for i in range(signal.shape[0]):
            coef[i], _ = curve_fit(log_model, t, signal[i], p0=[-1, 1, np.mean(signal[i])], maxfev=1000000)
            a, b, c = coef[i]

            if offset:
                f0 = getBaseResp(signal[i], t, **kwargs)[0]
                corrected_signal[i] = signal[i] - log_model(t, a, b, c) + f0
            else:
                corrected_signal[i] = signal[i] - log_model(t, a, b, c)

    # 3D signal case
    elif signal.ndim == 3:
        corrected_signal = np.zeros_like(signal)
        coef = np.zeros((signal.shape[0], signal.shape[1], 3))

        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                coef[i, j], _ = curve_fit(log_model, t, signal[i, j], p0=[-1, 1, np.mean(signal[i, j])], maxfev=1000000)
                a, b, c = coef[i, j]

                if offset:
                    f0 = getBaseResp(signal[i, j], t, **kwargs)[0]
                    corrected_signal[i, j] = signal[i, j] - log_model(t, a, b, c) + f0
                else:
                    corrected_signal[i, j] = signal[i, j] - log_model(t, a, b, c)

    else:
        raise ValueError("Signal array must be 1D, 2D, or 3D.")

    return corrected_signal, coef


def getBaseResp(signal: np.ndarray, t: np.ndarray, 
                t_base: tuple[float,float] = (2.2, 2.9),
                t_resp: tuple[float,float] = (3.0, 3.15),
                negResp: bool = False,
                calMeanResp: bool = False,
                avgAdjacentFrames: bool = False,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract average signal at t_base and max signal between t_resp.

    Args:
        signal (numpy array): signal array of shape [frame] or [traceNumber, frame]
                              or [neuronNumber, soundLevel, frame] (2-photon).
        t (list or array): time vector (in seconds).
        t_base (tuple, optional): time window (in seconds) for baseline.
        t_resp (tuple, optional): time window (in seconds) for response.
        negResp (bool, optional): whether to extract max signal between t_resp in either direction.
                                - 'True': Response with max absolute value is returned, whether positive or negative.
                                          Original sign of the response is preserved.
                                - 'False': Only max positive response is returned.
                                Defaults to 'False'.
        calMeanResp (bool, optional): whether to calculate mean response between t_resp instead of max response.
                                      Used in quantifying sustained tonic response in 2-photon imaging.
                                      Handles NaN values by ignoring them in mean calculation.
                                      Only works when 'negResp = False'.
                                      Defaults to 'False'.
        avgAdjacentFrames (bool, optional): whether to average peak amplitude with adjacent two frames (±1 frame) to reduce noise.
                                            Only works when 'negResp = False' and 'calMeanResp = False'.
                                            Defaults to 'False'.
        **kwargs: Optional arguments that will override default.

    Returns:
        tuple (base, resp) where:
            - base (numpy array): average signal between t_base for each trace.
                                  - 1D input: scalar
                                  - 2D input: array of shape [traceNumber]
                                  - 3D input: array of shape [neuronNumber, soundLevel]
            - resp (numpy array): max signal between t_resp for each trace.
                                  - 1D input: scalar
                                  - 2D input: array of shape [traceNumber]
                                  - 3D input: array of shape [neuronNumber, soundLevel]
    Notes:
        If negative response is calculated, 'negResp = True' only works for dFF response but not raw F.
    """

    # Optionally override parameters using kwargs
    t_base = kwargs.get('t_base',t_base)
    t_resp = kwargs.get('t_resp',t_resp)
    negResp = kwargs.get('negResp',negResp)
    calMeanResp = kwargs.get('calMeanResp',calMeanResp)
    avgAdjacentFrames = kwargs.get('avgAdjacentFrames', avgAdjacentFrames)

    # Get indices for baseline and response time windows
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
            if avgAdjacentFrames and not calMeanResp:
                # Average peak amplitude with adjacent two frames (±1 frame) to reduce noise
                resp_values = signal[resp_indices]
                peak_idx = np.argmax(resp_values)
                start_idx = max(0, peak_idx - 1)
                end_idx = min(len(resp_values), peak_idx + 2)
                resp = np.mean(resp_values[start_idx:end_idx])
            else:
                resp = signal[resp_indices].max() if not calMeanResp else np.nanmean(signal[resp_indices])
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
            if avgAdjacentFrames and not calMeanResp:
                # Average peak amplitude with adjacent two frames (±1 frame) to reduce noise
                resp = np.zeros(signal.shape[0])
                for i in range(signal.shape[0]):
                    resp_values = signal[i, resp_indices]
                    peak_idx = np.argmax(resp_values)
                    start_idx = max(0, peak_idx - 1)
                    end_idx = min(len(resp_values), peak_idx + 2)
                    resp[i] = np.mean(resp_values[start_idx:end_idx])
            else:
                resp = np.max(signal[:, resp_indices], axis=1) if not calMeanResp else np.nanmean(signal[:, resp_indices], axis=1)
    elif signal.ndim == 3:
        # If the signal is 3D, process each trace (the last dimension) re neuronNumber/soundLevel combination
        base = np.mean(signal[:, :, base_indices], axis=2)
        if negResp:
            # in each trace, find the responses with the max absolute values and keep their original signs
            resp_values = signal[:, :, resp_indices]
            resp = resp_values[np.arange(resp_values.shape[0])[:, np.newaxis],  # broadcasting for 2D indexing
                               np.arange(resp_values.shape[1]), 
                               np.argmax(np.abs(resp_values), axis=2)]
        else:
            # in each trace, find the responses with the max numeric values
            # may ignore negative responses
            if avgAdjacentFrames and not calMeanResp:
                # Average peak amplitude with adjacent two frames (±1 frame) to reduce noise
                resp = np.zeros((signal.shape[0], signal.shape[1]))
                for i in range(signal.shape[0]):
                    for j in range(signal.shape[1]):
                        resp_values = signal[i, j, resp_indices]
                        peak_idx = np.argmax(resp_values)
                        start_idx = max(0, peak_idx - 1)
                        end_idx = min(len(resp_values), peak_idx + 2)
                        resp[i, j] = np.mean(resp_values[start_idx:end_idx])
            else:
                resp = np.max(signal[:, :, resp_indices], axis=2) if not calMeanResp else np.nanmean(signal[:, :, resp_indices], axis=2)
    else:
        raise ValueError("Signal array must be 1D, 2D, or 3D.")
        
    return base, resp


def dFFcalc(signal, **kwargs):
    """
    Calculates dFF for a signal such as average fluorescence over time.

    Args:
        signal (numpy array): 1D, 2D, or 3D signal array (e.g., raw fluorescence).
                              Shape can be [frame] or [traceNumber, frame] or [traceNumber, neuronNumber, frame] (2-photon).
        **kwargs: Optional arguments that will override default.
            For example: t (numpy array): time vector (in seconds) corresponding to the last dimension of the signal. 
                         t_base (tuple): time window (in seconds) for baseline
                         t_resp (tuple): time window (in seconds) for response

    Returns:
        dFF (numpy array): deltaF/F of input signal (same shape as input signal).
        dF (numpy array): deltaF of input signal (same shape as input signal).
        f0 (float or numpy array): baseline signal (scalar for 1D, array for 2D or 3D).
    """

    t = kwargs.pop('t', getTimeVec(signal.shape[-1], **kwargs))

    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, **kwargs)[0]
    
    # Calculate dF and dFF
    if signal.ndim == 1:
        dF = signal - f0
        dFF = dF / f0
    elif signal.ndim == 2:
        dF = signal - f0[:, np.newaxis]
        dFF = dF / f0[:, np.newaxis]
    elif signal.ndim == 3:
        dF = signal - f0[:, :, np.newaxis]
        dFF = dF / f0[:, :, np.newaxis]
    else:
        raise ValueError("Signal array must be 1D, 2D, or 3D.")
    
    return dFF, dF, f0
     

def is_valid_resp(imgSeries: np.ndarray, subLinFit: bool = True, subLogFit: bool = False, dFResp: bool = False, 
                  t_base: tuple[float,float] = (2,3), t_resp_excl: tuple[float,float] = (3.3,4), **kwargs) -> bool:
    """
    Checks whether the response is a negative outlier.
    Negative outliers refer to traces whose Avg response is 3 sample standard deviations (SDs) below Avg baseline or peak response is below 0.

    Args:
        imgSeries (array): 3D array of shape (Y, X, frame)
        subLinFit (bool, optional): whether to subtract fitted line
        subLogFit (bool, optional): whether to subtract fitted logarithmic curve. If True, `subLinFit` will be ignored
        dFResp (bool, optional): if true, calculate dF response rather than dFF
        t_base (tuple, optional): time window (in seconds) for baseline
        t_resp_excl (tuple, optional): time window (in seconds) to exclude outliers (negative response)
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
    
    # whether to subtract fitted line/logarithmic curve
    if subLogFit:
        signal = subtractLogFit(t, signal, **kwargs)[0]
    elif subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]
    else:
        # photo-bleaching may cause unnecessary exclusion
        warnings.warn("Linear/Logarithmic fit subtraction is suggested before excluding outliers.")

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
    baseSD = resp[base_indices].std(ddof=1)

    # traces whose Avg response is 3 SDs below Avg baseline are extreme outliers even if no sound is played
    # if maxResp < meanBase, peak dFF response is negative -> makes no sense
    is_valid = (avgResp >= meanBase - 3*baseSD) and (maxResp >= meanBase)

    return is_valid


def is_significant_resp(imgSeries: np.ndarray, subLinFit: bool = True, subLogFit: bool = False, dFResp: bool = False, 
                        t_base: tuple[float,float] = (2,3), t_resp: tuple[float,float] = (3.3,4), 
                        butterFilt: bool = True, bidirect: bool = True, thres_2SD: bool = False, **kwargs) -> bool:
    """
    Checks whether the response is significant.
    Insigificant response refers to traces whose max response (and min response) is 
    within 3 sample standard deviations (2 sample standard deviations) range of Avg baseline.

    Args:
        imgSeries (array): 3D array of shape (Y, X, frame)
        subLinFit (bool, optional): whether to subtract fitted line
        subLogFit (bool, optional): whether to subtract fitted logarithmic curve. If True, `subLinFit` will be ignored
        dFResp (bool, optional): if true, calculate dF response rather than dFF
        t_base (tuple, optional): time window (in seconds) for baseline
        t_resp (tuple, optional): time window (in seconds) for response
        butterFilt (bool, optional): whether to apply low pass filter
        bidirect (bool, optional): whether to check response significance in both directions
                                   if false, assume positive response and test by the positive threshold only
        thres_2SD (bool, optional): if true, thresholds are set 2 SDs from Avg baseline rather than 3 SDs
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

    # whether to subtract fitted line/logarithmic curve
    if subLogFit:
        signal = subtractLogFit(t, signal, **kwargs)[0]
    elif subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]
    else:
        # photo-bleaching may cause bias
        warnings.warn("Linear/Logarithmic fit subtraction is suggested before testing insignificant responses.")

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
    base_indices = np.where((t >= t_base[0]) & (t <= t_base[1]))[0]
    resp_indices = np.where((t >= t_resp[0]) & (t <= t_resp[1]))[0]

    # equivalent to comparing by raw F (`signal`) as baseline F (f0) is consistently positive
    maxResp = resp[resp_indices].max()
    minResp = resp[resp_indices].min()
    meanBase = resp[base_indices].mean()
    baseSD = resp[base_indices].std(ddof=1)

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
             subLogFit: bool = False,
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
        subLinFit (bool, optional): whether to subtract fitted line
        subLogFit (bool, optional): whether to subtract fitted logarithmic curve. If True, `subLinFit` will be ignored
        butterFilt (bool, optional): whether to apply low pass filter
        dFResp (bool, optional): if true, calculate dF response rather than dFF
        negExcl (bool, optional): if true, exclude outliers whose Avg responses (within response time window) are 3 SDs below Avg baseline, 
                                  or whose max responses are below Avg baseline
        insigExcl (bool, optional): if true, convert insignificant traces whose max and min responses are within ±3 SDs of Avg baseline to 0
        sponCorrect (bool, optional): Used to correct for spontaneous activities or noise
                                      if true, subtract max spontaneous response (within baseline time window) from peak dFF response (within response time window)
        t_base (tuple, optional): time window (in seconds) for baseline
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
    
    # whether to subtract fitted line/logarithmic curve
    if subLogFit:
        signal = subtractLogFit(t, signal, **kwargs)[0]
    elif subLinFit:
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
        # subtract the max amplitude within baseline time window from the peak dFF response
        base_indices = np.where((t >= t_base[0]) & (t <= t_base[1]))[0]
        maxSpon = resp[base_indices].max() - pkBase_output
        pk -= maxSpon

    return pk


def meanPlusMinusSem(traceXtimeArray: np.ndarray, ignoreNaN: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mean, mean plus sample standard error of the mean (SEM), 
    and mean minus SEM along the first dimension of a 2D array.

    can use in plot like so:        
    
    u,upsem,umsem = meanPMstd(np.array(b[F].tolist()))
    ax.plot(t, u, '-', color = colors[i], label=a)
    ax.fill_between(t, umsem, upsem, alpha=0.2)

    Parameters:
    -----------
    traceXtimeArray (np.ndarray): A 2D NumPy array where rows correspond to individual traces 
                                  and columns correspond to time points.
    ignoreNaN (bool, optional): If True, ignores NaN values in traces. Defaults to False.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three 1D arrays:
        - Mean values across traces for each time point.
        - Mean values plus SEM across traces for each time point.
        - Mean values minus SEM across traces for each time point.
    """
    if ignoreNaN:
        u = np.nanmean(traceXtimeArray, axis=0)
        std = np.nanstd(traceXtimeArray, axis=0, ddof=1)
    else:
        u = traceXtimeArray.mean(axis=0)
        std = traceXtimeArray.std(axis=0, ddof=1)

    sem = std / np.sqrt(traceXtimeArray.shape[0])

    return u, u + sem, u - sem


def updateTable_signal(df: pd.DataFrame, qcam2img: dict, mask_name: str = 'response_mask', 
                       t_base: tuple = (2.0, 3.0), t_resp: tuple = (3.3, 4.0), 
                       subLogFit: bool = False, cutoff_freq: float = 2, add_filtered_data: bool = True, 
                       test_negative: bool = True, **kwargs) -> pd.DataFrame:
    """
    Update metadata dataframe with raw and processed signals within ROI.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'qcam', 'dir', and 'treatment'.
        qcam2img (dict): Dictionary mapping each qcam file path to its corresponding image data.
        mask_name (str, optional): Filename (case-sensitive) of the binary mask to search for. Defaults to 'response_mask'.
        t_base (tuple, optional): Baseline time window. Defaults to (2.0, 3.0).
        t_resp (tuple, optional): Response time window. Defaults to (3.3, 4.0).
        subLogFit (bool, optional): Whether to subtract fitted logarithmic curve instead of linear fit. 
                                    Defaults to 'False', so linear fit is subtracted.
        cutoff_freq (float, optional): Low-pass filter cutoff frequency (Hz). Defaults to 2.
        add_filtered_data (bool, optional): Whether to add columns including processed data after applying low-pass filter: 
                                            'F_ROI_linFilt_butterFilt', 'f0_ROI_linFilt_butterFilt', 
                                            'dF_ROI_linFilt_butterFilt', 'dF_ROI_linFilt_butterFilt_peak', 
                                            'dFF_ROI_linFilt_butterFilt', 'dFF_ROI_linFilt_butterFilt_peak'.
                                            If 'subLogFit' is 'True', these columns will be named with 'logFilt' instead of 'linFilt'.
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
                                   If 'subLogFit' is 'True', the columns will be named with 'logFilt' instead of 'linFilt'.
    
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
    # Subtracted linear/logarithmic fit
    if subLogFit:
        df_updated['F_ROI_logFilt'] = df_updated.apply(lambda x: subtractLogFit(x['time'], x['F_ROI_raw'], t_base=x['baseWindow'], **kwargs)[0], axis=1)
    else:
        df_updated['F_ROI_linFilt'] = df_updated.apply(lambda x: subtractLinFit(x['time'], x['F_ROI_raw'], t_base=x['baseWindow'], **kwargs)[0], axis=1)
    if add_filtered_data:
        # Processed data (subtracted linear/logarithmic fit and added low-pass filter)
        if subLogFit:
            df_updated['F_ROI_logFilt_butterFilt'] = df_updated['F_ROI_logFilt'].apply(lambda x: butterFilter(x, cutoff_freq=cutoff_freq, **kwargs))
        else:
            df_updated['F_ROI_linFilt_butterFilt'] = df_updated['F_ROI_linFilt'].apply(lambda x: butterFilter(x, cutoff_freq=cutoff_freq, **kwargs))

    # Add baseline fluorescence (f0) within ROI
    # Raw data
    df_updated['f0_ROI_raw'] = df_updated.apply(
        lambda x: getBaseResp(x['F_ROI_raw'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
    )
    # Subtracted linear/logarithmic fit
    if subLogFit:
        df_updated['f0_ROI_logFilt'] = df_updated.apply(
            lambda x: getBaseResp(x['F_ROI_logFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
        )
    else:
        df_updated['f0_ROI_linFilt'] = df_updated.apply(
            lambda x: getBaseResp(x['F_ROI_linFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
        )
    if add_filtered_data:
        # Processed data (subtracted linear/logarithmic fit and added low-pass filter)
        if subLogFit:
            df_updated['f0_ROI_logFilt_butterFilt'] = df_updated.apply(
                lambda x: getBaseResp(x['F_ROI_logFilt_butterFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
            )
        else:
            df_updated['f0_ROI_linFilt_butterFilt'] = df_updated.apply(
                lambda x: getBaseResp(x['F_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], **kwargs)[0], axis=1
            )

    # Add dF response within ROI
    # Raw data
    df_updated['dF_ROI_raw'] = df_updated.apply(lambda x: x['F_ROI_raw'] - x['f0_ROI_raw'], axis=1)
    df_updated['dF_ROI_raw_peak'] = df_updated.apply(
        lambda x: getBaseResp(x['dF_ROI_raw'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                  getBaseResp(x['dF_ROI_raw'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
        axis=1
    )
    # Subtracted linear/logarithmic fit
    if subLogFit:
        df_updated['dF_ROI_logFilt'] = df_updated.apply(lambda x: x['F_ROI_logFilt'] - x['f0_ROI_logFilt'], axis=1)
        df_updated['dF_ROI_logFilt_peak'] = df_updated.apply(
            lambda x: getBaseResp(x['dF_ROI_logFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                      getBaseResp(x['dF_ROI_logFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
            axis=1
        )
    else:
        df_updated['dF_ROI_linFilt'] = df_updated.apply(lambda x: x['F_ROI_linFilt'] - x['f0_ROI_linFilt'], axis=1)
        df_updated['dF_ROI_linFilt_peak'] = df_updated.apply(
            lambda x: getBaseResp(x['dF_ROI_linFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                      getBaseResp(x['dF_ROI_linFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
            axis=1
        )
    if add_filtered_data:
        # Processed data (subtracted linear/logarithmic fit and added low-pass filter)
        if subLogFit:
            df_updated['dF_ROI_logFilt_butterFilt'] = df_updated.apply(
                lambda x: x['F_ROI_logFilt_butterFilt'] - x['f0_ROI_logFilt_butterFilt'], axis=1
            )
            df_updated['dF_ROI_logFilt_butterFilt_peak'] = df_updated.apply(
                lambda x: getBaseResp(x['dF_ROI_logFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                          getBaseResp(x['dF_ROI_logFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
                axis=1
            )
        else:
            df_updated['dF_ROI_linFilt_butterFilt'] = df_updated.apply(
                lambda x: x['F_ROI_linFilt_butterFilt'] - x['f0_ROI_linFilt_butterFilt'], axis=1
            )
            df_updated['dF_ROI_linFilt_butterFilt_peak'] = df_updated.apply(
                lambda x: getBaseResp(x['dF_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                          getBaseResp(x['dF_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
                axis=1
            )

    # Add dFF response within ROI
    # Raw data
    df_updated['dFF_ROI_raw'] = df_updated.apply(lambda x: x['dF_ROI_raw'] / x['f0_ROI_raw'], axis=1)
    df_updated['dFF_ROI_raw_peak'] = df_updated.apply(
        lambda x: getBaseResp(x['dFF_ROI_raw'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                  getBaseResp(x['dFF_ROI_raw'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
        axis=1
    )
    # Subtracted linear/logarithmic fit
    if subLogFit:
        df_updated['dFF_ROI_logFilt'] = df_updated.apply(lambda x: x['dF_ROI_logFilt'] / x['f0_ROI_logFilt'], axis=1)
        df_updated['dFF_ROI_logFilt_peak'] = df_updated.apply(
            lambda x: getBaseResp(x['dFF_ROI_logFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                      getBaseResp(x['dFF_ROI_logFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
            axis=1
        )
    else:
        df_updated['dFF_ROI_linFilt'] = df_updated.apply(lambda x: x['dF_ROI_linFilt'] / x['f0_ROI_linFilt'], axis=1)
        df_updated['dFF_ROI_linFilt_peak'] = df_updated.apply(
            lambda x: getBaseResp(x['dFF_ROI_linFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                      getBaseResp(x['dFF_ROI_linFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
            axis=1
        )
    if add_filtered_data:
        # Processed data (subtracted linear/logarithmic fit and added low-pass filter)
        if subLogFit:
            df_updated['dFF_ROI_logFilt_butterFilt'] = df_updated.apply(
                lambda x: x['dF_ROI_logFilt_butterFilt'] / x['f0_ROI_logFilt_butterFilt'], axis=1
            )
            df_updated['dFF_ROI_logFilt_butterFilt_peak'] = df_updated.apply(
                lambda x: getBaseResp(x['dFF_ROI_logFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                          getBaseResp(x['dFF_ROI_logFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
                axis=1
            )
        else:
            df_updated['dFF_ROI_linFilt_butterFilt'] = df_updated.apply(
                lambda x: x['dF_ROI_linFilt_butterFilt'] / x['f0_ROI_linFilt_butterFilt'], axis=1
            )
            df_updated['dFF_ROI_linFilt_butterFilt_peak'] = df_updated.apply(
                lambda x: getBaseResp(x['dFF_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[1] - 
                          getBaseResp(x['dFF_ROI_linFilt_butterFilt'], x['time'], t_base=x['baseWindow'], t_resp=x['respWindow'], **kwargs)[0], 
                axis=1
            )

    if test_negative:
        # Identify whether each trace is an outlier with negative response (`False`)
        # Test sign of peak dFF response only, as the result is the same for peak dF response
        # Time window for exclusive thresholds is suggested overlapping with response time window
        df_updated['valid'] = df_updated.apply(
            lambda x: is_valid_resp(
                qcam2img[x['qcam']], ROImask=x['ROImask'], subLogFit=subLogFit, 
                t_base=x['baseWindow'], t_resp=x['respWindow'], t_resp_excl=x['respWindow'], **kwargs
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


def get_oddball_pos(oddball_pos: list | np.ndarray | dict[str, np.ndarray], 
                    stim_ISI: float = 0.5, 
                    stim_count: int = 120, 
                    stimStart: float = 3.0) -> tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert deviant and standard tone positions in the stimulus sequence to their corresponding onset times.

    Args:
        oddball_pos: Positions of deviant tones in the stimulus sequence. Can be:
                     - list or np.ndarray: Single sequence of deviant positions.
                     - dict: Multiple sequences with string keys and position arrays as values.
        stim_ISI (float, optional): Time (in seconds) between consecutive tone onsets.
        stim_count (int, optional): Total number of tones in the entire stimulus sequence.
        stimStart (float, optional): Time (in seconds) when the first stimulus starts.

    Returns:
        If input is list/np.ndarray:
            A tuple containing two arrays:
                - deviant_times (np.ndarray): Deviant tone onset times (in seconds).
                - standard_times (np.ndarray): Standard tone onset times (in seconds).
        If input is dict:
            A dictionary with the same structure as input, with keys mapping to (deviant_times, standard_times) tuples.
    """

    if isinstance(oddball_pos, (list, np.ndarray)):
        # Sort deviant positions in ascending order
        oddball_pos = np.sort(np.array(oddball_pos))
        deviant_times = stimStart + (oddball_pos-1)*stim_ISI
        total_times = np.arange(stimStart, stimStart + stim_count*stim_ISI, stim_ISI)
        # Standard positions are the difference between sets of total positions and deviant positions
        standard_times = np.setdiff1d(total_times, deviant_times)
        onset_times = (deviant_times, standard_times)

    elif isinstance(oddball_pos, dict):
        onset_times = {}
        for key, pos in oddball_pos.items():
            pos = np.sort(np.array(pos))
            deviant_times = stimStart + (pos-1)*stim_ISI
            total_times = np.arange(stimStart, stimStart + stim_count*stim_ISI, stim_ISI)
            standard_times = np.setdiff1d(total_times, deviant_times)
            onset_times[key] = (deviant_times, standard_times)

    else:
        raise TypeError("'oddball_pos' must be a list, numpy array, or dictionary.")
        
    return onset_times


def calc_train_peakDFF(signal: np.ndarray, t: np.ndarray, tone_onsets: np.ndarray, 
                       t_base: float = -0.025, t_resp: tuple = (0.3, 0.975), 
                       initial_t_excl: float = 5.0, **kwargs) -> tuple[np.ndarray, float | np.ndarray]:
    """
    Helper function to align ΔF/F signal to deviant (or standard) tone onset times, 
    extract response windows of each tone, subtract baseline, average across all windows, 
    and calculate the peak amplitude of this averaged Δ(ΔF/F) trace segment.

    Args:
        signal (np.ndarray): 1D or 2D ΔF/F signal array of shape [frame] or [traceNumber, frame].
        t (np.ndarray): 1D time vector (in seconds).
        tone_onsets (np.ndarray): 1D array of tone onset times (in seconds).
        t_base (float, optional): For each individual response, time point (relative to corresponding tone onset time) 
                                  at which baseline is calculated.
                                  If 'None', baseline is calculated at the last time frame before tone onset (-0.025 sec by default).
        t_resp (tuple, optional): For each individual response, response window (relative to corresponding tone onset time) 
                                  within which peak response is calculated.
                                  For zinc sensor with 1-sec ISI, defaults to 0.3-1 sec and exclude the baseline of next tone, 
                                  so the whole window (from t_base to t_resp[1]) is 1 sec.
        initial_t_excl (float, optional): Any response starting before this time in the trace will be excluded to avoid novelty effects.
                                          Defaults to 5 sec to exclude responses occurring in the first 2 sec after first tone onset at 3 sec.
                                          If None, no response will be excluded.
        **kwargs: Optional arguments that will override default.

    Returns:
        window_traces (np.ndarray): 2D or 3D array of baseline-subtracted ΔF/F traces (i.e., Δ(ΔF/F)) of shape 
                                    [toneNumber, frame] or [traceNumber, toneNumber, frame], in which 
                                    'toneNumber' is the number of deviant (or standard) tone onsets in the entire train.
        resp_peak (float or np.ndarray): Peak amplitude of the average Δ(ΔF/F) trace within the response window. 
                                         Scalar if input signal is 1D, or 1D array of shape [traceNumber] if input signal is 2D.
    """

    # Check the shape of input signal array
    if signal.ndim not in (1, 2):
        raise ValueError("Signal array must be 1D or 2D.")
    
    # Round time vector to 1 ms precision (3 decimal places) to avoid floating point issues
    t = np.round(t, 3)
    
    # Initialize a list to store all deviant (or standard) response traces within the entire train
    window_trace_list = []
    
    # Skip first few seconds to avoid novelty effects
    tone_onsets_filtered = tone_onsets[tone_onsets >= initial_t_excl] if initial_t_excl else tone_onsets
    
    for i in tone_onsets_filtered:
        
        base_indices = np.argmin(np.abs(t - (i + t_base))) if t_base is not None else np.where(t < i)[0][-1]  # Select the last time frame before sound onset as baseline
        window_indices = np.where((t >= t[base_indices]) & (t < i + t_resp[1]))[0]  # Including t_base time frame of this window but not of the next window
        
        # Subtracting baseline amplitude from ΔF/F trace re sound onset ('ΔF/F subtracting baseline'), i.e., Y = Δ(ΔF/F) = ΔF/F trace - baseline ΔF/F
        # Ensures each trace segment to be averaged starts strictly at 0
        if signal.ndim == 1:
            # signal of shape [frame]
            window_trace = signal[window_indices] - signal[base_indices]
        else:
            # signal of shape [traceNumber, frame]
            window_trace = signal[:, window_indices] - signal[:, base_indices][:, np.newaxis]
        window_trace_list.append(window_trace)

    # Average trace segments across all windows
    if signal.ndim == 1:
        window_traces = np.array(window_trace_list)  # of shape [toneNumber, frame]
        avg_window_trace = np.nanmean(window_traces, axis=0)  # Ignore NaN values (negative traces excluded when calculating ΔF/F traces before)
    else:
        window_traces = np.stack(window_trace_list, axis=1)  # of shape [traceNumber, toneNumber, frame]
        avg_window_trace = np.nanmean(window_traces, axis=1)

    # Generate time vector for averaged trace segment
    t_window = getTimeVec(avg_window_trace.shape[-1], 
                          delayAdjust = t_base if t_base is not None else t[base_indices] - i, **kwargs)  # First time frame starts at -0.025 sec (baseline)

    # Calculate peak response from averaged Δ(ΔF/F) trace segment
    resp_peak = getBaseResp(avg_window_trace, t_window, t_resp=t_resp,  # No need to subtract baseline amplitude again as baseline is already 0
                            t_base=(t_base, t_base) if t_base is not None else (t[base_indices] - i, t[base_indices] - i), **kwargs)[1]

    return window_traces, resp_peak


def add_oddball_respData(df_resp_trace: pd.DataFrame, 
                         t_base: float = -0.025, 
                         t_resp: tuple = (0.3, 0.975), 
                         initial_t_excl: float = 5.0, 
                         around_oddball_t_excl: tuple = (-2.0, 2.0), 
                         avg_across_randProtocol: bool = False, 
                         **kwargs) -> pd.DataFrame:
    """
    Align the response windows of deviant (or standard) tones, then pick the peak response of the average trace within the window.

    Args:
        df_resp_trace (pd.DataFrame): Dataframe including average trace re treatment (and pulse).
                                      Including columns: 'treatment', 'pulse' (optional), 'time', 'individual_traces', 
                                                         'avg_trace', 'deviant_times', and 'standard_times'.
                                      If column 'pulse' is not included, the function will only group by treatment and tone type (deviant/standard).
        t_base (float, optional): For each individual response, time point (relative to corresponding tone onset time) 
                                  at which baseline is calculated.
                                  If 'None', baseline is calculated at the last time frame before tone onset (-0.025 sec by default).
        t_resp (tuple, optional): For each individual response, response window (relative to corresponding tone onset time) 
                                  within which peak response is calculated.
        initial_t_excl (float, optional): Any response starting before this time in the trace will be excluded.
        around_oddball_t_excl (tuple, optional): Any standard response whose onset time falls into this window 
                                                 relative to (before, after) deviant tone onset times will be excluded.
        avg_across_randProtocol (bool, optional): Whether to average traces of different protocols re deviant/standard frequency. 
                                                  Only applicable if multiple protocols are used.
        **kwargs: Optional arguments that will override default.
    
    Returns:
        df_resp_peak (pd.DataFrame): Dataframe including individual response traces and peak amplitude of average trace
                                     re treatment, pulse (optional), and tone type (deviant/standard).
                                     Including columns: 'treatment', 'pulse' (optional), 'stimulus', 'trace', and 'response'.
    """
    
    # Filter standard responses not near deviant ones
    df_resp_trace['standard_times_filtered'] = df_resp_trace.apply(
        lambda x: reduce(
            np.intersect1d, 
            [
                x['standard_times'][
                    (x['standard_times'] <= (deviant_time + around_oddball_t_excl[0])) | 
                    (x['standard_times'] >= (deviant_time + around_oddball_t_excl[1]))
                ]
                for deviant_time in x['deviant_times']
            ]
        ), 
        axis=1
    )
    
    # Add deviant and standard peak response amplitude (of the average trace) to the Dataframe
    df_resp_trace[['Deviant_trace', 'Deviant_peak']] = df_resp_trace.apply(
        lambda x: calc_train_peakDFF(x['avg_trace'], x['time'], x['deviant_times'], 
                                     t_base=t_base, t_resp=t_resp, initial_t_excl=initial_t_excl, **kwargs), 
        axis=1, result_type='expand'
    )
    df_resp_trace[['Standard_trace', 'Standard_peak']] = df_resp_trace.apply(
        lambda x: calc_train_peakDFF(x['avg_trace'], x['time'], x['standard_times_filtered'], 
                                     t_base=t_base, t_resp=t_resp, initial_t_excl=initial_t_excl, **kwargs), 
        axis=1, result_type='expand'
    )

    if avg_across_randProtocol:
        # Incorporate all deviant (or standard) traces from randomized protocols into trace array re treatment and frequency
        # Then pick the pick amplitude of the RE-averaged trace
        # Will RE-formatting trace array into shape [totalNumber, frame], in which totalNumber = toneNumber * protocolNumber
        
        # Extract protocol type (without pulse ID)
        df_resp_trace['protocol_type'] = df_resp_trace['pulse'].str.extract(r'(Deviant: [^\n]+)') + '\n' + \
                                         df_resp_trace['pulse'].str.extract(r'(Standard: [^\n]+)')

        # Group by treatment and protocol type, then concatenate trace arrays
        df_resp_trace = df_resp_trace.groupby(['treatment', 'protocol_type'], as_index=False, sort=False).agg({
            'Deviant_trace': lambda x: np.vstack(x.tolist()),  # Combine all traces vertically
            'Standard_trace': lambda x: np.vstack(x.tolist()), 
            'time': 'first'  # Keep other columns as needed
        }).assign(pulse=lambda x: x['protocol_type']).drop(columns=['protocol_type'])

        # Add deviant and standard peak response amplitude (of the RE-averaged trace) back
        df_resp_trace['Deviant_peak'] = df_resp_trace['Deviant_trace'].apply(
            lambda x: getBaseResp(
                np.nanmean(x, axis=0), 
                getTimeVec(np.nanmean(x, axis=0).shape[0], delayAdjust = t_base, **kwargs),  # To-do: apply to t_base = None
                t_base = (t_base, t_base), 
                t_resp = t_resp, 
                **kwargs
            )[1], 
        )
        
        df_resp_trace['Standard_peak'] = df_resp_trace['Standard_trace'].apply(
            lambda x: getBaseResp(
                np.nanmean(x, axis=0), 
                getTimeVec(np.nanmean(x, axis=0).shape[0], delayAdjust = t_base, **kwargs),  # To-do: apply to t_base = None
                t_base = (t_base, t_base), 
                t_resp = t_resp, 
                **kwargs
            )[1], 
        )

    # Reshape the Dataframe into long format
    df_window_trace = pd.melt(
        df_resp_trace,
        id_vars=['treatment', 'pulse'] if 'pulse' in df_resp_trace.columns else 'treatment',  # Include 'pulse' as id variable if it exists
        value_vars=['Deviant_trace', 'Standard_trace'],
        var_name='stimulus',
        value_name='trace'
    )
    
    df_peak = pd.melt(
        df_resp_trace, 
        id_vars=['treatment', 'pulse'] if 'pulse' in df_resp_trace.columns else 'treatment', 
        value_vars=['Deviant_peak', 'Standard_peak'],
        var_name='stimulus',
        value_name='response'
    )
    
    # Clean stimulus names and merge
    df_window_trace['stimulus'] = df_window_trace['stimulus'].str.replace('_trace', '')
    df_peak['stimulus'] = df_peak['stimulus'].str.replace('_peak', '')
    df_resp_peak = pd.merge(df_window_trace, df_peak, 
                            on=['treatment', 'pulse', 'stimulus'] if 'pulse' in df_resp_trace.columns else ['treatment', 'stimulus'])
    
    return df_resp_peak


def align2pRawTraces(dPaths: list, lowpassFreq: int = 2.0, alignMethod: str = None, add0dB: bool = False, 
                     **kwargs) -> tuple[list, np.ndarray, np.ndarray, list]:
    """
    Align raw fluorescence traces from multiple 2-photon imaging files based on their metadata and sound onset times.

    Args:
        dPaths (list): List of file paths to 2-photon raw whole traces. Usually end in '_rawFluoWholeTraces.mat'.
        lowpassFreq (int, optional): Cutoff frequency for low-pass Butterworth filter in Hz. 
                                     If 'None', no filtering is applied. 
                                     Defaults to 2 Hz.
        alignMethod (str, optional): Method to align traces of different lengths. Can be 'pad', 'crop', or 'None'. 
                                     Use 'pad' to pad shorter traces with NaN at the end, or 'crop' to crop longer traces from the end. 
                                     If 'None', no alignment is performed. Defaults to 'None'.
        add0dB (bool, optional): Whether to add 0 dB to the end of level order as negative control. Defaults to False.
        **kwargs: Optional arguments that will override default.
    
    Returns:
        levelOrder (list): List of sound levels in the same order as the sound onsets.
        soundOnsets (np.ndarray): 1D array of sound onset times (in seconds).
        timeVector (np.ndarray): 1D array of time points corresponding to the frames of raw fluorescence traces (in seconds).
        rawWholeTraces_list (list): List of 3D raw fluorescence trace arrays from each file (animal), aligned according to the specified method. 
                                    Each array has shape [traceNumber, neuronNumber, frame].
    """

    # Extract metadata and rawFluoWholeTraces from each animal
    levelOrders_temp, timeVectors_temp, soundOnsets_temp, rawWholeTraces_list = [], [], [], []
    for path in dPaths:
        levelOrder, timeVector, soundOnsets, rawFluoWholeTraces = metadataProcess.getRawFluoWholeTraces(path)
        levelOrders_temp.append(levelOrder)
        timeVectors_temp.append(timeVector)
        soundOnsets_temp.append(soundOnsets)
        rawWholeTraces_list.append(rawFluoWholeTraces)

    # Check whether all levelOrders are identical
    for i in range(1, len(levelOrders_temp)):
        if levelOrders_temp[i] != levelOrders_temp[0]:
            raise ValueError(f"Mismatch detected in 'levelOrder' between file 0 and file {i}.")
    levelOrder = levelOrders_temp[0] + [0] if add0dB else levelOrders_temp[0]  # Add 0 dB to the end of level order

    # Check whether all soundOnsets are identical and align rawFluoWholeTraces on the first sound onset if not
    if len({a.shape for a in soundOnsets_temp}) != 1:
        raise ValueError("'soundOnsets' shapes are different across files. Cannot align on the first sound onset.")
    soundOnsets_arr = np.stack(soundOnsets_temp)  # of shape [n_files, n_onsets]
    if not np.allclose(soundOnsets_arr, soundOnsets_arr[0], atol=1e-8, rtol=0):  # broadcasting across rows
        warnings.warn("Mismatch detected in 'soundOnsets' across files.\n" \
                      "Cropping redundant 'rawFluoWholeTraces' from the beginning to align on the first sound onset.")
        soundOnsets_first = soundOnsets_arr[:, 0]
        soundOnsets_first_diff = soundOnsets_first - np.min(soundOnsets_first)
        for i in range(len(rawWholeTraces_list)):
            if soundOnsets_first_diff[i] > 1e-8:
                # Crop traces with more frames before the first sound onset
                cropIDX = np.argmin(np.abs(timeVectors_temp[i] - soundOnsets_first_diff[i])) + 1
                rawWholeTraces_list[i] = rawWholeTraces_list[i][:, :, cropIDX:]
                timeVectors_temp[i] = timeVectors_temp[i][:-cropIDX]  # timeVectors are cropped from the end to have the same starting point
                soundOnsets_temp[i] = soundOnsets_temp[i] - soundOnsets_first_diff[i]
        soundOnsets = soundOnsets_temp[np.argmin(soundOnsets_first)]
    else:
        soundOnsets = soundOnsets_temp[0]
    
    # Apply low-pass filter before padding to avoid being transformed to NaNs after filtering
    sample_freq = 1 / np.mean(np.diff(timeVectors_temp[0]))  # calculate sampling frequency using the first file (around 5.008 Hz)
    if lowpassFreq is not None:
        filtered_traces_temp = []
        for trace in rawWholeTraces_list:
            filtered_trace = butterFilter(trace, sample_freq = sample_freq, cutoff_freq = lowpassFreq, **kwargs)
            filtered_traces_temp.append(filtered_trace)
        rawWholeTraces_list = filtered_traces_temp

    timeVector_lengths = [len(tv) for tv in timeVectors_temp]
    if alignMethod == 'pad':
        # Pad shorter rawFluoWholeTraces with NaN at the end
        if len(set(timeVector_lengths)) > 1:
            warnings.warn("Different timeVector lengths detected.\n" \
                          "Padding shorter 'rawFluoWholeTraces' with NaN at the end to match longest length.")
            padded_traces = []
            for trace in rawWholeTraces_list:
                pad_len = max(timeVector_lengths) - trace.shape[2]
                if pad_len > 0:
                    # Pad trace with NaN along time dimension (axis=2)
                    pad_shape = (trace.shape[0], trace.shape[1], pad_len)
                    pad_array = np.full(pad_shape, np.nan)
                    trace_padded = np.concatenate((trace, pad_array), axis=2)
                else:
                    trace_padded = trace
                padded_traces.append(trace_padded)
            rawWholeTraces_list = padded_traces
        timeVector = timeVectors_temp[np.argmax(timeVector_lengths)]
    
    elif alignMethod == 'crop':
        # Crop any longer rawFluoWholeTraces from the end
        if len(set(timeVector_lengths)) > 1:
            warnings.warn("Different timeVector lengths detected.\n" \
                          "Cropping 'rawFluoWholeTraces' from the end to shortest length.")
            rawWholeTraces_list = [trace[:, :, :min(timeVector_lengths)] for trace in rawWholeTraces_list]
        timeVector = timeVectors_temp[np.argmin(timeVector_lengths)]

    else:
        # No alignment is performed
        if len(set(timeVector_lengths)) > 1:
            warnings.warn("Different timeVector lengths detected across files. Use the first file's timeVector by default.\n" \
                          "Please specify 'alignMethod' as 'pad' or 'crop' to align traces.")
        # USse the time vector from the first file by default
        timeVector = timeVectors_temp[0]
    
    if add0dB:
        # Add 0 dB sound onset at the end with equal interval (as negative control)
        soundOnset_0dB = timeVector[np.argmin(np.abs(timeVector - (soundOnsets[-1] + np.mean(np.diff(soundOnsets)))))]
        if soundOnset_0dB > timeVector[-1]:
            warnings.warn("Calculated 0 dB sound onset time exceeds the last time point in 'timeVector'.")
        soundOnsets = np.append(soundOnsets, soundOnset_0dB)
    
    # Check whether levelOrder and soundOnsets have the same length
    if len(levelOrder) != len(soundOnsets):
        raise ValueError("LevelOrder and soundOnsets must have the same length.")

    return levelOrder, soundOnsets, timeVector, rawWholeTraces_list


def compute_2pDFFtraces(rawWholeTraces: list[np.ndarray], t: np.ndarray, onset: float, 
                        t_base: tuple[float, float], t_resp: tuple[float, float], **kwargs) -> list[np.ndarray]:
    """
    Helper function to compute dF/F traces based on a given sound onset time in 2-photon.

    Args:
        rawWholeTraces (list): List of 3D raw fluorescence arrays of shape [traceNumber, neuronNumber, frame].
                               Each array corresponds to a different dataset (e.g., animal).
        t (np.ndarray): 1D array of time vector (in seconds).
        onset (float): Sound onset time (in seconds).
        t_base (tuple): Baseline window relative to sound onset (in seconds).
        t_resp (tuple): Response window relative to sound onset (in seconds).
        **kwargs: Optional arguments that will override default.

    Returns:
        dFFwholeTraces (list): List of 3D dF/F fluorescence arrays of shape [traceNumber, neuronNumber, frame].
    """

    dFFwholeTraces = []

    for rawFluo in rawWholeTraces:
        # Calculate dF/F based on the given sound onset time
        dFF, _, _ = dFFcalc(
            rawFluo, t=t,
            t_base=(onset + t_base[0], onset + t_base[1]),
            t_resp=(onset + t_resp[0], onset + t_resp[1]),
            **kwargs
        )

        dFFwholeTraces.append(dFF)

    return dFFwholeTraces


def process_2pRawTraces(rawWholeTraces: list[np.ndarray], 
                        t: np.ndarray, 
                        soundOnsets: np.ndarray, 
                        t_base: tuple[float, float] = (-1, 0), 
                        t_resp_phasic: tuple[float, float] = (0.3, 1.3), 
                        t_resp_tonic: tuple[float, float] = (2, 4), 
                        **kwargs) -> list[np.ndarray]:
    """
    Process raw fluorescence traces in 2-photon by detecting negative responses based on dF/F
    and setting corresponding raw trace segments to NaNs.
    
    Negative responses are defined as either:
        Average of 3 consecutive time points below 3 SDs of baseline, or
        maximum response below baseline average.

    Args:
        rawWholeTraces (list): List of 3D raw fluorescence arrays of shape [traceNumber, neuronNumber, frame].
        t (np.ndarray): 1D array of time vector (in seconds).
        soundOnsets (np.ndarray): 1D array of sound onset times (in seconds).
        t_base (tuple): Baseline window relative to sound onset (in seconds).
        t_resp_phasic (tuple): Phasic response window relative to sound onset (in seconds).
        t_resp_tonic (tuple): Tonic response window relative to sound onset (in seconds).
        **kwargs: Optional arguments that will override default.

    Returns:
        rawWholeTraces_processed (list): List of processed 3D raw fluorescence arrays with negative windows set to NaNs.
    """

    # Create a copy of the raw traces to store processed values
    rawWholeTraces_processed = [arr.copy() for arr in rawWholeTraces]

    # Compute dF/F re sound onset with shifted baseline window based on sound onset time
    dFFwholeTraces_all = [
        compute_2pDFFtraces(rawWholeTraces, t, onset, t_base, t_resp_phasic, **kwargs)
        for onset in soundOnsets
    ]
    
    for i in range(len(rawWholeTraces)):  # re dataset (animal)
        for j in range(len(soundOnsets)):
            # Average across all neurons
            dFFwholeTraces = np.nanmean(dFFwholeTraces_all[j][i], axis=1)  # of shape [traceNumber, frame]
            
            # Get time indices for baseline and response windows
            base_indices = np.where((t >= soundOnsets[j] + t_base[0]) & (t <= soundOnsets[j] + t_base[1]))[0]
            resp_indices = np.where((t >= soundOnsets[j] + t_resp_phasic[0]) & (t <= soundOnsets[j] + t_resp_tonic[1]))[0]  # Until start of the next baseline window
            window_indices = np.where((t >= soundOnsets[j] + t_base[0]) & (t <= soundOnsets[j] + t_resp_tonic[1]))[0]  # Baseline + response window
            
            if j < len(soundOnsets) - 1:
                # If not the last sound onset, also get indices for the next baseline and response windows
                next_base_indices = np.where((t >= soundOnsets[j+1] + t_base[0]) & (t <= soundOnsets[j+1] + t_base[1]))[0]
                next_window_indices = np.where((t >= soundOnsets[j+1] + t_base[0]) & (t <= soundOnsets[j+1] + t_resp_tonic[1]))[0]

            if j == 0:
                mean_ISI = np.mean(np.diff(soundOnsets))
                # If the first sound onset, estimate baseline indices from a virtual preceding onset
                if soundOnsets[0] - mean_ISI + t_base[0] < 0:
                    warnings.warn(
                        "Estimated pre-onset baseline window falls outside 'timeVector'. "
                        "Using the first baseline-length segment of the trace instead."
                    )
                    last_base_indices = np.where(t <= t[0] + (t_base[1]-t_base[0]))[0]
                else:
                    soundOnset_last = t[np.argmin(np.abs(t - (soundOnsets[0] - mean_ISI)))]
                    last_base_indices = np.where((t >= soundOnset_last + t_base[0]) & (t <= soundOnset_last + t_base[1]))[0]

            # Exclude windows with negative responses caused by motion artifacts
            for k in range(dFFwholeTraces.shape[0]):
                # re trial
                maxResp = np.nanmax(dFFwholeTraces[k, resp_indices])
                meanBase = np.nanmean(dFFwholeTraces[k, base_indices])
                baseSD = np.nanstd(dFFwholeTraces[k, base_indices], ddof=1)  # Use sample standard deviation (ddof=1)
                if j == 0:
                    meanBase_last = np.nanmean(dFFwholeTraces[k, last_base_indices])
                    baseSD_last = np.nanstd(dFFwholeTraces[k, last_base_indices], ddof=1)

                # avgResp = np.nanmean(dFFwholeTraces[k, resp_indices])
                # if (avgResp < meanBase - 3*baseSD) or (maxResp < meanBase):
                #     rawWholeTraces_list_processed[i][k, :, window_indices] = np.nan
                
                # If the average of three consecutive time points is 3 SDs below baseline, or
                # if maxResp < meanBase, peak dFF response is negative -> makes no sense
                cons_avg = np.correlate(dFFwholeTraces[k, resp_indices], np.ones(3)/3, mode='valid')
                if (np.any(cons_avg < meanBase - 3*baseSD)) or (maxResp < meanBase):
                    # Set traces within negative windows to NaNs
                    rawWholeTraces_processed[i][k, :, window_indices] = np.nan
                
                if j < len(soundOnsets) - 1:
                    # If trace within baseline window is negative, calculated peak dF/F response is biased
                    # Negative trace within baseline window cannot be captured unless calculating dF/F based on the last baseline window
                    next_cons_avg = np.correlate(dFFwholeTraces[k, next_base_indices], np.ones(3)/3, mode='valid')
                    if np.any(next_cons_avg < meanBase - 3*baseSD):
                        # Set traces within the next window to NaNs
                        rawWholeTraces_processed[i][k, :, next_window_indices] = np.nan

                if j == 0:
                    # If the first sound onset, also check whether trace within this baseline window is negative
                    # Calculate dF/F based on the baseline window of a virtual preceding onset
                    base_cons_avg = np.correlate(dFFwholeTraces[k, base_indices], np.ones(3)/3, mode='valid')
                    if np.any(base_cons_avg < meanBase_last - 3*baseSD_last):
                        # Set traces within the first window to NaNs
                        rawWholeTraces_processed[i][k, :, window_indices] = np.nan

    return rawWholeTraces_processed
