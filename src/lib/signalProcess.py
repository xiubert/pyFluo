from scipy.signal import butter, filtfilt
import numpy as np
from typing import Tuple


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

    # first frame acquired (1/fr) s after start
    t = (np.arange(1, nFrames + 1) * (1 / frameRate)) + delayAdjust
    # first frame acquired at start (starts at 0)
    if zeroStart:
        return t-(1/frameRate)
    return t


def butterFilter(signal: np.ndarray, 
                 sample_freq: int = 20, 
                 cutoff_freq: int = 5, 
                 order: int = 4,
                 **kwargs) -> np.ndarray:
    """
    Simple helper function for a lowpass butterworth filter.

    Args:
        signal (numpy array): signal array to be filtered
        sampleFreq (int): sampling frequency of the signal
        cutoff_freq: filter cutoff frequency
        order: filter order ('steepness' of signal drop-off at cutoff_freq)
        **kwargs: Optional arguments that will override default.

    Returns:
        filtered_signal (numpy array): lowpass filtered signal

    """
    # Optionally override parameters using kwargs
    sample_freq = kwargs.get('sample_freq', sample_freq)
    cutoff_freq = kwargs.get('cutoff_freq', cutoff_freq)

    b, a = butter(order, cutoff_freq/(sample_freq/2), 'lowpass') 

    return filtfilt(b,a,signal)


def subtractLinFit(t, signal: np.ndarray, offset: bool = True, **kwargs) -> np.ndarray:
    """
    Subtracts linear fit of signal from signal. 
    Useful to remove consistent signal drift in one direction.

    Args:
        t (list or array): time vector (in seconds).
        signal (numpy array): signal array.
        offset (bool, optional): whether to add baseline fluorescence (f0) back to the corrected signal as the offset.
                                Defaults to 'True'.
    
    Returns:
        filtered_signal (numpy array): signal array after removal of linear fit
    """
    # Optionally override parameters using kwargs
    offset = kwargs.get('offset',offset)
    
    X = np.vstack([t, np.ones(len(t))]).T
    slope,intercept = np.linalg.lstsq(X,signal, rcond=None)[0]

    if offset:
        # baseline fluorescence is added to bring corrected y-values back to approximately the same level of uncorrected ones
        # required if dFF is calculated using the corrected signal after linear fit
        f0 = getBaseResp(signal, t, **kwargs)[0]
        return signal-(t*slope+intercept)+f0, slope, intercept
    else:
        # output (signal - linear fit) directly
        # used to display how linear fit works
        return signal-(t*slope+intercept), slope, intercept


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
     

def pkDFFimg(imgSeries: np.ndarray,
                subLinFit: bool = True, 
                butterFilt: bool = False, 
                dFResp: bool = False, 
                **kwargs):
    """
    Calculates peak dFF response from image series array.
    
    Args:
        imgSeries (array): array of shape (Y, X, frame)
        subLinFit (bool): whether to subtract fitted line
        butterFilt (bool): whether to apply low pass filter
        dFResp (bool): if true, calculate dF response rather than dFF
        **kwargs: Optional arguments that will override default.
            example: ROImask (np.ndarray): 2D binary mask array specifying the region of interest
                     negResp (bool): whether to extract peak dFF response in either direction (original signal preserved) as 'pkResp'

    Returns:
        pk (float): peak dFF or dF response
    """
    
    t = getTimeVec(imgSeries.shape[-1],**kwargs)

    ROImask = kwargs.get('ROImask',np.ones(imgSeries.shape[:2]))
    signal = imgSeries[ROImask==1,:].mean(axis=0)
    
    # whether to subtract fitted line
    if subLinFit:
        signal = subtractLinFit(t, signal, **kwargs)[0]

    # whether to apply low pass filter
    if butterFilt:
        # cutoff_freq = kwargs.get('cutoff_freq', 4)  # Default cutoff_freq = 4
        # signal = butterFilter(signal, cutoff_freq=cutoff_freq)
        signal = butterFilter(signal, **kwargs)
    
    # baseline (f0) to be subtracted
    f0 = getBaseResp(signal, t, **kwargs)[0]
    
    # calculate dFF or dF response
    if dFResp:
        resp = signal-f0
    else:
        resp = (signal-f0)/f0

    # get baseline and peak from dFF or dF
    pkBase, pkResp = getBaseResp(resp, t, **kwargs)

    # calculate peak dFF reponse
    pk = pkResp-pkBase

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