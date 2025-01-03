from scipy.signal import butter, filtfilt
import numpy as np

def butterFilter(signal: np.ndarray, 
                 sample_freq: int = 20, 
                 cutoff_freq: int = 5, 
                 order: int = 4) -> np.ndarray:
    """
    Simple helper function for a lowpass butterworth filter.

    Args:
        signal (numpy array): signal array to be filtered
        sampleFreq (int): sampling frequency of the signal
        cutoff_freq: filter cutoff frequency
        order: filter order ('steepness' of signal drop-off at cutoff_freq)

    Returns:
        filtered_signal (numpy array): lowpass filtered signal

    """
    b, a = butter(order, cutoff_freq/(sample_freq/2), 'lowpass') 

    return filtfilt(b,a,signal)


def subtractLinFit(t,signal: np.ndarray) -> np.ndarray:
    """
    Subtracts linear fit of signal from signal. 
    Useful to remove consistent signal drift in one direction.

    Args:
        t (list or array): time vector (in seconds)
        signal (numpy array): signal array
    Returns:
        filtered_signal (numpy array): signal array after removal of linear fit
    """
    
    X = np.vstack([t, np.ones(len(t))]).T
    slope,intercept = np.linalg.lstsq(X,signal, rcond=None)[0]

    return signal-(t*slope+intercept),slope,intercept


def getBaseResp(signal: np.ndarray, t: np.ndarray, 
                t_base: tuple[float,float] = (2.2,2.9),
                t_resp: tuple[float,float] = (3.0,3.15)) -> tuple[float,float]:
        """
        Extract average signal at t_base and max signal between t_resp.

        Args:
            signal (numpy array): signal array
            t (list or array): time vector (in seconds)
            t_base: time window (in seconds) for baseline
            t_resp: time window (in seconds) for response

        Returns:
            base (float): average signal between t_base
            resp (float): max signal between t_resp
        """

        base = signal[np.where((t>=t_base[0]) & (t<=t_base[1]))].mean()
        resp = np.max(signal[np.where((t>=t_resp[0]) & (t<=t_resp[1]))])
        
        return base,resp


def pkDFF(img,t,
          subLinFit: bool = True, 
          butterFilt: bool = True, 
          **kwargs):

    signal = img.mean(axis=(0,1))
    
    if subLinFit:
        signal = subtractLinFit(t,signal)[0]

    if butterFilt:
        if 'cutoff_freq' not in kwargs:
            cutoff_freq = 4
        else:
            cutoff_freq = kwargs['cutoff_freq']
        signal = butterFilter(signal, cutoff_freq = cutoff_freq)
    
    f0 = getBaseResp(signal,t)[0]
    
    dFF = (signal-f0)/f0
    pkBase,pkResp = getBaseResp(dFF,t)

    # gets response in either direction
    pk = abs(pkResp-pkBase)

    return pk