from scipy.signal import butter, filtfilt

def butterFilter(signal, sampleFreq: int = 20, 
                 cutoff_freq: int = 5, order: int = 4):
    # filtered
    # Filter parameters
    # order = 4  # Filter order
    # cutoff_freq = 5  # Cutoff frequency in Hz
    # fs = 20  # Sampling frequency in Hz

    # Design the Butterworth filter
    b, a = butter(order, cutoff_freq/(sampleFreq/2), 'lowpass') 

    return filtfilt(b,a,signal)