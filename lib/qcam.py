import numpy as np
import re
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from lib.signals import butterFilter

def getTimeVec(nFrames: int, frameRate: int = 20, zeroStart: bool = True):
    """
    Generate time vector from frame count and rate.

    Args:
        nFrames (int): number of frames
        frameRate (int): number of frames acquired per second
        zeroStart (bool): whether first frame acquired at time 0.
    Returns:
        t (numpy array): vector of time values
    """
    # first frame acquired (1/fr) s after start
    t = (np.arange(1, nFrames + 1) * (1 / frameRate))
    # first frame acquired at start (starts at 0)
    if zeroStart:
        return t-(1/frameRate)
    return t


def extract_qcamraw(filepath: str) -> tuple[np.ndarray,dict]:
    """
    Extracts image data, header, and associated time vector from qcamraw file.

    Args:
        filepath (str): path to qcamraw file
        fr (int): frame rate of image acquisition
    
    Returns:
        img (numpy array): as Y x X x time
        header (dict): file header metadata
    """
    # Open the file for reading in binary mode
    with open(filepath, 'rb') as fid:
        # Read lines until an empty line is encountered to get the header
        headcount = 0
        header_lines = []
        while True:
            line = fid.readline().decode('utf-8').strip()
            if not line:
                break
            headcount += 1
            header_lines.append(line)
        
        # Parse the header into a dictionary
        header = {}
        for line in header_lines:
            colon_loc = line.find(':')
            if colon_loc == -1:
                continue
            field_name = line[:colon_loc].strip().replace('-', '_')
            field_value = line[colon_loc + 1:].strip()
            # Remove units if present in brackets
            match = re.search(r'\[.*\]', field_value)
            if match:
                field_value = field_value[:match.start()].strip()
            header[field_name] = field_value
        
        # Get the header size
        header_size = int(header['Fixed_Header_Size'])
        
        # Seek to the end of the file to determine total number of bytes
        fid.seek(0, 2)  # Move to the end of the file
        num_bytes = fid.tell()
        fid.seek(0)  # Reset pointer to the start
        
        # Read image data
        imgvec = np.fromfile(fid, dtype=np.uint16, offset=header_size)
    
    # Parse ROI to get image dimensions
    totROI = list(map(int, header['ROI'].split(',')))
    img_width = totROI[2]
    img_height = totROI[3]
    
    # # Calculate the number of frames
    n_frames = len(imgvec) // img_width // img_height
    expected_frames = (num_bytes - header_size) / int(header['Frame_Size'])
    
    if n_frames != expected_frames:
        print("Something went wrong w/r/t file size and pixel depth")
        return None, None
    
    # Reshape the image data into a 3D array
    img = imgvec.reshape((n_frames, img_height, img_width))
    img = np.transpose(img, (1, 2, 0))  # Reorder to [height, width, frames]
    
    return img,header


def getQCamHeader(filepath: str) -> dict:
    """
    Get qcam file header as dict.
    """
    with open(filepath, 'rb') as fid:
            # Read lines until an empty line is encountered to get the header
            headcount = 0
            header_lines = []
            while True:
                line = fid.readline().decode('utf-8').strip()
                if not line:
                    break
                headcount += 1
                header_lines.append(line)

    # Parse the header into a dictionary
    header = {}
    for line in header_lines:
        colon_loc = line.find(':')
        if colon_loc == -1:
            continue
        field_name = line[:colon_loc].strip().replace('-', '_')
        field_value = line[colon_loc + 1:].strip()
        # Remove units if present in brackets
        match = re.search(r'\[.*\]', field_value)
        if match:
            field_value = field_value[:match.start()].strip()
        header[field_name] = field_value
        
    return header

def calcSpatialDFFresp(img: np.ndarray, t: np.ndarray, 
                        frameRate: int = 20,
                        t_baseline: tuple[int] = (2,3),
                        stimlen: float = 0.4,
                        t_temporalAvg: tuple[float] = None,
                        temporalAvgFrameSpan: int = 10,
                        butterFilterParams: dict = {}):

    # Reshape to 2D: (number of pixels, time points)
    reshaped_data = img.reshape(-1,200)
    baselineIDX = np.where((t>=t_baseline[0]) & (t<=t_baseline[1]))[0]
    spatialbase = reshaped_data[:,baselineIDX].mean(axis=1).reshape(-1,1)
    spatialDFF = (reshaped_data-spatialbase)/spatialbase
    
    if butterFilterParams and isinstance(butterFilterParams,dict):
        spatialDFF = butterFilter(spatialDFF,**butterFilterParams)

    if t_temporalAvg is None:
        t_temporalAvg = (t_baseline[1]+stimlen,t_baseline[1]+stimlen+temporalAvgFrameSpan*(1/frameRate))

    spatialDFFresp = spatialDFF[:,np.where((t>=t_temporalAvg[0]) &
                                        (t<=t_temporalAvg[1]))[0]]\
                                            .mean(axis=1).reshape(*img.shape[:2])

    return spatialDFFresp


def plotAvgImg(img):
    ax = plt.imshow(img.mean(axis=2))

    return ax


def plotTraceAvgImg(t,img,cutoff_freq: float = 3):
    signal = np.reshape(img,(np.prod(img.shape[:2]),img.shape[2])).mean(axis=0)
    X = np.vstack([t, np.ones(len(t))]).T
    slope,intercept = np.linalg.lstsq(X,signal, rcond=None)[0]
    fig,ax = plt.subplots(3,1)
    ax[0].plot(t,signal)
    ax[0].set_title('raw trace with least-sq reg. fit')
    ax[0].plot(t,t*slope+intercept,'r')
    ax[1].plot(t,signal-(t*slope+intercept))
    ax[1].set_title('trace minus fit')
    ax[2].plot(t,butterFilter(signal-(t*slope+intercept),cutoff_freq=cutoff_freq))
    ax[2].set_title('filtered trace minus fit')

    return fig,ax


def getAllImgs(qFiles: list):
    imgs,headers = [],[]
    for q in qFiles:
        img,header = extract_qcamraw(q)
        imgs.append(img)
        headers.append(header)

    # exclude files where framecount is not most common
    nFrames = [i.shape[2] for i in imgs]
    nFrames = max(nFrames, key=nFrames.count)
    imgs,headers = zip(*[(i,h) for i,h in zip(imgs,headers) if i.shape[2]==nFrames])

    return imgs,headers


def experimentAvgPlot(dPath: str = None, qFiles: list = None):
    if qFiles is None:
        qFiles = glob.glob(os.path.join(dPath,'*.qcamraw'))

    imgs,headers = getAllImgs(qFiles)
    t = getTimeVec(imgs[0].shape[2],zeroStart=False)
    timeStamps = [h['File_Init_Timestamp'] for h in headers]
    timeStamps = [datetime.strptime(date, '%m-%d-%Y_%H:%M:%S') for date in timeStamps]

    fig,ax = plt.subplots(3,1,figsize=(12,10))
    ax[0].plot(t,butterFilter(np.array(imgs).mean(axis=(0,1,2))))
    ax[0].set_ylabel('raw F')
    ax[0].set_xlabel('t (s)')
    ax[0].set_xticks(np.arange(0,int(max(t))+1))

    ax[1].imshow(calcSpatialDFFresp(np.array(imgs).mean(axis=0).reshape(*imgs[0].shape),
                               t,stimlen=0.1, temporalAvgFrameSpan=8))

    ax[2].plot(timeStamps,np.array(imgs).mean(axis=(1,2,3)),'.')
    ax[2].set_ylabel('raw F')
    # Format the x-axis to show readable datetime labels
    # ax[1].gcf().autofmt_xdate()
    if dPath is None:
        fig.suptitle(os.path.dirname(qFiles[0]))
    else:
        fig.suptitle(dPath)

    fig.show()