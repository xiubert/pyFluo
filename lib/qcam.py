import numpy as np
import re
import matplotlib.pyplot as plt
from lib.signals import butterFilter

def extract_qcamraw(filepath: str, fr: int = 20) -> tuple[np.ndarray,np.ndarray,dict]:
    """
    Extracts image data, header, and associated time vector from qcamraw file.

    Args:
        filepath (str): path to qcamraw file
        fr (int): frame rate of image acquisition
    
    Returns:
        img (numpy array): as Y x X x time
        t (numpy array): time vector (in s) starting at 0
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
    
    # Generate time vector
    # first frame acquired (1/fr) s after start
    # t = np.arange(1, n_frames + 1) * (1 / fr)
    # first frame acquired at start (starts at 0)
    t = (np.arange(1, n_frames + 1) * (1 / fr))-(1/fr)
    
    return img,t,header


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

def calcSpatialDFFresp(img, t: np.ndarray, frameRate: int = 20,
               baseline: tuple[int] = (2,3),
               stimlen: float = 0.4,
               temporalAvgFrameSpan: int = 10,
               applyButterFilter: bool = True):
    # spatial baseline
    # baseline = (2,3)
    # stimlen = 0.4 #s
    # # average of 10 frames of spatial dff (re baseline) after stimulus ends
    # temporalAvgFrameSpan = 10

    # Reshape to 2D: (number of pixels, time points)
    reshaped_data = img.reshape(-1,200)
    baselineIDX = np.where((t>=baseline[0]) & (t<=baseline[1]))[0]
    spatialbase = reshaped_data[:,baselineIDX].mean(axis=1).reshape(-1,1)
    spatialDFF = (reshaped_data-spatialbase)/spatialbase
    if applyButterFilter:
        spatialDFF = butterFilter(spatialDFF)

    # _, ax = plt.subplots()

    # plt.imshow(spatialDFF[:,np.where((t>=baseline[1]+stimlen) &
    #                                     (t<=baseline[1]+stimlen+temporalAvgFrameSpan*(1/frameRate)))[0]]\
    #                                         .mean(axis=1).reshape(*img.shape[:2]))
    # plt.show()
    spatialDFFresp = spatialDFF[:,np.where((t>=baseline[1]+stimlen) &
                                        (t<=baseline[1]+stimlen+temporalAvgFrameSpan*(1/frameRate)))[0]]\
                                            .mean(axis=1).reshape(*img.shape[:2])

    return spatialDFFresp