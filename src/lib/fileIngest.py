import numpy as np
import re

"""
Functions for importing raw data from imaging data files.
"""

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

    Args:
        filepath (str): path to qcamraw file
    
    Returns:
        header (dict): file header metadata
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


def qcams2imgs(qFiles: list):
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