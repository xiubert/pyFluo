import numpy as np
import re
import pandas as pd
from glob import glob
import os

import lib.metadataProcess as metadataProcess


"""
Functions for importing raw data from imaging data files.
"""

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


def qcams2imgs(qFiles: list, consistentFrameCt: bool = True) -> tuple[list[np.ndarray],list[dict]]:
    """
    Helper to extract image data from list of qcamraw files. 
    Limits output to captures with most consistent frame count among files.

    Args:
        qFiles (list): list of qcamraw file paths to be imported.
        consistentFrameCt (bool): whether to limit captures to files with most consistent frame count.

    Returns:
        imgs (list of numpy arrays): list of image arrays
        headers (list of dicts): list of associated file headers
    """
    imgs,headers = [],[]
    for q in qFiles:
        img,header = extract_qcamraw(q)
        imgs.append(img)
        headers.append(header)

    if consistentFrameCt:
        # exclude files where framecount is not most common
        nFrames = [i.shape[2] for i in imgs]
        nFrames = max(nFrames, key=nFrames.count)
        imgs,headers = zip(*[(i,h) for i,h in zip(imgs,headers) if i.shape[2]==nFrames])

    return imgs,headers


def qcamPath2table(exprmntPaths: list[str], format: str = 'MAK') -> pd.DataFrame:
    """
    Generates a DataFrame mapping qcam files to corresponding XSG files and pulse metadata.

    Args:
        exprmntPaths (list[str]): List of experiment directory paths to search for `.qcamraw` files.
        format (str, optional): Format for extracting dB values from pulse metadata. 
                                - 'MAK': Matches patterns like "_XXdB_YYYmsTotal_".
                                - 'PAC': Matches patterns like "Hz_XXdB_TestTone_YYYmsPulse_".
                                Defaults to 'MAK'.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'qcam': Full paths to `.qcamraw` files.
            - 'dir': Corresponding experiment directory for each qcam file.
            - 'xsg': Full paths to `.xsg` files, assuming a 1:1 mapping with `.qcamraw` files.
                     If no corresponding `.xsg` file exists, the entry is dropped.
            - 'pulse': The first pulse name extracted from the XSG file's metadata.
            - 'dB': Decibel (dB) value extracted from the pulse metadata using the specified format.

    Notes:
        - This function assumes each `.qcamraw` file has a corresponding `.xsg` file in the same directory.
        - The pulse metadata extraction relies on the first pulse name in the XSG file.
        - Requires `metadataProcess.getPulseNames` and `metadataProcess.getPulseDB` to extract metadata.

    Examples:
        >>> exprmntPaths = ['exp1', 'exp2']
        >>> df = qcamPath2table(exprmntPaths, format='MAK')
        >>> print(df)
                     qcam   dir                         xsg           pulse   dB
        0  exp1/file1.qcamraw  exp1  exp1/file1.xsg  PulseName1   75
        1  exp2/file2.qcamraw  exp2  exp2/file2.xsg  PulseName2   80
    """
    qcams = []
    dirs = []

    for p in exprmntPaths:
        qpaths = glob(os.path.join(p,'*.qcamraw'))
        qcams.extend(qpaths)
        dirs.extend([p]*len(qpaths))

    df = pd.DataFrame(zip(qcams,dirs),columns=['qcam','dir'])

    # assume 1:1 mapping of qcamraw to XSG
    df['xsg'] = df['qcam'].apply(lambda x: x.replace('.qcamraw','.xsg') if os.path.exists(x.replace('.qcamraw','.xsg')) else None)
    df = df.dropna()

    # assume relevant pulse is first
    df['pulse'] = df['xsg'].apply(lambda x: metadataProcess.getPulseNames(x)[0])
    df['dB'] = df['pulse'].apply(lambda x: metadataProcess.getPulseDB(x,format=format))

    return df


def loadQCamTable(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Processes a DataFrame of qcam metadata and returns the updated DataFrame along with image and header data.

    Args:
        df (pd.DataFrame): A DataFrame containing qcam metadata, including a column named 'qcam' with paths to `.qcamraw` files.

    Returns:
        tuple:
            - pd.DataFrame: Updated DataFrame with additional columns:
                - 'nFrames': Number of frames in the qcam image data.
                - 'timestamp_init': Initialization timestamp of the qcam file (converted to datetime).
                - 'dim_YX': Tuple representing the image dimensions (height, width) as `(y, x)`.
            - dict: Dictionary mapping each qcam file path to its corresponding image data.
            - dict: Dictionary mapping each qcam file path to its metadata headers.

    Notes:
        - The function uses `extract_qcamraw` to load image data and header metadata for each qcam file.
        - Assumes the qcam header contains a 'ROI' field formatted as "startX,startY,endX,endY".
        - Timestamps in the qcam header are expected to follow the format '%m-%d-%Y_%H:%M:%S'.
        - Additional processing merges extracted metadata into the input DataFrame.

    Example:
        >>> df = pd.DataFrame({'qcam': ['path/to/file1.qcamraw', 'path/to/file2.qcamraw']})
        >>> updated_df, qcam2img, qcam2header = loadQCamTable(df)
        >>> print(updated_df)
                           qcam  nFrames         timestamp_init    dim_YX
        0  path/to/file1.qcamraw      500  2025-01-14 10:30:00  (512, 512)
        1  path/to/file2.qcamraw      600  2025-01-14 11:00:00  (256, 256)

    """
    qcam2img,qcam2header = {},{}
    timeStamps = []
    for _,b in df.iterrows():
        qcam2img[b.qcam],qcam2header[b.qcam] = extract_qcamraw(b.qcam)
        _, _, x, y = map(int, qcam2header[b.qcam]['ROI'].replace(' ','').split(','))

        timeStamps.append((b.qcam, qcam2img[b.qcam].shape[2], qcam2header[b.qcam]['File_Init_Timestamp'], (y,x)))

    df = df.merge(pd.DataFrame(timeStamps, columns=['qcam','nFrames','timestamp_init','dim_YX']), on='qcam')
    df['timestamp_init'] = pd.to_datetime(df['timestamp_init'], format='%m-%d-%Y_%H:%M:%S')

    return df,qcam2img,qcam2header
    
