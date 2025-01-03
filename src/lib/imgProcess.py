import numpy as np
import cv2

from lib.signalProcess import butterFilter

"""
Functions for processing imaging signals.
"""

def calcSpatialDFFresp(img: np.ndarray, t: np.ndarray, 
                        frameRate: int = 20,
                        t_baseline: tuple[int] = (2,3),
                        stimlen: float = 0.4,
                        t_temporalAvg: tuple[float] = None,
                        temporalAvgFrameSpan: int = 10,
                        butterFilterParams: dict = {}) -> np.ndarray:
    """
    Calculate spatial dFF (dFF at each pixel).
    """

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


def getImgEdges(img: np.ndarray) -> np.ndarray:
    """
    Isolates edges of an image using Canny edge detection.

    Args:
        img (numpy array): image array (assume grayscale)
    Returns:
        edges (numpy array): edge pixels take the value 255
    """
    # 1: Normalize
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # 2: Convert to grayscale (if needed)
    # gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if len(norm.shape) == 3 else norm
    # 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    # 4: Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)  # Tune thresholds as needed

    return edges


def getXYdisp(Xcoor: list[np.ndarray], Ycoor: list[np.ndarray],
              frameDiff = 1):
    """
    Calculates difference in X and Y pixel coordinates across coordinate lists.

    Args:
        Xcoor (list): list of X coordinates (such as for blood vessel edge for each frame)
        Ycoor (list): list of Y coordinates (such as for blood vessel edge for each frame)
        frameDiff (int): number of frames between which to calculate the difference in pixel coordinates.

    Returns:
        Xdiffs (numpy array): average difference in pixel coordinates along X
        Ydiffs (numpy array): average difference in pixel coordinates along Y
    """

    Xdiffs,Ydiffs = [],[]
    for i,(Xidx,Yidx) in enumerate(zip(Xcoor[:-frameDiff],Ycoor[:-frameDiff])):
        # get X coordinates at same Y coordinates between frames
        a,b = np.intersect1d(Ycoor[i],Ycoor[i+frameDiff],return_indices=True)[1:]
        # get difference in X coordinates at same Y coordinates
        Xdiffs.append(np.mean(Xcoor[i+frameDiff][b]-Xidx[a]))
        # Same, but for Y
        a,b = np.intersect1d(Xcoor[i],Xcoor[i+frameDiff],return_indices=True)[1:]
        Ydiffs.append(np.mean(Ycoor[i+frameDiff][b]-Yidx[a]))
        
    return np.array(Xdiffs),np.array(Ydiffs)


def getEdgeXYdisp(imgSeries,mask,frameDiff):

    YXcoors = []
    for frame in np.arange(imgSeries.shape[2]):        
        edge = getImgEdges(imgSeries[:,:,frame])
        # get X and Y edge coordinates within mask
        YXcoors.append(np.where(mask*edge))

    # lists where each element is edge X or Y coordinate
    # length can differ depending on how many edge pixels identified
    Ycoors,Xcoors = zip(*YXcoors)

    # Calculates difference in X and Y pixel coordinates across coordinate lists.
    Xdisp,Ydisp = getXYdisp(Xcoors, Ycoors, frameDiff = frameDiff)

    # get mean and median of coordinates for absolute changes
    muX = np.array(list(map(lambda x: (np.median(x),np.mean(x)),Xcoors)))
    muY = np.array(list(map(lambda x: (np.median(x),np.mean(x)),Ycoors)))
    medianX,meanX = muX[:,0],muX[:,1]
    medianY,meanY = muY[:,0],muY[:,1]

    return Xdisp, Ydisp, meanX, medianX, meanY, medianY








