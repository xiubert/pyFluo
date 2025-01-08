import numpy as np
import cv2
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import joblib
import os
from glob import glob

from lib.signalProcess import butterFilter
from lib.fileIngest import extract_qcamraw, getTimeVec, qcams2imgs


"""
Functions for processing imaging signals.
"""

def calcSpatialDFFresp(imgSeries: np.ndarray, t: np.ndarray, 
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
    reshaped_data = imgSeries.reshape(-1,200)
    baselineIDX = np.where((t>=t_baseline[0]) & (t<=t_baseline[1]))[0]
    spatialbase = reshaped_data[:,baselineIDX].mean(axis=1).reshape(-1,1)
    spatialDFF = (reshaped_data-spatialbase)/spatialbase
    
    if butterFilterParams and isinstance(butterFilterParams,dict):
        spatialDFF = butterFilter(spatialDFF,**butterFilterParams)

    if t_temporalAvg is None:
        t_temporalAvg = (t_baseline[1]+stimlen,t_baseline[1]+stimlen+temporalAvgFrameSpan*(1/frameRate))

    spatialDFFresp = spatialDFF[:,np.where((t>=t_temporalAvg[0]) &
                                        (t<=t_temporalAvg[1]))[0]]\
                                            .mean(axis=1).reshape(*imgSeries.shape[:2])

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


def getROImask(imgPath: str = None,
               qcamFiles: list = None, 
               processEdges: bool = False, processResponse: bool = False,
               saveMask: bool = True, **kwargs) -> np.ndarray:
    """
    Captures ROI mask via user drawn polygon.
    """
    if qcamFiles is not None:
        imgs = qcams2imgs(qcamFiles)[0]
        # take average across files
        imgSeries = np.array(imgs).mean(axis=0)
        # then avg across time
        img = imgSeries.mean(axis=2)
        savePath = ''
        saveMask=False
    
    else:
        if ".qcamraw" in imgPath and os.path.exists(imgPath):
            print("Showing temporal average of single qcamraw file for drawing ROI")
            imgSeries = extract_qcamraw(imgPath)[0]
            # take average across time
            img = imgSeries.mean(axis=2)
            savePath = imgPath.replace('.qcamraw','_mask.joblib')

        elif os.path.exists(imgPath):
            print("Showing temporal average of all .qcamraw files in directory for drawing ROI")
            qFiles = glob(os.path.join(imgPath,'*.qcamraw'))
            if len(qFiles)<1:
                ValueError("No .qcamraw files found in directory")
            imgs = qcams2imgs(qFiles)[0]
            # take average across files
            imgSeries = np.array(imgs).mean(axis=0)
            # then avg across time
            img = imgSeries.mean(axis=2)
            savePath = os.path.join(imgPath,'avgImg_mask.joblib')
            
        else:
            raise ValueError("Path does not exist")
    
    if not (processEdges ^ processResponse):
        return ValueError("Only one of processEdges or processResponse may be true")
    
    if processEdges:
        savePath = savePath.replace('_mask','_edge_mask')
        #Process frame for edges and visualize
        imgEdge = getImgEdges(img)
        plt.imshow(imgEdge+img*0.3, cmap='gray')

    elif processResponse:
        savePath = savePath.replace('_mask','_response_mask')
        t = getTimeVec(imgSeries.shape[2])

        # eventually use kwargs for calcSpatialDFFresp params
        spatialRespImg = calcSpatialDFFresp(imgSeries,
                        t, stimlen=0.1, temporalAvgFrameSpan=8)
        plt.imshow(spatialRespImg)
    else:
        # Display image
        plt.imshow(img, cmap='gray')

    # Prompt to draw a polygonal ROI
    print("Draw polygon... Double click to finish")
    roi = RoiPoly(color='r')

    # Get ROI mask
    mask = roi.get_mask((imgEdge if processEdges else img))
    
    if saveMask:
        if os.path.exists(savePath):
            mask_files = glob(savePath.replace('mask.joblib','mask*'))
            os.rename(savePath,savePath.replace('.joblib',f"_{len(mask_files):03}.joblib"))

        # save where image is saved
        joblib.dump(mask,savePath)

    # Display the ROI
    if processEdges:
        plt.imshow((img*.3+imgEdge)*(mask+0.6), cmap='gray')
        plt.show()

    if processResponse:
        _, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(spatialRespImg*(mask+0.5), cmap='viridis')
        plt.show()
        
    else:
        plt.imshow(img*(mask+0.6), cmap='gray')
        plt.show()

    return mask,imgSeries





