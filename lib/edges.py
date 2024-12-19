import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.qcam import getAllImgs

def getImgEdges(img):
    """
    Isolates edges of an image using Canny edge detection.
    """
    # 1: Convert to grayscale (if needed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3: Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)  # Tune thresholds as needed

    return edges

def getEdgeReTimeViaMask(img,mask):
    """
    Get image edges, masked edges, and X and Y coor of edges across image frames
    """
    edges, edgesMasked, XmaskedEdges, YmaskedEdges = [],[],[],[]
    for frame in np.arange(img.shape[2]):
        imgN = cv2.normalize(img[:,:,frame].squeeze(),
                              None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        edge = getImgEdges(imgN)

        edges.append(edge)
        edgesMasked.append(edge*mask)

        Y,X = np.where(mask*edge)
        XmaskedEdges.append(X)
        YmaskedEdges.append(Y)

    return edges,edgesMasked,XmaskedEdges,YmaskedEdges


def getExperimentEdgeDisp(qFiles: list, mask: np.ndarray):
    """
    Get average edge displacement across all files
    """
    uX,uY,Xdiff,Ydiff = [],[],[],[]

    imgs,_ = getAllImgs(qFiles)
    for im in imgs:
        _,_,x,y = getEdgeReTimeViaMask(im,mask)
        uX.append(np.array(list(map(np.mean,x))))
        uY.append(np.array(list(map(np.mean,y))))

        accumXdiff,accumYdiff = [],[]
        for i,(Xidx,Yidx) in enumerate(zip(x[:-1],y[:-1])):
            _,a,b = np.intersect1d(y[i],y[i+1],return_indices=True)
            accumXdiff.append(np.mean(abs(Xidx[a]-x[i+1][b])))
            _,a,b = np.intersect1d(x[i],x[i+1],return_indices=True)
            accumYdiff.append(np.mean(abs(Yidx[a]-y[i+1][b])))
        Xdiff.append(accumXdiff)
        Ydiff.append(accumYdiff)

    return uX,uY,Xdiff,Ydiff


def plotEdgeDisplacement(t,uX,uY,Xdiff,Ydiff):
    fig,ax = plt.subplots(4,1,figsize=(15,5))
    ax[0].plot(t,np.array(uX).mean(axis=0))
    ax[0].axvline(x=3,color='r')
    ax[0].set_title('X')

    ax[1].plot(t,np.array(uY).mean(axis=0))
    ax[1].axvline(x=3,color='r')
    ax[1].set_title('Y')

    ax[2].plot(t[:-1],np.array(Xdiff).mean(axis=0))
    ax[2].axvline(x=3,color='r')
    ax[2].set_title('X Diff')

    ax[3].plot(t[:-1],np.array(Ydiff).mean(axis=0))
    ax[3].axvline(x=3,color='r')
    ax[3].set_title('Y Diff')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    return fig,ax

