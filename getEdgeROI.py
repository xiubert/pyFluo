import glob
import os
import cv2
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import joblib
import sys

from lib.edges import *
from lib.qcam import *

# simple function to grab polygonal ROI mask from .qcamraw file
# after edge detection
def getEdgeROImask(dPath: str, file: str = None):

    if file is None:
        qFiles = glob.glob(os.path.join(dPath,'*.qcamraw'))
        imgs = []
        for q in qFiles:
            img,_ = extract_qcamraw(q)
            imgs.append(img)
        # exclude files where framecount is not most common
        nFrames = [i.shape[2] for i in imgs]
        nFrames = max(nFrames, key=nFrames.count)
        imgs = [i for i in imgs if i.shape[2]==nFrames]
        # save mask with name as first file
        filepath = os.path.join(dPath,qFiles[0])
        # take average across files then across time
        img = np.array(imgs).mean(axis=0).mean(axis=2)
    else:
        # example for one qcam file
        filepath = os.path.join(dPath,file)
        img,_ = extract_qcamraw(filepath)
        img = img.mean(axis=2)

    print(img.shape)

    # Normalize and convert image
    imgNorm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    #Process frame for edges and visualize
    edges = getImgEdges(imgNorm)
    # Display image
    plt.imshow(edges)

    # Draw a polygonal ROI
    roi = RoiPoly(color='r')

    # Get ROI mask
    mask = roi.get_mask(edges)

    # save where image is saved
    joblib.dump(mask,filepath.replace('.qcamraw','_mask.joblib'))

    # Display the ROI
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # path for single animal
    dPath = sys.argv[1]

    if len(sys.argv)>2:
        if os.path.exists(os.path.join(dPath,sys.argv[2])):
            file = sys.argv[2]
        else:
            raise("file does not exist")
    else:
        file = None
    getEdgeROImask(dPath=dPath, file=file)



