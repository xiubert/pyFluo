import glob
import os
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import joblib
import sys
import numpy as np

from lib.fileIngest import qcams2imgs,extract_qcamraw
from lib.imgProcess import getImgEdges

# simple function to grab polygonal ROI mask from .qcamraw file
# after edge detection
# for visualization uses mean image across all frames and all traces
def getEdgeROImask(dPath: str, file: str = None):

    if file is None:
        qFiles = glob.glob(os.path.join(dPath,'*.qcamraw'))

        imgs = qcams2imgs(qFiles)[0]
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

    #Process frame for edges and visualize
    edges = getImgEdges(img)
    # Display image
    plt.imshow(edges)

    # Draw a polygonal ROI
    roi = RoiPoly(color='r')

    # Get ROI mask
    mask = roi.get_mask(edges)

    savepath = filepath.replace('.qcamraw','_mask.joblib')

    if os.path.exists(savepath):
        mask_files = glob.glob(savepath.replace('mask.joblib','mask*'))
        os.rename(savepath,savepath.replace('.joblib',f"_{len(mask_files):03}.joblib"))

    # save where image is saved
    joblib.dump(mask,savepath)

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



