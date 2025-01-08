import os
import sys
import matplotlib.pyplot as plt

from lib.imgProcess import getROImask
from lib.fileIngest import getTimeVec

# Script for getting fluorescence response re time within selected ROI.

# Running this script from terminal will execute code below 'if __name__ == "__main__":'
# This script takes only one input: path to .qcamraw files or path to single .qcamraw files.
# eg. python ./getEdgeROI.py "/home/pac/Documents/Python/pyFluo/data/AA0308/"

# can run interactively and define qcam files here:
qFiles = [
        '/home/pac/Documents/Python/pyFluo/data/AA0308/AA0308AAAA0004.qcamraw',
        '/home/pac/Documents/Python/pyFluo/data/AA0308/AA0308AAAA0009.qcamraw',
        '/home/pac/Documents/Python/pyFluo/data/AA0308/AA0308AAAA0005.qcamraw',
        ]

if __name__ == "__main__":
    if len(sys.argv)<2:
        raise ValueError("Must provide a path to .qcamraw files or path to single .qcamraw files.")
    if len(sys.argv)>2:
        raise ValueError("Script takes only one input: path to .qcamraw files or path to single .qcamraw files.")
    imgPath = sys.argv[1]
    if not os.path.exists(imgPath):
        raise ValueError("Path does not exist")
    
    mask,imgSeries = getROImask(imgPath=imgPath, processResponse=True)
    t = getTimeVec(imgSeries.shape[2])
    plt.plot(t,imgSeries[mask==1,:].mean(axis=0))
    plt.show()

else:
    mask,imgSeries = getROImask(qcamFiles=qFiles, processResponse=True)
    t = getTimeVec(imgSeries.shape[2])
    plt.plot(t,imgSeries[mask==1,:].mean(axis=0))
    plt.show()




