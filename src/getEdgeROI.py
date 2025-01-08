import os
import sys

from lib.imgProcess import getROImask

# Script for getting ROI of edge for which movement will be tracked.

# Running this script from terminal will execute code below 'if __name__ == "__main__":'
# This script takes only one input: path to .qcamraw files or path to single .qcamraw files.
# eg. python ./getEdgeROI.py "/home/pac/Documents/Python/pyFluo/data/AA0308/"

if __name__ == "__main__":
    if len(sys.argv)<2:
        raise ValueError("Must provide a path to .qcamraw files or path to single .qcamraw files.")
    if len(sys.argv)>2:
        raise ValueError("Script takes only one input: path to .qcamraw files or path to single .qcamraw files.")
    imgPath = sys.argv[1]
    if not os.path.exists(imgPath):
        raise ValueError("Path does not exist")
    
    mask,_ = getROImask(imgPath=imgPath, processEdges=True)



