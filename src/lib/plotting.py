import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

from datetime import datetime
import os
from glob import glob
from operator import itemgetter

import lib.signalProcess as signalProcess
import lib.fileIngest as fileIngest
import lib.imgProcess as imgProcess

def plotAvgImg(img):
    ax = plt.imshow(img.mean(axis=2))

    return ax


def plotTraceAvgImg(t,img,cutoff_freq: float = 3):
    signal = np.reshape(img,(np.prod(img.shape[:2]),img.shape[2])).mean(axis=0)
    X = np.vstack([t, np.ones(len(t))]).T
    slope,intercept = np.linalg.lstsq(X,signal, rcond=None)[0]
    fig,ax = plt.subplots(3,1)
    ax[0].plot(t,signal)
    ax[0].set_title('raw trace with least-sq reg. fit')
    ax[0].plot(t,t*slope+intercept,'r')
    ax[1].plot(t,signal-(t*slope+intercept))
    ax[1].set_title('trace minus fit')
    ax[2].plot(t,signalProcess.butterFilter(signal-(t*slope+intercept),cutoff_freq=cutoff_freq))
    ax[2].set_title('filtered trace minus fit')

    return fig,ax


def experimentAvgPlot(dPath: str = None, qFiles: list = None,
                      suptitle: str = None, avgFperTrace: bool = True,
                      **kwargs):
    if qFiles is None:
        qFiles = glob(os.path.join(dPath,'*.qcamraw'))

    imgs,headers = fileIngest.qcams2imgs(qFiles)
    t = signalProcess.getTimeVec(imgs[0].shape[2],zeroStart=False)
    timeStamps = [h['File_Init_Timestamp'] for h in headers]
    timeStamps = [datetime.strptime(date, '%m-%d-%Y_%H:%M:%S') for date in timeStamps]

    if avgFperTrace:
        fig,ax = plt.subplots(3,1,figsize=(12,10))
        ax[2].plot(timeStamps,np.array(imgs).mean(axis=(1,2,3)),'.')
        ax[2].set_ylabel('raw F')
        ax[2].set_xlabel('experiment time')

    else:
        fig,ax = plt.subplots(2,1,figsize=(12,10))
    ax[0].plot(t,signalProcess.butterFilter(np.array(imgs).mean(axis=(0,1,2))))
    ax[0].set_ylabel('raw F')
    ax[0].set_xlabel('t (s)')
    ax[0].set_xticks(np.arange(0,int(max(t))+1))

    ax[1].imshow(imgProcess.calcSpatialDFFresp(np.array(imgs).mean(axis=0).reshape(*imgs[0].shape),
                                    **kwargs), cmap='jet')

    # Format the x-axis to show readable datetime labels
    # ax[1].gcf().autofmt_xdate()
    if suptitle is None:
        if dPath is None:
            fig.suptitle(os.path.dirname(qFiles[0]))
        else:
            fig.suptitle(dPath)
    else:
        fig.suptitle(suptitle)

    fig.show()

def plotDF_levelByTreatment(df: pd.DataFrame, qcam2img: dict, **kwargs):

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly  # Use Plotly's default qualitative colors
    label_to_color = {}  # Dictionary to store color mappings

    for i,(label, df_group) in enumerate(df.groupby(['dB','treatment'])):

        imgSeries = np.array(itemgetter(*df_group['qcam'])(qcam2img)) #shape [trace, Y, X, frame]
        roi_mask = kwargs.get('roi_mask',np.ones(imgSeries.shape[1:3]))
        signal = imgSeries[:,roi_mask==1,:].mean(axis=1)

        _,dF,_ = signalProcess.dFFcalc(signal,**kwargs)

        u,uPs,uMs = signalProcess.meanPlusMinusSem(dF)
        t = signalProcess.getTimeVec(len(u))
        label_str = str(list(map(str, label)))

        color = colors[i % len(colors)]  # Cycle through colors
        label_to_color[label_str] = color  # Store color for legend consistency

        fig.add_traces(
            [
                go.Scatter(
                    name=label_str,
                    x=t,
                    y=u,
                    mode='lines',
                    line=dict(color=color),
                    legendgroup=label_str,
                ),
                go.Scatter(
                    name=str(list(map(str,label))),
                    x=t,
                    y=uPs,
                    mode='lines',
                    line=dict(width=0),
                    legendgroup=label_str,
                    showlegend=False
                ),
                go.Scatter(
                    name=str(list(map(str,label))),
                    x=t,
                    y=uMs,
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=f"rgba{tuple(int(color.strip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",  # Convert hex to rgba with 20% opacity
                    legendgroup=label_str,
                    fill='tonexty',
                    showlegend=False
                )
            ]
        )

    # Label x-axis as "Time (s)"
    fig.update_layout(
                    title=f"{df.dir.unique()[0]}: dF at each sound level by treatment",
                    xaxis_title="time (s)", 
                    yaxis_title=("dF_roi" if "roi_mask" in kwargs else "dF")
                    )
    fig.show()