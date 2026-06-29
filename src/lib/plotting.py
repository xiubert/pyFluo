import numpy as np
import pandas as pd

import colorsys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime
import os
from glob import glob
from operator import itemgetter
import warnings
import math
from itertools import product

import lib.signalProcess as signalProcess
import lib.fileIngest as fileIngest
import lib.imgProcess as imgProcess

def saveMPLfig(fig, outputPath):
    # set export font to support editable text in vector graphics output
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    fig.savefig(outputPath, dpi=1000, 
                transparent=True, 
                format="pdf")


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
    
    # Optionally override parameters using kwargs
    dPath = kwargs.get('dPath',dPath)
    qFiles = kwargs.get('qFiles',qFiles)
    suptitle = kwargs.get('suptitle',suptitle)
    avgFperTrace = kwargs.get('avgFperTrace',avgFperTrace)
    
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


def plot_peakDFF_reTime(df: pd.DataFrame, 
                        time_col: str = 'timestamp_init', 
                        dB_plot: int = 80, 
                        resp_col: str = 'dFF_ROI_linFilt_butterFilt_peak'):
    """
    Plot peak response through experiment time to visualize potential time effects on response amplitudes.
    Connects different treatments with dashed lines if multiple treatments exist.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'dir', 'dB', specified 'time_col' and 'resp_col'.
        time_col (str, optional): Name of the column containing the experiment time.
                                  Defaults to 'timestamp_init'.
        dB_plot (int, optional): Sound intensity (in dB) for peak response to be plotted. Defaults to 80.
                                 If None, plot all sound intensities.
        resp_col (str, optional): Name of the column containing the response variable. 
                                  Defaults to 'dFF_ROI_linFilt_butterFilt_peak'.
    """
    
    # Check whether required columns exist
    required_cols = ['dir', 'dB', time_col, resp_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataframe must contain the following columns: {required_cols}")
    
    # Check whether specified sound level exists
    if dB_plot is not None and dB_plot not in df['dB'].unique():
        raise ValueError(f"{dB_plot} not found in the 'dB' column.")
    
    # Filter and sort data in chronological order
    df_filtered = df[df['dB'] == dB_plot] if dB_plot is not None else df
    df_filtered = df_filtered.sort_values(time_col).copy()
    
    fig,ax = plt.subplots(figsize=(16,4))
    colors = plt.cm.tab10.colors

    multi_animal = True if df['dir'].nunique() > 1 else False

    for i, (dir_name, dir_df_filtered) in enumerate(df_filtered.groupby('dir')):
        # Calculate relative time in minutes
        relative_time = (dir_df_filtered[time_col] - dir_df_filtered[time_col].iloc[0]).dt.total_seconds() / 60
        color = colors[i % len(colors)]

        if 'treatment' in df.columns and df['treatment'].nunique() > 1:
            # For multiple treatments
            treatments = dir_df_filtered['treatment'].unique()
            for j, treatment in enumerate(treatments):
                former_treat_mask = dir_df_filtered['treatment'] == treatments[j-1]
                latter_treat_mask = dir_df_filtered['treatment'] == treatment

                # Plot each treatment segment separately
                ax.plot(relative_time[latter_treat_mask], 
                        dir_df_filtered[resp_col][latter_treat_mask], 
                        '.-', color=color, markersize=6 if multi_animal else 10, 
                        label=dir_name if j == 0 else None)  # Only add label for first treatment of each animal
                
                # Connect different treatments with dashed lines
                if j > 0:
                    former_last_idx = dir_df_filtered.loc[former_treat_mask].index[-1]
                    latter_first_idx = dir_df_filtered.loc[latter_treat_mask].index[0]
                    
                    ax.plot([relative_time[former_last_idx], relative_time[latter_first_idx]],
                            [dir_df_filtered[resp_col].loc[former_last_idx], dir_df_filtered[resp_col].loc[latter_first_idx]],
                            '--', color=color)
        else:
            # For single treatment
            ax.plot(relative_time, dir_df_filtered[resp_col], '.-', color=color, markersize=10, label=dir_name)
    
    ax.set_ylabel(f'{resp_col}', fontsize=12)
    ax.set_xlabel('Time since experiment start (min)', fontsize=12)
    ax.set_title(f'Peak response at {dB_plot} dB through experiment time' if dB_plot is not None 
                 else 'Peak response at all intensities through experiment time', fontsize=14)
    ax.legend()

    plt.show()


def plot_respHeatmap(df: pd.DataFrame, dB_plot: int = 80, same_scale: bool = True, 
                     contrast_percentile: tuple[float, float] = None, 
                     ROIcontour: np.ndarray | dict[str, np.ndarray] | None = None, **kwargs):
    """
    Plot baseline gray-scaled wide-field images and response heatmaps for each treatment.
    Optionally add ROI mask contours to images for visualization.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'qcam', 'dB', and 'treatment'.
        dB_plot (int, optional): Sound intensity (in dB) to plot response heatmaps. Defaults to 80.
                                 If None, plot average across all intensities.
        same_scale (bool, optional): If 'True', use same color scaling in all heatmaps.
        contrast_percentile (tuple, optional): (Lower, upper) percentile range of the color scale that the heatmap is focusing on.
                                               Improve visibility of mid-range responses by preventing extreme values from compressing the color range.
                                               - Example: (0.1, 99.9) clips the color scale to exclude the darkest 0.1% and brightest 0.1% of pixels.
                                               - If None, use min/max scaling (may reduce contrast if outliers exist).
                                               Defaults to None.
        ROIcontour (np.ndarray | dict, optional): ROI's vertex coordinates, including a repeated first vertex to close the shape.
                                                  Either: - 2D numpy array (same ROI for all treatments).
                                                          - Dictionary mapping treatments to 2D arrays (different ROIs for each treatment).
                                                          - None (no contours shown).
        **kwargs: Optional arguments that will override default.
            examples: t_baseline (tuple): start and end time points (inclusive) of baseline to plot wide-field images.
                      t_temporalAvg (tuple): start and end time points (inclusive) of response to plot response heatmaps.
    """
    # Check whether required columns exist
    required_cols = ['qcam', 'dB', 'treatment']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataframe must contain the following columns: {required_cols}")

    # Validate argument 'ROIcontour'
    if ROIcontour is not None:
        if not isinstance(ROIcontour, (np.ndarray, dict)):
            raise ValueError("'ROIcontour' must be a numpy array or dictionary.")
        if isinstance(ROIcontour, dict):
            missing = set(df['treatment'].unique()) - set(ROIcontour.keys())
            if missing:
                raise ValueError(f"Dataframe contains treatments not in 'ROIcontour': {missing}")

    # Initialize figure
    treatments = df['treatment'].unique()
    nTreat = len(treatments)
    fig, ax = plt.subplots(nTreat, 2, figsize=(10, 3*nTreat))
    if nTreat == 1:
        # Ensure ax is 2D even for one treatment
        ax = ax.reshape(1, -1)

    if same_scale:
        # Get global min and max dFF for consistent color scaling in heatmaps
        all_spatialDFF = []
        for treatment, df_group in df.groupby('treatment', sort=False):
            qcams = (df_group['qcam'].tolist() if dB_plot is None 
                    else df_group[df_group['dB'] == dB_plot]['qcam'].tolist())
            _, _, _, spatialDFF = imgProcess.qcams2roiTrace(qcams, **kwargs)
            all_spatialDFF.append(spatialDFF)

        # Use percentiles for contrast enhancement if specified
        all_values = np.concatenate([dFF.flatten() for dFF in all_spatialDFF])
        dFF_min, dFF_max = np.percentile(all_values, contrast_percentile) if contrast_percentile else \
                           (min(dFF.min() for dFF in all_spatialDFF), max(dFF.max() for dFF in all_spatialDFF))

    # Plot baseline gray-scaled wide-field images and response heatmaps for each treatment
    for i, treatment in enumerate(treatments):
        qcams = (df[df['treatment'] == treatment]['qcam'].tolist() if dB_plot is None 
                else df[(df['treatment'] == treatment) & (df['dB'] == dB_plot)]['qcam'].tolist())

        # Calculate spatial dFF for response heatmap
        _, _, imgs, spatialDFF = imgProcess.qcams2roiTrace(qcams, **kwargs)

        # Determine heatmap color scaling
        if same_scale:
            vmin, vmax = dFF_min, dFF_max
        else:
            vmin, vmax = np.percentile(spatialDFF, contrast_percentile) if contrast_percentile else (None, None)
        
        # Plot images with labels/titles and colorbars
        ax[i,0].imshow(imgs.mean(axis=(0,-1)), 'gray')
        respHeat = ax[i,1].imshow(spatialDFF, cmap='jet', vmin=vmin, vmax=vmax)
        ax[i,0].set_ylabel(treatment, rotation=0, ha='right', va='center', fontsize=14)
        if i == 0:
            ax[i,0].set_title("Wide-field", fontsize=14)
            ax[i,1].set_title("Response heatmap", fontsize=14)
        plt.colorbar(respHeat, ax=ax[i,1])
    
        # Add contours if provided
        if ROIcontour is not None:
            contour = ROIcontour if isinstance(ROIcontour, np.ndarray) else ROIcontour[treatment]
            for j in range(2):
                ax[i,j].plot(contour[:,0], contour[:,1], 'w-', linewidth=2)
    
    plt.tight_layout()
    plt.show()


def plot_traces(df: pd.DataFrame, dB_plot: int | list[int] | None = 80, resp_col: str = 'dFF_ROI_raw', 
                sepPlot: bool = False, stimStart: float = 3.0, alpha_ind: float = 0.3, Yaxis_range: tuple[float,float] = None, **kwargs):
    """
    Plot individual and averaged traces for a given sound intensity across different treatments.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 
                           'dB', 'treatment', 'time' (or 'nFrames'), and column for response traces.
        dB_plot (int | list | None, optional): Sound intensity (in dB) for traces to be plotted. Defaults to 80.
                                               Either: - Integer representing the sound level in dB.
                                                       - List including all sound levels to be plotted.
                                                       - None, to plot all traces.
        resp_col (str, optional): Column name for response traces. Defaults to 'dFF_ROI_raw'.
        sepPlot (bool, optional): If True, plot treatments in separate subplots; otherwise, plot in one plot. 
                                  Defaults to 'False'.
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        alpha_ind (float, optional): Transparency for individual traces. Defaults to 0.3.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, Y-axis is auto-scaled.
        **kwargs: Optional arguments that will override default.
    """
    
    # Check whether required columns exist
    required_cols = ['dB', 'treatment', resp_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")
    
    # Check whether specified sound level(s) exist 
    if isinstance(dB_plot, int):
        if dB_plot not in df['dB'].unique():
            raise ValueError(f"dB_plot={dB_plot} not found in the 'dB' column.")
    elif isinstance(dB_plot, list):
        missing_dBs = [db for db in dB_plot if db not in df['dB'].unique()]
        if missing_dBs:
            raise ValueError(f"dB_plot contains values not found in the 'dB' column: {missing_dBs}")
    elif dB_plot is not None:
        raise TypeError(f"dB_plot must be an int, list[int], or None, got {type(dB_plot)}")

    # Filter the DataFrame for the specified sound intensity(s)
    if isinstance(dB_plot, int):
        filtered_df = df[df['dB'] == dB_plot].reset_index(drop=True)
    elif isinstance(dB_plot, list):
        filtered_df = df[df['dB'].isin(dB_plot)].reset_index(drop=True)
    else:
        filtered_df = df.reset_index(drop=True)
    
    # Get time vectors
    if 'time' in filtered_df.columns:
        time_vectors = filtered_df['time'].tolist()
    elif 'nFrames' in filtered_df.columns:
        time_vectors = [signalProcess.getTimeVec(nFrames, **kwargs) for nFrames in filtered_df['nFrames']]
    else:
        raise ValueError("Cannot access time vector. Neither 'time' nor 'nFrames' column exist in dataframe.")

    # Ensure sound stimuli start at the same time for all traces
    if 'baseWindow' not in filtered_df.columns and 'respWindow' not in filtered_df.columns:
        if pd.Series([len(time_vector) for time_vector in time_vectors]).nunique() > 1:
            # Raise error when traces have more than one length and cannot be aligned according to time window
            raise ValueError("Traces have different lengths. Unable to align them based on baseline or response time windows.")
    else:
        # Align traces based on time windows if multiple trace lengths exist
        windows = filtered_df['baseWindow'].tolist() if 'baseWindow' in filtered_df.columns else filtered_df['respWindow'].tolist()
        # Set the latest time window as reference (to avoid negative values in time vectors)
        ref_window = max(windows, key=lambda x: x[0])    # With the latest start time
        for i, (window, time_vector) in enumerate(zip(windows, time_vectors)):
            if window != ref_window:
                shift = window[0] - ref_window[0]
                # Shift the time vector
                time_vectors[i] = time_vector - shift
    
    # Determine the full time range of time vectors after alignment
    min_time = 0    # Bacause reference time vector starts at 0, and all time vectors are non-negative after alignment
    max_time = max(max(time_vector) for time_vector in time_vectors)

    # Pad the time vector and traces to match the full time range
    padded_traces = []
    for time_vector, trace in zip(time_vectors, filtered_df[resp_col]):
        # Calculate padding needed at beginning and end
        start_pad = int((min(time_vector) - min_time) / np.mean(np.diff(time_vector)))
        end_pad = int((max_time - max(time_vector)) / np.mean(np.diff(time_vector)))
        
        # Pad the trace with NaNs
        padded_trace = np.pad(trace, (start_pad, end_pad), mode='constant', constant_values=np.nan)
        
        # Pad the time vector
        padded_time = np.pad(time_vector, (start_pad, end_pad), mode='constant', constant_values=np.nan)
        
        padded_traces.append((padded_time, padded_trace))
    
    # Update time_vectors and traces with paddings
    time_vectors = [arr[0] for arr in padded_traces]
    responses = [arr[1] for arr in padded_traces]

    # Extract traces for each treatment
    traces = {}
    treatments = filtered_df['treatment'].unique()
    for treatment in treatments:
        treatment_mask = filtered_df['treatment'] == treatment
        treatment_responses = [responses[i] for i in range(len(responses)) if treatment_mask[i]]
        traces[treatment] = {
            'individual': np.array(treatment_responses).T,
            'averaged': np.nanmean(np.array(treatment_responses).T, axis=1)
        }
    
    # Create the figure and axes
    if sepPlot:
        fig, ax = plt.subplots(len(treatments), 1, figsize=(10, 4 * len(treatments)), sharex=True)
        if len(treatments) == 1:
            ax = [ax]  # Ensure ax is always a list for consistency
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax = [ax]  # Ensure ax is a list for consistency
    
    # Use the first time vector (all should be the same after padding)
    time = time_vectors[0]
    
    # Plot the traces
    for i, treatment in enumerate(treatments):
        # Use discrete colormap (from matplotlib) for different treatments
        color = plt.cm.tab10(i)
        current_ax = ax[i] if sepPlot else ax[0]
        
        # Plot individual traces
        for j in range(traces[treatment]['individual'].shape[1]):
            current_ax.plot(time, traces[treatment]['individual'][:, j], color='gray' if sepPlot else color, 
                            alpha=alpha_ind, label=f'{treatment} Individual' if j == 0 else "")
        
        # Plot averaged trace
        current_ax.plot(time, traces[treatment]['averaged'], color=color, linewidth=2, 
                        label=f'{treatment} Averaged')
        
        # Add labels, title, and stimulus line
        current_ax.set_xlabel('time (s)', size=12)
        current_ax.set_ylabel(resp_col, size=12)
        current_ax.set_title(f"{treatment}", size=12) if sepPlot else None
        current_ax.axvline(x=stimStart, color='k', linestyle='--')
        current_ax.legend(loc='upper right')
        if Yaxis_range is not None:
            current_ax.set_ylim(Yaxis_range)
    
    # Format the title based on 'dB_plot' type
    if isinstance(dB_plot, int):
        fig.suptitle(f"Individual and Averaged Traces: {dB_plot} dB", size=14)
    elif isinstance(dB_plot, list):
        dB_str = ', '.join(map(str, dB_plot))  # Convert list to comma-separated string
        fig.suptitle(f"Individual and Averaged Traces: {dB_str} dB", size=14)
    
    plt.tight_layout()
    plt.show()


def plotDF_levelByTreatment(df: pd.DataFrame, qcam2img: dict = None, resp_col: str = None, dFResp: bool = False, 
                            sepPlot: bool = True, errBar: bool = True, Yaxis_range: tuple[float,float] = None, **kwargs):
    """
    Plot fluorescence response (dFF or dF) traces by treatment and dB.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns `dB` and `treatment`, and either column `qcam` or specified `resp_col`.
        qcam2img (dict, optional): Dictionary mapping each qcam file path to its corresponding image data. Defaults to None.
        resp_col (str, optional): Column name for response traces. Defaults to None.
        dFResp (bool, optional): If true, calculate dF response rather than dFF.
        sepPlot (bool, optional): Whether to create separate subplots for each treatment.
                                  - `True`: For each subplot, lower dBs are in cooler colors and higher dBs are in warmer colors.
                                  - `False`: Treatments are distinguished by different color types, 
                                             with lower dBs in lighter colors and higher dBs in darker colors.
        errBar (bool, optional): If true, add error bars to curves.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, Y-axis is auto-scaled.
        **kwargs: Optional keyword arguments.
            example: roi_mask (np.ndarray): 2D binary mask array specifying the region of interest.
                     t_base (tuple): Time window (in seconds) for baseline.

    Notes:
        - If `resp_col` is specified, traces are extracted directly from the dataframe rather than `qcam2img`.
        - If `resp_col` is None, `qcam2img` must be provided.
    """
    
    # Check whether either 'qcam2img' or 'resp_col' is provided
    if qcam2img is None and resp_col is None:
        raise ValueError("Either 'qcam2img' or 'resp_col' must be provided.")

    # Check whether column 'resp_col' exists in the dataframe if provided
    if resp_col is not None and resp_col not in df.columns:
        raise ValueError(f"DataFrame must contain column: {resp_col}")
    
    if resp_col is None:
        # Check whether column 'qcam' exists in the dataframe if 'resp_col' is not provided
        if 'qcam' not in df.columns:
            raise ValueError("DataFrame must contain column 'qcam' when resp_col is None.")
        # Check for missing qcam keys in 'qcam2img'
        missing_qcams = set(df['qcam']) - set(qcam2img.keys())
        if missing_qcams:
            raise ValueError(f"qcam2img missing keys: {missing_qcams}")

    # Sort dataframe by treatment (consistent with initial order) and dB (in ascending order)
    df_sorted = pd.concat([group.sort_values(by='dB') for _, group in df.groupby('treatment', sort=False)])

    # Initialize figure
    if sepPlot:
        # Map cooler colors to lower dBs and warmer colors to higher dBs
        dB_values = df_sorted['dB'].unique()
        colors = cm.coolwarm(np.linspace(0, 1, len(dB_values)))
        colors = [f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b, _ in colors]
        dB2color = {dB: colors[i] for i, dB in enumerate(dB_values)}

        # Create subplots with one row per treatment
        treat_values = df_sorted['treatment'].unique()
        fig = make_subplots(rows=len(treat_values), cols=1, vertical_spacing=0.2,
                            subplot_titles=[f"{treatment}" for treatment in treat_values])
    else:
        # Use discrete colormap (from matplotlib) for different treatments
        colors = plt.cm.tab10.colors
        max_dB = df_sorted['dB'].max()  # Maximum dB value for normalization
        treat_values = df_sorted['treatment'].unique()
        treat2color = {treat: f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for treat, (r, g, b) in zip(treat_values, colors)}
        fig = go.Figure()

    # Calculate fluorescence response within each treatment/dB combination
    for (treatment, dB), df_group in df_sorted.groupby(['treatment', 'dB'], sort=False):
        if resp_col is not None:
            response = np.vstack(df_group[resp_col])
        else:
            # If 'resp_col' is not given, extract image data from qcam files
            imgSeries = np.array(itemgetter(*df_group['qcam'])(qcam2img))  # Shape: [trace, Y, X, frame]
            if imgSeries.ndim == 3:  # If only one trace, add a dimension
                imgSeries = np.expand_dims(imgSeries, axis=0)
            roi_mask = kwargs.get('roi_mask', np.ones(imgSeries.shape[1:3]))
            signal = imgSeries[:, roi_mask == 1, :].mean(axis=1)
            dFF, dF, _ = signalProcess.dFFcalc(signal, **kwargs)
            response = dF if dFResp else dFF
        mean, upper, lower = signalProcess.meanPlusMinusSem(response)
        t = signalProcess.getTimeVec(len(mean), **kwargs)

        label_str = f"{treatment}, {dB} dB"

        if sepPlot:
            # Get cool-to-warm color based on dB level
            color = dB2color[dB]
            rgba_fill = color.replace("rgb", "rgba").replace(")", ", 0.1)")  # Add transparency of 10%
            row = list(treat_values).index(treatment) + 1  # +1 because subplot rows are 1-indexed
        else:
            # Adjust lightness based on dB level
            base_color = treat2color[treatment]
            r, g, b = [int(val) for val in base_color.strip("rgb(").strip(")").split(",")]
            lightness = 2.5 - 2 * (dB / max_dB)  # Normalize dB to [0.5, 2.5] for lightness
            h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
            r_new, g_new, b_new = colorsys.hls_to_rgb(h, l*lightness, s)
            color = f"rgb({int(r_new*255)}, {int(g_new*255)}, {int(b_new*255)})"
            rgba_fill = color.replace("rgb", "rgba").replace(")", ", 0.1)")  # Add transparency of 10%
            row, col = None, None  # No subplots

        # Add mean traces
        fig.add_trace(
            go.Scatter(
                name=label_str,
                x=t,
                y=mean,
                mode='lines',
                line=dict(color=color),
                legendgroup=label_str,
                showlegend=True
            ),
            row=row, col=1 if sepPlot else None
        )

        if errBar:
            # Add upper bound of error bar
            fig.add_trace(
                go.Scatter(
                    name=label_str,
                    x=t,
                    y=upper,
                    mode='lines',
                    line=dict(width=0),
                    legendgroup=label_str,
                    showlegend=False
                ),
                row=row, col=1 if sepPlot else None
            )

            # Add lower bound of error bar
            fig.add_trace(
                go.Scatter(
                    name=label_str,
                    x=t,
                    y=lower,
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=rgba_fill,
                    legendgroup=label_str,
                    fill='tonexty',
                    showlegend=False
                ),
                row=row, col=1 if sepPlot else None
            )

    # Update layout
    fig.update_layout(
        title=f"{df_sorted.dir.unique()[0]}: Fluorescence response at each sound level by treatment | signal: {(resp_col if resp_col else '')}",
        xaxis_title="time (s)",
        yaxis_title=("dF" if dFResp else "dFF")
    )

    if sepPlot:
        # Adjust layout for subplots
        fig.update_layout(height=410 * len(treat_values))  # Adjust height based on the number of treatments
        for i in range(1, len(treat_values) + 1):  # Add X- and Y-axis legends for each subplot
            fig.update_xaxes(title_text="time (s)", row=i, col=1)
            fig.update_yaxes(title_text=("dF" if dFResp else "dFF"), row=i, col=1)
    
    # Set Y-axis range if specified
    if Yaxis_range is not None:
        if sepPlot:
            for i in range(1, len(treat_values) + 1):
                fig.update_yaxes(range=Yaxis_range, row=i, col=1)
        else:
            fig.update_yaxes(range=Yaxis_range)

    fig.show()


def plotTrace_reAnimal(df: pd.DataFrame, dB_plot: int = 80, resp_col: str = 'dFF_ROI_raw', 
                       sepPlot: bool = True, **kwargs):
    """
    Plot averaged traces with error bars for a given sound intensity re treatments and animals.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 
                           'dir', 'dB', 'treatment', 'time' (or 'nFrames'), and column for response traces.
        dB_plot (int, optional): Sound intensity (in dB) for traces to be plotted. Defaults to 80.
        resp_col (str, optional): Column name for response traces. Defaults to 'dFF_ROI_raw'.
        sepPlot (bool, optional): If True, plot treatments in separate subplots; otherwise, plot in one plot. 
                                  Defaults to 'True'.
        **kwargs: Optional arguments that will override default.
    """

    # Check whether required columns exist
    required_cols = ['dir', 'dB', 'treatment', resp_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataframe must contain the following columns: {required_cols}")
    
    # Check whether specified sound level exists
    if dB_plot not in df['dB'].unique():
        raise ValueError(f"{dB_plot} not found in the 'dB' column.")
    
    # Filter the DataFrame for the specified sound intensity
    filtered_df = df[df['dB'] == dB_plot].reset_index(drop=True)
    
    # Create a list including time vectors of all trials
    if 'time' in filtered_df.columns:
        time_vectors = filtered_df['time'].tolist()
    elif 'nFrames' in filtered_df.columns:
        time_vectors = [signalProcess.getTimeVec(nFrames, **kwargs) for nFrames in filtered_df['nFrames']]
    else:
        raise ValueError("Cannot access time vector. Neither 'time' nor 'nFrames' column found in dataframe.")
    
    # Ensure sound stimuli start at the same time for all traces
    trace_lengths = filtered_df['time'].apply(len) if 'time' in filtered_df.columns else filtered_df['nFrames']
    if 'baseWindow' not in filtered_df.columns and 'respWindow' not in filtered_df.columns:
        if trace_lengths.nunique() > 1:
            # Raise error when traces have more than one length and cannot be aligned according to time window
            raise ValueError("Traces have different lengths. Unable to align them based on baseline or response time windows.")
    else:
        # Align the baseline or response time window across traces from different animals
        windows = filtered_df['baseWindow'].tolist() if 'baseWindow' in filtered_df.columns else filtered_df['respWindow'].tolist()
        # Set the latest time window as reference (to avoid negative values in time vectors)
        ref_window = max(windows, key=lambda x: x[0])    # With the latest start time
        for i, (window, time_vector) in enumerate(zip(windows, time_vectors)):
            if window != ref_window:
                shift = window[0] - ref_window[0]
                # Shift the time vector
                time_vectors[i] = time_vector - shift
    
    # Initialize data for plotting
    plot_data = {}

    # Determine the full time range of time vectors after alignment
    min_time = 0    # Bacause reference time vector starts at 0, and all time vectors are non-negative after alignment
    max_time = max(max(time_vector) for time_vector in time_vectors)

    # Group by animal and treatment
    for (animal, treatment), group in filtered_df.groupby(['dir', 'treatment'], sort=False):
        traces = np.array(group[resp_col].tolist())
        mean, upper, lower = signalProcess.meanPlusMinusSem(traces)
        time_vector = time_vectors[group.index[0]]  # Use the time vector of the first trace in the group

        # Pad the time vector and traces to match the full time range
        if min(time_vector) > min_time:
            # Pad at the beginning
            padding_length = int((min(time_vector) - min_time) / np.mean(np.diff(time_vector)))
            mean = np.pad(mean, (padding_length, 0), constant_values=np.nan)
            upper = np.pad(upper, (padding_length, 0), constant_values=np.nan)
            lower = np.pad(lower, (padding_length, 0), constant_values=np.nan)
            time_vector = np.pad(time_vector, (padding_length, 0), constant_values=np.nan)

        if max(time_vector) < max_time:
            # Pad at the end
            padding_length = int((max_time - max(time_vector)) / np.mean(np.diff(time_vector)))
            mean = np.pad(mean, (0, padding_length), constant_values=np.nan)
            upper = np.pad(upper, (0, padding_length), constant_values=np.nan)
            lower = np.pad(lower, (0, padding_length), constant_values=np.nan)
            time_vector = np.pad(time_vector, (0, padding_length), constant_values=np.nan)

        # Store the padded data
        plot_data[(animal, treatment)] = (mean, upper, lower, time_vector)
    
    # Create the plot
    treatments = filtered_df['treatment'].unique()
    if sepPlot:
        # Create subplots for each treatment
        fig = make_subplots(rows=len(treatments), cols=1, subplot_titles=[f"{treat}" for treat in treatments],
                            vertical_spacing=0.15)
    else:
        # Create a single plot
        fig = go.Figure()
    
    # Assign colors
    animals = filtered_df['dir'].unique()
    animal_colors = {animal: f"hsl({(i * 360 / len(animals)) % 360}, 50%, 50%)" for i, animal in enumerate(animals)}
    
    # Track which animals have already been added to the legend
    legend_added = set()

    # Plotting
    for i, ((animal, treatment), (mean, upper, lower, time_vector)) in enumerate(plot_data.items()):
        if sepPlot:
            # In separate subplots, use the same color for the same animal across treatments
            color = animal_colors[animal]
            label = f"{animal}" if animal not in legend_added else None  # Add label only once
            if animal not in legend_added:
                legend_added.add(animal)
        else:
            # In one plot, use the same base color for the same animal, but vary lightness for treatments
            base_color = animal_colors[animal]
            # Extract hue from the base color
            hue = float(base_color.split('(')[1].split(',')[0])  # Extract hue as a float
            lightness = 0.3 + 0.4 * list(treatments).index(treatment) / (len(treatments) - 1)  # Vary lightness
            color = f"hsl({int(hue)}, 50%, {int(lightness * 100)}%)"  # Convert hue to integer
            label = f"{animal} {treatment}"  # Show full label for each trace
        
        if sepPlot:
            row = list(treatments).index(treatment) + 1
            col = 1
        else:
            row, col = None, None
        
        # Add mean trace
        fig.add_trace(
            go.Scatter(
                x=time_vector,
                y=mean,
                mode='lines',
                name=label,
                line=dict(color=color),
                legendgroup=animal if sepPlot else f"{animal}_{treatment}",  # Link traces for the same animal in sepPlot
                showlegend=label is not None  # Show legend only for the first occurrence of each animal
            ),
            row=row, col=col
        )
        
        # Add error bands (upper and lower bounds)
        fig.add_trace(
            go.Scatter(
                x=time_vector,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                legendgroup=animal if sepPlot else f"{animal}_{treatment}"
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=time_vector,
                y=lower,
                mode='lines',
                line=dict(width=0),
                fillcolor=color.replace("hsl", "hsla").replace(")", ", 0.2)"),  # Set transparency = 0.2
                fill='tonexty',
                showlegend=False,
                legendgroup=animal if sepPlot else f"{animal}_{treatment}"
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=f"Averaged traces across animals at dB={dB_plot}",
        xaxis_title="Time (s)",
        yaxis_title=resp_col,
        height=450 * len(treatments) if sepPlot else 500, 
        legend=dict(y=0.5)
    )
    
    if sepPlot:
        # Update subplot titles and axes
        for i, treatment in enumerate(treatments):
            fig.update_xaxes(title_text="Time (s)", row=i+1, col=1)
            fig.update_yaxes(title_text=resp_col, row=i+1, col=1)
    
    fig.show()


def barplot_avgDFF_singleDB(df: pd.DataFrame, 
                            dB_plot: int = 80, 
                            resp_col: str = 'dFF_ROI_linFilt_butterFilt_peak', 
                            ctrl_treat: str = None, 
                            avgAnimal: bool = True, 
                            normalize: str = None, 
                            SEMbar: bool = True, 
                            include_0dB: bool = False) -> pd.DataFrame:
    """
    Plot barplots for averaged fluorescence peak response across animals (with individual animal data points) at specified sound intensity.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'dir', 'dB', 'treatment', and column for the response variable.
        dB_plot (int, optional): Sound intensity (in dB) for traces to be plotted. Defaults to 80.
        resp_col (str, optional): Name of the column containing the response variable. Defaults to 'dFF_ROI_linFilt_butterFilt_peak'.
        ctrl_treat (str, optional): Name of the control treatment in column 'treatment'. Normalize other treatments to it if normalization is performed.
                                    - None: The first treatment that appears in the dataframe is assumed to be control treatment (CTRL).
        avgAnimal (bool, optional): Whether to average peak response across animals or individual trials.
                                    - 'True': Average in two steps:
                                              First average peak responses within each animal, then average the mean across animals.
                                              Error bars represent SEM or SD across animals.
                                    - 'False': Average in one step:
                                               Average all individual trials from all animals.
                                               Error bars represent SEM or SD across trials.
                                    Defaults to 'True'.
        normalize (str, optional): Whether to normalize peak response to the CTRL (max) response (in percentage). 
                                   - None: No normalization is applied.
                                   - 'byTrial': For each animal, normalize all individual trials to the trial with max response.
                                   - 'byGroup': For each animal, calculate the mean for each treatment, then normalize these means to the mean of CTRL (max mean).
                                                Only applicable when 'avgAnimal' is 'True'.
                                   - 'trial2group': For each animal, calculate the mean of CTRL treatment (max mean), then normalize each single trial to this mean.
        SEMbar (bool, optional): If 'True', use standard error (SEM) for error bars. If 'False', use standard deviation (SD).
        include_0dB (bool, optional): Include 0 dB intensity even if it is not among the common 'dB' levels across all animals.

    Returns:
        df_stats (pd.DataFrame): Dataframe including statistics computed across animals ('dir's). 
                                 Including columns: 'treatment', 'count', 'mean', 'std', and 'sem'.
    
    Notes:
        - Data including only one treatment type may raise error.
        - If normalize == 'byGroup' or 'trial2group', the first treatment that appears in the dataframe is assumed to be CTRL.
        - If 'avgAnimal' is 'False', animals may have different weights due to varying trial counts for each animal.
        - If 'avgAnimal' is 'True', normalize == 'byGroup' or normalize == 'trial2group' will create the same barplot.
          The CTRL treatment can get a zero error bar.
        - normalize == 'byGroup' is only applicable when 'avgAnimal' is 'True'.
    """

    # Check whether required columns exist
    required_cols = ['dir', 'dB', 'treatment', resp_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataframe must contain the following columns: {required_cols}")
    
    # Check whether specified sound level exists
    if dB_plot not in df['dB'].unique():
        raise ValueError(f"{dB_plot} not found in the 'dB' column.")
    
    # Check whether 0 dB exists if it is to be included
    if include_0dB and 0 not in df['dB'].unique():
        raise ValueError("0 not found in the 'dB' column.")
    
    # Check whether control treatment exists if specified
    if ctrl_treat is not None and ctrl_treat not in df['treatment'].unique():
        raise ValueError(f"{ctrl_treat} not found in the 'treatment' column.")

    # Check whether normalization is applicable
    if normalize == 'byGroup' and not avgAnimal:
        warnings.warn("'byGroup' normalization requires 'avgAnimal=True' - skipping normalization")
        normalize = None

    # Filter data for the specified dB level
    if include_0dB:
        df_filtered = df[df['dB'].isin([dB_plot, 0])].copy().reset_index(drop=True)
    else:
        df_filtered = df[df['dB'] == dB_plot].copy().reset_index(drop=True)
    
    # Apply normalization if specified
    if normalize == 'byTrial':
        # Normalize each trial's response to the max response within the same animal
        df_filtered[resp_col] = df_filtered.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / x.max()) * 100)

    elif normalize == 'byGroup' and avgAnimal:
        # Get the first treatment that appears in the dataframe as control if not specified
        control_treatment = ctrl_treat if ctrl_treat is not None else df_filtered['treatment'].iloc[0]
        # Verify control treatment exists for all animals at 'dB_plot'
        missing_ctrl = set(df_filtered['dir']) - set(df_filtered[(df_filtered['treatment'] == control_treatment) & 
                                                                 (df_filtered['dB'] == dB_plot)]['dir'])
        if missing_ctrl:
            raise ValueError(f"Animals {missing_ctrl} missing control treatment '{control_treatment}' at {dB_plot} dB")
        # Calculate mean response for each animal/treatment/dB combination
        temp_means = df_filtered.groupby(['dir', 'treatment', 'dB'], sort=False)[resp_col].mean().reset_index()
        # Get control group means for each animal at 'dB_plot'
        control_means = temp_means[(temp_means['treatment'] == control_treatment) & (temp_means['dB'] == dB_plot)].set_index('dir')[resp_col]
        # Normalize all treatments to the control group mean for each animal
        temp_means[resp_col] = temp_means.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / control_means[x.name]) * 100)
        # Merge the normalized values back to the original dataframe
        df_filtered = df_filtered.drop(resp_col, axis=1).merge(temp_means, on=['dir', 'treatment', 'dB'])

    elif normalize == 'trial2group':
        # Get the first treatment that appears in the dataframe as control if not specified
        control_treatment = ctrl_treat if ctrl_treat is not None else df_filtered['treatment'].iloc[0]
        # Verify control treatment exists for all animals at 'dB_plot'
        missing_ctrl = set(df_filtered['dir']) - set(df_filtered[(df_filtered['treatment'] == control_treatment) & 
                                                                 (df_filtered['dB'] == dB_plot)]['dir'])
        if missing_ctrl:
            raise ValueError(f"Animals {missing_ctrl} missing control treatment '{control_treatment}' at {dB_plot} dB")
        # Calculate mean response for control group of each animal at 'dB_plot' only
        control_means = df_filtered[(df_filtered['treatment'] == control_treatment) & (df_filtered['dB'] == dB_plot)].groupby('dir')[resp_col].mean()
        # Normalize all trials to the corresponding animal's control group mean at 'dB_plot'
        df_filtered[resp_col] = df_filtered.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / control_means[x.name]) * 100)
    
    if avgAnimal:
        # Calculate mean response for each animal under each treatment and dB
        plot_data = df_filtered.groupby(['dir', 'treatment', 'dB'], sort=False)[resp_col].mean().reset_index()
    else:
        # Use all individual trials
        plot_data = df_filtered[['dir', 'treatment', 'dB', resp_col]].copy()
    
    # Calculate group statistics (mean and SEM/SD) for each treatment and dB
    df_stats = plot_data.groupby(['treatment', 'dB'], sort=False)[resp_col].agg(['count', 'mean', 'std', 'sem']).reset_index()
    
    # Initialize barplot
    fig, ax = plt.subplots(figsize=(5, 6)) if include_0dB else plt.subplots(figsize=(4, 6))
    palette = plt.cm.tab10.colors
    # palette = sns.color_palette("Set1")
    nTreat = len(df_stats['treatment'])
    
    # Plot bars with error
    if include_0dB:
        # Plot bars at both 'dB_plot' and 0 dB
        # Create positions for the bars
        bar_width = 0.2  # Width of bars
        gap_width = 0.05  # Spacing between dB_plot and 0 dB bars
        treatments = df_stats['treatment'].unique()
        x = np.arange(len(treatments)) * 0.8  # Spacing between treatment groups
        bar_positions = []
        for i, treat in enumerate(treatments):
            bar_positions.extend([x[i] - bar_width/2 - gap_width/2, x[i] + bar_width/2 + gap_width/2])
        
        # Store bar information for tick labeling
        bar_info = []
        for i, (treat, dB) in enumerate(product(treatments, [dB_plot, 0])):
            subset = df_stats[(df_stats['treatment'] == treat) & (df_stats['dB'] == dB)]
            if not subset.empty:
                error = subset['sem'].values[0] if SEMbar else subset['std'].values[0]
                ax.bar(bar_positions[i], subset['mean'].values[0], 
                       width=bar_width, yerr=error, capsize=5,
                       color=palette[i//2], alpha=0.5 if dB == 0 else 1)
                bar_info.append((bar_positions[i], treat, dB))
        
        # Set up two rows of tick labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f"{dB} dB" for _, _, dB in bar_info], fontsize=12)
        
        # Add treatment names as secondary labels
        for i, t in enumerate(treatments):
            # Position in middle of the two bars for this treatment
            treatment_xpos = x[i]
            ax.text(treatment_xpos, -0.08*ax.get_ylim()[1], t, 
                    ha='center', va='top', fontsize=12)
    
    else:
        # Only plot bars at specified 'dB_plot'
        error = df_stats['sem'] if SEMbar else df_stats['std']
        ax.bar(df_stats['treatment'], df_stats['mean'], 
               yerr=error, capsize=5, color=palette[:nTreat], width=0.3)
    
    # Add points representing the data of each animal
    for animal in plot_data['dir'].unique():
        animal_data = plot_data[plot_data['dir'] == animal]
        
        if include_0dB:
            for t in treatments:
                for j, dB in enumerate([dB_plot, 0]):
                    subset = animal_data[(animal_data['treatment'] == t) & (animal_data['dB'] == dB)]
                    if not subset.empty:
                        # Position points with the same offset as bars
                        x_offset = -bar_width/2 - gap_width/2 if j == 0 else bar_width/2 + gap_width/2
                        x_pos = x[treatments.tolist().index(t)] + x_offset
                        y_vals = subset[resp_col].values
                        for y in y_vals:
                            x_jitter = x_pos + np.random.normal(0, 0.015)
                            point_size = 40 if not avgAnimal else 60
                            ax.scatter(x_jitter, y, facecolors='none', edgecolors='k', s=point_size, linewidth=1.5)
        else:
            # Create a dictionary mapping treatments to x-positions
            treatment_pos = {t: i for i, t in enumerate(df_stats['treatment'])}
            
            # If there are exactly two treatments, add offsets to dots
            if nTreat==2:
                offsets = [0.3 if t == df_stats['treatment'][0] else -0.3 for t in animal_data['treatment']]
                x_pos = [treatment_pos[t] + offset for t, offset in zip(animal_data['treatment'], offsets)]
            else:
                x_pos = [treatment_pos[t] for t in animal_data['treatment']]
            
            y_vals = animal_data[resp_col].values
            
            # Add connecting lines for paired data when averaging by animal
            if avgAnimal:
                ax.plot(x_pos, y_vals, color='gray', alpha=0.8, linestyle='--', linewidth=1.5)
            
            # If each dot represents a single trial, plot them with jitters 
            for x, y, t in zip(x_pos, y_vals, animal_data['treatment']):
                x_jitter = x + (np.random.normal(0, 0.015) if not avgAnimal else 0)
                edge_color = palette[treatment_pos[t]] if nTreat==2 else 'k'
                point_size = 40 if not avgAnimal else 60
                ax.scatter(x_jitter, y, facecolors='none', edgecolors=edge_color,
                           s=point_size, linewidth=2 if avgAnimal else 1.5)
    
    # Add labels and title
    if not include_0dB:
        ax.set_xlabel('Treatment', fontsize=12)
    ax.set_ylabel(f'Normalized {resp_col} Response (%)' if normalize else f'{resp_col} Response', fontsize=12)
    ax.set_title(f'Fluorescence Peak Response at {dB_plot} and 0 dB' if include_0dB 
                 else f'Fluorescence Peak Response at {dB_plot} dB', fontsize=14, pad=20)
    ax.tick_params(axis='both', labelsize=12)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    return df_stats


def plot_avgDFF_acrossAnimal(df: pd.DataFrame, 
                             measure_col: str = 'dB', 
                             resp_col: str = 'dFF_ROI_linFilt_butterFilt_peak', 
                             ctrl_treat: str = None, 
                             avgAnimal: bool = True, 
                             normalize: str = None, 
                             SEMbar: bool = True, 
                             plotAvg_reLevel: str = 'line', 
                             include_0dB: bool = False, 
                             **kwargs) -> pd.DataFrame:
    """
    Plot barplots or lineplots with error bars for fluorescence peak response averaged across animals re sound intensities.
    Create a new dataframe including the mean, SD, and SEM of peak response for each sound level.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'dir', 'treatment', 'dB', and column for the response variable.
        measure_col (str, optional): Column name for the independent variable. Can be sound intensity or frequency.
                                     Defaults to 'dB' (sound intensity).
        resp_col (str, optional): Column name for the response variable. Defaults to 'dFF_ROI_linFilt_butterFilt_peak'.
        ctrl_treat (str, optional): Name of the control treatment in column 'treatment'. Normalize other treatments to it if normalization is performed.
                                    - None: The first treatment that appears in the dataframe is assumed to be control treatment (CTRL).
        avgAnimal (bool, optional): Whether to average peak response across animals or individual trials.
                                    - 'True': Average in two steps:
                                              First average peak responses within each animal, then average the mean across animals.
                                              Error bars represent SEM or SD across animals.
                                    - 'False': Average in one step:
                                               Average peak responses of all individual trials from all animals.
                                               Error bars represent SEM or SD across trials.
        normalize (str, optional): Whether to normalize peak response to the max response (in percentage). 
                                   - 'byGroup': For each animal, calculate the mean for each sound level, then normalize these means to the max mean in CTRL.
                                                Only applicable when 'avgAnimal' is 'True'.
                                   - 'byTrial': For each animal, normalize all individual trials to the trial with the max response.
                                   - None: No normalization is applied.
        SEMbar (bool, optional): If 'True', use standard error (SEM) for error bars. If 'False', use standard deviation (SD).
        plotAvg_reLevel (str, optional): Whether to plot averaged response across animals re sound level.
                                         - 'line': Plot treatments in different colors in one lineplot.
                                         - 'bar': Plot treatments in multiple barplots sharing the same Y-axis scale.
                                         - None: Only return the statistics dataframe.
        include_0dB (bool, optional): Include 0 dB intensity even if it is not among the common 'dB' levels across all animals.
        **kwargs: Optional arguments that will override default.
            example: capsize (float, optional): Error bar cap size. Defaults to no caps.

    Returns:
        df_avg (pd.DataFrame): Dataframe including statistics computed across animals ('dir's).
                               Including columns: 'treatment', 'dB', 'count', 'mean', 'std', and 'sem'.
    
    Notes:
        - If 'avgAnimal' is 'False', animals may have different weights due to varying trial counts for each animal.
        - normalize='byGroup' is only applicable when 'avgAnimal' is 'True'.
    """

    # Check whether required columns exist
    required_col = ['dir', 'treatment', measure_col, resp_col]
    if not all(col in df.columns for col in required_col):
        raise ValueError(f"DataFrame must contain the following columns: {required_col}")
    
    # Check whether 0 dB exists if it is to be included
    if include_0dB and 0 not in df['dB'].unique():
        raise ValueError("0 not found in the 'dB' column.")

    # Check whether normalization is applicable
    if normalize == 'byGroup' and not avgAnimal:
        warnings.warn("'byGroup' normalization requires 'avgAnimal=True' - skipping normalization")
    
    # Group by 'dir' and 'treatment' and extract unique 'dB' values
    dB_lists = list(df.groupby(['dir', 'treatment'], sort=False)[measure_col].unique())

    # Find the common 'dB' levels across all animals
    common_dB = set(dB_lists[0])    # Start with the first list
    for item in dB_lists[1:]:    # Iterate over the remaining lists
        common_dB.intersection_update(item)
    if include_0dB and 0 not in common_dB:
        common_dB.add(0)  # Add 0 dB if not already present
    if not common_dB:
        raise ValueError(f"No common '{measure_col}' values found across all animals and treatments.")
    common_dB = sorted(common_dB)   # Sort 'dB' values in ascending order

    # Filter rows where 'dB' is in the common_dB list
    df_filtered = df[df[measure_col].isin(common_dB)].reset_index(drop=True)

    if normalize == 'byTrial':
        # Normalize each trial's response to the max response within the same animal
        df_filtered[resp_col] = df_filtered.groupby(['dir'], sort=False)[resp_col].transform(lambda x: (x / x.max()) * 100)
        print("Normalized on the trial basis.")

    if avgAnimal:
        # Calculate the mean within each dir for each treatment/dB combination
        df_grouped = df_filtered.groupby(['dir', 'treatment', measure_col], as_index=False, sort=False)[resp_col].mean().reset_index(drop=True)
        if normalize == 'byGroup':
            # Normalize mean response of each treatment/dB combination group to the max mean response in CTRL within the same animal
            # Get the first treatment that appears in the dataframe as control if not specified
            control_treatment = ctrl_treat if ctrl_treat is not None else df_filtered['treatment'].iloc[0]
            # Verify control treatment exists for all animals
            missing_ctrl = set(df_filtered['dir']) - set(df_filtered[df_filtered['treatment'] == control_treatment]['dir'])
            if missing_ctrl:
                raise ValueError(f"Animals {missing_ctrl} missing control treatment '{control_treatment}'")
            # Calculate mean response for each animal/treatment/dB combination
            temp_means = df_filtered.groupby(['dir', 'treatment', measure_col], sort=False)[resp_col].mean().reset_index()
            # Get max mean response in control treatment for each animal
            control_max = temp_means[temp_means['treatment'] == control_treatment].groupby('dir', sort=False)[resp_col].max()
            # Normalize all treatments to the control group's max mean for each animal
            temp_means[resp_col] = temp_means.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / control_max[x.name]) * 100)
            # Replace the grouped means with normalized values
            df_grouped = temp_means
            print('Normalized to max response in control treatment on the treatment/dB combination group basis.')
    else:
        # Maintain the unaveraged original data of each individual trial
        df_grouped = df_filtered.loc[:, ['treatment', measure_col, resp_col]]

    # Calculate the mean, standard deviation (SD), and standard error (SEM) across dirs for each treatment/dB combination
    agg_dict = {
        f'count_{resp_col}': (resp_col, 'count'), 
        f'mean_{resp_col}': (resp_col, 'mean'), 
        f'std_{resp_col}': (resp_col, 'std')
    }
    df_avg = df_grouped.groupby(['treatment', measure_col], as_index=False, sort=False).agg(**agg_dict)
    df_avg[f'sem_{resp_col}'] = df_avg[f'std_{resp_col}'] / np.sqrt(df_avg[f'count_{resp_col}'])

    # Fill NaN standard deviations or errors with 0 (if there's only one dir for a treatment/dB combination)
    df_avg[f'std_{resp_col}'] = df_avg[f'std_{resp_col}'].fillna(0)
    df_avg[f'sem_{resp_col}'] = df_avg[f'sem_{resp_col}'].fillna(0)

    # Sort the DataFrame by 'dB' in ascending order while keeping the original order of 'treatment'
    # Otherwise, x-ticks cannot match 'dB' values in the correct order
    df_avg = pd.concat([group.sort_values(by=measure_col) for _, group in df_avg.groupby('treatment', sort=False)]).reset_index(drop=True)

    if plotAvg_reLevel == 'line':
        # Plot lineplot
        plt.figure(figsize=(8,6))
        for treatment, group in df_avg.groupby('treatment', sort=False):
            plt.errorbar(group[measure_col], group[f'mean_{resp_col}'], 
                         yerr=group[f'sem_{resp_col}'] if SEMbar else group[f'std_{resp_col}'], 
                         label=treatment, marker='o', **kwargs)
        plt.xlabel("Sound Intensity (dB SPL)", fontsize=12, labelpad=10)
        plt.ylabel(f"Normalized {resp_col} Response (%)" if normalize else f"{resp_col} Response", fontsize=12, labelpad=10)
        plt.legend(title="Treatment", title_fontsize=12, fontsize=12)
        plt.title("Response by Sound Intensity and Treatment", fontsize=14, pad=20)
        plt.tick_params(axis='both', labelsize=12)

        # Remove the top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    elif plotAvg_reLevel == 'bar':
        # Plot barplots
        g = sns.FacetGrid(df_avg, col='treatment', sharey=True, height=4)

        # Define the function to plot bars with error bars
        def plot_bars_with_error(data, measure_col, **kwargs):
            x = np.arange(len(data[measure_col]))
            yerr = data[f'sem_{resp_col}'] if SEMbar else data[f'std_{resp_col}']
            plt.bar(x, data[f'mean_{resp_col}'], yerr=yerr, **kwargs)
            plt.xticks(x, data[measure_col])

        # Map the plotting function to the FacetGrid
        g.map_dataframe(plot_bars_with_error, measure_col=measure_col, **kwargs)

        # Add labels and titles
        g.set_axis_labels("Sound Intensity (dB SPL)", f"{resp_col} Response" if normalize is None else f"{resp_col} Response (%)")
        g.set_titles(col_template="{col_name}")
        plt.show()

    else:
        raise ValueError("'plotAvg_reLevel' must be 'line' or 'bar'.")

    return df_avg


def plotDFFSeriesMask(imgSeries: np.ndarray, 
                      Xcoor: float, Ycoor: float, 
                      width: float, height: float, 
                      Xshift_step: float = 3, 
                      Yshift_step: float = 2, 
                      shift_direct: float = None, 
                      dFResp: bool = False, 
                      displayContour: bool = True, 
                      baseline: bool = True, 
                      stimStart: float = 3.0, 
                      Yaxis_range: tuple[float,float] = None, 
                      Xshift_Num: int = None, 
                      Yshift_Num: int = None, 
                      gif_frameDur: float = 500, 
                      gif_name: str = "response_series.gif", 
                      **kwargs):
    """
    Creates an interactive fluorescence response plot within a shifting mask.
    Supports manual movement via sliders and optional GIF export.

    Args:
        imgSeries (array): 4D or 3D signal array of shape (traceNumber, Y, X, frame) or (Y, X, frame).
        Xcoor (float): X-coordinate of the top-left vertex of the binary mask at its initial position.
        Ycoor (float): Y-coordinate of the top-left vertex of the binary mask at its initial position.
        width (float): Distance between left and right sides of the binary mask.
        height (float): Distance between top and bottom sides of the binary mask.
        Xshift_step (float, optional): Step size along the X-axis. 
        Yshift_step (float, optional): Step size along the Y-axis.
        shift_direct (float, optional): Angle (in degrees) that determines mask movement direction relative to the axis.
                                        - `None`: X-slider moves the mask horizontally (along X-axis) and Y-slider moves it vertically (along Y-axis).
                                        - Positive angles: Moving either slider shifts the mask diagonally in specified direction (clockwise rotation from X-axis).
                                        - Negative angles: Same as positive but counter-clockwise rotation.
                                        - 0 degree: Equivalent to `None` (pure X/Y axis movement).
                                        Defaults to `None`.
        dFResp (bool, optional): Whether to calculate dF (`True`) or dFF (`False`).
        displayContour (bool, optional): Whether to show mask as contour (`True`) or shaded region (`False`).
        baseline (bool, optional): Whether to move the mask on a spatial baseline fluorescence heatmap.
                                   - `True`: Background heatmap indicates spatial baseline fluorescence.
                                   - `False`: Background heatmap indicates spatial dFF response.
                                   Defaults to `True`.
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, Y-axis is auto-scaled.
        Xshift_Num (int, optional): Number of steps along the X-axis for GIF movement.
        Yshift_Num (int, optional): Number of steps along the Y-axis for GIF movement.
        gif_frameDur (float, optional): Frame duration (milliseconds) for GIF export.
        gif_name (str, optional): Filename for saved GIF.
        **kwargs: Optional keyword arguments.

    Notes:
        - Before calling this function in Jupyter Notebook, use magic commands `%matplotlib widget` to set interactive backend.
        - After calling, use `%matplotlib inline` to return to inline backend and render following plots as static images.
        - Negative `Xshift_step` and `Yshift_step` move the mask in opposite directions in GIF.
        - Arguments `Xshift_Num`, `Yshift_Num`, `gif_frameDur`, `gif_name` are only necessary while generating and saving the GIF.
        - `Xshift_Num` and `Yshift_Num` must be positive, 0, or None. If both positive, they must be equal.
        - Move the mask in any directions in GIF by changing the ratio `Xshift_step / Yshift_step`.
    """

    # Raise error for image signal of improper dimensions
    if imgSeries.ndim not in (3, 4):
        raise ValueError("Image signal array must be 3D or 4D.")

    # Raise error for invalid shifting directions
    if shift_direct is not None and (shift_direct <= -90 or shift_direct >= 90):
        raise ValueError("`shift_direct` must be between -90 and 90 degrees (exclusive).")
    
    # Create the time vector
    t = signalProcess.getTimeVec(imgSeries.shape[-1], **kwargs)

    # Generate a binary mask at a self-defined initial position
    mask_init = imgProcess.getSquareMask(Xcoor, Ycoor, width, height, **kwargs)

    # Compute the averaged response within the initial mask
    # Adapt to both 3D and 4D signal arrays
    avgImg_init = imgSeries[..., mask_init['mask'] == 1, :].mean(axis=-2)

    # Calculate dFF and dF response at initial position
    dFF_init, dF_init, _ = signalProcess.dFFcalc(avgImg_init, **kwargs)

    # Calculate error bars for 4D signal arrays
    if imgSeries.ndim == 4:
        dFF_init, dFFpsem_init, dFFmsem_init = signalProcess.meanPlusMinusSem(dFF_init)
        dF_init, dFpsem_init, dFmsem_init = signalProcess.meanPlusMinusSem(dF_init)

    # Set plots backend to widget mode
    # Render plots as an interactive window rather than PNG formats
    # %matplotlib widget

    # Initialize figure with two subplots (response curve + baseline heatmap)
    fig, (ax_curve, ax_img) = plt.subplots(1, 2, figsize=(12, 5))

    # Adjust the main plots to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.3)

    # Plot dF or dFF response at the initial ROI
    if dFResp:
        line, = ax_curve.plot(t, dF_init, lw=2)
        semBar = ax_curve.fill_between(t, dFpsem_init, dFmsem_init, color='r', alpha=0.2) if imgSeries.ndim == 4 else None
        ax_curve.set_ylabel('Fluorescence Intensity (dF)')
    else:
        line, = ax_curve.plot(t, dFF_init, lw=2)
        semBar = ax_curve.fill_between(t, dFFpsem_init, dFFmsem_init, color='r', alpha=0.2) if imgSeries.ndim == 4 else None
        ax_curve.set_ylabel('Fluorescence Intensity (dFF)')
    
    ax_curve.set_xlabel('Time (s)')
    ax_curve.set_title('Fluorescence Traces: Position Slider')
    ax_curve.axvline(x=stimStart, color='k', linestyle='--')

    # Transform signal array into 3D if it is initially 4D
    avgImg_map = imgSeries.mean(axis=0) if imgSeries.ndim == 4 else imgSeries

    # Load heatmap of baseline or dFF response fluorescence as static background
    ax_img.imshow(imgProcess.calcSpatialBaseFluo(avgImg_map, **kwargs) if baseline 
                  else imgProcess.calcSpatialDFFresp(avgImg_map, **kwargs), cmap='jet')

    # Show initial mask overlay against baseline heatmap
    if displayContour:
        # Display mask contour in black
        mask_overlay, = ax_img.plot(mask_init['ROIcontour'][:, 0], mask_init['ROIcontour'][:, 1], color='w', linewidth=2)
    else:
        # Display mask as a translucent shade
        mask_overlay = ax_img.imshow(mask_init['mask'], cmap='gray', alpha=0.5)
    ax_img.set_title('Baseline Fluorescence Heatmap')

    # Set X- and Y-slider
    ax_x_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_y_slider = plt.axes([0.1, 0.25, 0.0225, 0.63])
    X_slider = Slider(ax_x_slider, "X position", valmin=0, valmax=imgSeries.shape[-2], 
                      valinit=Xcoor, valstep=Xshift_step)
    Y_slider = Slider(ax_y_slider, "Y position", valmin=0, valmax=imgSeries.shape[-3], 
                      valinit=Ycoor, valstep=Yshift_step, orientation="vertical")

    # Update function for X- and Y-slider
    def update(event):
        # Treat `semBar`, `mask_overlay` as a global variable and modify it from the outer function
        nonlocal semBar, mask_overlay

        # Generate new binary masks with current slider values
        # Move the mask diagonally if specified 
        tan_theta_shift = math.tan(math.radians(shift_direct)) if shift_direct is not None else 0
        Xcoor_new = X_slider.val - (Y_slider.val - Ycoor) * tan_theta_shift
        Ycoor_new = Y_slider.val + (X_slider.val - Xcoor) * tan_theta_shift
        mask = imgProcess.getSquareMask(Xcoor_new, Ycoor_new, width, height, **kwargs)
        
        # Ensure contour does not exceed image boundaries
        if np.any(mask['ROIcontour'][:,0] < 0) or np.any(mask['ROIcontour'][:,0] > imgSeries.shape[-2]):
            warnings.warn("Contour exceeds X boundaries.")
        if np.any(mask['ROIcontour'][:,1] < 0) or np.any(mask['ROIcontour'][:,1] > imgSeries.shape[-3]):
            warnings.warn("Contour exceeds Y boundaries.")
        
        # Compute dFF and dF response within new ROIs
        avgImg = imgSeries[..., mask['mask'] == 1, :].mean(axis=-2)
        dFF, dF, _ = signalProcess.dFFcalc(avgImg, **kwargs)
        
        # Calculate error bars for 4D signal arrays
        if imgSeries.ndim == 4:
            dFF, dFFpsem, dFFmsem = signalProcess.meanPlusMinusSem(dFF)
            dF, dFpsem, dFmsem = signalProcess.meanPlusMinusSem(dF)

        # Update the response curves
        line.set_ydata(dF if dFResp else dFF)
        
        # Replace old error bars with new ones for 4D arrays
        if semBar:
            semBar.remove()
            semBar = ax_curve.fill_between(
                t, 
                dFpsem if dFResp else dFFpsem, 
                dFmsem if dFResp else dFFmsem, 
                color='r', 
                alpha=0.2
            )

        if Yaxis_range is None:
            # Autoscale Y-axis
            ax_curve.relim()
            ax_curve.autoscale_view()
        else:
            # Manually set Y-axis range
            ax_curve.set_ylim(Yaxis_range)
        
        # Update the mask display with new shifting
        if displayContour:
            mask_overlay.set_data(mask['ROIcontour'][:, 0], mask['ROIcontour'][:, 1])
        else:
            mask_overlay.set_data(mask['mask'])
        
        fig.canvas.draw_idle()

    # Connect sliders to update function
    # X_slider.on_changed(lambda val: update(val))
    # Y_slider.on_changed(lambda val: update(val))
    # Use `motion_notify_event` to reduce computational burden and avoid sliders getting stuck
    fig.canvas.mpl_connect("motion_notify_event", update)

    if not (Xshift_Num in [None, 0] and Yshift_Num in [None, 0]):
        # Raise error for negative `Xshift_Num` or `Yshift_Num`
        if (Xshift_Num is not None and Xshift_Num < 0) or (Yshift_Num is not None and Yshift_Num < 0):
            raise ValueError("`Xshift_Num` and `Yshift_Num` must be positive, 0, or None.")
        
        # For movements not parallel to either axis, raise error for unequal steps along X and Y
        if Xshift_Num and Yshift_Num and Xshift_Num != Yshift_Num:
            raise ValueError("`Xshift_Num` and `Yshift_Num` must be equal if both are set.")
        
        # Add a button to generate GIF
        ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(ax_button, 'Save GIF')

        def save_gif_callback(event):
            # Generate the range of X and Y positions for the GIF
            X_pos = np.arange(Xcoor, Xcoor + Xshift_Num * Xshift_step, Xshift_step) if Xshift_Num else None
            Y_pos = np.arange(Ycoor, Ycoor + Yshift_Num * Yshift_step, Yshift_step) if Yshift_Num else None
            
            # Define function to update the plot for each frame of the GIF
            def animate(frame):
                # Ensure semBar refers to the outer scope
                nonlocal semBar

                # Generate binary masks corresponding to frame counts
                x = X_pos[frame] if X_pos is not None else Xcoor
                y = Y_pos[frame] if Y_pos is not None else Ycoor
                mask = imgProcess.getSquareMask(x, y, width, height, **kwargs)

                # Compute dFF and dF response within new ROIs
                avgImg = imgSeries[..., mask['mask'] == 1, :].mean(axis=-2)
                dFF, dF, _ = signalProcess.dFFcalc(avgImg, **kwargs)

                # Calculate error bars for 4D signal arrays
                if imgSeries.ndim == 4:
                    dFF, dFFpsem, dFFmsem = signalProcess.meanPlusMinusSem(dFF)
                    dF, dFpsem, dFmsem = signalProcess.meanPlusMinusSem(dF)

                # Update the response curves for each frame
                ydata = dF if dFResp else dFF
                line.set_data(t, ydata)
                ax_curve.set_title(f'Fluorescence Traces: X={x}, Y={y}')
                
                if Yaxis_range is None:
                    # Autoscale Y-axis
                    ax_curve.relim()
                    ax_curve.autoscale_view()
                else:
                    # Manually set Y-axis range
                    ax_curve.set_ylim(Yaxis_range)

                # Update the error bars
                if semBar:
                    semBar.remove()
                    semBar = ax_curve.fill_between(
                        t, 
                        dFpsem if dFResp else dFFpsem, 
                        dFmsem if dFResp else dFFmsem, 
                        color='r', 
                        alpha=0.2
                    )

                # Update the mask display with new shifting
                if displayContour:
                    mask_overlay.set_data(mask['ROIcontour'][:, 0], mask['ROIcontour'][:, 1])
                else:
                    mask_overlay.set_data(mask['mask'])

                return line, mask_overlay, semBar

            # Create the animation
            ani = FuncAnimation(fig, animate, frames = Xshift_Num if Xshift_Num else Yshift_Num, 
                                blit=False, interval=gif_frameDur)

            # Save the animation as a GIF
            gif_fps = max(1, round(1000 / gif_frameDur))
            ani.save(gif_name, writer=PillowWriter(fps=gif_fps))
            print(f"GIF saved as {gif_name}")

        # Connect the button to the `save_gif_callback` function
        button.on_clicked(save_gif_callback)
    
    plt.show()

    # Automatically switch back to inline backend
    # Render the following plots as static images instead of interactive widgets
    # %matplotlib inline


def plot_ROI_trace(roi_avg: np.ndarray, normalize: bool = True, plot_trace: bool = True, **kwargs) -> np.ndarray:
    """
    Plot fluorescence traces within a series of sweeping ROIs before and after subtracting linear fit.

    Args:
        roi_avg (np.ndarray): 2D or 3D array of ROI average fluorescence traces.
                              Shape should be [maskNumber, frame] or [traceNumber, maskNumber, frame].
        normalize (bool, optional): Whether to normalize each trace to the mean fluorescence over time.
        plot_trace (bool, optional): Whether to plot traces before and after removal of linear fit.
        **kwargs: Optional keyword arguments.

    Returns:
        roi_avg_filt (np.ndarray): ROI fluorescence traces after removal of linear fit (same shape as input array 'roi_avg').
    """

    # Check the dimension of input array
    if roi_avg.ndim not in (2, 3):
        raise ValueError("Trace array must be 2D or 3D.")
    
    # Get the time vector
    t = signalProcess.getTimeVec(roi_avg.shape[-1], **kwargs)

    # Filter to center around 0 fluorescence intensity
    if not normalize:
        # Directly subtract linear fit from the raw fluorescence traces
        roi_avg_filt = signalProcess.subtractLinFit(t, roi_avg, offset=False)[0]

    else:
        # Normalize each trace to the mean fluorescence across all time frames
        roi_avg_filt = signalProcess.subtractLinFit(t, roi_avg, offset=False)[0] / np.mean(roi_avg, axis=-1, keepdims=True)

    if plot_trace:
        fig, ax = plt.subplots(1, 2, figsize=(16,8))

        if roi_avg.ndim == 2:
            ax[0].plot(t, roi_avg.T)
            ax[1].plot(t, roi_avg_filt.T)
        else:
            ax[0].plot(t, np.mean(roi_avg, axis=0).T)
            ax[1].plot(t, np.mean(roi_avg_filt, axis=0).T)

        ax[0].set_title('Raw Traces re ROIs', fontsize=14)
        ax[0].set_ylabel('raw F', fontsize=12)
        
        if not normalize:
            ax[1].set_title('Traces re ROIs After Subtracting Linear Fit', fontsize=14)
            ax[1].set_ylabel('raw F - linear fit', fontsize=12)
        else:
            ax[1].set_title('Normalized Traces re ROIs After Subtracting Linear Fit', fontsize=14)
            ax[1].set_ylabel('normalized F - linear fit', fontsize=12)

        for axn in ax:
            axn.set_xlabel('time (s)', fontsize=12)

    return roi_avg_filt


def plot_hierarchical_cluster(linkage_matrix: np.ndarray, 
                              roi_trace: np.ndarray, 
                              stimStart: float = 3.0, 
                              timeVector: np.ndarray = None, 
                              n_clusters: int = None, 
                              Yaxis_label: str = 'ROI', 
                              **kwargs) -> tuple:
    """
    Visualizes hierarchical clustering results with dendrogram and sorted ROI traces.
    
    This function creates a two-panel figure showing:
    1. A dendrogram displaying the hierarchical clustering structure.
    2. A heatmap of ROI traces sorted by cluster similarity.
    
    Args:
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix of shape [maskNumber-1, 4].
        roi_trace (np.ndarray): 2D or 3D array of ROI fluorescence traces.
                                Shape should be [maskNumber, frame] or [traceNumber, maskNumber, frame].
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        timeVector (np.ndarray, optional): 1D array of time points corresponding to frames.
                                           If provided, overrides time vector generated from roi_trace shape.
        n_clusters (int, optional): Desired number of clusters. If provided, overrides 'color_threshold' logic. Defaults to None.
        Yaxis_label (str, optional): Label for the Y-axis of the dendrogram and trace heatmap. Defaults to 'ROI'.
        **kwargs: Optional keyword arguments.
            - Example: Additional arguments passed to dendrogram() such as:
                       color_threshold (float, optional): Distance threshold for cluster coloring.
                                                          Clusters with distance greater than the threshold are in different colors.
                                                          eg. color_threshold=0.5, color_threshold=0.5*max(linkage_matrix[:,2]), etc.
                                                          Defaults to '0.7*max(linkage_matrix[:,2])'.
    
    Returns:
        tuple: Contains:
               - color_map (dict): Mapping of cluster IDs to colors.
               - leaves_color_list (np.ndarray): Color assignments for each leaf.
               - leaf_order (np.ndarray): ROI indices in dendrogram order.
    
    Notes:
        - Uses matplotlib's default color cycle for cluster coloring.
    """
    
    # Check the shape of input array
    if roi_trace.ndim not in (2, 3):
        raise ValueError("Trace array must be 2D or 3D.")
    
    # Average across different trials (repetitions) for 3D array
    roi_trace = np.mean(roi_trace, axis=0) if roi_trace.ndim == 3 else roi_trace
    
    # Get the time vector
    t = timeVector if timeVector is not None else signalProcess.getTimeVec(roi_trace.shape[-1], **kwargs)

    # Remove 'color_threshold' from kwargs if exists
    color_threshold = kwargs.pop('color_threshold', 0.7*max(linkage_matrix[:,2]))

    # Override 'color_threshold' if 'n_clusters' is provided
    if n_clusters is not None:
        # Sort cluster distances in descending order
        cluster_dists = np.sort(linkage_matrix[:, 2])[::-1]
        color_threshold = cluster_dists[n_clusters-2] if n_clusters >= 2 else 0

    # Generate the dendrogram without plotting to get metadata
    dendro = dendrogram(linkage_matrix, no_plot=True, color_threshold=color_threshold, **kwargs)
    
    # Extract leaf order and colors
    leaf_order = dendro['leaves']
    leaves_color_list = dendro['leaves_color_list']
    
    # Get the default matplotlib color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Get the unique colors and create a cluster-to-color mapping
    unique_colors = np.unique(leaves_color_list)

    # Create a mapping from cluster number to color
    color_map_legend = {i + 1: color for i, color in enumerate(unique_colors)}
    color_map = {str(val):{'cluster_id':k, 'cluster_color':color_cycle[k]} for k,val in color_map_legend.items()}
    
    # Sort ROI traces by dendrogram order
    sorted_roi_trace = roi_trace[leaf_order]
    
    # Create figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot dendrogram (left panel)
    dendrogram(
        linkage_matrix,
        orientation='left',
        leaf_font_size=5,
        no_labels=True,
        color_threshold=color_threshold,
        ax=ax[0], 
        **kwargs
    )
    ax[0].set_title('Hierarchical Clustering Dendrogram', fontsize=14)
    ax[0].set_xlabel('Distance', fontsize=12)
    ax[0].set_ylabel(f'{Yaxis_label} Index', fontsize=12)
    
    # Add cluster legend
    legend_handles = [mpatches.Patch(color=color_map_legend[i], 
                                     label=f'Cluster {i}') for i in color_map_legend]
    ax[0].legend(handles=legend_handles, loc='upper left', fontsize=12)
    
    # Plot sorted ROI responses (right panel)
    roi_trace_img = ax[1].imshow(
        sorted_roi_trace, 
        aspect='auto', 
        extent=[t[0], t[-1], 0, sorted_roi_trace.shape[0]], 
        origin='lower', 
        cmap='coolwarm'
    )
    
    ax[1].axvline(x=stimStart, color='k', linestyle='--')
    ax[1].set_title(f'{Yaxis_label} Sorted by Clustering Similarity', fontsize=14)
    ax[1].set_xlabel('time (s)', fontsize=12)
    ax[1].set_ylabel(f'{Yaxis_label} Index', fontsize=12)

    # Add colorbar
    colorbar = fig.colorbar(roi_trace_img, ax=ax[1])
    colorbar.set_label('Fluorescence Intensity', fontsize=12)
    # ax[0].legend([1,2,3],loc='upper left')
    
    return color_map, np.array(leaves_color_list), np.array(leaf_order)


def plot_cluster_roi(img_series: np.ndarray, 
                     masks: np.ndarray, 
                     roi_trace: np.ndarray, 
                     color_map: dict, 
                     leaves_color_list: np.ndarray, 
                     leaf_order: np.ndarray, 
                     background_contrast: float = 0.2, 
                     stimStart: float = 3, 
                     Yaxis_range: tuple[float,float] = None, 
                     plot_traces: bool = False, 
                     alpha_traces: float = 0.2, 
                     plot_errBar: bool = False, 
                     alpha_errBar: float = 0.2, 
                     **kwargs) -> dict[str, np.ndarray]:
    """
    Plot clustered ROIs and their corresponding fluorescence traces for each cluster.
    
    For each cluster, creates a two-panel figure showing:
    1. Spatial map of ROIs belonging to each cluster overlaid on wide-field image.
    2. The average fluorescence trace for ROIs in the corresponding cluster.

    Args:
        img_series (np.ndarray): 3D or 4D image array of shape [Y, X, frame] or [traceNumber, Y, X, frame].
        masks (np.ndarray): 3D array of binary masks (ROIs) of shape [maskNumber, Y, X].
        roi_trace (np.ndarray): 2D or 3D array of ROI fluorescence traces.
                                Shape should be [maskNumber, frame] or [traceNumber, maskNumber, frame].
        color_map (dict): Mapping of cluster IDs to colors.
        leaves_color_list (np.ndarray): Color assignments for each leaf.
        leaf_order (np.ndarray): ROI indices in dendrogram order.
        background_contrast (float, optional): Controls ROI visibility against background wide-field image.
                                               ROI:background = (1 + background_contrast):background_contrast
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.
        plot_traces (bool, optional): Whether to plot individual traces.
        alpha_traces (float, optional): Transparency for individual traces. Defaults to 0.2.
        plot_errBar (bool, optional): Whether to plot SEM as shading (error bars).
        alpha_errBar (float, optional): Transparency for error bars. Defaults to 0.2.
        **kwargs: Optional keyword arguments.

    Returns:
        cluster_trace (dict): Dictionary mapping each cluster ID to the corresponding fluorescence trace
                              (one lower dimension than input signal 'roi_trace').
                              Example: 'CLUSTER: 1': np.ndarray

    Notes:
        - Individual traces or error bars can be plotted only when 'roi_trace' is 3D (multi-trial data).
    """

    # Check the shape of input arrays
    if img_series.ndim not in (3, 4):
        raise ValueError("Image array must be 3D or 4D.")
    if masks.ndim != 3:
        raise ValueError("Mask array must be 3D.")
    if roi_trace.ndim not in (2, 3):
        raise ValueError("Trace array must be 2D or 3D.")

    # Ensure that the shapes of image and trace arrays match
    if (img_series.ndim == 3 and roi_trace.ndim == 3) or (img_series.ndim == 4 and roi_trace.ndim == 2):
        raise ValueError("'img_series' and 'roi_trace' do not match in shape.")

    # Convert 'img_series' to 3D by averaging across trials
    img_series = np.mean(img_series, axis=0) if img_series.ndim == 4 else img_series

    # Get the time vector
    t = signalProcess.getTimeVec(img_series.shape[-1], **kwargs)

    # Initialize a dictionary to store traces of each cluster
    cluster_trace = {}
    
    for cluster_color, cluster_d in color_map.items():
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        cluster_name = f"CLUSTER: {cluster_d['cluster_id']}"

        # Display corresponding ROIs against wide-field image by adding contrast
        ax[0].imshow((masks[leaf_order[leaves_color_list==cluster_color]].sum(axis=0) + background_contrast)*img_series.mean(axis=-1))

        # Average fluorescence traces across corresponding ROIs
        roi_trace_cluster = roi_trace[:, leaf_order[leaves_color_list==cluster_color]].mean(axis=-2) if roi_trace.ndim == 3 \
                            else roi_trace[leaf_order[leaves_color_list==cluster_color]].mean(axis=-2)

        # Add fluorescence traces into dictionary
        cluster_trace[cluster_name] = roi_trace_cluster
        
        # Average fluorescence traces across trials if input trace array is 3D
        roi_trace_cluster_avg = np.mean(roi_trace_cluster, axis=0) if roi_trace.ndim == 3 else roi_trace_cluster

        ax[1].plot(t, roi_trace_cluster_avg)
        ax[1].set_ylabel('Fluorescence Intensity', fontsize=14)
        ax[1].set_xlabel('time (s)', fontsize=14)
        ax[1].axvline(x=stimStart, color='k', linestyle='--')
        fig.suptitle(cluster_name, color=cluster_d['cluster_color'], fontsize=18, fontweight='bold')
        
        if plot_traces is True and roi_trace.ndim == 3:
            # Plot individual traces only when input array includes multiple trials
            ax[1].plot(t, roi_trace_cluster.T, color='gray', alpha=alpha_traces)
        
        if plot_errBar is True and roi_trace.ndim == 3:
            # Plot error bars only when input array includes multiple trials
            _, uFpsem, uFmsem = signalProcess.meanPlusMinusSem(roi_trace_cluster)
            ax[1].fill_between(t, uFpsem, uFmsem, alpha=alpha_errBar)

        if Yaxis_range is not None:
            ax[1].set_ylim(Yaxis_range)

    return cluster_trace


def plot_oddball_wholeTrace(df: pd.DataFrame, 
                            onset_times: tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]], 
                            resp_col: str = 'dFF_ROI_linFilt_butterFilt', 
                            negative_exclude: bool = False, 
                            t_base: float = -0.025, 
                            t_resp: tuple = (0.3, 0.975), 
                            plot_traces: bool = True, 
                            plot_errBar: bool = False, 
                            stimStart: float = 3.0, 
                            Xaxis_range: tuple[float,float] = None, 
                            Yaxis_range: tuple[float,float] = None, 
                            Yaxis_label: str = None, 
                            show_deviant_dots: bool = False, 
                            show_standard_dots: bool = False, 
                            trace_color: str = 'k',
                            stim_colors: dict = {'Deviant': 'r', 'Standard': 'k'}, 
                            **kwargs) -> pd.DataFrame:
    """
    Plot (individual and average) traces with marked deviant and standard tone onset times.

    Args:
        df (pd.DataFrame): Metadata Dataframe including columns: 'pulse', 'treatment', 'time', and column for response traces.
        onset_times (tuple | dict): Deviant and standard tone onset times. Can be:
                                    - tuple: (deviant_times, standard_times).
                                    - dict: with string keys mapping to tuples (deviant_times, standard_times).
        resp_col (str, optional): Column name for response traces. Defaults to 'dFF_ROI_linFilt_butterFilt'.
        negative_exclude (bool, optional): Whether to exclude negative responses (resp peak - base < 0) before averaging.
                                           If True, criterion is applied to each trace individually after fit subtraction and before low-pass filtering.
                                           After low-pass filtering, negative response windows are set to NaN, and then averaged.
                                           This will not change the trace plot, but will modify the dataframe output.
        t_base (float, optional): For each individual response, time point (relative to corresponding tone onset time) 
                                  at which baseline is calculated.
                                  If 'None', baseline is calculated at the last time frame before tone onset (-0.025 sec by default).
                                  Only used if 'negative_exclude' is True.
        t_resp (tuple, optional): For each individual response, response window (relative to corresponding tone onset time) 
                                  within which peak response is calculated.
                                  Only used if 'negative_exclude' is True.
        plot_traces (bool, optional): Whether to plot individual and average traces.
                                      If False, only return the dataframe with response traces and tone onset times without plotting.
        plot_errBar (bool, optional): Whether to plot error bars.
                                      If True, SEM is plotted as shaded area instead of individual traces.
        stimStart (float, optional): Time (in seconds) when the first stimulus starts.
        Xaxis_range (tuple, optional): Set fixed X-axis range as (x_min, x_max). If None, auto-scales X-axis.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.
        Yaxis_label (str, optional): Label for Y-axis. If None, defaults to the column name for response traces in df.
        show_deviant_dots (bool, optional): If True, show deviant tone onset as red dots beneath the traces.
        show_standard_dots (bool, optional): If True, show standard tone onset as black dots beneath the traces.
        trace_color (str, optional): Color for average trace and individual traces/error bars. Defaults to 'k' (black).
        stim_colors (dict, optional): Dictionary mapping marker colors to each tone type (deviant/standard).
                                      If None, defaults to {'Deviant': 'r', 'Standard': 'k'}.
        **kwargs: Optional keyword arguments.

    Returns:
        df_resp_trace (pd.DataFrame): Dataframe including individual and average response traces re treatment and pulse.
                                      Including columns: 'treatment', 'pulse', 'time', 'individual_traces', 
                                                         'avg_trace', 'deviant_times', and 'standard_times'.
    """
    
    df = df.copy()

    # Add deviant and standard tone onset times to the Dataframe based on the pulse name
    if isinstance(onset_times, tuple):
        # If 'onset_times' is a tuple, all pulse names must NOT end with digits
        if df["pulse"].str.contains(r'\d+$', regex=True).any():
            raise ValueError("Pulse names end with digits but 'onset_times' is a tuple (expected dict).")
        # Get all tone onset times (deviant + standard)
        total_times = np.sort(np.concatenate((onset_times[0], onset_times[1])))
        # Assume positions of deviant tones are the same for all traces
        df['deviant_times'] = [onset_times[0]] * len(df)
        df['standard_times'] = [onset_times[1]] * len(df)
        # Simplify pulse names
        df['pulse'] = df['pulse'].str.replace(
            'oddball_15422Hz_std_7711Hz.*', 
            'Deviant: 16kHz\nStandard: 8kHz', 
            regex=True
        )
        df['pulse'] = df['pulse'].str.replace(
            'oddball_7711Hz_std_15422Hz.*', 
            'Deviant: 8kHz\nStandard: 16kHz', 
            regex=True
        )
    
    elif isinstance(onset_times, dict):
        # If 'onset_times' is a dictionary, all pulse names must end with digits
        if not df["pulse"].str.contains(r'\d+$', regex=True).all():
            raise ValueError("Not all pulse names end with digits but 'onset_times' is a dict.")
        # Get all tone onset times (deviant + standard) by assuming they are the same across all pulse names
        total_times = np.sort(np.concatenate((list(onset_times.values())[0][0], list(onset_times.values())[0][1])))
        # Extract suffix digits from pulse names as pulse ID
        df['pulse_id'] = df['pulse'].str.extract(r'(\d+)$')
        # Check all pulse IDs exist in 'onset_times' keys
        missing_pulse_id = set(df['pulse_id']) - set(onset_times.keys())
        if missing_pulse_id:
            raise ValueError(f"Pulse IDs {missing_pulse_id} not found in 'onset_times' dictionary keys.")
        # Map each oddball pulse train to each trace
        df['deviant_times'] = df['pulse_id'].apply(lambda x: onset_times[x][0])
        df['standard_times'] = df['pulse_id'].apply(lambda x: onset_times[x][1])
        # Simplify pulse names while preserving pulse ID
        df['pulse'] = df.apply(lambda x: 
            'Deviant: 16kHz\nStandard: 8kHz\nProtocol: ' + x['pulse_id'] 
            if 'oddball_15422Hz_std_7711Hz' in x['pulse'] 
            else 'Deviant: 8kHz\nStandard: 16kHz\nProtocol: ' + x['pulse_id'],
            axis=1
        )
        # Drop temporary column
        df = df.drop(columns=['pulse_id'])
    
    else:
        raise TypeError("'onset_times' must be either a tuple or dict.")
    
    # Round time vector to 3 decimal places (assume time is the same for all traces)
    # time = df['time'].iloc[0]
    time = np.round(df['time'].iloc[0], 3)

    # Get stimulus marker colors
    dev_color = stim_colors.get('Deviant', 'r') if stim_colors else 'r'
    std_color = stim_colors.get('Standard', 'k') if stim_colors else 'k'

    # Initialize a list to store response trace data
    resp_trace = []

    for treat in df['treatment'].unique():
        n_pulses = df[df['treatment'] == treat]['pulse'].nunique()
        if plot_traces:
            fig, ax = plt.subplots(n_pulses, 1, figsize=(16, 4*n_pulses))
            if n_pulses == 1:
                ax = [ax]
            if n_pulses == 1:
                # Increase vertical space between subplot title and figure title
                plt.subplots_adjust(top=0.8)
            if isinstance(onset_times, tuple):
                plt.subplots_adjust(hspace=0.3)
            else:
                # Reduce space between title and the first subplot
                plt.subplots_adjust(hspace=0.4, top=0.95)

        for i, (pulse, group) in enumerate(df[df['treatment'] == treat].groupby('pulse')):
            traces = np.array(group[resp_col].tolist())
            traces_mean, traces_mean_psem, traces_mean_msem = signalProcess.meanPlusMinusSem(traces)
            deviant_times = group['deviant_times'].iloc[0]  # Take first value (all same in group)
            standard_times = group['standard_times'].iloc[0]
            
            if plot_traces:
                ax[i].plot(time, traces_mean, linewidth=2, color=trace_color)
                if plot_errBar:
                    ax[i].fill_between(time, traces_mean_psem, traces_mean_msem, color='gray' if trace_color == 'k' else trace_color, alpha=0.1)
                else:
                    for j in range(traces.shape[0]):
                        ax[i].plot(time, traces[j, :], linewidth=1.5, color='gray' if trace_color == 'k' else trace_color, alpha=0.3)
                ax[i].set_ylabel(Yaxis_label if Yaxis_label is not None else resp_col, fontsize=16)
                ax[i].axvline(x=stimStart, color='k', linestyle='--', linewidth=2)

                y_min, y_max = ax[i].get_ylim()
                dot_y = y_min + 0.1 * (y_max - y_min)  # slightly above bottom

                if show_deviant_dots:
                    ax[i].scatter(deviant_times, [dot_y]*len(deviant_times),
                                  color=dev_color, edgecolors=dev_color, s=20, zorder=1, alpha=1, label="Deviant tone")
                if show_standard_dots:
                    ax[i].scatter(standard_times, [dot_y]*len(standard_times),
                                  color=std_color, edgecolors=std_color, s=20, zorder=1, alpha=0.2, label="Standard tone")
                if not (show_deviant_dots or show_standard_dots):
                    for m, deviant in enumerate(deviant_times):
                        ax[i].axvline(x=deviant, color=dev_color, alpha=0.4, label="Deviant tone" if m == 0 else None, linewidth=2)
                    for n, standard in enumerate(standard_times):
                        ax[i].axvline(x=standard, color=std_color, alpha=0.08, label="Standard tone" if n == 0 else None, linewidth=2)
                if Yaxis_range is not None:
                    ax[i].set_ylim(Yaxis_range)
                ax[i].set_title(f"{pulse}", fontsize=14)
                ax[i].legend(fontsize=14, loc='upper right')
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['bottom'].set_linewidth(2)
                ax[i].spines['left'].set_linewidth(2)
                ax[i].tick_params(axis='both', labelsize=14, width=2)
                if Xaxis_range is not None:
                    ax[i].set_xlim(Xaxis_range)

            if negative_exclude:
                if 'butterFilt' in resp_col:
                    # Apply criterion after linear/logrithmic fit subtraction but before low-pass filtering
                    # Search for dataframe column only including remaining substrings ('linFilt' or 'logFilt')
                    resp_col_beforeButterFilt = resp_col.replace('_butterFilt', '')
                    if resp_col_beforeButterFilt not in df.columns:
                        raise ValueError(f"Column '{resp_col_beforeButterFilt}' not found in DataFrame.")
                    traces_beforeButterFilt = np.array(group[resp_col_beforeButterFilt].tolist())
                    for k in range(traces_beforeButterFilt.shape[0]):
                        for stim_time in total_times:
                            base, resp = signalProcess.getBaseResp(traces_beforeButterFilt[k, :], time,
                                                                   t_base=(stim_time + t_base, stim_time + t_base) if t_base else \
                                                                          (time[np.where(time < stim_time)[0][-1]], time[np.where(time < stim_time)[0][-1]]),
                                                                   t_resp=(stim_time + t_resp[0], stim_time + t_resp[1]),
                                                                   **kwargs)
                            if resp - base < 0:
                                # Set negative response windows to NaN (including t_base time frame of this window but not of the next window)
                                traces[k, :][np.where((time >= stim_time + t_base) & (time < stim_time + t_resp[1]))[0]] = np.nan
                    traces_negExcl_mean = np.nanmean(traces, axis=0)  # Ignore NaN values (negative responses)
                elif 'linFilt' in resp_col or 'logFilt' in resp_col:
                    # Directly apply criterion without low-pass filtering
                    for k in range(traces.shape[0]):
                        for stim_time in total_times:
                            base, resp = signalProcess.getBaseResp(traces[k, :], time,
                                                                   t_base=(stim_time + t_base, stim_time + t_base) if t_base else \
                                                                          (time[np.where(time < stim_time)[0][-1]], time[np.where(time < stim_time)[0][-1]]),
                                                                   t_resp=(stim_time + t_resp[0], stim_time + t_resp[1]),
                                                                   **kwargs)
                            if resp - base < 0:
                                traces[k, :][np.where((time >= stim_time + t_base) & (time < stim_time + t_resp[1]))[0]] = np.nan
                    traces_negExcl_mean = np.nanmean(traces, axis=0)
                else:
                    raise ValueError("For 'negative_exclude=True', 'resp_col' must include either 'butterFilt', 'linFilt', or 'logFilt'.")
                
            # Store all relevant data
            resp_trace.append({'treatment': treat, 'pulse': pulse, 'time': time, 
                               'individual_traces': traces, 
                               'avg_trace': traces_negExcl_mean if negative_exclude else traces_mean, 
                               'deviant_times': deviant_times, 
                               'standard_times': standard_times})

        if plot_traces:
            ax[-1].set_xlabel('time (s)', fontsize=16)
            fig.suptitle(f"Oddball Paradigm: {treat}", fontsize=16)

    if plot_traces:
        plt.show()

    # Convert response trace list to DataFrame
    df_resp_trace = pd.DataFrame(resp_trace)

    return df_resp_trace


def plot_oddball_trace_reTone(df_resp_peak: pd.DataFrame, 
                              trace_col: str = 'trace', 
                              x_col: str = 'treatment', 
                              y_col: str = 'pulse', 
                              within_col: str = 'stimulus', 
                              x_order: list = None, 
                              y_order: list = None, 
                              within_order: list = None, 
                              color_palette: str | dict = None, 
                              Yaxis_label: str = 'ΔF/F subtracting baseline', 
                              Xaxis_range: tuple[float,float] = None, 
                              Yaxis_range: tuple[float,float] = None, 
                              plot_traces: bool = False, 
                              alpha_traces: float = 0.1, 
                              plot_errBar: bool = False, 
                              alpha_errBar: float = 0.05, 
                              stimStart: float = 0.0, 
                              **kwargs) -> plt.Figure:
    """
    Plot average traces across multiple conditions for comparison. 

    This function visualizes up to three experimental factors simultaneously:
    (1) x-axis grouping (e.g., treatment),
    (2) y-axis grouping (e.g., pulse protocol),
    (3) within-subplot grouping (e.g., stimulus type).
    
    Example:
    - treatment: preZX1 vs postZX1
    - pulse: 8/16 kHz vs 16/8 kHz protocols
    - stimulus: Deviant vs Standard tones

    Args:
        df_resp_peak (pd.DataFrame): Dataframe including individual response traces and experimental conditions.
                                     Including columns referenced by: trace_col, x_col, y_col, within_col.
        trace_col (str, optional): Column name containing response traces. Each cell should contain either:
                                   - a 1D array of shape [frame], representing a single trace, or
                                   - a 2D array of shape [traceNumber, frame], representing multiple traces.
                                   If multiple rows belong to the same plotting condition, traces from all
                                   rows are combined using np.vstack() before plotting. Defaults to 'trace'.
        x_col (str, optional): Column used for x-axis grouping (e.g., treatment). Defaults to 'treatment'. 
                               If None, plots collapse to a single column. 
                               If 'post' is included in x_col value, the corresponding trace will be plotted with alpha=0.5.
        y_col (str, optional): Column used for y-axis grouping (e.g., pulse protocol). Defaults to 'pulse'. 
                               If None, plots collapse to a single row.
        within_col (str, optional): Column used for within-subplot grouping (e.g., stimulus type).
                                    Defaults to 'stimulus'.
                                    If None, only one trace is plotted in each subplot.
        x_order (list, optional): Order of values in x_col for arranging subplot columns.
                                  If None, uses the order of first appearance in df_resp_peak.
        y_order (list, optional): Order of values in y_col for arranging subplot rows.
                                  If None, uses the order of first appearance in df_resp_peak.
        within_order (list, optional): Order of values in within_col for displaying legend entries within each subplot.
                                       If None, uses the order of first appearance in df_resp_peak.
        color_palette (str | dict, optional): Either a single color string applied to all traces, or a dict mapping condition keys to colors.
                                              Supported dict key formats are:
                                                  within, (x, within), (y, within) or (x, y, within), 
                                                  where x, y and within are values from x_col, y_col and within_col, respectively.
                                              Defaults to 'k' if within_col is None, otherwise {'Deviant': 'r', 'Standard': 'k'}.
        Yaxis_label (str, optional): Label for Y-axis. Defaults to 'ΔF/F subtracting baseline' 
                                     representing Y = Δ(ΔF/F) = ΔF/F trace - baseline ΔF/F.
        Xaxis_range (tuple, optional): Set fixed X-axis range as (x_min, x_max). If None, auto-scales X-axis.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.
        plot_traces (bool, optional): Whether to plot individual traces.
        alpha_traces (float, optional): Transparency for individual traces.
        plot_errBar (bool. optional): Whether to plot SEM as shading (error bars).
        alpha_errBar (float, optional): Transparency for error bars.
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 0.0.
        **kwargs: Optional keyword arguments.
            - Example: delayAdjust (float, optional): Adjustment in time (s) for frame data acquisition.
                                                      Defaults to -0.025 sec.

    Returns:
        fig (plt.Figure): Figure object containing the plotted traces.
    """

    # Check if at least one grouping variable is provided
    if all(v is None for v in [x_col, y_col, within_col]):
        raise ValueError("At least one grouping variable must be provided.")
   
    # First time frame starts at -0.025 sec by default (sound onset: 0 sec)
    delayAdjust = kwargs.pop('delayAdjust', -0.025)  # remove from kwargs if present

    # If any grouping column is not provided, collapse the plot to fewer dimensions
    df_plot = df_resp_peak.copy()
    x_col_input = x_col
    if x_col is None:
        df_plot["_x"] = ""
        x_col = "_x"

    y_col_input = y_col
    if y_col is None:
        df_plot["_y"] = ""
        y_col = "_y"

    within_col_input = within_col  # Store original within_col input for legend adding
    if within_col is None:
        df_plot["_within"] = "all"
        within_col = "_within"

        if color_palette is None:
            # Plot all traces in black by default if only one trace is in each subplot
            color_palette = {"all": "k"}
        elif isinstance(color_palette, str):
            color_palette = {"all": color_palette}
    else:
        if color_palette is None:
            color_palette = {'Deviant': 'r', 'Standard': 'k'}
        elif isinstance(color_palette, str):
            color_palette = {stim: color_palette for stim in df_plot[within_col].unique()}

    x_label = x_order if x_order is not None else df_plot[x_col].unique()
    y_label = y_order if y_order is not None else df_plot[y_col].unique()
    within_label = within_order if within_order is not None else df_plot[within_col].unique()
    n_x_label = len(x_label)
    n_y_label = len(y_label)
    
    fig, ax = plt.subplots(
        n_y_label, n_x_label, 
        figsize=(
            8+2*n_x_label if x_col_input is not None else 8, 
            4+2*n_y_label if y_col_input is not None else 5
        )
    )
    
    # Force ax to be 2D of shape (n_y_label, n_x_label)
    if n_y_label == 1 and n_x_label == 1:
        ax = np.array([[ax]])
    elif n_y_label == 1:
        ax = ax.reshape(1, -1)
    elif n_x_label == 1:
        ax = ax.reshape(-1, 1)

    for i, y_val in enumerate(y_label):
        for j, x_val in enumerate(x_label):
            for stim in within_label:
                df_subset = df_plot[
                    (df_plot[x_col] == x_val) & 
                    (df_plot[y_col] == y_val) & 
                    (df_plot[within_col] == stim)
                ]
                if df_subset.empty:
                    # Skip plotting if no data for this combination of x, y, and within values  
                    continue
                elif len(df_subset) == 1:
                    trace = df_subset[trace_col].iloc[0]
                else:
                    # Vstack traces across rows if multiple rows exist
                    trace = np.vstack(df_subset[trace_col].tolist())
                
                time = signalProcess.getTimeVec(trace.shape[-1], delayAdjust = delayAdjust, **kwargs)

                trace_color = (
                    color_palette.get((x_val, y_val, stim))
                    or color_palette.get((x_val, stim))
                    or color_palette.get((y_val, stim))
                    or color_palette.get(stim)
                )
                if trace_color is None:
                    raise KeyError(
                        f"No color specified for x='{x_val}', y='{y_val}', within='{stim}'."
                    )
                
                if trace.ndim == 1:
                    # Directly plot without error bars if trace is 1D of shape [frame]
                    ax[i, j].plot(time, trace, color=trace_color, alpha=0.5 if 'post' in str(x_val).lower() else 1.0, label=f'{stim}', linewidth=3)
                elif trace.ndim == 2:
                    # Average across trials and plot error bars if trace is 2D of shape [traceNumber, frame]
                    trace_mean, trace_psem, trace_msem = signalProcess.meanPlusMinusSem(trace, ignoreNaN=True)  # Ignore NaN values (negative responses excluded)
                    ax[i,j].plot(time, trace_mean, color=trace_color, alpha=0.5 if 'post' in str(x_val).lower() else 1.0, label=f'{stim}', linewidth=3)
                    if plot_errBar:
                        # Add error bars
                        ax[i,j].fill_between(time, trace_psem, trace_msem, color=trace_color, alpha=alpha_errBar)
                    if plot_traces:
                        # Add individual traces
                        for k in range(trace.shape[0]):
                            ax[i,j].plot(time, trace[k, :], color=trace_color, alpha=alpha_traces, linewidth=2)
                else:
                    raise ValueError(f"Traces must be either 1D (frame,) or 2D (traceNumber, frame), got shape {trace.shape}.")
            
            ax[i,j].axvline(x=stimStart, color='k', linestyle='--', linewidth=2)
            if Xaxis_range is not None:
                ax[i,j].set_xlim(Xaxis_range)
            if Yaxis_range is not None:
                ax[i,j].set_ylim(Yaxis_range)
            if within_col_input is not None:
                if any(isinstance(key, tuple) and len(key) > 1 for key in color_palette.keys()):
                    # Add legend in every subplot if trace color sets are different across subplots
                    ax[i, j].legend(fontsize=16, loc='upper right')
                elif (within_col_input is not None) and (i == 0) and (j == n_x_label - 1):
                    # Add legend only in the top-right subplot if within_col is provided (multiple traces in each subplot)
                    ax[i, j].legend(fontsize=16, loc='upper right')
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_linewidth(2)
            ax[i, j].spines['left'].set_linewidth(2)
            ax[i, j].tick_params(axis='both', labelsize=18, width=2)

            if i == 0:
                # Add treatment titles
                ax[i,j].set_title(x_val, pad=15, fontsize=20, fontweight='bold')
            if i == n_y_label - 1:
                # Add X-axis labels
                ax[i,j].set_xlabel('time (s)', fontsize=18)
            if j == 0:
                ax[i,j].set_ylabel(Yaxis_label, fontsize=18)
                # Add pulse protocol names
                ax[i,j].text(-0.6, 0.5, y_val, fontsize=20, fontweight='bold', rotation=0, 
                             ha='center', va='center', transform=ax[i,j].transAxes)

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # extra left margin for exporting to pdf
    plt.show()

    return fig


def plot_blurred_respSpatialDFF(freq2dFFresp_calcium: dict, freq2dFFresp_zinc: dict, ksize: tuple = (7, 7), 
                                plot_figures: bool = True, display_wideField: bool = True, display_traces: bool = True, 
                                mask: np.ndarray = None, mask_contour: np.ndarray = None, 
                                stimStart: float = 3.0, palette: list = ['r', 'g'], 
                                Yaxis_range: tuple[float,float] = None) -> tuple[dict, dict, dict]:
    """
    Plot Gaussian blurred response spatialDFF images and traces within ROI for calcium and zinc, 
    and compute Spearman correlation coefficients between calcium and zinc spatialDFF re frequency.

    Args:
        freq2dFFresp_calcium (dict): Dictionary mapping frequency to 
                                     tuple(raw spatialDFF image of shape (Y, X), 
                                           subset DataFrame at the frequency) for calcium.
        freq2dFFresp_zinc (dict): Dictionary mapping frequency to 
                                  tuple(raw spatialDFF image of shape (Y, X), 
                                        subset DataFrame at the frequency) for zinc.
        ksize (tuple, optional): Kernel size for Gaussian blur. 
                                 Suggested minimal size (7, 7) to effectively reduce noise (default).
                                 A larger size (e.g., (15, 15)) is suggested for stronger noise reduction.
        plot_figures (bool, optional): Whether to plot spatialDFF images and traces. Defaults to True.
        display_wideField (bool, optional): Whether to display gray-scaled wide-field images for calcium and zinc. Defaults to True.
        display_traces (bool, optional): Whether to display traces within ROI stored in the DataFrames. Defaults to True.
        mask (np.ndarray, optional): Binary mask to filter pixels in flattened arrays in 'freq2dFFrespArray_calcium' 
                                     and 'freq2dFFrespArray_zinc', on which Spearman correlation is computed.
                                     If None, no mask is applied and arrays in 'freq2dFFrespArray' include all pixels. Defaults to None.
        mask_contour (np.ndarray, optional): Mask's vertex coordinates to overlay on images, including a repeated first vertex to close the shape.
                                             If None, no contour is plotted. Defaults to None.
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        palette (list, optional): List of colors for plotting traces. Defaults to red and green for calcium and zinc, respectively.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max) for traces. If None, Y-axis is auto-scaled.
    
    Returns:
        tuple[dict, dict, dict]:
            - freq2dFFrespArray_calcium: Dictionary mapping frequency to blurred, normalized and flattened spatialDFF within mask for calcium (1D array).
            - freq2dFFrespArray_zinc: Dictionary mapping frequency to blurred, normalized and flattened spatialDFF within mask for zinc (1D array).
            - freq2spearman: Dictionary mapping frequency to Spearman correlation coefficient between calcium and zinc spatialDFF (float).
    """

    # Check keys of freq2dFFresp_calcium and freq2dFFresp_zinc are identical
    if set(freq2dFFresp_calcium.keys()) != set(freq2dFFresp_zinc.keys()):
        raise ValueError("Keys of freq2dFFresp_calcium and freq2dFFresp_zinc must be identical.")
    
    # Initialize dictionaries to store blurred and normalized images/flattened arrays for calcium and zinc
    freq2dFFrespImg_calcium, freq2dFFrespArray_calcium = imgProcess.blurImg(freq2dFFresp_calcium, ksize=ksize, mask=mask)
    freq2dFFrespImg_zinc, freq2dFFrespArray_zinc = imgProcess.blurImg(freq2dFFresp_zinc, ksize=ksize, mask=mask)

    # Initialize a dictionary mapping frequency to Spearman correlation coefficient
    freq2spearman = {}
    for freq in freq2dFFrespArray_calcium.keys():
        corr, _ = spearmanr(freq2dFFrespArray_calcium[freq], freq2dFFrespArray_zinc[freq])
        freq2spearman[freq] = corr

    # Overview processed calcium and zinc spatialDFF and traces within ROI at A1
    if plot_figures:
        for freq in freq2dFFrespImg_calcium.keys():
            if display_wideField:
                # Display gray-scaled wide-field images
                fig = plt.figure(figsize=(20, 6)) if display_traces else plt.figure(figsize=(10, 6))
                gs = GridSpec(2, 3, width_ratios=[1, 1, 2]) if display_traces else GridSpec(2, 2, width_ratios=[1, 1])

                ax_calcium_wf = fig.add_subplot(gs[0, 0])
                ax_zinc_wf = fig.add_subplot(gs[0, 1])
                ax_traces = fig.add_subplot(gs[:, 2]) if display_traces else None
                ax_calcium_resp = fig.add_subplot(gs[1, 0])
                ax_zinc_resp = fig.add_subplot(gs[1, 1])
                
                # Plot gray-scaled wide-field images
                qcams_calcium = freq2dFFresp_calcium[freq][1]['qcam'].tolist()
                imgs_calcium = np.array(fileIngest.qcams2imgs(qcams_calcium)[0])
                ax_calcium_wf.imshow(imgs_calcium.mean(axis=(0,-1)), 'gray')

                qcams_zinc = freq2dFFresp_zinc[freq][1]['qcam'].tolist()
                imgs_zinc = np.array(fileIngest.qcams2imgs(qcams_zinc)[0])
                ax_zinc_wf.imshow(imgs_zinc.mean(axis=(0,-1)), 'gray')
                
                # Plot color-scaled blurred response spatialDFF heatmaps
                heatmap_calcium = ax_calcium_resp.imshow(freq2dFFrespImg_calcium[freq], cmap='jet')
                heatmap_zinc = ax_zinc_resp.imshow(freq2dFFrespImg_zinc[freq], cmap='jet')
                
                # Add colorbars
                fig.colorbar(heatmap_calcium, cax=make_axes_locatable(ax_calcium_resp).append_axes("right", size="5%", pad=0.2))
                fig.colorbar(heatmap_zinc, cax=make_axes_locatable(ax_zinc_resp).append_axes("right", size="5%", pad=0.2))
                # fig.colorbar(ax_calcium, ax=ax_calcium_resp, fraction=0.035, pad=0.04)
                # fig.colorbar(ax_zinc, ax=ax_zinc_resp, fraction=0.035, pad=0.04)

            else:
                # Only plot color-scaled blurred response spatialDFF heatmaps
                fig, ax = (plt.subplots(1, 3, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 1, 2]}) if display_traces 
                           else plt.subplots(1, 2, figsize=(10, 4)))
                ax_calcium_resp = ax[0]
                ax_zinc_resp = ax[1]
                ax_traces = ax[2] if display_traces else None

                # Plot color-scaled blurred response spatialDFF heatmaps
                heatmap_calcium = ax_calcium_resp.imshow(freq2dFFrespImg_calcium[freq], cmap='jet')
                heatmap_zinc = ax_zinc_resp.imshow(freq2dFFrespImg_zinc[freq], cmap='jet')

                # Add colorbars
                fig.colorbar(heatmap_calcium, cax=make_axes_locatable(ax_calcium_resp).append_axes("right", size="5%", pad=0.2))
                fig.colorbar(heatmap_zinc, cax=make_axes_locatable(ax_zinc_resp).append_axes("right", size="5%", pad=0.2))

            if mask_contour is not None:
                # Add mask contours if Spearman correlation is computed within a polygonal mask
                if display_wideField:
                    ax_calcium_wf.plot(mask_contour[:,0], mask_contour[:,1], 'w-', linewidth=2)
                    ax_zinc_wf.plot(mask_contour[:,0], mask_contour[:,1], 'w-', linewidth=2)
                ax_calcium_resp.plot(mask_contour[:,0], mask_contour[:,1], 'w-', linewidth=2)
                ax_zinc_resp.plot(mask_contour[:,0], mask_contour[:,1], 'w-', linewidth=2)

            if display_wideField:
                ax_calcium_wf.set_title('Calcium', color=palette[0], fontsize=18, fontweight='bold')
                ax_zinc_wf.set_title('Zinc', color=palette[1], fontsize=18, fontweight='bold')
            else:
                ax_calcium_resp.set_title('Calcium', color=palette[0], fontsize=18, fontweight='bold')
                ax_zinc_resp.set_title('Zinc', color=palette[1], fontsize=18, fontweight='bold')

            if display_traces:
                # Plot traces within ROI stored in the DataFrames
                resp_calcium = np.array(freq2dFFresp_calcium[freq][1]['dFF_ROI_linFilt'].tolist())
                ax_traces.plot(freq2dFFresp_calcium[freq][1]['time'].iloc[0], np.mean(resp_calcium, axis=0), color=palette[0], label='Calcium')
                for i in range(resp_calcium.shape[0]):
                    ax_traces.plot(freq2dFFresp_calcium[freq][1]['time'].iloc[0], resp_calcium[i, :], color=palette[0], alpha=0.1)

                resp_zinc = np.array(freq2dFFresp_zinc[freq][1]['dFF_ROI_logFilt_butterFilt'].tolist())
                ax_traces.plot(freq2dFFresp_zinc[freq][1]['time'].iloc[0], np.mean(resp_zinc, axis=0), color=palette[1], label='Zinc')
                for j in range(resp_zinc.shape[0]):
                    ax_traces.plot(freq2dFFresp_zinc[freq][1]['time'].iloc[0], resp_zinc[j, :], color=palette[1], alpha=0.1)
                
                ax_traces.set_xlabel('Time (s)', fontsize=14)
                ax_traces.set_ylabel('ΔF/F', fontsize=14)
                ax_traces.axvline(x=stimStart, color='k', linestyle='--')
                ax_traces.legend(loc='upper right', fontsize=14)
                ax_traces.tick_params(axis='both', labelsize=12)
                ax_traces.set_title('Traces within ROI stored in DataFrames', fontsize=16)
                if Yaxis_range is not None:
                    ax_traces.set_ylim(Yaxis_range)
            
            fig.subplots_adjust(wspace=0.3, top=0.85)
            freq_display = freq/1000 if freq > 500 else freq  # Assume freq is in Hz and convert to kHz for title if freq > 500
            fig.suptitle(f'{freq_display} kHz: Spearman r = {freq2spearman[freq]:.2f}', fontsize=18, fontweight='bold')

        plt.show()

    return freq2dFFrespArray_calcium, freq2dFFrespArray_zinc, freq2spearman


def plot_boxplot(df: pd.DataFrame, x: str, y: str, 
                 group: str | None = None, 
                 id: str | list[str] = None, 
                 x_order: list = None, 
                 group_order: list = None, 
                 palette: list = ['k', 'gray'], 
                 offset: float = 0.3, 
                 jitter: float = 0, 
                 show_xlabel: bool = False, 
                 Yaxis_label: str = None, 
                 Yaxis_range: tuple[float,float] = None, 
                 axhline: None | int | float | list = 0, 
                 title_color: str = None):

    """
    Plot box-and-whisker plot with optionally paired data points.

    Args:
        df (pd.DataFrame): Dataframe including columns for X-axis, Y-axis, and pairing ID.
        x (str): Column name for X-axis categories (e.g., treatment or stimulus type).
        y (str): Column name for Y-axis values (e.g., response).
        group (str, optional): Column defining the horizontal subplots (e.g., stimulus type). 
                               If None, only one subplot is plotted.
        id (str or list[str], optional): Column name(s) for pairing ID (e.g., subject ID). 
                                         If None, no paired connections are drawn.
        x_order (list, optional): Order of values in x for arranging X-axis categories.
                                  If None, uses the order of first appearance in df.
        group_order (list, optional): Order of values in group for arranging subplot columns.
                                      If None, uses the order of first appearance in df.
        palette (list, optional): List of colors for each X-axis category. Defaults to ['k', 'gray'].
        offset (float, optional): Horizontal offset for paired data points to avoid overlap. Defaults to 0.3.
        jitter (float, optional): Amount of jitter to apply to individual data points. Defaults to 0 (no jitter).
        show_xlabel (bool, optional): Whether to display X-axis label. Defaults to False.
        Yaxis_label (str, optional): Label for Y-axis. If None, uses the column name y.
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.
        axhline (None or int or float or list, optional): Horizontal line(s) to draw on the plot. 
                                                          If a list, horizontal lines are drawn at each value in the list.
                                                          If None, no line is drawn.
        title_color (str, optional): Color of the subplot titles. If None, uses the same color as the left-most bar in the subplot.
    """
    
    x_labels = x_order if x_order is not None else df[x].unique()
    if group is None:
        groups = [None]
    else:
        if group_order is not None:
            groups = group_order
        else:
            groups = df[group].unique()

    fig, axes = plt.subplots(1, len(groups), 
                             figsize=(5+4*(len(groups)-1)+1.5*(len(x_labels)-2), 6), 
                             sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for i, g in enumerate(groups):
        ax = axes[i]
        df_sub = df if g is None else df[df[group] == g]

        # Select palette colors re subplot
        colors = palette if group is None else palette[i*len(x_labels):(i+1)*len(x_labels)]

        # Box-and-whisker plot
        sns.boxplot(
            data=df_sub, x=x, y=y, 
            order=x_labels, width=0.45 if id is None else 0.3, showfliers=False, 
            showmeans=True, meanline=True, ax=ax
        )

        # Overlay individual data points
        np.random.seed(0)  # Make the jitter reproducible
        sns.stripplot(
            data=df_sub, x=x, y=y, 
            order=x_labels, jitter=jitter, size=8, 
            linewidth=2, color='white', zorder=5, ax=ax
        )

        # Draw paired connections
        if id is not None:
            x_pos = {t: j for j, t in enumerate(x_labels)}
            for _, df_pair in df_sub.groupby(id, sort=False):
                y_pre = df_pair[df_pair[x] == x_labels[0]][y].values[0]
                y_post = df_pair[df_pair[x] == x_labels[1]][y].values[0]
                ax.plot(
                    [x_pos[x_labels[0]] + offset, x_pos[x_labels[1]] - offset],
                    [y_pre, y_post],
                    color='gray', linewidth=1.5, alpha=0.6, zorder=3
                )

        # Re-style boxplot and points
        for j in range(len(x_labels)):
            # Set boxplot color re treatment
            ax.patches[j].set_facecolor('white')
            ax.patches[j].set_edgecolor(colors[j])
            ax.patches[j].set_linewidth(3)
            for line in ax.lines[j*6:(j+1)*6]:  # Each box contributes 6 Line2D objects: whisker, whisker, cap, cap, median, mean
                line.set_color(colors[j])
                line.set_linewidth(3)

            # Offset stripplot points and set color re treatment
            offsets = ax.collections[j].get_offsets()
            if j == 0:
                offsets[:, 0] += offset
            else:
                offsets[:, 0] -= offset
            ax.collections[j].set_offsets(offsets)
            ax.collections[j].set_edgecolor(colors[j])

        # Formatting
        if axhline is None:
            pass
        elif isinstance(axhline, (int, float)):
            ax.axhline(axhline, color='gray', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
        elif isinstance(axhline, list):
            for y_val in axhline:
                ax.axhline(y_val, color='gray', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
        else:
            raise ValueError("axhline must be None, a number, or a list of numbers.")
        ax.set_xlabel('' if not show_xlabel else x, fontsize=20)
        ax.tick_params(axis='x', labelsize=20, width=2)
        ax.tick_params(axis='y', labelsize=16, width=2)

        if not show_xlabel:
            for j, label in enumerate(ax.get_xticklabels()):
                label.set_color(colors[j])  # Color x-axis tick labels
                label.set_fontweight('bold')
        
        if g is not None:
            ax.set_title(g, fontsize=22, fontweight='bold', color=title_color if title_color is not None else colors[0])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.grid(False)

        if i == 0:
            ax.set_ylabel(Yaxis_label if Yaxis_label is not None else y, fontsize=20)
        
        if Yaxis_range is not None:
            ax.set_ylim(Yaxis_range)

    plt.show()


def plot_barplot(df: pd.DataFrame, x: str, y: str, 
                 group: str | None = None, 
                 id: str | list[str] = None, 
                 x_order: list = None, 
                 group_order: list = None, 
                 palette: list = ['k', 'gray', 'k', 'gray'], 
                 offset: float = 0.3, 
                 jitter: float = 0, 
                 Yaxis_range: tuple[float,float] = None, 
                 title_color: str = None):

    """
    Plot barplot with SEM error bars and optionally paired data points.

    Args:
        df (pd.DataFrame): Dataframe including columns for X-axis, Y-axis, grouping variable, and pairing ID.
        x (str): Column name for bar categories (e.g., treatment).
        y (str): Column name for Y-axis values (e.g., response).
        group (str, optional): Column defining the horizontal subplots (e.g., stimulus type). 
                               If None, only one subplot is plotted.
        id (str or list[str], optional): Column name(s) for pairing ID (e.g., subject ID). 
                                         If None, no paired connections are drawn.
        x_order (list, optional): Order of values in x for arranging X-axis categories.
                                  If None, uses the order of first appearance in df.
        group_order (list, optional): Order of values in group for arranging subplot columns.
                                      If None, uses the order of first appearance in df.
        palette (list, optional): List of colors for each bar and corresponding datapoints. 
                                  Defaults to ['k', 'gray', 'k', 'gray'].
        offset (float, optional): Horizontal offset for paired datapoints to avoid overlap. Defaults to 0.3.
        jitter (float, optional): Amount of jitter to apply to individual datapoints. Defaults to 0 (no jitter).
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.
        title_color (str, optional): Color of the subplot titles. If None, uses the same color as the left-most bar in the subplot.
    """

    x_labels = x_order if x_order is not None else df[x].unique()
    if group is None:
        groups = [None]
    else:
        if group_order is not None:
            groups = group_order
        else:
            groups = df[group].unique()

    fig, axes = plt.subplots(1, len(groups), 
                             figsize=(5+4*(len(groups)-1)+1.5*(len(x_labels)-2), 6), 
                             sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for i, g in enumerate(groups):
        ax = axes[i]
        df_sub = df if g is None else df[df[group] == g]

        # Select palette colors re subplot
        colors = palette if group is None else palette[i*len(x_labels):(i+1)*len(x_labels)]

        # Barplot
        sns.barplot(data=df_sub, x=x, y=y, order=x_labels, 
                    errorbar='se', capsize=0.08, width=0.4+0.08*(len(x_labels)-2), ax=ax)

        # Extract datapoints
        data_points = []
        for xi in x_labels:
            vals = df_sub[df_sub[x] == xi][y].values
            data_points.append(vals)

        bar_positions = np.arange(len(x_labels))

        # Plot individual datapoints
        np.random.seed(0)  # Make the jitter reproducible
        for j in range(len(x_labels)):
            x_pos = bar_positions[j] + offset if j == 0 else bar_positions[j] - offset
            jitter_vals = np.random.normal(0, jitter, len(data_points[j]))
            ax.scatter(
                np.full(len(data_points[j]), x_pos) + jitter_vals, 
                data_points[j], 
                facecolors='white', edgecolors=colors[j], 
                s=80, linewidth=2, zorder=3
            )

        # Draw paired connections
        if id is not None:
            for _, df_pair in df_sub.groupby(id, sort=False):
                y_pre = df_pair[df_pair[x] == x_labels[0]][y].values[0]
                y_post = df_pair[df_pair[x] == x_labels[1]][y].values[0]
                ax.plot(
                    [bar_positions[0] + offset, bar_positions[1] - offset], 
                    [y_pre, y_post], 
                    color='gray', linewidth=1.5, alpha=0.6, zorder=1
                )

        # Re-style bars
        for j in range(len(x_labels)):
            ax.patches[j].set_facecolor(colors[j])

        # Formatting
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(x_labels, fontsize=20)
        ax.set_xlabel('')
        ax.tick_params(axis='x', width=2)
        ax.tick_params(axis='y', labelsize=16, width=2)

        for j, label in enumerate(ax.get_xticklabels()):
            label.set_color(colors[j])  # Color x-axis tick labels
            label.set_fontweight('bold')

        if g is not None:
            ax.set_title(g, fontsize=22, fontweight='bold', color=title_color if title_color is not None else colors[0])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.grid(False)

        if i == 0:
            ax.set_ylabel(y, fontsize=20)

        if Yaxis_range is not None:
            ax.set_ylim(Yaxis_range)

    plt.show()


def plot_lineplot_adaptation(df: pd.DataFrame, x: str, y: str, 
                             group: str | None = None, 
                             exp_fit: bool = False, 
                             palette: list = ['k', 'gray'], 
                             markers: list = ['o', 'o'], 
                             Yaxis_range: tuple[float, float] = None) -> None | dict[str, tuple[float, float, float]]:
    """
    Plot line plot of averaged responses to a train of stimuli (e.g., adaptation to consecutive standard tones).
    Optionally break X-axis to show response amplitudes at the beginning and end of the train.

    Args:
        df (pd.DataFrame): Dataframe including columns for X-axis (Tone Position), Y-axis (Response ΔF/F) and grouping variable.
        x (str): Column name for X-axis variable (e.g., standard tone position indices).
        y (str): Column name for Y-axis values (e.g., response amplitude).
        group (str, optional): Column defining different conditions to plot in separate lines (e.g., Treatment). 
                               If None, all data are plotted in one line.
        exp_fit (bool, optional): Whether to fit an exponential decay function, y = A * exp(-t/tau) + C.
                                  Only fit if no gap is detected in tone positions (i.e., no break in X-axis).
                                  If True, returns a dict of parameters re fitted line.
        palette (list, optional): List of colors for each line. Defaults to ['k', 'gray'].
        markers (list, optional): Marker styles for each group. Must have the same order as `palette`.
                                  Defaults to ['o', 'o'].
        Yaxis_range (tuple, optional): Set fixed Y-axis range as (y_min, y_max). If None, auto-scales Y-axis.

    Returns:
        None or dict: If `exp_fit` is True, returns a dict mapping group labels to fitted parameters (A, tau, C). 
                      Otherwise, returns None.
    """

    # Detect gaps in X-axis variable and optionally break X-axis to show responses at the beginning and end of the train
    pos = df[x].iloc[0]  # Assume pos is identical across rows
    pos_diff = np.diff(pos)
    pos_unique, pos_counts = np.unique(pos_diff, return_counts=True)

    if len(pos_unique) > 2:
        raise ValueError("Multiple gaps detected in time positions. At most one gap is allowed.")
    if len(pos_unique) == 2 and exp_fit:
        raise ValueError("Exponential fitting is not supported when a gap is detected in time positions.")
    if exp_fit:
        # Initialize a dict to store fitted parameters re curve
        fit_params = {}
        # Define the exponential decay function
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C
    
    no_gap = (len(pos_unique) == 1)

    # Assume the most frequent spacing as the regular tick spacing
    tick_spacing = pos_unique[pos_counts.argmax()]

    if no_gap:
        fig, ax1 = plt.subplots(figsize=(pos.shape[0], 6))
        ax2 = None
    else:
        # Assume the least frequent spacing as the gap where X-axis is broken
        split_idx = np.where(pos_diff == pos_unique[pos_counts.argmin()])[0][0]  # Index of the last element in the left Axis

        # Calculate the range of X-axis for both Axes to set width ratios
        range_left = split_idx + tick_spacing
        range_right = pos.shape[0] - split_idx - 2 + tick_spacing

        # Break the X-axis into two portions by creating two subplots with shared Y-axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pos.shape[0], 6), sharey=True, 
                                       gridspec_kw={'width_ratios': [range_left, range_right]})  # Equal tick spacing
        fig.subplots_adjust(wspace=0.1)  # Adjust spacing between Axes

    if group is None:
        # Plot all data in one line
        response = np.array(df[y].tolist())
        # Not including NaN values in mean and SEM calculation if any
        response_mean = np.nanmean(response, axis=0)
        response_sem = np.nanstd(response, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(response), axis=0))  # Sample SEM (ddof=1)

        if no_gap:
            ax1.errorbar(pos, response_mean, yerr=response_sem, 
                         capsize=3, capthick=1.5, marker=markers[0], 
                         linestyle='none' if exp_fit else '-', 
                         linewidth=2.5, color=palette[0])
            if exp_fit:
                # Fit an exponential decay function to averaged data: y = A * exp(-t/tau) + C
                params, _ = curve_fit(exp_decay, pos.astype(np.float64), response_mean, maxfev=1000000, 
                                      p0 = [response_mean[0]-response_mean[-1], 1, response_mean[-1]])  # Initial guess
                A_fit, tau_fit, C_fit = params
                # Plot fitted curve
                t_smooth = np.linspace(pos.min(), pos.max(), 100*(pos.shape[0]-1)+1)
                y_smooth = exp_decay(t_smooth, A_fit, tau_fit, C_fit)
                ax1.plot(t_smooth, y_smooth, linewidth=2.5, color=palette[0])
                fit_params['all'] = params
        else:
            # Left Axis: standard responses before the first deviant response
            ax1.errorbar(pos[:split_idx+1], response_mean[:split_idx+1], yerr=response_sem[:split_idx+1], 
                         capsize=3, capthick=1.5, marker=markers[0], linewidth=2.5, color=palette[0])
            # Right Axis: standard responses after the last deviant response
            ax2.errorbar(pos[split_idx+1:], response_mean[split_idx+1:], yerr=response_sem[split_idx+1:], 
                         capsize=3, capthick=1.5, marker=markers[0], linewidth=2.5, color=palette[0])
    else:
        # Plot different lines re group
        for i, g in enumerate(df[group].unique()):
            response = np.array(df[df[group] == g][y].tolist())
            response_mean = np.nanmean(response, axis=0)
            response_sem = np.nanstd(response, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(response), axis=0))

            if no_gap:
                ax1.errorbar(pos, response_mean, yerr=response_sem, 
                             capsize=3, capthick=1.5, marker=markers[i], 
                             linestyle='none' if exp_fit else '-', 
                             linewidth=2.5, color=palette[i], label=g)
                if exp_fit:
                    # Fit an exponential decay function to averaged data: y = A * exp(-t/tau) + C
                    params, _ = curve_fit(exp_decay, pos.astype(np.float64), response_mean, maxfev=1000000, 
                                          p0 = [response_mean[0]-response_mean[-1], 1, response_mean[-1]])  # Initial guess
                    A_fit, tau_fit, C_fit = params
                    # Plot fitted curve
                    t_smooth = np.linspace(pos.min(), pos.max(), 100*(pos.shape[0]-1)+1)
                    y_smooth = exp_decay(t_smooth, A_fit, tau_fit, C_fit)
                    ax1.plot(t_smooth, y_smooth, linewidth=2.5, color=palette[i])
                    fit_params[g] = params
            else:
                ax1.errorbar(pos[:split_idx+1], response_mean[:split_idx+1], yerr=response_sem[:split_idx+1], 
                             capsize=3, capthick=1.5, marker=markers[i], linewidth=2.5, color=palette[i], label=g)
                ax2.errorbar(pos[split_idx+1:], response_mean[split_idx+1:], yerr=response_sem[split_idx+1:], 
                             capsize=3, capthick=1.5, marker=markers[i], linewidth=2.5, color=palette[i], label=g)

    if no_gap:
        ax1.set_xlim(pos[0] - tick_spacing*0.5, pos[-1] + tick_spacing*0.5)
        ax1.set_xticks(np.arange(pos[0], pos[-1] + tick_spacing, tick_spacing))
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        if group is not None:
            ax1.legend(fontsize=16, loc='upper right')
    else:
        # Set X-axis limits corresponding to width ratios
        ax1.set_xlim(pos[0] - tick_spacing*0.5, pos[split_idx] + tick_spacing*0.5)  
        ax2.set_xlim(pos[split_idx+1] - tick_spacing*0.5, pos[-1] + tick_spacing*0.5)

        # Set identical tick spacing for both panels
        ax1.set_xticks(np.arange(pos[0], pos[split_idx] + tick_spacing, tick_spacing))
        ax2.set_xticks(np.arange(pos[split_idx+1], pos[-1] + tick_spacing, tick_spacing))

        # Hide the spines in ax1 and ax2
        for spine in ["right", "top"]:
            ax1.spines[spine].set_visible(False)
        for spine in ["left", "right", "top"]:
            ax2.spines[spine].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        
        # Add slanted break marks at corners of Axes
        d = 2  # Proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=14,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot(1, 0, transform=ax1.transAxes, **kwargs)
        ax2.plot(0, 0, transform=ax2.transAxes, **kwargs)

        # Remove left ticks in ax2
        ax2.tick_params(left=False, labelleft=False)

        if group is not None:
            ax2.legend(fontsize=16, loc='upper right')

    ax1.set_ylabel(y, fontsize=18)
    for ax in [ax1] if no_gap else [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=18, width=2)
        if Yaxis_range is not None:
            ax.set_ylim(Yaxis_range)

    # Add shared X-axis label centered below the two panels
    fig.text(0.5, 0.01, x, ha='center', fontsize=18)

    plt.show()

    if exp_fit:
        return fit_params
