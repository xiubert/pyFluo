import numpy as np
import pandas as pd

import colorsys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns

from datetime import datetime
import os
from glob import glob
from operator import itemgetter
import warnings
import math

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


def plot_respHeatmap(df: pd.DataFrame, dB_plot: int = 80, same_scale: bool = True, 
                     ROIcontour: np.ndarray | dict[str, np.ndarray] | None = None, **kwargs):
    """
    Plot baseline gray-scaled wide-field images and response heatmaps for each treatment.
    Optionally add ROI mask contours to images for visualization.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'qcam', 'dB', and 'treatment'.
        dB_plot (int, optional): Sound intensity (in dB) to plot response heatmaps. Defaults to 80.
                                 If None, plot average across all intensities.
        ROIcontour (dict, optional): ROI's vertex coordinates, including a repeated first vertex to close the shape.
                                     Either: - 2D numpy array (same ROI for all treatments).
                                             - Dictionary mapping treatments to 2D arrays (different ROIs for each treatment).
                                             - None (no contours shown).
        same_scale (bool, optional): If 'True', use same color scaling in all heatmaps.
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
        dFF_min, dFF_max = min(dFF.min() for dFF in all_spatialDFF), max(dFF.max() for dFF in all_spatialDFF)

    # Plot baseline gray-scaled wide-field images and response heatmaps for each treatment
    for i, treatment in enumerate(treatments):
        qcams = (df[df['treatment'] == treatment]['qcam'].tolist() if dB_plot is None 
                else df[(df['treatment'] == treatment) & (df['dB'] == dB_plot)]['qcam'].tolist())

        # Calculate spatial dFF for response heatmap
        _, _, imgs, spatialDFF = imgProcess.qcams2roiTrace(qcams, **kwargs)

        # Plot images with labels/titles and colorbars
        ax[i,0].imshow(imgs.mean(axis=(0,-1)), 'gray')
        vmin, vmax = (dFF_min, dFF_max) if same_scale else (None, None)
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


def plot_traces(df: pd.DataFrame, dB_plot: int = 80, resp_col: str = 'dFF_ROI_raw', 
                sepPlot: bool = False, stimStart: float = 3.0, alpha_ind: float = 0.3, **kwargs):
    """
    Plot individual and averaged traces for a given sound intensity across different treatments.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 
                           'dB', 'treatment', 'time' (or 'nFrames'), and column for response traces.
        dB_plot (int, optional): Sound intensity (in dB) for traces to be plotted. Defaults to 80.
        resp_col (str, optional): Column name for response traces. Defaults to 'dFF_ROI_raw'.
        sepPlot (bool, optional): If True, plot treatments in separate subplots; otherwise, plot in one plot. 
                                  Defaults to 'False'.
        stimStart (float, optional): Stimulus start time (in seconds). Defaults to 3.0.
        alpha_ind (float, optional): Transparency for individual traces. Defaults to 0.3.
        **kwargs: Optional arguments that will override default.
    """
    
    # Check whether required columns exist
    required_cols = ['dB', 'treatment', resp_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")
    
    # Check whether specified sound level exists
    if dB_plot not in df['dB'].unique():
        raise ValueError(f"dB_plot={dB_plot} not found in the 'dB' column.")
    
    # Filter the DataFrame for the specified sound intensity
    filtered_df = df[df['dB'] == dB_plot].reset_index(drop=True)
    
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
        current_ax.legend()
    
    fig.suptitle(f"Individual and Averaged Traces: {dB_plot} dB", size=14)
    plt.tight_layout()
    plt.show()


def plotDF_levelByTreatment(df: pd.DataFrame, qcam2img: dict = None, resp_col: str = None, dFResp: bool = False, 
                            sepPlot: bool = True, errBar: bool = True, **kwargs):
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
                            SEMbar: bool = True) -> pd.DataFrame:
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
    
    # Check whether control treatment exists if specified
    if ctrl_treat is not None and ctrl_treat not in df['treatment'].unique():
        raise ValueError(f"{ctrl_treat} not found in the 'treatment' column.")

    # Check whether normalization is applicable
    if normalize == 'byGroup' and not avgAnimal:
        warnings.warn("'byGroup' normalization requires 'avgAnimal=True' - skipping normalization")
        normalize = None

    # Filter data for the specified dB level
    df_filtered = df[df['dB'] == dB_plot].copy().reset_index(drop=True)
    
    # Apply normalization if specified
    if normalize == 'byTrial':
        # Normalize each trial's response to the max response within the same animal
        df_filtered[resp_col] = df_filtered.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / x.max()) * 100)

    elif normalize == 'byGroup' and avgAnimal:
        # Get the first treatment that appears in the dataframe as control if not specified
        control_treatment = ctrl_treat if ctrl_treat is not None else df_filtered['treatment'].iloc[0]
        # Verify control treatment exists for all animals
        missing_ctrl = set(df_filtered['dir']) - set(df_filtered[df_filtered['treatment'] == control_treatment]['dir'])
        if missing_ctrl:
            raise ValueError(f"Animals {missing_ctrl} missing control treatment '{control_treatment}'")
        # Calculate mean response for each animal/treatment combination
        temp_means = df_filtered.groupby(['dir', 'treatment'], sort=False)[resp_col].mean().reset_index()
        # Get control group means for each animal
        control_means = temp_means[temp_means['treatment'] == control_treatment].set_index('dir')[resp_col]
        # Normalize all treatments to the control group mean for each animal
        temp_means[resp_col] = temp_means.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / control_means[x.name]) * 100)
        # Merge the normalized values back to the original dataframe
        df_filtered = df_filtered.drop(resp_col, axis=1).merge(temp_means, on=['dir', 'treatment'])

    elif normalize == 'trial2group':
        # Get the first treatment that appears in the dataframe as control if not specified
        control_treatment = ctrl_treat if ctrl_treat is not None else df_filtered['treatment'].iloc[0]
        # Verify control treatment exists for all animals
        missing_ctrl = set(df_filtered['dir']) - set(df_filtered[df_filtered['treatment'] == control_treatment]['dir'])
        if missing_ctrl:
            raise ValueError(f"Animals {missing_ctrl} missing control treatment '{control_treatment}'")
        # Calculate mean response for control group of each animal
        control_means = df_filtered[df_filtered['treatment'] == control_treatment].groupby('dir')[resp_col].mean()
        # Normalize all trials to their animal's control group mean
        df_filtered[resp_col] = df_filtered.groupby('dir', sort=False)[resp_col].transform(lambda x: (x / control_means[x.name]) * 100)
    
    if avgAnimal:
        # Calculate mean response for each animal under each treatment
        plot_data = df_filtered.groupby(['dir', 'treatment'], sort=False)[resp_col].mean().reset_index()
    else:
        # Use all individual trials
        plot_data = df_filtered[['dir', 'treatment', resp_col]].copy()
    
    # Calculate group statistics (mean and SEM/SD) for each treatment
    df_stats = plot_data.groupby('treatment', sort=False)[resp_col].agg(['count', 'mean', 'std', 'sem']).reset_index()
    
    # Initialize barplot
    fig, ax = plt.subplots(figsize=(4, 6))
    palette = plt.cm.tab10.colors
    # palette = sns.color_palette("Set1")
    error = df_stats['sem'] if SEMbar else df_stats['std']
    nTreat = len(df_stats['treatment'])
    istwo_treatments = nTreat==2
    
    # Plot bars with error
    ax.bar(df_stats['treatment'], df_stats['mean'], 
           yerr=error, capsize=5, color=palette[:nTreat], width=0.3)
    
    # Create a dictionary mapping treatments to x-positions
    treatment_pos = {t: i for i, t in enumerate(df_stats['treatment'])}
    
    for animal in plot_data['dir'].unique():
        animal_data = plot_data[plot_data['dir'] == animal]
        
        # If there are exactly two treatments, add offsets to dots
        if istwo_treatments:
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
            x_jitter = x + (np.random.normal(0, 0.03) if not avgAnimal else 0)
            edge_color = palette[treatment_pos[t]] if istwo_treatments else 'k'
            point_size = 40 if not avgAnimal else 60
            ax.scatter(x_jitter, y, facecolors='none', edgecolors=edge_color,
                       s=point_size, linewidth=2 if avgAnimal else 1.5)
    
    # Add labels and title
    ax.set_xlabel('Treatment', fontsize=12)
    ylabel = f'{resp_col} Response (%)' if normalize else f'{resp_col} Response'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Fluorescence Peak Response at {dB_plot} dB', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()

    return df_stats


def plot_avgDFF_acrossAnimal(df: pd.DataFrame, 
                             measure_col: str = 'dB', 
                             resp_col: str = 'dFF_ROI_linFilt_butterFilt_peak', 
                             avgAnimal: bool = True, 
                             normalize: str = None, 
                             SEMbar: bool = True, 
                             plotAvg_reLevel: str = 'line', 
                             **kwargs) -> pd.DataFrame:
    """
    Plot barplots or lineplots with error bars for fluorescence peak response averaged across animals re sound intensities.
    Create a new dataframe including the mean, SD, and SEM of peak response for each sound level.
    Optionally export the dataframe as an excel file.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns: 'dir', 'treatment', 'dB', and column for the response variable.
        measure_col (str, optional): Column name for the independent variable. Can be sound intensity or frequency.
                                     Defaults to 'dB' (sound intensity).
        resp_col (str, optional): Column name for the response variable. Defaults to 'dFF_ROI_linFilt_butterFilt_peak'.
        avgAnimal (bool, optional): Whether to average peak response across animals or individual trials.
                                    - 'True': Average in two steps:
                                              First average peak responses within each animal, then average the mean across animals.
                                              Error bars represent SEM or SD across animals.
                                    - 'False': Average in one step:
                                               Average peak responses of all individual trials from all animals.
                                               Error bars represent SEM or SD across trials.
        normalize (str, optional): Whether to normalize peak response to the max response (in percentage). 
                                   - 'byGroup': For each animal, calculate the mean for each sound level, then normalize these means to the max mean.
                                                Only applicable when 'avgAnimal' is 'True'.
                                   - 'byTrial': For each animal, normalize all individual trials to the trial with the max response.
                                   - None: No normalization is applied.
        SEMbar (bool, optional): If 'True', use standard error (SEM) for error bars. If 'False', use standard deviation (SD).
        plotAvg_reLevel (str, optional): Whether to plot averaged response across animals re sound level.
                                         - 'line': Plot treatments in different colors in one lineplot.
                                         - 'bar': Plot treatments in multiple barplots sharing the same Y-axis scale.
                                         - None: Only return the statistics dataframe.
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
    
    # Check whether normalization is applicable
    if normalize == 'byGroup' and not avgAnimal:
        warnings.warn("'byGroup' normalization requires 'avgAnimal=True' - skipping normalization")
    
    # Group by 'dir' and 'treatment' and extract unique 'dB' values
    dB_lists = list(df.groupby(['dir', 'treatment'], sort=False)[measure_col].unique())

    # Find the common 'dB' levels across all animals
    common_dB = set(dB_lists[0])    # Start with the first list
    for item in dB_lists[1:]:    # Iterate over the remaining lists
        common_dB.intersection_update(item)
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
            # Normalize mean response of each treatment/dB combination group to the max mean response within the same animal
            df_grouped[resp_col] = df_grouped.groupby(['dir'], sort=False)[resp_col].transform(lambda x: (x / x.max()) * 100)
            print('Normalized on the treatment/dB combination group basis.')
    else:
        # Maintain the unaveraged original data of each individual trial
        df_grouped = df_filtered.loc[:, ['treatment', measure_col, resp_col]]

    # Calculate the mean, standard deviation (SD), and standard error (SEM) across dirs for each treatment/dB combination
    agg_dict = {
        f'count_{resp_col}': (resp_col, 'size'), 
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
        plt.xlabel("Sound Intensity (dB)")
        plt.ylabel(f"{resp_col} Response" if normalize is None else f"{resp_col} Response (%)")
        plt.legend(title="Treatment")
        plt.title("Response by Sound Intensity and Treatment")
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
        g.set_axis_labels("Sound Intensity (dB)", f"{resp_col} Response" if normalize is None else f"{resp_col} Response (%)")
        g.set_titles(col_template="{col_name}")
        plt.show()

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
    if not (-90 < shift_direct < 90):
        raise ValueError("`shift_direct` must be between -90 and 90 degrees.")
    
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