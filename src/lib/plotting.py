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

from datetime import datetime
import os
from glob import glob
from operator import itemgetter
import warnings

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


def plotDF_levelByTreatment(df: pd.DataFrame, qcam2img: dict, dFResp: bool = False, 
                            sepPlot: bool = True, errBar: bool = True, **kwargs):
    """
    Plot fluorescence response (dFF or dF) traces by treatment and dB.

    Args:
        df (pd.DataFrame): Metadata dataframe including columns `dB` and `treatment`.
        qcam2img (dict): Dictionary mapping each qcam file path to its corresponding image data.
        dFResp (bool, optional): If true, calculate dF response rather than dFF.
        sepPlot (bool, optional): Whether to create separate subplots for each treatment.
                                  - `True`: For each subplot, lower dBs are in cooler colors and higher dBs are in warmer colors.
                                  - `False`: Treatments are distinguished by different color types, 
                                             with lower dBs in lighter colors and higher dBs in darker colors.
        errBar (bool, optional): If true, add error bars to curves.
        **kwargs: Optional keyword arguments.
            example: roi_mask (np.ndarray): 2D binary mask array specifying the region of interest.
    """

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
        # Use qualitative colors for treatments
        colors = px.colors.qualitative.Set1
        max_dB = df_sorted['dB'].max()  # Maximum dB value for normalization
        treat_values = df_sorted['treatment'].unique()
        treat2color = {treat: colors[i] for i, treat in enumerate(treat_values)}
        fig = go.Figure()

    # Calculate fluorescence response within each treatment/dB combination
    for (treatment, dB), df_group in df_sorted.groupby(['treatment', 'dB'], sort=False):
        imgSeries = np.array(itemgetter(*df_group['qcam'])(qcam2img))  # Shape: [trace, Y, X, frame]
        roi_mask = kwargs.get('roi_mask', np.ones(imgSeries.shape[1:3]))
        signal = imgSeries[:, roi_mask == 1, :].mean(axis=1)

        dFF, dF, _ = signalProcess.dFFcalc(signal, **kwargs)
        response = dF if dFResp else dFF
        mean, upper, lower = signalProcess.meanPlusMinusSem(response)
        t = signalProcess.getTimeVec(len(mean))

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
        title=f"{df_sorted.dir.unique()[0]}: Fluorescence response at each sound level by treatment",
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


def plotDFFSeriesMask(imgSeries: np.ndarray, 
                      Xcoor: float, Ycoor: float, 
                      width: float, height: float, 
                      Xshift_step: float = 3, 
                      Yshift_step: float = 2, 
                      dFResp: bool = False, 
                      displayContour: bool = True, 
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
        dFResp (bool, optional): Whether to calculate dF (`True`) or dFF (`False`).
        displayContour (bool, optional): Whether to show mask as contour (`True`) or shaded region (`False`).
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

    # Transform signal array into 3D if it is initially 4D
    avgImg_map = imgSeries.mean(axis=0) if imgSeries.ndim == 4 else imgSeries

    # Load heatmap of baseline fluorescence as static background
    ax_img.imshow(imgProcess.calcSpatialBaseFluo(avgImg_map, **kwargs), cmap='jet')

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
        mask = imgProcess.getSquareMask(X_slider.val, Y_slider.val, width, height, **kwargs)
        
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