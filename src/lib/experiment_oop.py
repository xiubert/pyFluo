import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from operator import itemgetter

import lib.fileIngest as fileIngest
import lib.plotting as plotting
import lib.metadataProcess as metadataProcess
import lib.imgProcess as imgProcess
import lib.signalProcess as signalProcess

class Experiment:
    def __init__(self, directory, 
                 parent = None, 
                 format: str = 'MAK', 
                 subfolder: bool = False, 
                 drop_missing_dB: bool = True,
                 t_base: tuple = (2.0, 3.0), t_resp: tuple = (3.3, 4.0)
                 ):
        """
        Initializes an Experiment instance, either standalone or as part of an ExperimentGroup.

        Args:
            directory (str): Path to the experiment directory.
            parent (ExperimentGroup, optional): Reference to the parent ExperimentGroup, if any.
            format (str): Format for extracting dB values from pulse metadata ('MAK' or 'PAC').
            subfolder (bool): Whether to search recursively within subfolders.
            drop_missing_dB (bool): Whether to automatically drop traces where dB was not identified.
        """
        self.directory = directory
        self.parent = parent  # Parent experiment group (optional)
        self.format = format
        self.subfolder = subfolder
        self.t_base = t_base
        self.t_resp = t_resp

        if self.parent is not None:
            # If part of a group, reference the group's DataFrame
            self.df = self.parent.df[self.parent.df['dir'] == self.directory]
            self.qcam2img = self.parent.qcam2img  # Shared reference
            self.qcam2header = self.parent.qcam2header  # Shared reference
        else:
            # If standalone, create its own DataFrame and storage
            self.df = fileIngest.qcamPath2table([self.directory], self.format, self.subfolder)
            if drop_missing_dB:
                self.df = self.df[~self.df['dB'].isna()]
            # Load treatment / injection metadata
            self.df['treatment'] = metadataProcess.getInjectionCond(self.df)
            
            self.qcam2img = {}
            self.qcam2header = {}

    def _repr_html_(self):
        return self.df._repr_html_()

    def load_qcam_data(self, **kwargs):
        """Loads qcam data, either independently or using the parent ExperimentGroup."""
        if self.parent:
            # eventually may need method here for adding experiments to experiment group
            pass
            # self.parent.load_qcam_data()  # Load centrally for all experiments
        else:
            self.df, self.qcam2img, self.qcam2header = fileIngest.loadQCamTable(self.df, **kwargs)

    def process_signal(self, **kwargs):
        t_base = kwargs.get('t_base', self.t_base)
        t_resp = kwargs.get('t_resp', self.t_resp)
        self.df = signalProcess.updateTable_signal(self.df, self.qcam2img, 
                                                   t_base=t_base, t_resp=t_resp, **kwargs)

    def plot_average_fluorescence(self):
        """Plots the average fluorescence trace for this experiment."""
        fig = sp.make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        # Handle filtered dataframe
        if self.parent is not None:
            df_plot = self.parent.df[self.parent.df['dir'] == self.directory]
        else:
            df_plot = self.df

        # Generate colormap for the number of traces
        colors = plotly.colors.sample_colorscale(plotly.colors.get_colorscale('RdBu'), np.linspace(0, 1, len(df_plot)))[::-1]

        # Plot each trace with its corresponding color
        for (_,df_time_file), color in zip(df_plot.sort_values(by='timestamp_init')[['timestamp_init','qcam']].iterrows(), colors):
            fig.add_trace(go.Scatter(y=self.qcam2img[df_time_file['qcam']].mean(axis=(0, 1)), 
                                    name=f"{pd.to_datetime(df_time_file['timestamp_init']).strftime('%H:%M:%S')}_{os.path.basename(df_time_file['qcam'])}",
                                    line=dict(color=color, width=1)),
                        row=1, col=1)
        
        # Plot the experiment average
        fig.add_trace(
            go.Scatter(y=np.array(itemgetter(*df_plot['qcam'].tolist())(self.qcam2img)).mean(axis=(0,1,2)),
                    name="experiment average",
                    line=dict(color='black', width=2)),  # Black dashed for contrast
            row=2, col=1
        )

        # Update layout
        fig.update_layout(title=f"Avg Fluorescence - {self.directory}",
                        xaxis1=dict(title="average across all traces"),
                        xaxis2=dict(title='frame'),
                        yaxis=dict(title='rawF'),
                        yaxis2=dict(title='rawF')
        )

        fig.show()
    
    def plot_experiment_overview(self, **kwargs):
        # in case ExperimentGroup df was filtered:
        if self.parent is not None:
            qFiles = self.parent.df[self.parent.df['dir'] == self.directory]['qcam'].tolist()
        else:
            qFiles = self.df['qcam'].tolist()
         
        plotting.experimentAvgPlot(qFiles=qFiles,**kwargs)

    def plot_experiment_segments(self, n_segments, **kwargs):
        colors = cm.coolwarm(np.linspace(0, 1, n_segments))

        df_plot = self.df
        t = signalProcess.getTimeVec(df_plot.iloc[0]['nFrames'])
        df_plot['rawF'] = df_plot['qcam'].apply(lambda x: self.qcam2img[x].mean(axis=(0,1)))
        df_plot['F_linFilt'] = df_plot.apply(lambda x: signalProcess.subtractLinFit(t,x['rawF'], offset=False)[0],axis=1)

        for d, df_dir in df_plot.groupby('dir'):

            # Convert groupby object to list and sort
            treatments = [(treatment, df) for treatment, df in df_dir.groupby('treatment')]
            # Sort so that pre comes first
            treatments.sort(key=lambda x: 0 if 'pre' in x[0] else 1)

            _, ax = plt.subplots(2, len(treatments), figsize=(14, 8))

            for t_i, (treatment, df_treat) in enumerate(treatments):
                # Calculate elapsed_s within the treatment
                df_treat['elapsed_s'] = (df_treat['timestamp_init'] - df_treat['timestamp_init'].min()).dt.total_seconds()
                
                # Dynamically calculate the quartiles (qcut) for the elapsed time
                if n_segments==5:
                    df_treat['elapsed_q'] = pd.qcut(df_treat['elapsed_s'], q=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
                elif n_segments==4:
                    df_treat['elapsed_q'] = pd.qcut(df_treat['elapsed_s'], q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
                elif n_segments==3:
                    df_treat['elapsed_q'] = pd.qcut(df_treat['elapsed_s'], q=3, labels=['0-33%', '33-66%', '66-100%'])
                elif n_segments==2:
                    df_treat['elapsed_q'] = pd.qcut(df_treat['elapsed_s'], q=2, labels=['0-50%', '50-100%'])

                for i, (a, b) in enumerate(df_treat.groupby('elapsed_q')):
                    u, upsem, umsem = signalProcess.meanPlusMinusSem(np.array(b['F_linFilt'].tolist()))
                    ax_p1 = (ax[0,t_i] if len(treatments)>1 else ax[0])
                    ax_p1.plot(t, u, '-', color=colors[i], label=a)
                    ax_p1.fill_between(t, umsem, upsem, alpha=0.2)
                    ax_p1.set_xlabel('time (s)')
                    ax_p1.set_ylabel('F')
                    ax_p1.set_title(treatment)
                ax_p1.legend(title="segment",loc='lower left', bbox_to_anchor=(1, 0.5))
                
                ax_p2 = (ax[1,t_i] if len(treatments)>1 else ax[1])
                sns.countplot(
                    data=df_treat,
                    x='elapsed_q',
                    hue='dB',
                    palette='Reds',
                    saturation=1,
                    ax = ax_p2
                )

                # Improve the appearance
                ax_p2.set_title(treatment, fontsize=12)
                ax_p2.set_xlabel('Elapsed Time Segment', fontsize=10)
                ax_p2.set_ylabel('Count', fontsize=10)
                ax_p2.legend(title='dB', bbox_to_anchor=(1.15, 1), loc='upper right')

                # Add some spacing for the legend
                plt.tight_layout()
        
            plt.suptitle(f"{d}")  # Title includes `dir` only
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            plt.show()

    
    def plotDF_levelByTreatment(self, **kwargs):
        # in case ExperimentGroup df was filtered:
        if self.parent is not None:
            df_plot = self.parent.df[self.parent.df['dir'] == self.directory]
        else:
            df_plot = self.df
            
        plotting.plotDF_levelByTreatment(df_plot,self.qcam2img,**kwargs)


    def plot_respHeatmap(self, **kwargs):
        t_base = kwargs.get('t_base', self.t_base)
        t_resp = kwargs.get('t_resp', self.t_resp)
        plotting.plot_respHeatmap(self.df,  t_baseline = t_base, t_temporalAvg = t_resp, **kwargs)


    def get_ROI_mask(self, condition: str = None, **kwargs):
        if condition:
            qcams = self.df.loc[self.df['treatment'].str.startswith(condition),'qcam'].tolist()
            avgImgSeries = np.array(itemgetter(*qcams)(self.qcam2img)).mean(axis=0)
            saveName = f"response_mask_{condition}"
        else:
            avgImgSeries = np.array(list(self.qcam2img.values())).mean(axis=0)
            saveName = "response_mask"

        spatialDFF = imgProcess.calcSpatialDFFresp(avgImgSeries, **kwargs)
        ui, mask_output = imgProcess.getROImaskUI(spatialDFF, expDir = self.directory, saveName=saveName, **kwargs)
        
        return ui, mask_output, avgImgSeries, spatialDFF

    def plot_ROI_mask(self, mask_output, avgImgSeries, spatialDFF):
        _,ax = plt.subplots(1,2)
        ax[0].imshow(avgImgSeries.mean(axis=-1),cmap='gray')
        ax[1].imshow(spatialDFF,cmap='jet')
        for axi in ax:
            axi.plot(mask_output['ROIcontour'][:,0],mask_output['ROIcontour'][:,1],'w-',linewidth=2)

    def plot_avgDFF(self, measure_col='dB', resp_col="dFF_ROI_linFilt_butterFilt_peak", avgAnimal=False, normalize=None, **kwargs):
        capsize = kwargs.get('capsize', 3)
        return plotting.plot_avgDFF_acrossAnimal(self.df, measure_col=measure_col, resp_col=resp_col, 
                                                 avgAnimal=avgAnimal, normalize=normalize, capsize=capsize, **kwargs)



class ExperimentGroup:
    def __init__(self, experiment_dirs: list, format: str = 'MAK', subfolder: bool = False, drop_missing_dB: bool = True):
        """
        Initializes an ExperimentGroup that contains multiple experiments.

        Args:
            experiment_dirs (list): List of directories corresponding to individual experiments.
            format (str): Format for extracting dB values from pulse metadata ('MAK' or 'PAC').
            subfolder (list[bool], optional): Whether to search recursively within subfolders.
            drop_missing_dB (bool): Whether to automatically drop traces where dB was not identified.

        """
        self.experiment_dirs = experiment_dirs
        self.format = format
        self.subfolder = subfolder
        self.qcam2img = {}  # Centralized storage
        self.qcam2header = {}

        # Generate metadata table for all experiments at once
        self.df = fileIngest.qcamPath2table(self.experiment_dirs, self.format, self.subfolder)
        if drop_missing_dB:
            self.df = self.df[~self.df['dB'].isna()]

        # Load treatment / injection metadata
        self.df['treatment'] = metadataProcess.getInjectionCond(self.df)

        # Initialize experiment objects (referencing the same dataframe and shared storage)
        self.experiments = [Experiment(directory, self) for directory in experiment_dirs]

    def _repr_html_(self):
        return self.df._repr_html_()

    def load_qcam_data(self, **kwargs):
        """Loads qcam data for all experiments efficiently."""
        self.df, qcam2img, qcam2header = fileIngest.loadQCamTable(self.df, **kwargs)

        # Update shared storage
        self.qcam2img.update(qcam2img)
        self.qcam2header.update(qcam2header)

    def plot_all_experiments(self):
        """Plots average fluorescence traces for all experiments in the group."""
        fig = go.Figure()

        for dir_name, group_df in self.df.groupby('dir'):
            dir_traces = np.array(itemgetter(*group_df['qcam'].tolist())(self.qcam2img))
            fig.add_trace(go.Scatter(y=dir_traces.mean(axis=(0,1,2)), name=dir_name))

        fig.update_layout(
            title="Avg Fluorescence - All Experiments",
            xaxis_title="Frame",
            yaxis_title="Raw Fluorescence"
        )
        fig.show()


# in case of adding plot_experiment method to experiment group
# #         if self.parent:
#             experiment_index = self.parent.experiments.index(self)
#             self.parent.plot_experiment_fluorescence(experiment_index)

#     def plot_experiment_fluorescence(self, experiment_index: int):
#         """Plots average fluorescence for a specific experiment in the group."""
#         if 0 <= experiment_index < len(self.experiments):
#             experiment = self.experiments[experiment_index]
#             fig = sp.make_subplots(
#                 rows=2, cols=1, 
#                 shared_xaxes=True,
#                 vertical_spacing=0.1,
#             )

#             # Use group's qcam2img since it's centralized
#             exp_files = [f for f in self.qcam2img if f.startswith(experiment.directory)]
#             exp_imgs = {f: self.qcam2img[f] for f in exp_files}

#             for file, img_data in exp_imgs.items():
#                 avg_trace = img_data.mean(axis=(0, 1))  # Average over spatial dimensions
#                 fig.add_trace(go.Scatter(y=avg_trace, name=os.path.basename(file)), row=1, col=1)

#             fig.add_trace(
#                 go.Scatter(y=np.array(list(exp_imgs.values())).mean(axis=(0,1,2))),
#                 row=2, col=1
#             )

#             fig.update_layout(title=f"Avg Fluorescence - {experiment.directory}",
#                               xaxis1=dict(title="average across all traces"),
#                               xaxis2=dict(title='frame'),
#                               yaxis=dict(title='rawF'),
#                               yaxis2=dict(title='rawF'))
#             fig.show()
#         else:
#             print(f"Invalid experiment index: {experiment_index}. Choose between 0 and {len(self.experiments)-1}.")