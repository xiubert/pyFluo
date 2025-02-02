import pandas as pd
import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from operator import itemgetter

import lib.fileIngest as fileIngest

class Experiment:
    def __init__(self, directory, format: str = 'MAK', subfolder: bool = False):
        """
        Initializes an Experiment instance.

        Args:
            directory (str): Path to the experiment directory.
            format (str): Format for extracting dB values from pulse metadata ('MAK' or 'PAC').
            subfolder (bool): Whether to search recursively within subfolders.
        """
        self.directory = directory
        self.format = format
        self.subfolder = subfolder
        self.df = self._generate_metadata_table()
        self.qcam2img = {}
        self.qcam2header = {}
    
    def _repr_html_(self):
        return self.df._repr_html_()

    def _generate_metadata_table(self) -> pd.DataFrame:
        """Loads qcam metadata."""
        df = fileIngest.qcamPath2table([self.directory], self.format, self.subfolder)  # Generates metadata table

        return df
    
    def load_qcam_data(self):
        """Extracts qcam image/header data."""        
        # Extract image and header data
        self.df, self.qcam2img, self.qcam2header = fileIngest.loadQCamTable(self.df)

    def plot_average_fluorescence(self):
        """Plots average fluorescence trace for this experiment."""
        fig = sp.make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        for file, img_data in self.qcam2img.items():
            avg_trace = img_data.mean(axis=(0, 1))  # Average over spatial dimensions
            fig.add_trace(go.Scatter(y=avg_trace, name=os.path.basename(file)), row=1, col=1)
        
        fig.add_trace(
            go.Scatter(y=np.array(list(self.qcam2img.values())).mean(axis=(0,1,2))),
            row=2, col=1
        )
        # fig.update_yaxes(title_text="average across all traces", row=2, col=1)
        
        fig.update_layout(title=f"Avg Fluorescence - {self.directory}",
                          xaxis1=dict(title="average across all traces"),
                          xaxis2=dict(title='frame'),
                          yaxis=dict(title='rawF'),
                          yaxis2=dict(title='rawF')
        )
        fig.show()


class ExperimentGroup:
    def __init__(self, experiment_dirs: list, format: str = 'MAK', subfolder: list[bool] = None):
        self.experiments = [Experiment(d) for d in experiment_dirs]
        self.format = format
        self.subfolder = subfolder
        self.df = self._generate_metadata_table()
        self.qcam2img = {}
        self.qcam2header = {}

    def _generate_metadata_table(self) -> pd.DataFrame:
        """Loads qcam metadata."""
        """Combines metadata from all experiments into a single DataFrame."""
        all_dfs = []
        for experiment in self.experiments:
            if experiment.df is not None:
                df_copy = experiment.df.copy()
                all_dfs.append(df_copy)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
        
    def _repr_html_(self):
        return self.df._repr_html_()

    def load_qcam_data(self):
        """Extracts qcam image/header data."""        
        # Extract image and header data
        self.df, self.qcam2img, self.qcam2header = fileIngest.loadQCamTable(self.df)

    def plot_all_experiments(self):
        """Plots average fluorescence traces for all experiments on the same plot."""
        fig = go.Figure()
        for a,b in self.df.groupby('dir'):
            dirTraces = np.array(itemgetter(*b['qcam'].tolist())(self.qcam2img))
            fig.add_trace(go.Scatter(y=dirTraces.mean(axis=(0,1,2)), name=a))

        fig.update_layout(title="Avg Fluorescence - All Experiments",
                            xaxis_title="Frame",
                            yaxis_title="rawF")
        fig.show()