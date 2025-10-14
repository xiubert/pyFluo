import tkinter as tk
from tkinter import filedialog, messagebox
import lib.experiment_oop as exp
import joblib
import os
from collections import Counter

def select_folder(entry):
    folder_path = filedialog.askdirectory(title="Select a Folder")
    entry.delete(0, 'end')
    entry.insert(0,folder_path)


def load_exp():
    global experiment
    experiment = exp.Experiment(entry_folder.get(), subfolder=var_subfolder.get(), t_base = (float(entry_t_base_min.get()),float(entry_t_base_max.get())), t_resp = (float(entry_t_resp_min.get()),float(entry_t_resp_max.get())), format="PAC")
    experiment=check_filesize(experiment)
    print(experiment.df["treatment"].value_counts())
    experiment.load_qcam_data()
    joblib.dump(experiment.df,entry_folder.get()'../data/cache/TLB10-2-4-0128_zincResp.joblib')


def check_filesize(e):
    filelist=e.df["qcam"]
    filesize=[]
    for file in filelist:
        filesize.append(os.path.getsize(file))
    c=Counter(filesize)
    common_value,_=c.most_common()[0]

    for i in range(0,len(filesize)):
        if filesize[i]!=common_value:
            e.df=e.df.drop(e.df.index[i])
            messagebox.showinfo("Info", "Remove a file")

    return e

def draw_ROI():
    global experiment, mask_output, imgSeries, spatialDFF
    ui, mask_output, imgSeries, spatialDFF = experiment.get_ROI_mask(dB_plot=int(entry_dB_plot.get()), contrast_percentile=(0.1, 99.9))
    ui.show(threaded=True)

def process_ROI():
    global experiment, mask_output, imgSeries, spatialDFF
    experiment.plot_ROI_mask(mask_output, imgSeries, spatialDFF, contrast_percentile=(0.1, 99.9))
    experiment.process_signal()
    experiment.df = experiment.df[experiment.df.valid]


global experiment

root = tk.Tk()
root.title("Data_Processing_Jinbo")

var_subfolder=tk.IntVar()
var_subfolder.set(1)

entry_folder=tk.Entry(root)
entry_folder.grid(row=0,column=0)
button_folder=tk.Button(root,text="select folder",command=lambda:select_folder(entry_folder))
button_folder.grid(row=0,column=1)
label_t_base = tk.Label(root, text="t_base")
label_t_base.grid(row=1, column=0)
entry_t_base_min=tk.Entry(root)
entry_t_base_min.grid(row=1, column=1)
entry_t_base_max=tk.Entry(root)
entry_t_base_max.grid(row=1, column=2)
entry_t_base_min.insert(0, "2")
entry_t_base_max.insert(0, "3")
label_t_resp = tk.Label(root, text="t_resp")
label_t_resp.grid(row=1, column=3)
entry_t_resp_min=tk.Entry(root)
entry_t_resp_min.grid(row=1, column=4)
entry_t_resp_max=tk.Entry(root)
entry_t_resp_max.grid(row=1, column=5)
entry_t_resp_min.insert(0, "3.3")
entry_t_resp_max.insert(0, "4")
check_subfolder=tk.Checkbutton(root, text="search recursively within subfolders", variable=var_subfolder)
check_subfolder.grid(row=0, column=4)
button_load=tk.Button(root,text="Load experiment data", command=lambda: load_exp())
button_load.grid(row=0, column=5)
button_plot_average_fluorescence=tk.Button(root, text="plot_average_fluorescence", command=lambda: experiment.plot_average_fluorescence())
button_plot_average_fluorescence.grid(row=2, column=0)
label_n_segment = tk.Label(root, text="n_segment")
label_n_segment.grid(row=2, column=1)
entry_n_segment=tk.Entry(root)
entry_n_segment.grid(row=2, column=2)
entry_n_segment.insert(0,"5")
button_plot_experiment_segments=tk.Button(root, text="plot_experiment_segments", command=lambda: experiment.plot_experiment_segments(int(entry_n_segment.get())))
button_plot_experiment_segments.grid(row=2, column=3)
button_plot_experiment_overview=tk.Button(root, text="plot_experiment_overview", command=lambda: experiment.plot_experiment_overview())
button_plot_experiment_overview.grid(row=2, column=4)
button_plotDF_levelByTreatment=tk.Button(root, text="plotDF_levelByTreatment", command=lambda: experiment.plotDF_levelByTreatment())
button_plotDF_levelByTreatment.grid(row=2, column=5)
label_dB_plot = tk.Label(root, text="dB_plot")
label_dB_plot.grid(row=3, column=0)
entry_dB_plot=tk.Entry(root)
entry_dB_plot.grid(row=3, column=1)
entry_dB_plot.insert(0,"60")
button_plot_respHeatmap=tk.Button(root, text="plot_respHeatmap", command=lambda: experiment.plot_respHeatmap(dB_plot=int(entry_dB_plot.get()), contrast_percentile=(0.1, 99.9)))
button_plot_respHeatmap.grid(row=3, column=2)
label_condition=tk.Label(root, text="condition")
label_condition.grid(row=4, column=0)
entry_condition_1=tk.Entry(root)
entry_condition_1.grid(row=4, column=1)
entry_condition_2=tk.Entry(root)
entry_condition_2.grid(row=5, column=1)
button_draw_ROI=tk.Button(root, text="draw ROI", command=lambda: draw_ROI())
button_draw_ROI.grid(row=4, column=2)


button_process_ROI=tk.Button(root, text="process ROI", command=lambda: process_ROI())
button_process_ROI.grid(row=7, column=2)
root.mainloop()