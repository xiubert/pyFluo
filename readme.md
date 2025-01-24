# Scripts for low-level analysis of fluorescence imaging data

## Getting started
1. Clone repository `https://github.com/xiubert/pyFluo.git` and change to respository directory (`cd pyFluo`).
2. Create python venv for running these scripts to isolate dependencies: `python -m venv env`
3. Activate virtual environment:
    - Unix: `source env/bin/activate`
    - Windows: 
        - VSCode terminal defaults to PowerShell: `.\env\Scripts\Activate.ps1`
        - If in command prompt `.\env\bin\activate.bat`
4. Install dependencies: `pip install -r requirements.txt`

## Quick overview of experiment average raw fluorescence
- Notebook: `src/avgExperiment.ipynb`

## Organizing metadata of multiple experiment directories
- Notebook: `src/getExprmntMetadata.ipynb`

## Plotting mean fluorescence within selectable ROI across traces
- Notebook: `src/getROI_fluoResp.ipynb`

## Relating blood vessel movement with fluorescence responses
- Notebook: `src/FandXY_reSoundStim.ipynb`

## Exploring effects of experiment duration on fluorescence traces
- Notebook: `src/F_exprmnt_duration.ipynb`

## Example exploratory data analysis on a single experiment
- Notebook: `src/single_experiment_EDA.ipynb`

## Considering less biased approaches to selecting response areas / ROIs
- Notebook: `src/unbiasedROI.ipynb`

