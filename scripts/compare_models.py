# %load_ext autoreload
# %autoreload 2

from sklearn.metrics import r2_score, f1_score, confusion_matrix

import xarray as xr
import os
from matplotlib import pyplot as plt

#import torch
import numpy as np

import seaborn as sns
sns.set_style('ticks')

import argparse

pred_charts = [os.path.join('../data/arctic_sea_ice/predict/', ff) for ff in os.listdir('../data/arctic_sea_ice/predict/') if ff.endswith('_prep.nc')]
len(pred_charts)

def plot_prediction(ice_chart, plot_dir='compare-final-models', keys=['terramind-base', 'terramind-tim', 'unet'], targets=['SIC', 'SOD', 'FLOE']):

    # Create the plot
    nr = len(targets) + 1
    nc = len(keys) + 2
    fig, ax = plt.subplots(nr, nc, sharex=False, sharey=False, figsize=(nc*3, nr*3))
    shrink = 0.90

    plotvals = {'SIC': {'label':'SIC (10/10)', 'cmap':'Blues_r', 'levels':list(range(11)),
                        'ticklabels':list(range(11))},
                'SOD': {'label':'SOD', 'cmap':'rainbow', 'levels':list(range(6)),
                        'ticklabels':['Open water', 'New ice', 'Young ice', 'Thin First-year ice', 'Thick First-year ice', 'Old ice']},
                'FLOE': {'label':'FLOE', 'cmap':'rainbow', 'levels':list(range(7)),
                        'ticklabels':['Open water', 'Cake ice', 'Small floe', 'Medium floe', 'Big floe', 'Vast floe', 'Bergs']},
                }

    # Hide all plots in the first row except the first two (SAR images)
    for j in range(2, nc):
        ax[0, j].set_visible(False)
    
    # --------------------------------------------------------------
    # Reference data
    # --------------------------------------------------------------

    with xr.open_dataset(ice_chart) as ds:

        sar1 = ds["nersc_sar_primary"].values
        sar2 = ds["nersc_sar_secondary"].values
    sar1 = np.where(sar1 == 0, None, sar1).astype(float)
    sar2 = np.where(sar2 == 0, None, sar2).astype(float)

    img = ax[0, 0].imshow(sar1, cmap='Greys')
    plt.colorbar(img, orientation="vertical", shrink=shrink)
    ax[0, 0].set_title("Sentinel-1 HH")
    img = ax[0, 1].imshow(sar2, cmap='Greys')
    plt.colorbar(img, orientation="vertical", shrink=shrink)
    ax[0, 1].set_title("Sentinel-1 HV")

    with xr.open_dataset(ice_chart.replace('_prep.nc', '_prep_reference.nc')) as reference_ds:
    
        print(f'Opening reference for {ice_chart}')
    
        true_sic = reference_ds["SIC"].values
        true_sod = reference_ds["SOD"].values
        true_floe = reference_ds["FLOE"].values

    # Mask invalid values
    true_sic = np.where(true_sic == 255, None, true_sic).astype(float)
    true_sod = np.where(true_sod == 255, None, true_sod).astype(float)
    true_floe = np.where(true_floe == 255, None, true_floe).astype(float)

    true_masks = {'SIC': true_sic, 'SOD': true_sod, 'FLOE': true_floe}

    for i, target in enumerate(targets):
        img = ax[i+1, 0].contourf(true_masks[target], cmap=plotvals[target]["cmap"], levels=plotvals[target]["levels"])
        cbar = plt.colorbar(img, ax=ax[i+1, -1], shrink=shrink, label=plotvals[target]["label"])
        cbar.set_ticks(ticks=plotvals[target]["levels"], labels=plotvals[target]["ticklabels"])
        ax[i+1, 0].set_title(f"Ground truth - {target}")

    # --------------------------------------------------------------
    # Terramind-base
    # --------------------------------------------------------------

    for i, target in enumerate(targets):
        print(target)
        #if target == 'FLOE':
        #    continue
        
        for j, key in enumerate(keys):
            print(' ', key)

            with xr.open_dataset(os.path.join('../output/predictions', key, target, os.path.basename(ice_chart))) as pred_ds:
                val = np.where(np.isnan(true_masks[target]), None, pred_ds[target].values).astype(float)

            img = ax[i+1, j+1].contourf(val, cmap=plotvals[target]["cmap"], levels=plotvals[target]["levels"])
            #cbar = plt.colorbar(img, ax=ax[i+1, j+1], shrink=shrink)
            #cbar.set_ticks(ticks=plotvals[target]["levels"], labels=plotvals[target]["ticklabels"])

    # Set column titles (keys) and row titles (targets)
    
    # Set titles for each subplot with model+target
    for i, target in enumerate(targets):
        for j, key in enumerate(keys):
            if i == 0:  # Only set column headers once
                ax[0, j+1].set_title(key)
            ax[i+1, j+1].set_title(f"{key} - {target}")

    [axx.set_aspect('equal') for axx in ax.flatten()]
    [axx.set_xticks([]) for axx in ax.flatten()]
    [axx.set_yticks([]) for axx in ax.flatten()]
    [axx.axis('off') for axx in ax[:, -1].flatten()]
    fig.suptitle(os.path.basename(ice_chart))

    fig.tight_layout()

    if not os.path.exists(f"../output/plots/{plot_dir.replace(' ', '-').lower()}"):
        os.mkdir(f"../output/plots/{plot_dir.replace(' ', '-').lower()}")

    plt.savefig(os.path.join(f"../output/plots/{plot_dir.replace(' ', '-').lower()}", os.path.basename(ice_chart).replace(".nc", ".png")), bbox_inches='tight')

# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Compare model predictions on sea ice charts')
    parser.add_argument('--keys', nargs='+', default=['terramind-base', 'terramind-tim', 'unet'],
                        help='List of model keys to compare')
    parser.add_argument('--index', type=int, default=1,
                        help='Index of the prediction chart to plot')
    parser.add_argument('--plot-dir', type=str, default='compare-final-models',
                        help='Directory name for saving plots')
    parser.add_argument('--targets', nargs='+', default=['SIC', 'SOD', 'FLOE'],
                        help='List of target variables to compare')
    
    args = parser.parse_args()
    
    pred_charts = [os.path.join('../data/arctic_sea_ice/predict/', ff) for ff in os.listdir('../data/arctic_sea_ice/predict/') if ff.endswith('_prep.nc')]
    
    if len(pred_charts) > 0:
        if args.index < 0 or args.index >= len(pred_charts):
            print(f"Error: Index {args.index} out of range. Available range: 0-{len(pred_charts)-1}")
        else:
            print(f"Plotting chart at index {args.index} with models: {', '.join(args.keys)}")
            print(f"Targets: {', '.join(args.targets)}")
            print(f"Plot directory: {args.plot_dir}")
            plot_prediction(pred_charts[args.index], plot_dir=args.plot_dir, 
                           keys=args.keys, targets=args.targets)
    else:
        print("No prediction charts found.")
