#!/usr/bin/env python
"""
Script for running tiled inference on Arctic Sea Ice data.
Converted from the predict.ipynb notebook.
"""

import os
import sys
import argparse
import xarray as xr
import torch
import terratorch
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.datasets.arctic_sea_ice import ArcticSeaIceDataModule


def setup_inference_parameters():
    """Set up the tiled inference parameters."""
    return TiledInferenceParameters(
        h_crop=768, 
        h_stride=128, 
        w_crop=768, 
        w_stride=128, 
        average_patches=True, 
        batch_size=8, 
        delta=8,
        verbose=False
    )


def load_model(checkpoint_path):
    """Load the model from checkpoint and move to GPU if available."""
    model = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(checkpoint_path).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"Model loaded and running on {model.device}")
    return model


def setup_datamodule(target, renormalize):
    """Set up the data module for inference."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/arctic_sea_ice/')
    datamodule = ArcticSeaIceDataModule(
        data_path, 
        target=target,
        fill_values_to_nan=False,
        renormalize=renormalize,
        return_chart_name=True
    )
    datamodule.setup(stage="predict")
    return datamodule


def create_output_directory(predictions_path, key, target):
    """Create output directory if it doesn't exist."""
    target_dir = os.path.join(predictions_path, key, target)
    print(f"Creating output directory: {target_dir}")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    return target_dir


def model_forward(x, model, **kwargs):
    """Wrapper function for model forward pass."""
    return model(x, **kwargs).output


def run_predictions(model, datamodule, tiled_inference_parameters, predictions_path, key, target, renormalize, checkpoint):
    """Run predictions on all samples in the dataset."""
    for idx in range(len(datamodule.predict_ds)):
        sample = datamodule.predict_ds.__getitem__(idx)
        ice_chart = sample["ice_chart"]

        # Check if it exists already
        target_path = os.path.join(predictions_path, key, target, ice_chart)
        if os.path.exists(target_path):
            print(f"Warning: {target_path} is not empty. Will continue")

        # Wrap the model_forward function to include our model
        forward_fn = lambda x, **kwargs: model_forward(x, model, **kwargs)
        
        # Run tiled inference
        pred = tiled_inference(
            forward_fn,
            sample["image"].unsqueeze(0).to(model.device),
            model._hparams["model_args"]["num_classes"],
            tiled_inference_parameters
        )
        pred = pred.squeeze(0).argmax(dim=0).to("cpu")

        # Create xarray dataset
        ds = xr.Dataset(data_vars={target: (("sar_lines", "sar_samples"), pred.cpu().numpy())})
        ds.attrs.update({
            "checkpoint": os.path.abspath(checkpoint),
            "key": key,
            "renormalize": str(renormalize)
        })

        # Save predictions
        ds.to_netcdf(target_path)

        del sample, pred

        print(f"Processed {ice_chart}")


def main():
    """Main function to parse arguments and run the prediction pipeline."""
    parser = argparse.ArgumentParser(description='Run tiled inference on Arctic Sea Ice data')
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint')
    parser.add_argument('--renormalize', action='store_true', help='Whether to renormalize the data')
    parser.add_argument('--key', default='terramind-base-renormalize-true', help='Experiment description key')
    parser.add_argument('--target', default='SIC', help='Target variable name')
    parser.add_argument('--predictions_path', default='./output/predictions', help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Setup
    tiled_inference_parameters = setup_inference_parameters()
    
    # Ensure output directory exists
    create_output_directory(args.predictions_path, args.key, args.target)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Setup datamodule
    datamodule = setup_datamodule(args.target, args.renormalize)
    
    # Run predictions
    run_predictions(
        model,
        datamodule,
        tiled_inference_parameters,
        args.predictions_path,
        args.key,
        args.target,
        args.renormalize,
        args.checkpoint
    )


if __name__ == "__main__":
    main()
