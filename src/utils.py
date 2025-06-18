from lightning.pytorch.callbacks import Callback
import wandb

class WandBConfigCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        """Log all relevant parameters to WandB at the start of training"""
        if not trainer.logger:
            return
            
        # Find WandB logger if present
        for logger in trainer.loggers:
            if 'WandbLogger' in str(type(logger)):
                # Create config dict with actual values (no ${} substitution needed)
                config = {
                    "features": trainer.datamodule.features,
                    "target": trainer.datamodule.target,
                    "patch_size": trainer.datamodule.patch_size,
                    "batch_size": trainer.datamodule.batch_size,
                    "num_workers": trainer.datamodule.num_workers,
                    "data_root": trainer.datamodule.data_root,
                    "seed": trainer.datamodule.seed,
                    "fill_values_to_nan": trainer.datamodule.fill_values_to_nan,
                    "max_nan_frac": trainer.datamodule.max_nan_frac,
                    "val_split": trainer.datamodule.val_split,
                    "shuffle": trainer.datamodule.shuffle,
                    "means": trainer.datamodule.means,
                    "stds": trainer.datamodule.stds,
                    # Add other parameters you want to log
                }
                
                # Use wandb directly
                if wandb.run is not None:
                    wandb.config.update(config)
                
                break

class EpochShuffle(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Set epoch in the dataset
        if hasattr(trainer.datamodule, "train_ds") and hasattr(trainer.datamodule.train_ds, "set_epoch"):
            trainer.datamodule.train_ds.set_epoch(trainer.current_epoch)