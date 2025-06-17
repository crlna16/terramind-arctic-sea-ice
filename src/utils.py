from lightning.pytorch.callbacks import Callback

class WandBConfigCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        """Log all relevant parameters to WandB at the start of training"""
        if not trainer.logger:
            return
            
        # Find WandB logger if present
        for logger in trainer.loggers:
            if 'WandbLogger' in str(type(logger)):
                # Access datamodule parameters directly
                datamodule = trainer.datamodule
                
                # Create config dict with actual values (no ${} substitution needed)
                config = {
                    "features": datamodule.features,
                    "target": datamodule.target,
                    "patch_size": datamodule.patch_size,
                    "batch_size": datamodule.batch_size,
                    "num_workers": datamodule.num_workers,
                    "data_root": datamodule.data_root,
                    "seed": datamodule.seed,
                    "fill_values_to_nan": datamodule.fill_values_to_nan,
                    "max_nan_frac": datamodule.max_nan_frac,
                    "val_split": datamodule.val_split,
                    "shuffle": datamodule.shuffle,
                    "means": datamodule.means,
                    "stds": datamodule.stds,
                    # Add other parameters you want to log
                }
                
                # Update WandB config with resolved values
                logger.experiment.config.update(config)
                break