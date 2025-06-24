'''Arctic sea ice data handling'''

from typing import List
import xarray as xr
import os
from abc import ABC

import torch
from torch.utils.data import Dataset, IterableDataset
import lightning as L
import numpy as np
import pandas as pd

from torchvision import tv_tensors
from torchvision.transforms import v2


import logging

logger = logging.getLogger(__name__)

class ArcticSeaIceBaseDataset(ABC):
    '''
    Dataset for ready-to-train Arctic sea ice charts.
    '''
    def __init__(self,
                 ice_charts: List[str],
                 features: List[str] = ['nersc_sar_primary', 'nersc_sar_secondary'],
                 target: str = "SIC", 
                 seed = None,
                 patch_size: int = 256,
                 fill_values_to_nan: bool = True,
                 max_nan_frac: float = 0.25,
                 renormalize: bool = False,
                 return_chart_name: bool = False,
                ):
        """ArcticSeaIceBaseDataset

        Args:
            ice_charts (list[str]): List of ice chart file paths.
            features (list[str], optional): List of input features. Defaults to ['nersc_sar_primary', 'nersc_sar_secondary'].
            target (str, optional): Target, choice of SIC, SOD, FLOE. Defaults to "SIC".
            seed (_type_, optional): Random seed. Defaults to None.
            patch_size (int, optional): Sample patch size. Defaults to 256.
            fill_values_to_nan (bool, optional): If True, replace fill values with NaN. Defaults to True.
            max_nan_frac (float, optional): Maximum fraction of NaN pixels accepted in a random patch. Defaults to 0.25.
            renormalize (bool, optional): If True, renormalize tensors with Terramind stats. Defaults to False.
            return_chart_name (bool, optional): If True, return the ice chart name in the output. Defaults to False.
        """        
        super().__init__()

        self.ice_charts = ice_charts
        self.features = features
        self.target = target
        self.seed = seed
        self.patch_size = patch_size
        self.fill_values_to_nan = fill_values_to_nan
        self.max_nan_frac = max_nan_frac
        self.renormalize = renormalize
        self.return_chart_name = return_chart_name

        # random number generator for iterator
        self.rng = np.random.default_rng(seed=self.seed)

        # fixed values
        self.norm_values = {
            'nersc_sar_primary': {'mean': -14.508, 'std': 5.660},
            'nersc_sar_secondary': {'mean': -24.701, 'std': 4.747},
            'terramind_VV': {'mean': -12.599, 'std': 5.195},
            'terramind_VH': {'mean': -20.293, 'std': 5.890},
        }


    @staticmethod
    def _load_dataset(path, features, target, fill_values_to_nan=True):
        """Load dataset from path and return selected features and target.

        Args:
            path (str): File path (netcdf).
            features (list[str]): List of input features.
            target (str): Target variable, choice of SIC, SOD, FLOE.
            fill_values_to_nan (bool, optional): If True, replace fill values with NaN. Defaults to True.

        Returns:
            xr.Dataset: Dataset with selected features and target variable.
        """        
        with xr.open_dataset(path) as ds:

            if fill_values_to_nan:
                ds['nersc_sar_primary'] = xr.where(ds['nersc_sar_primary'] != 0, ds['nersc_sar_primary'], None)
                ds['nersc_sar_secondary'] = xr.where(ds['nersc_sar_secondary'] != 0, ds['nersc_sar_secondary'], None)
                ds[target] = xr.where(ds[target] != 255, ds[target], None)

            return ds[features + [target]]

    @staticmethod
    def _select_patch(ds, patch_size, var='nersc_sar_primary', seed=None, max_nan_frac=0.25):
        '''Select a small patch from ds. Resample until we find a sample with less than max_nan_frac empty pixels.'''

        xmax = len(ds.sar_samples) - patch_size
        ymax = len(ds.sar_lines) - patch_size

        # fall back patch
        patch = ds.isel(sar_samples=slice(0, patch_size), sar_lines=slice(0, patch_size))
        max_lookups = 100

        rng = np.random.default_rng(seed=seed)

        looking_for_patch = True
        count = 0
        while looking_for_patch and count < max_lookups:
            i = int(rng.random() * xmax)
            j = int(rng.random() * ymax)

            patch = ds.isel(sar_samples=slice(i, i+patch_size), sar_lines=slice(j, j+patch_size))
            
            nan_frac = np.isnan(patch[var]).sum().item() / patch_size / patch_size
            if nan_frac < max_nan_frac:
                looking_for_patch = False
            else:
                count += 1

        if count == max_lookups:
            logger.warning(f'Could not find a patch with less than {max_nan_frac} NaN pixels in {max_lookups} attempts. Returning None.') 
            return None
        
        return patch

    @staticmethod
    def _extract_tensors(ds, features, target):
        '''Extract features and targets from dataset and convert to torch tensors'''
        x = []
        for feat in features:
            x.append(torch.from_numpy(ds[feat].values).to(torch.float32))

        x = torch.stack(x)

        y = torch.from_numpy(ds[target].values)

        # Missing values are NaN, convert for loss function
        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=-1).to(torch.long)

        return x, y

    @staticmethod
    def _renormalize_tensors(x, norm_values):
        """Renormalize tensors with the stats from Terramind."""

        # SAR channels were already mean/std normalized
        # Retrieve original values
        x[0] = x[0] * norm_values['nersc_sar_primary']['std'] + norm_values['nersc_sar_primary']['mean']
        x[1] = x[1] * norm_values['nersc_sar_secondary']['std'] + norm_values['nersc_sar_secondary']['mean']

        # Re-normalize to Terramind stats
        x[0] = (x[0] - norm_values['terramind_VV']['mean']) / norm_values['terramind_VV']['std']
        x[1] = (x[1] - norm_values['terramind_VH']['mean']) / norm_values['terramind_VH']['std']

        return x

class ArcticSeaIceValidationDataset(ArcticSeaIceBaseDataset, Dataset):
    '''Dataset for Arctic sea ice charts. Validation: Cover all ice charts.'''
    def __init__(self, 
                 ice_charts: List[str],
                 features: List[str] = ['nersc_sar_primary', 'nersc_sar_secondary'],
                 target: str = "SIC", 
                 seed = None,
                 patch_size: int = 256,
                 fill_values_to_nan: bool = True,
                 max_nan_frac: float = 0.25,
                 renormalize: bool = False,
                 return_chart_name: bool = False,
                ):
        super().__init__(ice_charts=ice_charts,
                         features=features,
                         target=target,
                         seed=seed,
                         patch_size=patch_size,
                         fill_values_to_nan=fill_values_to_nan,
                         max_nan_frac=max_nan_frac,
                         renormalize=renormalize,
                         return_chart_name=return_chart_name,
                         )

        # read in all ice charts
        print(f"Loading {len(self.ice_charts)} ice charts for validation dataset")
        tiles = []
        for ice_chart in self.ice_charts:
            ds = self._load_dataset(ice_chart, 
                                    self.features, 
                                    self.target,
                                    fill_values_to_nan=self.fill_values_to_nan
                                   )

            tiles.extend(self._get_tiles(ds))
        
        self.tiles = tiles
        print(f"Number of tiles in validation dataset: {len(self.tiles)}")

    def _get_tiles(self, ds):
        """Get tiles from dataset. Last pixels may be missing."""
        xmax = len(ds.sar_samples) - self.patch_size
        ymax = len(ds.sar_lines) - self.patch_size

        tiles = []
        for i in range(0, xmax, self.patch_size // 2):
            for j in range(0, ymax, self.patch_size // 2):
                tile = ds.isel(sar_samples=slice(i, i+self.patch_size), sar_lines=slice(j, j+self.patch_size))
                if np.all(np.isnan(tile[self.target].values)):
                    logger.debug(f"Skipping tile at ({i}, {j}) with all NaN values in target.")
                    continue
                tiles.append(tile)

        return tiles

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.tiles)

    def __getitem__(self, idx):
        """Return a single sample from the dataset."""
        x, y = self._extract_tensors(self.tiles[idx], self.features, self.target)

        if self.renormalize:
            x = self._renormalize_tensors(x, self.norm_values)

        if self.return_chart_name:
            return {"image": x, "mask": y.squeeze(), "ice_chart": os.path.basename(self.ice_charts[idx])}

        return {"image": x, "mask": y.squeeze()}


class ArcticSeaIceIterableDataset(ArcticSeaIceBaseDataset, IterableDataset):
    '''Iterable dataset for Arctic sea ice charts.'''
    def __init__(self, 
                 ice_charts: List[str],
                 features: List[str] = ['nersc_sar_primary', 'nersc_sar_secondary'],
                 target: str = "SIC", 
                 seed = None,
                 patch_size: int = 256,
                 fill_values_to_nan: bool = True,
                 max_nan_frac: float = 0.25,
                 samples_per_chart: int = 10,
                 augment: bool = True,
                 renormalize: bool = False,
                 return_chart_name: bool = False,
                 chart_weights: str = 'chart_weights.csv',
                ):
        super().__init__(ice_charts=ice_charts,
                         features=features,
                         target=target,
                         seed=seed,
                         patch_size=patch_size,
                         fill_values_to_nan=fill_values_to_nan,
                         max_nan_frac=max_nan_frac,
                         renormalize=renormalize,
                         return_chart_name=return_chart_name
                         )
        
        self.samples_per_chart = samples_per_chart  # Number of samples to take from each chart
        self.augment = augment  # Whether to apply data augmentation
        self.chart_weights = chart_weights

        if self.chart_weights is not None:
            print(f"Loading chart weights from {chart_weights} and normalizing them")
            self.df_chart_weights = pd.read_csv(chart_weights, index_col=0)
            self.df_chart_weights["weight"] = self.df_chart_weights["weight"] / self.df_chart_weights["weight"].sum()
            print(self.df_chart_weights.head())
        self.epoch = 0  # Initialize epoch for reshuffling

        print(f"Using {self.samples_per_chart} samples per chart with {len(self.ice_charts)} ice charts.")

        self.orig_ice_charts = self.ice_charts.copy()  # Store original ice charts for reshuffling


        if self.augment:
            print("Data augmentation: horizontal and vertical flips")
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ])
        else:
            self.transforms = None

    def set_epoch(self, epoch):
        """Set the epoch number to enable reshuffling."""
        self.epoch = epoch
        # Reshuffle ice charts using the epoch number and base seed
        base_seed = self.seed if self.seed is not None else 42
        epoch_seed = base_seed + self.epoch
        
        # Repeat chart based on weights
        if self.df_chart_weights is None:
            self.ice_charts = self.orig_ice_charts * self.samples_per_chart
        else:
            # Repeat each ice chart according to its weight
            tmp_ice_charts = []
            for ice_chart in self.orig_ice_charts:
                if not os.path.basename(ice_chart) in self.df_chart_weights.index:
                    raise ValueError(f"Ice chart {os.path.basename(ice_chart)} not found in chart weights.")

                weight = self.df_chart_weights.loc[os.path.basename(ice_chart), 'weight']
                num_samples = int(np.round(weight * self.samples_per_chart * len(self.orig_ice_charts)))
                num_samples = max(num_samples, 1)  # Ensure at least one sample per chart
                logger.debug(f"Processing ice chart: {os.path.basename(ice_chart)} with weight {weight}: {num_samples} samples")
                tmp_ice_charts.extend([ice_chart] * num_samples)
            self.ice_charts = tmp_ice_charts
        print(f"Total number of ice charts after valid pixel weighting: {len(self.ice_charts)}")

        # Shuffle using a deterministic random state based on epoch
        rng = np.random.RandomState(epoch_seed)
        rng.shuffle(self.ice_charts)
        
        print(f"Dataset reshuffled for epoch {epoch}, first chart: {os.path.basename(self.ice_charts[0])}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.orig_ice_charts)

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which charts this worker should process
        if worker_info is None:  # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:  # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Split ice charts among workers
        per_worker = int(np.ceil(len(self.ice_charts) / num_workers))
        start_idx = worker_id * per_worker
        end_idx = min(start_idx + per_worker, len(self.ice_charts))
        
        # Use only this worker's subset of charts
        worker_charts = self.ice_charts[start_idx:end_idx]
        logger.debug(f"Worker {worker_id} processing {len(worker_charts)} charts out of {len(self.ice_charts)} total charts.")
        
        # Now iterate over just this worker's charts
        for ice_chart in worker_charts:
            #print(f"Worker {worker_id} processing chart: {os.path.basename(ice_chart)}")
            ds = self._load_dataset(ice_chart, 
                            self.features, 
                            self.target,
                            fill_values_to_nan=self.fill_values_to_nan
                           )
            
            patch = self._select_patch(ds, self.patch_size, var=self.target, seed=self.seed, max_nan_frac=self.max_nan_frac)
            if patch is None:
                continue
            x, y, = self._extract_tensors(patch, self.features, self.target)

            # Data augmentation
            if self.augment and self.transforms is not None:
                x = tv_tensors.Image(x)
                y = tv_tensors.Mask(y)

                x, y = self.transforms(x, y)

            if self.renormalize:
                x = self._renormalize_tensors(x, self.norm_values)

            if self.return_chart_name:
                yield {"image": x, "mask": y.squeeze(), "ice_chart": os.path.basename(ice_chart)}

            yield {"image": x, "mask": y.squeeze()}

class ArcticSeaIceDataset(ArcticSeaIceBaseDataset, Dataset):
    '''Dataset for inference on Arctic sea ice charts.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        """Return the number of ice charts in the dataset."""
        return len(self.ice_charts)

    def __getitem__(self, idx):
        """Get a full patch from the dataset"""

        if self.target is not None:
            # labels available
            ds = self._load_dataset(self.ice_charts[idx],
                                    self.features, 
                                    self.target,
                                    fill_values_to_nan=self.fill_values_to_nan
                                    )

            # convert to tensors
            x, y = self._extract_tensors(ds, self.features, self.target)
    
            if self.renormalize:
                x = self._renormalize_tensors(x, self.norm_values)
    
            if self.return_chart_name:
                return {"image": x, "mask": y.squeeze(), "ice_chart": os.path.basename(self.ice_charts[idx])}

            return {"image": x, "mask": y.squeeze()}

        else:
            # labels not available
            ds = self._load_dataset(self.ice_charts[idx],
                                    self.features, 
                                    self.features[0],  # Use the first feature as a placeholder
                                    fill_values_to_nan=self.fill_values_to_nan
                                    )

            # convert to tensors
            x, y = self._extract_tensors(ds, self.features, self.features[0])
    
            if self.renormalize:
                x = self._renormalize_tensors(x, self.norm_values)
    
            if self.return_chart_name:
                return {"image": x, "ice_chart": os.path.basename(self.ice_charts[idx])}

            return {"image": x}


class ArcticSeaIceDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str,
                 features: List[str] = ['nersc_sar_primary', 'nersc_sar_secondary'],
                 target: str = "SIC", #['SIC', 'SOD', 'FLOE'],
                 seed = None,
                 patch_size: int = 256,
                 fill_values_to_nan: bool = True,
                 max_nan_frac: float = 0.25,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 shuffle: bool = True,
                 renormalize: bool = True,
                 samples_per_chart: int = 10, # Only used in ArcticSeaIceIterableDataset
                 augment: bool = True, # Only used in ArcticSeaIceIterableDataset
                 return_chart_name: bool = False,
                 chart_weights: str = 'chart_weights.csv',
                ):
        super().__init__()
        self.data_root = data_root
        self.features = features
        self.target = target
        self.seed = seed
        self.patch_size = patch_size
        self.fill_values_to_nan = fill_values_to_nan
        self.max_nan_frac = max_nan_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.renormalize = renormalize
        self.samples_per_chart = samples_per_chart  # Only used in ArcticSeaIceIterableDataset
        self.augment = augment  # Only used in ArcticSeaIceIterableDataset
        self.return_chart_name = return_chart_name
        self.chart_weights = chart_weights


        # assigned in setup
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.rng = np.random.default_rng(self.seed)

    def prepare_data(self):
        pass

    def setup(self, stage="fit"):        
        if stage == "fit":
            train_ice_charts = [os.path.join(self.data_root, "train", f) for f in os.listdir(os.path.join(self.data_root, "train")) if f.endswith('.nc')]
            val_ice_charts = [os.path.join(self.data_root, "val", f) for f in os.listdir(os.path.join(self.data_root, "val")) if f.endswith('.nc')]

            if self.shuffle:
                self.rng.shuffle(train_ice_charts)
            
            self.train_ds = ArcticSeaIceIterableDataset(train_ice_charts,
                                                        features=self.features,
                                                        target=self.target,
                                                        seed=self.seed,
                                                        patch_size=self.patch_size,
                                                        fill_values_to_nan=self.fill_values_to_nan,
                                                        max_nan_frac=self.max_nan_frac,
                                                        samples_per_chart=self.samples_per_chart,
                                                        augment=self.augment,
                                                        renormalize=self.renormalize,
                                                        return_chart_name=self.return_chart_name,
                                                        chart_weights=self.chart_weights,
                                                       )
            self.val_ds = ArcticSeaIceValidationDataset(val_ice_charts,
                                                        features=self.features,
                                                        target=self.target,
                                                        seed=self.seed,
                                                        patch_size=self.patch_size,
                                                        fill_values_to_nan=self.fill_values_to_nan,
                                                        max_nan_frac=self.max_nan_frac,
                                                        renormalize=self.renormalize,
                                                        return_chart_name=self.return_chart_name,
                                                       )

            print(f"Number of ice charts in train: {len(self.train_ds.ice_charts)}")
            print(f"Number of ice charts in validation: {len(self.val_ds.ice_charts)}")
            
                                                
        elif stage == "test":
            test_ice_charts = [os.path.join(self.data_root, "test", f) for f in os.listdir(os.path.join(self.data_root, "test")) if f.endswith('.nc')]
            self.test_ds = ArcticSeaIceDataset(test_ice_charts,
                                               features=self.features,
                                               target=self.target,
                                               seed=self.seed,
                                               patch_size=self.patch_size,
                                               fill_values_to_nan=self.fill_values_to_nan,
                                               max_nan_frac=self.max_nan_frac,
                                               renormalize=self.renormalize,
                                               return_chart_name=self.return_chart_name,
                                              )
        elif stage == "predict":
            predict_ice_charts = [os.path.join(self.data_root, "predict", f) for f in os.listdir(os.path.join(self.data_root, "predict")) if f.endswith('_prep.nc')]
            self.predict_ds = ArcticSeaIceDataset(predict_ice_charts,
                                                  features=self.features,
                                                  target=None,
                                                  seed=self.seed,
                                                  patch_size=self.patch_size,
                                                  fill_values_to_nan=self.fill_values_to_nan,
                                                  max_nan_frac=self.max_nan_frac,
                                                  renormalize=self.renormalize,
                                                  return_chart_name=self.return_chart_name,
                                                 )

    def train_dataloader(self):
        # shuffling is handled in iterable dataset
        return torch.utils.data.DataLoader( self.train_ds, batch_size=self.batch_size, shuffle=False,  num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)