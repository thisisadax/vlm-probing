import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod


class TensorDataset(Dataset):
    '''Dataset wrapper for tensor features and labels with metadata.'''
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, df: pd.DataFrame = None):
        self.features = features
        self.labels = labels
        self.df = df
        
        if len(self.features) != len(self.labels):
            raise ValueError(
                f'Mismatch between features ({len(self.features)}) and '
                f'labels ({len(self.labels)}) dimensions'
            )
        
        self.feature_dim = features.shape[-1] if len(features.shape) > 1 else 1
        self.label_dim = labels.shape[-1] if len(labels.shape) > 1 else 1
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.features[idx], self.labels[idx], self.df.iloc[idx].to_dict()
    
    @property
    def metadata(self) -> dict:
        '''Return dataset metadata.'''
        return {
            'size': len(self),
            'feature_dim': self.feature_dim,
            'label_dim': self.label_dim,
            'feature_dtype': self.features.dtype,
            'label_dtype': self.labels.dtype
        }


class ProbeDatasets(ABC):
    '''Base class for loading and managing probe datasets.'''
    
    def __init__(
        self,
        task_name: str,
        model_name: str,
        layer_name: str,
        train_prop: float = 0.7,
        test_prop: float = 0.15,
        val_prop: float = 0.15,
        batch_size: int = 256,
        random_seed: int = 42,
        exclude_text: bool = False
    ):
        self.task_name = task_name
        self.model_name = model_name
        self.layer_name = layer_name
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.exclude_text = exclude_text
        self.image_mask = None
        
        # Validate split proportions
        if not np.isclose(train_prop + test_prop + val_prop, 1.0):
            raise ValueError(f'Split proportions must sum to 1, got {train_prop + test_prop + val_prop}')
    
    @property
    @abstractmethod
    def target_column(self) -> str:
        '''Name of the target column in the CSV.'''
        pass
    
    @abstractmethod
    def _encoder_fn(self, target_values: pd.Series) -> torch.Tensor:
        '''Encode target values into labels.'''
        pass
    
    def _load_features(self, files: List[str]) -> torch.Tensor:
        '''Load and concatenate feature files.'''
        features = torch.cat([torch.load(file) for file in files], dim=0).float()

        # Select only the image token features for each sample
        if self.exclude_text:
            mask_path = f'output/{self.task_name}/{self.model_name}/image_mask.pt'
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'Image mask file not found: {mask_path}')
            self.image_mask = torch.load(mask_path, weights_only=True)
            image_indices = torch.where(self.image_mask[0] == 1)[0]  # TODO: change this to be more general (currently only using first mask)
            features = features[:, image_indices.to(features.device)]
        return features
    
    def _create_split_indices(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Create indices for train/test/val split using numpy.split.
        '''
        torch.manual_seed(self.random_seed)
        indices = torch.randperm(n_samples)
        train_idx = round(n_samples * self.train_prop)
        test_idx = train_idx + round(n_samples * self.test_prop)
        train_indices, test_indices, val_indices = np.split(indices, [train_idx, test_idx])
        return train_indices, test_indices, val_indices
    
    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        '''Load and split data into train, test, and validation dataloaders.'''
        # Load the CSV file
        csv_path = f'output/{self.task_name}/{self.model_name}/model_responses.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'CSV file not found: {csv_path}')
        df = pd.read_csv(csv_path)
        
        # Verify target column exists
        if self.target_column not in df.columns:
            raise ValueError(f'Target column "{self.target_column}" not found in CSV')
        
        # Load and concatenate .pt files
        pattern = f'output/{self.task_name}/{self.model_name}/activations/{self.layer_name}/*.pt'
        pt_files = sorted(glob.glob(pattern))
        
        if not pt_files:
            raise FileNotFoundError(f'No .pt files found matching pattern: {pattern}')
        
        # Load features and encode labels
        features = self._load_features(pt_files)
        labels = self._encoder_fn(df[self.target_column])
        
        if not isinstance(labels, torch.Tensor):
            raise ValueError('_encoder_fn must return a torch.Tensor')
        
        # Create split indices
        train_idx, test_idx, val_idx = self._create_split_indices(len(features))
        
        # Create dataloaders
        train_loader = DataLoader(
            TensorDataset(features[train_idx], labels[train_idx], df=df.iloc[train_idx]),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            TensorDataset(features[test_idx], labels[test_idx], df=df.iloc[test_idx]),
            batch_size=self.batch_size
        )
        
        val_loader = DataLoader(
            TensorDataset(features[val_idx], labels[val_idx], df=df.iloc[val_idx]),
            batch_size=self.batch_size
        )
        return train_loader, test_loader, val_loader
