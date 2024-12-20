import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

from datasets.dataset_utils import ProbeDatasets


class NumerosityClassificationDatasets(ProbeDatasets):
    '''Implementation for numerosity estimation task.'''
    
    @property
    def target_column(self) -> str:
        return 'n_dots'
    
    def _encoder_fn(self, target_values: pd.Series) -> torch.Tensor:
        unique_values = sorted(target_values.unique())
        indices = torch.tensor([unique_values.index(val) for val in target_values], dtype=torch.long)
        return F.one_hot(indices, num_classes=len(unique_values)).float()


class NumerosityRegressionDatasets(ProbeDatasets):
    '''Implementation for numerosity estimation task.'''
    
    @property
    def target_column(self) -> str:
        return 'n_dots'
    
    def _encoder_fn(self, target_values: pd.Series) -> torch.Tensor:
        unique_values = sorted(target_values.unique())
        indices = torch.tensor([unique_values.index(val) for val in target_values], dtype=torch.float)
        return indices.reshape(-1,1) / indices.max()