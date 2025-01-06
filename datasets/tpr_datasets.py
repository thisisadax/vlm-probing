import ast
from typing import Dict
import torch
import torch.nn.functional as F
import pandas as pd

from datasets.dataset_utils import ProbeDatasets


class TPRDatasets(ProbeDatasets):
    '''Dataset class for tensor product representation probing of feature conjunctions.'''
    
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
        super().__init__(
            task_name, model_name, layer_name, train_prop, test_prop,
            val_prop, batch_size, random_seed, exclude_text
        )
    
    @property
    def target_column(self) -> str:
        return 'features'
    
    def _encoder_fn(self, target_values: pd.Series) -> torch.Tensor:
        '''Encode feature dictionaries into binary probe labels.
        
        Args:
            target_values: Series of feature dictionaries
            
        Returns:
            Tensor of shape [n_samples, n_probes] where each column is a binary
            indicator for the presence of a feature conjunction
        '''
        # Convert string representations to dictionaries
        df = pd.DataFrame({'features': target_values.apply(ast.literal_eval)})
        
        # Create conjunction strings for each object
        # Use this version if you have keys for the objects
        #df['conjunctions'] = df.features.apply(
        #    lambda objs: [f'{obj["shape"]}_{obj["color"]}' for obj in objs.values()]
        #)
        df['conjunctions'] = df.features.apply(
            lambda objs: [f'{obj["shape"]}_{obj["color"]}' for obj in objs]
        )
        
        # Get unique conjunctions across all trials
        unique_conjunctions = sorted(df.conjunctions.explode().unique())
        self.feature_conjunctions = unique_conjunctions
        n_classes = len(unique_conjunctions)
        
        # Create conjunction to index mapping and encoder
        conjunction_to_idx = {conj: idx for idx, conj in enumerate(unique_conjunctions)}
        one_hot_encoder = lambda x: F.one_hot(torch.tensor(x), num_classes=n_classes)
        
        # Convert conjunctions to indices in the dataframe
        df['conjunction_indices'] = df.conjunctions.apply(
            lambda x: [conjunction_to_idx[conj] for conj in x]
        )
        
        # Create one-hot encodings using pre-defined encoder
        #one_hots = torch.stack([
        #    one_hot_encoder(indices).sum(0)
        #    for indices in df.conjunction_indices
        #]).float()
        #print('one_hots shape: ', one_hots.shape)
        #print('one_hots[0]: ', one_hots[0])

        test = []
        for indices in df.conjunction_indices:
            temp = torch.zeros([n_classes, n_classes])
            temp[indices, indices] = 1
            test.append(temp)
        test = torch.stack(test).float()
        #print(test.shape)
        #print('!!!!!!!!!!!!!!!!!')
        return test #one_hots
    
    @property
    def probe_info(self) -> Dict:
        '''Return information about the probes and their feature conjunctions'''
        if not hasattr(self, 'feature_conjunctions'):
            raise ValueError('Dataset must be loaded before accessing probe_info')
            
        return {
            'n_probes': len(self.feature_conjunctions),
            'conjunctions': [
                {'shape': conj.split('_')[0], 'color': conj.split('_')[1]}
                for conj in self.feature_conjunctions
            ]
        }