import os
import warnings
import hydra
import gc
import torch
from omegaconf import DictConfig
import pyrootutils
from typing import Tuple, List
import pytorch_lightning as pl
from glob import glob
from pathlib import Path

from utils import instantiate_modules


def get_layer_names(cfg: DictConfig) -> List[str]:
    '''Get all layer names from activation directory.'''
    activation_path = f"output/{cfg.probe.task_name}/{cfg.probe.model_name}/activations/*"
    layer_dirs = glob(activation_path)
    return [Path(d).name for d in layer_dirs]

def setup(cfg: DictConfig, layer_name: str = None) -> Tuple:
    '''Setup training components.'''
    # Set seeds
    pl.seed_everything(cfg.seed)
    
    # Update layer name in dataset config if provided
    if layer_name is not None:
        cfg.dataset.layer_name = layer_name
    
    # Load dataset first to determine dimensions
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_loader, test_loader, val_loader = dataset.load()
    
    # Get dimensions from the dataset
    input_dim = train_loader.dataset.features.shape[-1]
    output_dim = train_loader.dataset.label_dim
    
    # Initialize model with dimensions from dataset
    probe = hydra.utils.instantiate(
        cfg.probe,
        input_dim=input_dim,
        hidden_dim=input_dim,  # Hidden dim for pooled probe is same as input
        output_dim=output_dim
    )
    
    return probe, train_loader, test_loader, val_loader

# Project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path='config', config_name='probing')
def run(cfg: DictConfig) -> None:
    '''Main training routine.'''
    
    # Determine which layers to process
    layer_names = [cfg.dataset.layer_name] if cfg.dataset.layer_name else get_layer_names(cfg)
    
    for layer_name in layer_names:
        print(f"\nProcessing layer: {layer_name}")
        
        # Setup components for this layer
        model, train_loader, test_loader, val_loader, full_loader = setup(cfg, layer_name)
        
        # Initialize callbacks and logger
        callbacks = instantiate_modules(cfg.get('callbacks'))
        logger = instantiate_modules(cfg.get('logger'))
        
        if not logger:
            logger = True  # Enable default logger if none specified
            
        # Initialize trainer
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger
        )
        
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Test model on test set
        trainer.test(dataloaders=test_loader)
        
        # Run inference on full dataset with mask saving enabled
        trainer.test(model=model, dataloaders=full_loader, ckpt_path="best")
        
        # Cleanup
        if logger and not isinstance(logger, bool):
            if hasattr(logger, 'experiment'):
                logger.experiment.finish()
        
        # Clear memory
        del model, train_loader, test_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run()
