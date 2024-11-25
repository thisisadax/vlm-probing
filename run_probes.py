import os
import warnings
import hydra
from omegaconf import DictConfig
import pyrootutils
from typing import Tuple
import pytorch_lightning as pl

from utils import instantiate_modules


def setup(cfg: DictConfig) -> Tuple:
    '''Setup training components.'''
    # Set seeds
    pl.seed_everything(cfg.seed)
    
    # Load dataset first to determine dimensions
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_loader, test_loader, val_loader = dataset.load()
    
    # Get dimensions from the dataset
    input_dim = train_loader.dataset.features.shape[-1]
    output_dim = train_loader.dataset.label_dim
    
    # Initialize model with dimensions from dataset
    model = hydra.utils.instantiate(
        cfg.probe,
        input_dim=input_dim,
        hidden_dim=input_dim,  # Hidden dim for pooled probe is same as input
        output_dim=output_dim
    )
    
    return model, train_loader, test_loader, val_loader

# Project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path='config', config_name='probing')
def run(cfg: DictConfig) -> None:
    '''Main training routine.'''
    # Setup components
    model, train_loader, test_loader, val_loader = setup(cfg)
    
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
    
    # Test model
    trainer.test(dataloaders=test_loader)
    
    # Cleanup
    if logger and not isinstance(logger, bool):
        if hasattr(logger, 'experiment'):
            logger.experiment.finish()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run()
