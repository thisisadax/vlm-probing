import os
import warnings
import hydra
from omegaconf import DictConfig
import pyrootutils
import torch
from typing import List, Tuple
import pytorch_lightning as pl

warnings.filterwarnings('ignore')

def instantiate_modules(module_cfgs: DictConfig) -> List:
    """Instantiate all modules with _target_ defined in config."""
    modules = []
    for _, module in module_cfgs.items():
        if '_target_' in module:
            modules.append(hydra.utils.instantiate(module))
    return modules

def setup(cfg: DictConfig) -> Tuple:
    """Setup training components."""
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
        cfg.model,
        input_dim=input_dim,
        hidden_dim=input_dim//2,  # Using half of input_dim as hidden_dim
        output_dim=output_dim
    )
    
    return model, train_loader, test_loader, val_loader

# Project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path='config', config_name='probing')
def run(cfg: DictConfig) -> None:
    """Main training routine."""
    # Setup components
    model, train_loader, test_loader, val_loader = setup(cfg)
    
    # Initialize callbacks and logger
    callbacks = instantiate_modules(cfg.get('callbacks'))
    loggers = instantiate_modules(cfg.get('logger'))
    
    # Initialize trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers
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
    if loggers:
        for logger in loggers:
            if hasattr(logger, 'experiment'):
                logger.experiment.finish()

if __name__ == '__main__':
    run()
