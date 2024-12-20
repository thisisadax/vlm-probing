from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Union
import pandas as pd
import os
import numpy as np


class MetricsLogger(TensorBoardLogger):
    def __init__(self, save_dir: str, name: str = 'default', version: Optional[Union[int, str]] = None, **kwargs):
        '''
        Custom Logger that stores metrics, creates plots, and logs to TensorBoard.
        
        Args:
            save_dir (str): Directory to save plots and TensorBoard logs
            name (str): Name of the experiment
            version (Optional[Union[int, str]]): Version of the experiment
            **kwargs: Additional arguments passed to TensorBoardLogger
        '''
        # Initialize TensorBoardLogger
        super().__init__(save_dir=save_dir, name=name, version=version, **kwargs)
        
        # Dictionary to store metrics
        self.metrics: Dict[str, list] = {}
        self.steps: Dict[str, list] = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        '''
        Log metrics to both TensorBoard and our custom storage.
        '''
        # Log to TensorBoard
        super().log_metrics(metrics, step)
        
        # Store in our custom format
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
                self.steps[metric_name] = []
            self.metrics[metric_name].append(value)
            self.steps[metric_name].append(step if step is not None else len(self.metrics[metric_name]))


    def finalize(self, status: str) -> None:
        '''
        Called at the end of training to save all plots and metrics.
        '''
        super().finalize(status)
        self.save_metrics_plot()
        self.save_csv()
    

    def save_metrics_plot(self) -> None:
        '''
        Create and save plots for training metrics.
        '''
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        if 'train_loss' in self.metrics:
            ax1.plot(self.steps['train_loss'], self.metrics['train_loss'], 
                    label='Training Loss', color='blue')
        if 'val_loss' in self.metrics:
            ax1.plot(self.steps['val_loss'], self.metrics['val_loss'], 
                    label='Validation Loss', color='red')
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        if 'train_acc' in self.metrics:
            ax2.plot(self.steps['train_acc'], self.metrics['train_acc'], 
                    label='Training Accuracy', color='blue')
        if 'val_acc' in self.metrics:
            ax2.plot(self.steps['val_acc'], self.metrics['val_acc'], 
                    label='Validation Accuracy', color='red')
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the figure to TensorBoard
        self.experiment.add_figure('Training Metrics', fig, close=False)
        print(f'Saved metrics plot to {save_path} and TensorBoard')
    
    def save_csv(self) -> None:
        '''
        Save metrics to CSV file with proper alignment of steps and values.
        '''
        # First, get all unique steps across all metrics
        all_steps = sorted(set(step for steps in self.steps.values() for step in steps))
        
        # Create a dictionary to store aligned data
        aligned_data = {'step': all_steps}
        
        # For each metric, create an array of values aligned with all_steps
        for metric_name in self.metrics:
            # Create a mapping of step to value for this metric
            step_to_value = dict(zip(self.steps[metric_name], self.metrics[metric_name]))
            
            # Create aligned array with NaN for missing steps
            aligned_values = [step_to_value.get(step, np.nan) for step in all_steps]
            aligned_data[metric_name] = aligned_values
        
        # Create DataFrame and save
        df = pd.DataFrame(aligned_data)
        csv_path = os.path.join(self.log_dir, 'metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved metrics to {csv_path}')