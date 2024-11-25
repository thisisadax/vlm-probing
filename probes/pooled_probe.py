import os
import json
import datetime
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch import optim

from probes.probe_utils import AttentionPooler, SimpleMLP


class Probe(torch.nn.Module):
    def __init__(self, pooler, probe_mlp):
        super().__init__()
        self.pooler = pooler
        self.probe = probe_mlp
        
    def forward(self, x):
        xs, attention = self.pooler(x, return_att_vectors=True)
        ys = self.probe(xs)
        return ys, attention


class LightningPooledProbe(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int, 
        output_dim: int,
        task_name: str,
        model_name: str,
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        max_epochs: int = 20,
        visualize_attention: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create the network
        pooler = AttentionPooler(input_dim, hidden_dim)
        probe_mlp = SimpleMLP(hidden_dim, hidden_dim, output_dim)
        self.net = Probe(pooler, probe_mlp)

        # Used to store attention patterns for visualization.
        self.attention_patterns = []
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr/self.hparams.max_epochs
        )
        return [optimizer], [lr_scheduler]
    
    def forward(self, xs):
        return self.net(xs)

    def loss(self, batch, mode='train'):
        xs, ys = batch
        logits, _ = self.forward(xs)
        loss = F.cross_entropy(logits, ys)
        
        # Log metrics with more explicit parameters
        accuracy = (logits.argmax(dim=1) == ys.argmax(dim=1)).float().mean()
        self.log(
            f'{mode}_loss', 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            f'{mode}_acc',
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, attention = self.forward(xs)
        loss = F.cross_entropy(logits, ys)
        self.log('val_loss', loss)
        accuracy = (logits.argmax(dim=1) == ys.argmax(dim=1)).float().mean()
        self.log('val_acc', accuracy)
        
        if self.hparams.visualize_attention:
            self.attention_patterns.append(attention)
            
        return loss

    def on_validation_epoch_end(self):
        if not self.hparams.visualize_attention:
            return
            
        # Collect attention patterns from first 25 samples
        attention_patterns = torch.cat(self.attention_patterns)[:25]
        attention_patterns = attention_patterns[:, 1:-1] # Remove first and last token attention
        
        # Check if remaining sequence length is square
        seq_len = attention_patterns.shape[1]
        side_len = int(np.sqrt(seq_len))
        if side_len * side_len != seq_len:
            print(f'Warning: Sequence length {seq_len} is not a perfect square')
            return
            
        # Create 5x5 grid of attention patterns
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for attention, ax in zip(attention_patterns, axes.flat):
            # Reshape to square
            att_map = attention.reshape(side_len, side_len).cpu().numpy()
            _ = ax.imshow(att_map, cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        
        # Save plot to file
        output_dir = os.path.join('output', self.hparams.task_name, self.hparams.model_name, 'attention_patterns')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'epoch-{self.current_epoch}.png')
        plt.savefig(output_path)
        plt.close()
        return None

    def test_step(self, batch, batch_idx):
        return self.loss(batch, mode='test')
        
    def save_run_results(self):
        '''Save model results and attention patterns'''
        base_path = os.path.join('output', self.hparams.task_name, self.hparams.model_name)
        results_path = os.path.join(base_path, 'results')
        attention_path = os.path.join(base_path, 'attention_patterns')
        
        # Create directories
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(attention_path, exist_ok=True)
        
        # Save metrics
        metrics = {
            'final_train_acc': self.trainer.callback_metrics.get('train_acc').item(),
            'final_val_acc': self.trainer.callback_metrics.get('val_acc').item(),
            'final_test_acc': self.trainer.callback_metrics.get('test_acc', 0.0),
            'final_train_loss': self.trainer.callback_metrics.get('train_loss').item(),
            'final_val_loss': self.trainer.callback_metrics.get('val_loss').item(),
            'final_test_loss': self.trainer.callback_metrics.get('test_loss', 0.0),
            'epochs': self.current_epoch,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save metrics as JSON
        metrics_file = os.path.join(results_path, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save attention patterns if available
        if self.attention_patterns:
            attention_file = os.path.join(attention_path, 'attention_patterns.npy')
            patterns = torch.cat(self.attention_patterns).cpu().numpy()
            np.save(attention_file, patterns)
            
    def on_fit_end(self):
        '''Called at the end of training'''
        self.save_run_results()
