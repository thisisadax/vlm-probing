import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from probes.probe_utils import SimpleMLP, PooledAttentionProbe
from utils import collect_outputs


class TensorProductProbe(pl.LightningModule):
    '''
    A single probe for the TPR (Token-Pair Representations) model.
    '''
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 task_name: str, model_name: str, lr: float = 5e-4,
                 weight_decay: float = 1e-6, max_epochs: int = 20,
                 visualize_attention: bool = False, sparsity_lambda: float = 0,
                 probe_name: str = None, layer_name: str = None):
        super().__init__()
        self.save_hyperparameters()
        
        pooler = MultiProbeAttentionPooler(input_dim, output_dim)
        probe_mlp = SimpleMLP(input_dim, input_dim, output_dim)
        self.net = PooledAttentionProbe(pooler, probe_mlp, softmax=False)

        # Track examples from the validation loop.
        self.output_dir = os.path.join('output', task_name, model_name, probe_name, layer_name, 'attention_patterns')
        
        # Initialize lists to store outputs from the test phase
        self.save_outputs = False # Set to True to save attention masks
        self.outputs = []
        self.test_attention_masks = []
        self.test_predictions = []
        self.test_labels = []

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

    def training_step(self, batch, batch_idx):
        return self.loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        output = self.loss(batch, mode='val')
        self.outputs.append(output)  # collect the validation epoch info
        return output

    def test_step(self, batch, batch_idx):
        output = self.loss(batch, mode='test')
        
        if self.save_outputs:
            # Store the outputs
            self.test_attention_masks.append(output['attention'].detach().cpu())
            self.test_predictions.append(output['predictions'].detach().cpu())
            self.test_labels.append(output['targets'].detach().cpu())
            
        return output
        
    def on_test_epoch_end(self):
        if self.test_attention_masks:  # Only save if we collected data
            # Stack all tensors
            attention_masks = torch.cat(self.test_attention_masks, dim=0)
            predictions = torch.cat(self.test_predictions, dim=0)
            labels = torch.cat(self.test_labels, dim=0)
            
            # Create save directory
            save_dir = os.path.join('output', self.hparams.task_name, 
                                  self.hparams.model_name, 
                                  self.hparams.probe_name,
                                  self.hparams.layer_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save tensors
            torch.save(attention_masks, os.path.join(save_dir, 'attention_masks.pt'))
            torch.save(predictions, os.path.join(save_dir, 'predictions.pt'))
            torch.save(labels, os.path.join(save_dir, 'labels.pt'))

    def loss(self, batch, mode='train'):
        xs, ys, metadata, masks = batch
        logits, attention = self.forward(xs)
        # extract the diagonal elements from the logits and ys
        logits = torch.diagonal(logits, dim1=-2, dim2=-1)
        ys_diag = torch.diagonal(ys, dim1=-2, dim2=-1)
        logits = F.sigmoid(logits)
        class_loss = F.mse_loss(logits, ys_diag)
        loss = class_loss + self.hparams.sparsity_lambda * attention.abs().mean()
        targets = ys.float()
        predictions = logits > 0.5
        accuracy = (predictions == ys_diag).float().mean() * 100

        self.log(f'{mode}_loss', loss, on_epoch=True)
        self.log(f'{mode}_acc', accuracy, on_epoch=True)
        
        return {
            'loss': loss,
            'predictions': logits,
            'targets': targets,
            'attention': attention,
            'metadata': metadata,
            'masks': masks
        }

    def loss_OLD(self, batch, mode='train'):
        xs, ys, metadata, masks = batch
        logits, attention = self.forward(xs)
        logits = F.sigmoid(logits)
        class_loss = F.binary_cross_entropy(logits, ys)
        loss = class_loss + self.hparams.sparsity_lambda * attention.abs().mean()
        targets = ys.float()
        
        # Calculate overall accuracy across all classes
        target_classes = targets.argmax(dim=-1)  # [batch, n_probes]
        # Get predicted classes
        predictions = F.softmax(logits, dim=-1).argmax(dim=-1)  # [batch, n_probes]
        # Create mask for valid rows (rows with a positive label)
        valid_mask = targets.sum(dim=-1) > 0  # [batch, n_probes]
        # Calculate accuracy only for valid positions
        correct = (predictions == target_classes) & valid_mask
        total_valid = valid_mask.sum()
        accuracy = 100.0 * correct.sum() / total_valid

        self.log(f'{mode}_loss', loss, on_epoch=True)
        self.log(f'{mode}_acc', accuracy, on_epoch=True)
        
        return {
            'loss': loss,
            'predictions': logits,
            'targets': targets,
            'attention': attention,
            'metadata': metadata,
            'masks': masks
        }

    def on_validation_epoch_end(self):
        # Collect the results from the validation loop.
        outputs = collect_outputs(self.outputs)
        
        # Add assertions to catch misalignment
        n_samples = len(outputs['predictions'])
        assert len(outputs['metadata']) == n_samples, f"Metadata length ({len(outputs['metadata'])}) != predictions length ({n_samples})"
        assert len(outputs['targets']) == n_samples, f"Targets length ({len(outputs['targets'])}) != predictions length ({n_samples})"
        assert len(outputs['attention']) == n_samples, f"Attention length ({len(outputs['attention'])}) != predictions length ({n_samples})"
        assert len(outputs['masks']) == n_samples, f"Masks length ({len(outputs['masks'])}) != predictions length ({n_samples})"
        
        # Visualize attention maps and 2D feature conjunctions.
        if self.hparams.visualize_attention:
            self.visualize_attention_patterns(outputs)
        #self._plot_conjunction_accuracies(mode='val')
        self.outputs.clear()

    def _plot_conjunction_accuracies(self, mode='val'):
        '''Visualize feature conjunction accuracies as a bar plot'''
        preds = self.outputs['predictions']
        targets = self.outputs['targets']
        preds = F.softmax(preds, dim=-1).argmax(dim=-1)

        # Calculate per-class accuracies
        class_accuracies = (preds == targets).float().mean(dim=0)
        num_classes = class_accuracies.shape[0]
        
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(num_classes)
        ax.bar(x_pos, class_accuracies.numpy())
        
        # Customize the plot
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Per-class Accuracies - Epoch {self.current_epoch}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
        
        # Add value labels on top of each bar
        for i, v in enumerate(class_accuracies):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        # Log to tensorboard
        self.logger.experiment.add_figure(
            f'{mode}_class_accuracies',
            fig,
            global_step=self.current_epoch
        )
        plt.close(fig)

    def visualize_attention_patterns(self, outputs):
        # Choose a random index
        idx = np.random.choice(len(outputs['attention']))
        
        # Get number of feature conjunctions present in this example
        targets = outputs['targets'][idx]
        feature_indices = torch.where(targets.sum(dim=-1) > 0)[0]
        n_features = len(feature_indices)
        
        if n_features == 0:
            return
            
        # Get corresponding attention maps and image path
        attention_maps = outputs['attention'][idx][feature_indices].cpu().numpy()
        image_path = outputs['metadata']['path'].values[idx]
        
        # Plot the attention maps
        fig = self._plot_masked_data(attention_maps, image_path)
        
        # Save the figure
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f'epoch-{self.current_epoch}.png')
        plt.savefig(output_path)
        plt.close()

    def _plot_masked_data(self, attention_maps, image_path):
        '''
        Create a figure with attention maps for each feature present in the image
        alongside the input image.
        
        Parameters:
        -----------
        attention_maps : array-like
            Array of shape [N, T] containing N attention maps over T tokens
        image_path : str
            Path to the image file to display
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure object
        '''
        n_features = len(attention_maps)
        
        # Create figure with n_features + 1 subplots (including the input image)
        fig, axes = plt.subplots(1, n_features + 1, figsize=(4*(n_features + 1), 4))
        
        # Plot input image in the first subplot
        img = Image.open(image_path)
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        # Plot attention maps
        for i, attention in enumerate(attention_maps):
            # Remove image delimiter tokens and reshape to square
            attention = attention[1:-1]  # remove delimiter tokens
            size = int(np.ceil(np.sqrt(len(attention))))
            square_attention = attention.reshape(size, size)
            
            # Plot attention heatmap
            im = axes[i+1].imshow(square_attention, cmap='RdBu_r')
            fig.colorbar(im, ax=axes[i+1])
            axes[i+1].set_title(f'Feature {i+1} Attention', fontsize=14)
            axes[i+1].axis('off')
        
        plt.tight_layout()
        return fig


class MultiProbeAttentionPooler(nn.Module):
    '''
    Learns separate attention pooling operations for each feature conjunction using
    a single linear projection to compute attention weights for all probes in parallel.
    '''
    def __init__(self, input_dim: int, n_probes: int):
        '''
        Args:
            input_dim: Dimension of the input embeddings (D)
            n_probes: Number of feature conjunction probes (P)
        '''
        super().__init__()
        self.input_dim = input_dim
        self.n_probes = n_probes
        self.attention_proj = nn.Linear(input_dim, n_probes)
        
    def forward(self, x, return_att_vectors=True):
        '''
        Applies attention pooling for each probe in parallel.
        
        Args:
            x: Input tensor of shape [B, T, D] where:
               B = batch size
               T = number of tokens
               D = embedding dimension
            
        Returns:
            Tuple containing:
            - Pooled representations: Shape [B, P, D] where P = number of probes
            - Attention weights: Shape [B, P, T] showing where each probe attends
        '''
        # Project each token's embedding to n_probes attention scores
        # Input shape: [B, T, D]
        # Output shape: [B, T, P]
        attention_logits = self.attention_proj(x)
        
        # Apply softmax over token dimension (dim=1)
        # Input/Output shape: [B, T, P]
        attention = F.softmax(attention_logits, dim=1)
        
        # Pool representations using attention weights
        # attention: [B, T, P]
        # x: [B, T, D]
        # Output: [B, P, D]
        pooled = torch.einsum('btp,btd->bpd', attention, x)
        return pooled, attention.transpose(1, 2)  # Return [B, P, T] for consistency
