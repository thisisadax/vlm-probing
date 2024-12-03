import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

from probes.probe_utils import SimpleMLP, PooledAttentionProbe, TenneyMLP
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
        #probe_mlp = TenneyMLP(input_dim, hidden_dim, output_dim)
        self.net = PooledAttentionProbe(pooler, probe_mlp, softmax=False)

        # Track examples from the validation loop.
        self.outputs = []

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

    def loss(self, batch, mode='train'):
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
            'predictions': predictions,
            'targets': targets,
            'attention': attention,
            'metadata': metadata,
            'masks': masks
        }

    def on_validation_epoch_end(self):
        # Collect the results from the validation loop.
        outputs = collect_outputs(self.outputs)
        
        # Visualize attention maps and 2D feature conjunctions.
        if self.hparams.visualize_attention:
            self.visualize_attention_patterns(outputs)
        #self._plot_conjunction_accuracies(mode='val')
        self.outputs.clear()

    def _plot_conjunction_accuracies(self, mode='val'):
        '''Visualize feature conjunction accuracies as a bar plot'''
        preds = self.outputs['predictions']
        targets = self.outputs['targets']

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
        mask = outputs['masks'][idx].cpu().numpy()
        attention = outputs['attention'][idx].cpu().numpy()
        image_path = outputs['metadata']['path'].values[idx]
        fig = self._plot_masked_data(attention, mask, image_path)
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f'epoch-{self.current_epoch}.png')
        plt.savefig(output_path)
        plt.close()

    def _plot_masked_data(self, data, mask, image_path):
        '''
        Create a figure with three subplots: a line plot with masked background,
        a square image visualization of the masked data, and the input image.
        
        Parameters:
        -----------
        data : array-like
            The input data to plot
        mask : array-like
            Binary mask of same length as data, where 1 indicates masked regions
        image_path : str
            Path to the image file to display
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure object
        '''
        
        # Validate inputs
        if len(data) != len(mask):
            raise ValueError('Data and mask must have the same length')
            
        # Extract masked data
        masked_data = data[mask == 1]
        if len(masked_data) == 0:
            raise ValueError('No masked data found (mask contains all zeros)')
        masked_data = masked_data[1:-1] # remove image delimeter tokens
            
        # Calculate optimal square dimensions
        size = int(np.ceil(np.sqrt(len(masked_data))))
        
        # Pre-generate figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        
        # First subplot: Original plot with masked background
        ymin, ymax = data.min(), data.min()
        padding = (ymax - ymin) * 0.1
        ymin -= padding
        ymax += padding
        x = np.arange(len(data))
        ax1.fill_between(x, ymin, ymax,
                        where=mask==1,
                        color='red',
                        alpha=0.25,
                        label='Image Token Weights')
        ax1.plot(data, label='Attention Mask', c='black')
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        ax1.set_title('Attention Weights', fontsize=14)
        ax1.set_xlabel('Token Index', fontsize=13)
        ax1.set_ylabel('Weight', fontsize=13)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlim(0, len(data))
        
        # Create heatmap of the square data
        square_data = masked_data.reshape(size, size)
        im = ax2.imshow(square_data, cmap='RdBu_r')
        fig.colorbar(im, ax=ax2)
        ax2.set_title(f'Image Token Attention Weights', fontsize=14)
        ax2.axis('off')
        
        # Add image
        img = Image.open(image_path)
        ax3.imshow(img)
        ax3.set_title('Input Image', fontsize=14)
        ax3.axis('off')
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
        #print(f'n_probes: {n_probes}')
        
        # Single linear projection to compute attention scores for all probes
        # Maps from input_dim -> n_probes
        self.attention_proj = nn.Sequential(nn.Linear(input_dim, input_dim//2),
        									 nn.ReLU(),
                                             nn.Linear(input_dim//2, n_probes))
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
        #print(f'attention_logits.shape: {attention_logits.shape}')
        
        # Apply softmax over token dimension (dim=1)
        # Input/Output shape: [B, T, P]
        #attention = F.softmax(attention_logits / torch.sqrt(torch.tensor(self.input_dim)), dim=1)
        attention = F.softmax(attention_logits, dim=1)
        #print(f'attention post softmax shape: {attention_logits.shape}')
        
        # Pool representations using attention weights
        # attention: [B, T, P]
        # x: [B, T, D]
        # Output: [B, P, D]
        pooled = torch.einsum('btp,btd->bpd', attention, x)
        #print(f'pooled shape: {pooled.shape}')
        #print('\n\n\n\n')
        return pooled, attention.transpose(1, 2)  # Return [B, P, T] for consistency