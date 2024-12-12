import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from PIL import Image

from probes.probe_utils import AttentionPooler, SimpleMLP, PooledAttentionProbe
from utils import collect_outputs


class BasePooledProbe(pl.LightningModule):
    '''Base class containing shared functionality'''
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 task_name: str, model_name: str, lr: float = 5e-4,
                 weight_decay: float = 1e-6, max_epochs: int = 20,
                 visualize_attention: bool = False, sparsity_lambda: float = 0, 
                 probe_name: str = None, layer_name: str = None):
        super().__init__()
        self.save_hyperparameters()
        
        pooler = AttentionPooler(input_dim, input_dim)
        probe_mlp = SimpleMLP(input_dim, input_dim, output_dim)
        self.net = PooledAttentionProbe(pooler, probe_mlp)
        self.outputs = []
        self.output_dir = os.path.join('output', task_name, model_name, probe_name, layer_name, 'attention_patterns')
        
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
        return self.loss(batch, mode='test')

    def _visualize_attention_patterns(self, outputs):
        if not self.hparams.visualize_attention:
            return
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
        size = int(np.ceil(np.sqrt(len(masked_data)))) # TODO: MAKE GENERAL FOR RECTANGULAR IMAGES
        
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

    def _visualize_attention_patterns_OLD(self, outputs):
        if not self.hparams.visualize_attention:
            return
        
        attention_patterns = outputs['attention'][:25, 1:-1] # exclude delimiter image tokens
        seq_len = attention_patterns.shape[1]
        side_len = int(np.sqrt(seq_len))
        if side_len * side_len != seq_len:
            print(f'Warning: Sequence length {seq_len} is not a perfect square')
            return
            
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for attention, ax in zip(attention_patterns, axes.flat):
            att_map = attention.reshape(side_len, side_len).cpu().numpy()
            _ = ax.imshow(att_map, cmap='viridis')
            ax.axis('off')
        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f'epoch-{self.current_epoch}.png')
        plt.savefig(output_path)
        plt.close()


class PooledRegressionProbe(BasePooledProbe):

    def loss(self, batch, mode='train'):
        xs, ys, metadata, masks = batch
        predictions, attention = self.forward(xs)
        loss = F.mse_loss(predictions, ys)
        loss = loss + self.hparams.sparsity_lambda * attention.abs().mean()
        self.log(f'{mode}_loss', loss, on_epoch=True)
        self.log(f'{mode}_r2', self._r2_score(predictions, ys), on_epoch=True)
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': ys,
            'attention': attention,
            'metadata': metadata,
            'masks': masks
        }
    
    def _r2_score(self, y_pred, y_true):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot
    
    def on_validation_epoch_end(self):
        # Collect the results from the validation loop.
        outputs = collect_outputs(self.outputs)
        
        # Visualize attention maps and model predictions.
        if self.hparams.visualize_attention:
            self._visualize_attention_patterns(outputs)
        self._plot_prediction_scatter(outputs, mode='val')
        self.outputs.clear()

    def _plot_prediction_scatter(self, outputs, mode='val'):
        preds = outputs['predictions']
        targets = outputs['targets']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(targets, preds, alpha=0.5)
        lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('Ground Truth Numerosity')
        ax.set_ylabel('Model Prediction')
        ax.set_title(f'Predictions vs Ground Truth ({mode.capitalize()})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        self.logger.experiment.add_figure(
            f'{mode}_prediction_scatter',
            fig,
            global_step=self.current_epoch
        )
        plt.close(fig)


class PooledClassificationProbe(BasePooledProbe):

    def loss(self, batch, mode='train'):
        xs, ys, metadata, masks = batch
        logits, attention = self.forward(xs)
        class_loss = F.cross_entropy(logits, ys)
        loss = class_loss + self.hparams.sparsity_lambda * attention.abs().mean()
        predictions = logits.argmax(dim=1)
        targets = ys.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
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
        
        # Visualize attention maps and class accuracies.
        if self.hparams.visualize_attention:
            self._visualize_attention_patterns(outputs)
        self._plot_class_accuracies(outputs, mode='val')
        self.outputs.clear()

    def _plot_class_accuracies(self, outputs, mode='val'):
        '''Visualize per-class accuracies as a bar plot'''
        preds = outputs['predictions']
        targets = outputs['targets']
        num_classes = self.hparams.output_dim
        accuracies = torch.zeros(num_classes)
        
        for i in range(num_classes):
            mask = (targets == i)
            if mask.sum() > 0:
                accuracies[i] = (preds[mask] == i).float().mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(num_classes), accuracies.cpu().numpy())
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Per-Class Accuracies ({mode.capitalize()})')
        ax.set_xticks(range(num_classes))
        ax.grid(True, alpha=0.3)
        self.logger.experiment.add_figure(
            f'{mode}_class_accuracies',
            fig,
            global_step=self.current_epoch
        )
        plt.close(fig)