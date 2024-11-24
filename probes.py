from abc import ABC, abstractmethod
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torch.nn import Module, Linear, ModuleList, Sequential, Tanh, LayerNorm, Dropout, ReLU
from torch import optim


class Probe(torch.nn.Module):
    def __init__(self, pooler, probe_mlp):
        super().__init__()
        self.pooler = pooler
        self.probe = probe_mlp
        
    def forward(self, x, return_attention=False):
        if return_attention:
            xs, attention = self.pooler(x, return_att_vectors=True)
            ys = self.probe(xs)
            return ys, attention
        xs = self.pooler(x)
        ys = self.probe(xs)
        return ys

class LightningProbe(pl.LightningModule):
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
    
    def forward(self, xs, visualize_attention=False):
        if visualize_attention:
            out, attention = self.net(xs, return_attention=True)
            return out, attention
        return self.net(xs)

    def loss(self, batch, mode='train'):
        xs, ys = batch
        logits = self.forward(xs)
        loss = F.cross_entropy(logits, ys)
        
        # Log metrics
        self.log(f'{mode}_loss', loss)
        accuracy = (logits.argmax(dim=1) == ys.argmax(dim=1)).float().mean()
        self.log(f'{mode}_acc', accuracy)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        if self.hparams.visualize_attention:
            xs, ys = batch
            logits, attention = self.forward(xs, visualize_attention=True)
            loss = F.cross_entropy(logits, ys)
            self.log('val_loss', loss)
            accuracy = (logits.argmax(dim=1) == ys.argmax(dim=1)).float().mean()
            self.log('val_acc', accuracy)
            self.attention_patterns.append(attention)
        return self.loss(batch, mode='val')

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
        for idx, (attention, ax) in enumerate(zip(attention_patterns, axes.flat)):
            # Reshape to square
            att_map = attention.reshape(side_len, side_len).cpu().numpy()
            im = ax.imshow(att_map, cmap='viridis')
            ax.axis('off')
            
        plt.tight_layout()
        
        # Save plot to file
        output_dir = Path(f"output/{self.hparams.task_name}/{self.hparams.model_name}/attention_patterns")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"epoch-{self.current_epoch}.png"
        plt.savefig(output_path)
        plt.close()
        return None

    def test_step(self, batch, batch_idx):
        return self.loss(batch, mode='test')


class AttentionPooler(Module):
    '''Attention pooling as described in https://arxiv.org/pdf/1905.06316.pdf (page 14, C).
    '''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_projections = Linear(input_dim, hidden_dim)
        self.attention_scorers = Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, return_att_vectors=False):
        '''
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :param return_att_vectors: if True, return attention vectors along with pooled embeddings
        :return: a tensor of pooled span embeddings of dimension `hidden_dim`, and optionally attention vectors
        '''
        return self._pool(hidden_states, return_att_vectors=return_att_vectors)

    def _pool(self, hidden_states, return_att_vectors=False):
        '''
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :return: a tensor of pooled span embeddings.
        '''

        # apply projections with parameters for span target k
        embed_spans = [self.in_projections(span) for span in hidden_states]
        att_vectors = [self.attention_scorers(span).softmax(0)
                       for span in embed_spans]

        pooled_spans = [att_vec.T @ embed_span
                        for att_vec, embed_span in zip(embed_spans, att_vectors)]
        if return_att_vectors:
            return torch.stack(pooled_spans).squeeze(-1), torch.stack(att_vectors).squeeze(-1)
        else:
            return torch.stack(pooled_spans).squeeze(-1)
    

class MLP(Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.25, single_span=True):
        super().__init__()

        if not single_span:
            input_dim *= 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout_prob

        self.classifier = self._build_mlp()

    def forward(self, inputs):
        return self.classifier(inputs)

    @abstractmethod
    def _build_mlp(self):
        '''build the mlp classifier

        :rtype: Module
        :return: the mlp module.
        '''


class TenneyMLP(MLP):
    '''The 2 layer MLP used by Tenney et al. in https://arxiv.org/abs/1905.06316.

    https://github.com/nyu-mll/jiant/blob/ead63af002e0f755c6418478ec3cabb4062a601e/jiant/modules/simple_modules.py#L49
    '''

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            Tanh(),
            LayerNorm(self.hidden_dim),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )


class SimpleMLP(MLP):

    def _build_mlp(self):
        return Sequential(
            LayerNorm(self.input_dim),
            Dropout(self.dropout),
            Linear(self.input_dim, self.output_dim)
        )


class HewittMLP(MLP):
    '''MLP-2 from Hewitt and Liang: https://arxiv.org/abs/1909.03368.
    '''

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )
