from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import Module, Linear, ModuleList, Sequential, Tanh, LayerNorm, Dropout, ReLU
from torch import optim


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
        span's position in the batch, and [a, b) the span interval along the sequence dimension.
        :return: a tensor of pooled span embeddings of dimension `hidden_dim`.
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
class Probe(torch.nn.Module):
    def __init__(self, pooler, probe_mlp):
        super().__init__()
        self.pooler = pooler
        self.probe = probe_mlp
        
    def forward(self, x):
        xs = self.pooler(x)
        ys = self.probe(xs)
        return ys

class LightningProbe(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int, 
        output_dim: int,
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        max_epochs: int = 20
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create the network
        pooler = AttentionPooler(input_dim, hidden_dim)
        probe_mlp = SimpleMLP(hidden_dim, hidden_dim, output_dim)
        self.net = Probe(pooler, probe_mlp)
        
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
        return self.loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.loss(batch, mode='test')
