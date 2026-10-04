import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .attention_block import AttentionBlock
from memory import MemoryLayer


class HybridMemLM(nn.Module):
    """
    在 Hybrid 基础上，把某一层替换成 MemoryLayer。

    默认层序：
        ssm ssm ssm  mem  ssm ssm ssm
    """

    def __init__(
        self,
        vocab_size=34,
        d_model=128,
        n_head=4,
        window=64,
        layer_kinds=None,
        n_mem_slots=64,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        if layer_kinds is None:
            layer_kinds = ["ssm", "ssm", "ssm", "mem", "ssm", "ssm", "ssm"]
        self.layer_kinds = layer_kinds
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layers = []
        for k in layer_kinds:
            if k == "attn":
                layers.append(AttentionBlock(d_model, n_head, window, dropout))
            elif k == "mem":
                layers.append(MemoryLayer(d_model, n_mem_slots, dropout))
            else:
                layers.append(MambaBlock(d_model, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        assert L <= self.max_len
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))