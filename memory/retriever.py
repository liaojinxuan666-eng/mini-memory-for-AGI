import torch
import torch.nn as nn
import torch.nn.functional as F


class Retriever(nn.Module):
    """把 query 和 key 投到同一检索空间，用 InfoNCE 训练对齐。"""

    def __init__(self, d_model=128, d_ret=64):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(d_model, d_ret), nn.GELU(), nn.Linear(d_ret, d_ret),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(d_model, d_ret), nn.GELU(), nn.Linear(d_ret, d_ret),
        )

    def similarity(self, query, keys):
        """
        query: [B, d_model]
        keys:  [B, N, d_model]
        返回:  [B, N]  相似度
        """
        q = F.normalize(self.q_proj(query), dim=-1)          # [B, d_ret]
        k = F.normalize(self.k_proj(keys), dim=-1)           # [B, N, d_ret]
        return torch.einsum("bd,bnd->bn", q, k)