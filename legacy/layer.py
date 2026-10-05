import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLayer(nn.Module):
    """
    可微外挂记忆层。

    每个 batch 内部维护一个 [B, n_slots, d] 的记忆库。
    逐位置处理：
        1. 读取：query 与所有 slot 的 key 算相似度，softmax 加权取值
        2. 写入：write gate 控制写入多少，新 key/value 与旧槽位软混合
        3. 融合：读到的内容加回当前隐状态

    每个 batch 起始时记忆重置为可学习初值。
    """

    def __init__(self, d_model=128, n_slots=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots

        self.ln = nn.LayerNorm(d_model)

        # 写入侧
        self.write_key = nn.Linear(d_model, d_model)
        self.write_val = nn.Linear(d_model, d_model)
        self.write_gate = nn.Linear(d_model, 1)

        # 读取侧
        self.read_query = nn.Linear(d_model, d_model)

        # 输出
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 记忆初始化（可学习）
        self.key_init = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.val_init = nn.Parameter(torch.zeros(n_slots, d_model))

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        residual = x
        h = self.ln(x)

        # 每个 batch 一份记忆
        keys = self.key_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        vals = self.val_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        scale = D ** 0.5
        reads = []
        for t in range(L):
            ht = h[:, t]                       # [B, D]

            # --- 读取 ---
            q = self.read_query(ht)            # [B, D]
            sim = torch.einsum("bd,bnd->bn", q, keys) / scale
            read_w = torch.softmax(sim, dim=-1)             # [B, N]
            read = torch.einsum("bn,bnd->bd", read_w, vals) # [B, D]
            reads.append(read)

            # --- 写入 ---
            wk = self.write_key(ht)            # [B, D]
            wv = self.write_val(ht)            # [B, D]
            g = torch.sigmoid(self.write_gate(ht))          # [B, 1]

            # 每个 slot 的写入权重
            wsim = torch.einsum("bd,bnd->bn", wk, keys) / scale
            w_attn = torch.softmax(wsim, dim=-1)            # [B, N]
            write_amt = g * w_attn                          # [B, N]

            # 软混合更新
            wa = write_amt.unsqueeze(-1)                    # [B, N, 1]
            keys = (1 - wa) * keys + wa * wk.unsqueeze(1)
            vals = (1 - wa) * vals + wa * wv.unsqueeze(1)

        reads = torch.stack(reads, dim=1)                   # [B, L, D]
        out = residual + self.dropout(self.out_proj(reads))
        return out