import torch
import torch.nn.functional as F


def chunk_and_encode(h_all, x, chunk_size):
    """
    h_all: [B, L, d]  encoder 输出的 hidden
    x:     [B, L]     token 序列
    返回:
        keys:   [B, n_chunks, d]
        values: [B, n_chunks, chunk_size]
    """
    B, L, d = h_all.shape
    n_chunks = (L + chunk_size - 1) // chunk_size

    pad_len = n_chunks * chunk_size - L
    if pad_len > 0:
        x_pad = torch.cat(
            [x, torch.zeros(B, pad_len, dtype=x.dtype, device=x.device)], dim=1
        )
    else:
        x_pad = x

    values = x_pad.view(B, n_chunks, chunk_size)

    key_idx = torch.arange(n_chunks, device=x.device) * chunk_size + (chunk_size - 1)
    key_idx = torch.clamp(key_idx, max=L - 1)
    keys = h_all[:, key_idx, :]

    return keys, values


def retrieve(query, keys, top_k):
    """query [B,d], keys [B,N,d] -> idx [B,K]"""
    q = F.normalize(query, dim=-1)
    k = F.normalize(keys, dim=-1)
    sim = torch.einsum("bd,bnd->bn", q, k)
    K = min(top_k, keys.shape[1])
    _, idx = sim.topk(K, dim=-1)
    return idx


def build_augmented_before_query(x, values, idx, n_ignore_tail=3):
    """
    把检索到的 chunk 插在序列尾部 QUERY 之前。

    x: [B, L]
    values: [B, N, C]
    idx: [B, K]
    n_ignore_tail: 序列末尾保留的 token 数（QUERY, bx, by 三个）

    返回:
        x_aug: [B, L + K*C]
        q_pos: [B]  QUERY 在 x_aug 里的位置索引
    """
    B, L = x.shape
    N, C = values.shape[1], values.shape[2]
    K = idx.shape[1]

    retrieved = torch.gather(
        values, dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, C),
    )
    retrieved_flat = retrieved.reshape(B, K * C)          # [B, K*C]

    head = x[:, :L - n_ignore_tail]                        # [B, L-3]
    tail = x[:, L - n_ignore_tail:]                        # [B, 3]  QUERY, bx, by

    x_aug = torch.cat([head, retrieved_flat, tail], dim=1)
    q_pos = torch.full((B,), L - n_ignore_tail + K * C,
                       dtype=torch.long, device=x.device)
    return x_aug, q_pos