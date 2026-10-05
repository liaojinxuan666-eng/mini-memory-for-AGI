import torch
import torch.nn.functional as F


def chunk_and_encode(h_all, x, chunk_size, n_ignore_tail=3):
    """
    只对前 L - n_ignore_tail 个 token 做 chunk 编码。
    排除末尾的 [QUERY, bx, by]，避免答案泄漏进 memory。

    h_all: [B, L_full, d]
    x:     [B, L_full]
    返回:
        keys:   [B, n_chunks, d]
        values: [B, n_chunks, chunk_size]
    """
    B, L_full, d = h_all.shape
    L = L_full - n_ignore_tail

    h_all = h_all[:, :L]
    x = x[:, :L]

    n_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - L

    if pad_len > 0:
        x_pad = torch.cat(
            [x, torch.zeros(B, pad_len, dtype=x.dtype, device=x.device)], dim=1
        )
        h_pad = torch.cat(
            [h_all, torch.zeros(B, pad_len, d, dtype=h_all.dtype, device=h_all.device)],
            dim=1,
        )
        mask = torch.cat(
            [torch.ones(B, L, device=x.device),
             torch.zeros(B, pad_len, device=x.device)],
            dim=1,
        )
    else:
        x_pad, h_pad = x, h_all
        mask = torch.ones(B, L, device=x.device)

    h_view = h_pad.view(B, n_chunks, chunk_size, d)
    m_view = mask.view(B, n_chunks, chunk_size).unsqueeze(-1)
    keys = (h_view * m_view).sum(dim=2) / (m_view.sum(dim=2) + 1e-6)
    values = x_pad.view(B, n_chunks, chunk_size)
    return keys, values


def retrieve(query, keys, top_k):
    q = F.normalize(query, dim=-1)
    k = F.normalize(keys, dim=-1)
    sim = torch.einsum("bd,bnd->bn", q, k)
    K = min(top_k, keys.shape[1])
    _, idx = sim.topk(K, dim=-1)
    return idx


def build_augmented_before_query(x, values, idx, n_ignore_tail=3):
    """
    把检索到的 chunk 插在序列尾部 QUERY 之前。
    返回:
        x_aug: [B, L + K*C]
        q_pos: [B]
    """
    B, L = x.shape
    N, C = values.shape[1], values.shape[2]
    K = idx.shape[1]

    retrieved = torch.gather(
        values, dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, C),
    )
    retrieved_flat = retrieved.reshape(B, K * C)

    head = x[:, :L - n_ignore_tail]
    tail = x[:, L - n_ignore_tail:]

    x_aug = torch.cat([head, retrieved_flat, tail], dim=1)
    q_pos = torch.full((B,), L - n_ignore_tail + K * C,
                       dtype=torch.long, device=x.device)
    return x_aug, q_pos