import torch

SEP = 32
QUERY = 33


def make_recall_batch(batch_size, n_noise, device, vocab_coord=32):
    """
    GridWorld Recall 任务。

    输入布局:
        [bx, by, SEP, n_1, ..., n_L, SEP, QUERY, bx, by]
        长度 N = n_noise + 7

    目标:
        target[:, -3] = bx
        target[:, -2] = by
        其它位置 = -100
    """
    B = batch_size
    bx = torch.randint(0, vocab_coord, (B,))
    by = torch.randint(0, vocab_coord, (B,))
    noise = torch.randint(0, vocab_coord, (B, n_noise))
    sep = torch.full((B, 1), SEP)
    query = torch.full((B, 1), QUERY)

    x = torch.cat([
        bx.unsqueeze(1),
        by.unsqueeze(1),
        sep,
        noise,
        sep,
        query,
        bx.unsqueeze(1),
        by.unsqueeze(1),
    ], dim=1)

    target = torch.full_like(x, -100)
    target[:, -3] = x[:, -2]
    target[:, -2] = x[:, -1]

    return x.to(device), target.to(device)