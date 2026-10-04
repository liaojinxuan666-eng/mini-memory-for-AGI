import sys
import time
import torch
import torch.nn.functional as F

from config import CFG
from model import HybridLM
from data import make_recall_batch
from memory.rag import chunk_and_encode, retrieve, build_augmented_before_query


CHUNK_SIZE = 8
TOPK = 4


def _eval_from_logits(logits, q_pos, x):
    """logits: [B, L_aug, V], q_pos: [B], x: [B, L] 原序列"""
    B = x.shape[0]
    ar = torch.arange(B, device=logits.device)
    logit_q = logits[ar, q_pos]                    # 预测 bx
    logit_bx = logits[ar, q_pos + 1]               # 预测 by
    gt_bx = x[:, -2]
    gt_by = x[:, -1]
    pred_bx = logit_q.argmax(-1)
    pred_by = logit_bx.argmax(-1)
    return (
        (pred_bx == gt_bx).float().mean().item(),
        (pred_by == gt_by).float().mean().item(),
        ((pred_bx == gt_bx) & (pred_by == gt_by)).float().mean().item(),
    )


def evaluate(model, n_noise, device, use_rag):
    model.eval()
    with torch.no_grad():
        x, y = make_recall_batch(64, n_noise, device)
        if use_rag:
            h = model.forward_hidden(x)
            keys, values = chunk_and_encode(h, x, CHUNK_SIZE)
            idx = retrieve(h[:, x.shape[1] - 3], keys, TOPK)
            x_aug, q_pos = build_augmented_before_query(x, values, idx)
            logits = model(x_aug)
            return _eval_from_logits(logits, q_pos, x)
        else:
            logits = model(x)
            B, L = x.shape
            q_pos = torch.full((B,), L - 3, dtype=torch.long, device=x.device)
            return _eval_from_logits(logits, q_pos, x)


def train_one(use_rag, steps, n_noise, cfg=CFG, device="cuda"):
    torch.manual_seed(cfg.train.seed)
    model = HybridLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        window=cfg.model.window,
        layer_kinds=["ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"],
        max_len=cfg.model.max_len,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = "rag" if use_rag else "base"

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    t0 = time.time()
    losses = []

    for step in range(1, steps + 1):
        model.train()
        x, y = make_recall_batch(cfg.train.batch_size, n_noise, device)

        if use_rag:
            with torch.no_grad():
                h = model.forward_hidden(x)
                keys, values = chunk_and_encode(h, x, CHUNK_SIZE)
                idx = retrieve(h[:, x.shape[1] - 3], keys, TOPK)
            x_aug, q_pos = build_augmented_before_query(x, values, idx)
            logits = model(x_aug)
        else:
            logits = model(x)
            B, L = x.shape
            q_pos = torch.full((B,), L - 3, dtype=torch.long, device=x.device)

        B = x.shape[0]
        ar = torch.arange(B, device=device)
        loss = (
            F.cross_entropy(logits[ar, q_pos], x[:, -2])
            + F.cross_entropy(logits[ar, q_pos + 1], x[:, -1])
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        losses.append(loss.item())

        if step % cfg.train.log_every == 0:
            avg = sum(losses[-cfg.train.log_every:]) / cfg.train.log_every
            print(f"[{name}] step {step}/{steps}  loss {avg:.4f}  "
                  f"elapsed {time.time()-t0:.0f}s")

    acc_bx, acc_by, acc_both = evaluate(model, n_noise, device, use_rag)
    print(f"[{name}] params={n_params:,}  acc_bx={acc_bx:.4f}  "
          f"acc_by={acc_by:.4f}  acc_both={acc_both:.4f}  "
          f"time={time.time()-t0:.0f}s")
    return dict(name=name, acc_both=acc_both, time=time.time() - t0)


if __name__ == "__main__":
    use_rag = len(sys.argv) > 1 and sys.argv[1] == "rag"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_noise = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    train_one(use_rag, steps, n_noise)