import sys
import time
import torch
import torch.nn.functional as F

from config import CFG
from model import HybridLM
from model.hybrid_mem import HybridMemLM
from data import make_recall_batch


def build(use_mem, cfg):
    kinds = ["ssm", "ssm", "ssm", "mem", "ssm", "ssm", "ssm"] if use_mem \
            else ["ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"]
    cls = HybridMemLM if use_mem else HybridLM
    kwargs = dict(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        window=cfg.model.window,
        layer_kinds=kinds,
        max_len=cfg.model.max_len,
        dropout=cfg.model.dropout,
    )
    if use_mem:
        kwargs["n_mem_slots"] = cfg.model.n_mem_slots
    return cls(**kwargs)


def evaluate(model, n_noise, cfg, device):
    model.eval()
    with torch.no_grad():
        x, y = make_recall_batch(64, n_noise, device)
        logits = model(x)
        shifted_logits = logits[:, :-1]
        shifted_y = y[:, 1:]
        # bx 在 y[-3]，by 在 y[-2]
        pred_bx = shifted_logits[:, -4].argmax(-1)
        pred_by = shifted_logits[:, -3].argmax(-1)
        gt_bx = shifted_y[:, -4]
        gt_by = shifted_y[:, -3]
        acc_bx = (pred_bx == gt_bx).float().mean().item()
        acc_by = (pred_by == gt_by).float().mean().item()
        acc_both = ((pred_bx == gt_bx) & (pred_by == gt_by)).float().mean().item()
    return acc_bx, acc_by, acc_both


def train_one(use_mem, steps, n_noise, cfg=CFG, device="cuda"):
    torch.manual_seed(cfg.train.seed)
    model = build(use_mem, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = "mem" if use_mem else "base"

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        x, y = make_recall_batch(cfg.train.batch_size, n_noise, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.model.vocab_size),
            y[:, 1:].reshape(-1),
            ignore_index=-100,
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

    acc_bx, acc_by, acc_both = evaluate(model, n_noise, cfg, device)
    print(f"[{name}] params={n_params:,}  acc_bx={acc_bx:.4f}  "
          f"acc_by={acc_by:.4f}  acc_both={acc_both:.4f}  "
          f"time={time.time()-t0:.0f}s")
    return dict(name=name, params=n_params, acc_bx=acc_bx, acc_by=acc_by,
                acc_both=acc_both, time=time.time() - t0)


if __name__ == "__main__":
    use_mem = len(sys.argv) > 1 and sys.argv[1] == "mem"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_noise = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    train_one(use_mem, steps, n_noise)