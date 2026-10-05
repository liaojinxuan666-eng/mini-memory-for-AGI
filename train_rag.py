import sys
import time
import torch
import torch.nn.functional as F

from config import CFG
from model import HybridLM
from data import make_recall_batch
from memory.rag import chunk_and_encode, build_augmented_before_query
from memory import Retriever


CHUNK_SIZE = 8
TOPK = 4
LAMBDA_RET = 1.0          # InfoNCE 权重


def _eval_from_logits(logits, q_pos, x):
    B = x.shape[0]
    ar = torch.arange(B, device=logits.device)
    pred_bx = logits[ar, q_pos].argmax(-1)
    pred_by = logits[ar, q_pos + 1].argmax(-1)
    gt_bx = x[:, -2]
    gt_by = x[:, -1]
    return (
        (pred_bx == gt_bx).float().mean().item(),
        (pred_by == gt_by).float().mean().item(),
        ((pred_bx == gt_bx) & (pred_by == gt_by)).float().mean().item(),
    )


def _one_step(model, retriever, x, y, cfg, device, train=True):
    with torch.no_grad():
        h = model.forward_hidden(x)
        keys, values = chunk_and_encode(h, x, CHUNK_SIZE)
        query = h[:, x.shape[1] - 3]

    # --- retriever：InfoNCE，正样本 = chunk 0（含坐标） ---
    sim = retriever.similarity(query, keys)                  # [B, N]
    target = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    loss_ret = F.cross_entropy(sim, target)

    # --- 用 retriever 的相似度选 top-k ---
    _, idx = sim.topk(TOPK, dim=-1)

    # --- 主干：CE loss ---
    x_aug, q_pos = build_augmented_before_query(x, values, idx)
    logits = model(x_aug)
    B = x.shape[0]
    ar = torch.arange(B, device=device)
    loss_ce = (
        F.cross_entropy(logits[ar, q_pos], x[:, -2])
        + F.cross_entropy(logits[ar, q_pos + 1], x[:, -1])
    )

    loss = loss_ce + LAMBDA_RET * loss_ret
    return loss, loss_ce.item(), loss_ret.item(), idx


def train_one(steps, n_noise, cfg=CFG, device="cuda"):
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
    retriever = Retriever(cfg.model.d_model, 64).to(device)

    params = list(model.parameters()) + list(retriever.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train(); retriever.train()
        x, y = make_recall_batch(cfg.train.batch_size, n_noise, device)
        loss, ce, ret, _ = _one_step(model, retriever, x, y, cfg, device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % cfg.train.log_every == 0:
            avg = sum(losses[-cfg.train.log_every:]) / cfg.train.log_every
            print(f"[rag] step {step}/{steps}  total {avg:.4f}  "
                  f"ce {ce:.4f}  ret {ret:.4f}  elapsed {time.time()-t0:.0f}s")

    # 评估
    model.eval(); retriever.eval()
    with torch.no_grad():
        x, y = make_recall_batch(64, n_noise, device)
        h = model.forward_hidden(x)
        keys, values = chunk_and_encode(h, x, CHUNK_SIZE)
        query = h[:, x.shape[1] - 3]
        sim = retriever.similarity(query, keys)
        _, idx = sim.topk(TOPK, dim=-1)
        hit = (idx == 0).any(dim=-1).float().mean().item()

        x_aug, q_pos = build_augmented_before_query(x, values, idx)
        logits = model(x_aug)
        acc_bx, acc_by, acc_both = _eval_from_logits(logits, q_pos, x)

    print(f"[rag] params={n_params:,}  hit@4={hit:.4f}  "
          f"acc_bx={acc_bx:.4f}  acc_by={acc_by:.4f}  acc_both={acc_both:.4f}  "
          f"time={time.time()-t0:.0f}s")
    return dict(hit=hit, acc_both=acc_both)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_noise = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    train_one(steps, n_noise)