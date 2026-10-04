# memory/bank.py
import torch


class MemoryBank:
    """
    非参数化记忆库。每条记忆包含：
        - key:   [d] 向量，用于检索
        - value: [C] token 序列（原始 token，可拼回输入）
        - meta:  dict，随便挂信息（比如 chunk 位置、surprise 分数）
    """

    def __init__(self, capacity=None):
        self.capacity = capacity
        self.keys = []
        self.values = []
        self.meta = []

    def __len__(self):
        return len(self.keys)

    def clear(self):
        self.keys = []
        self.values = []
        self.meta = []

    def add(self, key, value, meta=None):
        """key: [d]，value: [C]"""
        if self.capacity is not None and len(self.keys) >= self.capacity:
            # 简单策略：满了丢最旧的
            self.keys.pop(0)
            self.values.pop(0)
            self.meta.pop(0)
        self.keys.append(key.detach())
        self.values.append(value.detach())
        self.meta.append(meta or {})

    def retrieve(self, query, top_k=1):
        """
        query: [d]
        返回: (values list, scores)
        """
        if len(self.keys) == 0:
            return [], torch.tensor([])
        keys = torch.stack(self.keys, dim=0)             # [N, d]
        q = query / (query.norm() + 1e-8)
        k = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        sim = k @ q                                       # [N]
        kk = min(top_k, len(self.keys))
        scores, idx = sim.topk(kk)
        return [self.values[i] for i in idx.tolist()], scores