# mini-memory-for-AGI

最小认知架构，手机 + 免费 Colab 实现，一步步搭建、评估、迭代。

## 总路线

1. **混合序列引擎（SSM + Attention）** ✅
2. **外部记忆系统** ✅
3. 世界模型
4. 持续学习
5. 内在动机 + 元认知
6. 全局工作空间
7. 行动闭环

## 环境约束

- 手机写代码，Colab 免费 T4 GPU 训练
- 不碰危险自我复制、不碰关键基础设施
- 每个模块都能跑、能评估、能迭代

---

## 阶段 1：混合序列引擎 ✅

**架构**：3 × SSM → 1 × Attention → 3 × SSM，d_model=128，约 1.2M 参数。

**任务**：Copy（half_len=32，序列长 65，500 步）。

**结果**（T4 GPU，seed=42）：

| 模型 | 参数量 | acc | 训练时间 |
|---|---:|---:|---:|
| 纯 SSM | 1,176,xxx | 0.0225 | 189s |
| 纯 Attention | 509,616 | 1.0000 | 12s |
| Hybrid (3:1:3) | 1,221,184 | 1.0000 | 162s |

**结论**：
- 纯 SSM 无精确检索能力，短序列 Copy 失败
- 纯 Attention 短序列高效，但受窗口限制
- Hybrid 兼顾流式与检索，序列变长时优势才显现

**踩坑记录**：
- MambaBlock 串行扫描 → 改并行扫描（倍增法），提速 2.7 倍
- `_init_weights` 会覆盖 A_log，导致 SSM 失效
- Colab Python 3.13 + torch 默认无 CUDA，需切 T4 GPU

---

## 阶段 2：外部记忆系统 ✅

**目标**：把工作记忆里重要的片段按时间线刻进外部存储，突破 SSM 递归隐状态和 Attention 窗口的容量限制。

**任务**：GridWorld Recall。
 
输入: [bx, by, SEP, n_1, ..., n_L, SEP, QUERY, bx, by]
目标: 只在 QUERY 后两个位置预测 bx, by

**架构演进**：

| 版本 | 机制 | L=100 acc_both |
|---|---|---:|
| base（无记忆） | Hybrid | 0.0000 |
| MemoryLayer | 可微神经图灵机，软写 | 0.0312 |
| RAG v1 | last-token key, CHUNK=16, TOPK=2 | 0.3281 |
| **RAG v2** | **mean pool key, CHUNK=8, TOPK=4** | **1.0000** |

**RAG v2 关键设计**：
1. **冻结 encoder**：Hybrid 只做特征提取，不参与检索梯度
2. **Chunk 切分**：每 8 个 token 为一个 chunk，用 chunk 内 hidden 的 mean 作为 key
3. **非参数化存储**：key 和 value 都是张量，存在 CPU/GPU 内存
4. **注入位置**：把检索回的 chunk 插在 QUERY 前（而非序列开头），让 Attention 窗口能直接看到
5. **直接 CE loss**：在 QUERY 位置直接算两个 token 的交叉熵，不做全序列 shift

**泛化结果**（500 步）：

| 序列长度 | base acc_both | RAG v2 acc_both |
|---|---:|---:|
| L=100 | 0.0000 | 1.0000 |
| L=500 | 0.0000 | 1.0000 |

**关键体会**：
- 软写入的 MemoryLayer 会把信息在 EMA 中稀释掉，长序列必崩
- 记忆应该是**非参数化的外挂硬盘**，而不是内嵌在主干里的脑组织
- 检索位置比检索本身更重要：插入位置不对，信息等于没送进模型

---

## 仓库结构
 
mini-memory-for-AGI/
├── config.py
├── data/
│   ├── init.py
│   └── gridworld.py          # GridWorld Recall 数据生成
├── model/
│   ├── init.py
│   ├── mamba_block.py        # 并行扫描版 SSM
│   ├── attention_block.py    # 滑动窗口 + RoPE
│   └── hybrid.py             # SSM + Attention 主干
├── memory/
│   ├── init.py
│   ├── bank.py               # 非参数化记忆库
│   └── rag.py                # chunk 编码、检索、注入
├── train_rag.py              # 阶段 2 训练脚本
└── README.md

---

## 下一步：阶段 3 世界模型

在潜空间预测下一状态，为内在动机和行动闭环打基础。
 
 
操作
1. 手机浏览器打开 github.com/liaojinxuan666-eng/mini-memory-for-AGI
2. 点 README.md
3. 铅笔图标编辑
4. 全选删掉 → 粘贴上面内容
5. Commit 