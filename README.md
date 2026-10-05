# mini-memory-for-AGI

最小认知架构，手机 + 免费 Colab/Kaggle 逐步搭建、评估、迭代。

## 总路线

1. **混合序列引擎（SSM + Attention）** ✅
2. **外部记忆系统** ✅
3. 世界模型
4. 持续学习
5. 内在动机 + 元认知
6. 全局工作空间
7. 行动闭环

## 环境约束

- 手机写代码，Colab / Kaggle 免费 T4 GPU 训练
- 不碰危险自我复制、不碰关键基础设施
- 每个模块都能跑、能评估、能迭代

---

## 阶段 1：混合序列引擎 ✅

**架构**：3 × SSM → 1 × Attention → 3 × SSM，d_model=128，约 1.2M 参数。

**任务**：Copy（half_len=32，序列长 65，500 步）。

**结果**（T4 GPU，seed=42）：

| 模型 | 参数量 | acc | 训练时间 |
|---|---:|---:|---:|
| 纯 SSM | 1,176,640 | 0.0225 | 189s |
| 纯 Attention | 509,616 | 1.0000 | 12s |
| Hybrid (3:1:3) | 1,221,184 | 1.0000 | 162s |

**结论**：
- 纯 SSM 无精确检索能力，短序列 Copy 失败
- 纯 Attention 短序列高效，但受窗口限制
- Hybrid 兼顾流式与检索

**踩坑**：
- MambaBlock 串行扫描 → 改并行扫描（倍增法），提速 2.7x
- `_init_weights` 会覆盖 A_log，导致 SSM 失效
- Colab Python 3.13 + torch 默认无 CUDA，需切 T4 GPU

---

## 阶段 2：外部记忆系统 ✅

**目标**：把重要的片段写入外部存储，突破 SSM 隐状态和 Attention 窗口的容量限制。

**任务**：GridWorld Recall。
 
输入: [bx, by, SEP, n_1, ..., n_L, SEP, QUERY, bx, by]
目标: 只在 QUERY 后两个位置预测 bx, by

**三个版本的演进（重要，记录失败路径）**：

| 版本 | 机制 | L=100 acc_both |
|---|---|---:|
| base（无记忆） | Hybrid | 0.0000 |
| v1: MemoryLayer | 可微 NTM，软写 | 0.0312 |
| v2: RAG 无训练 | 冻结 encoder + 硬 top-k | 0.0312 |
| v2 泄漏版（已废弃） | 未排除尾部 [QUERY, bx, by] | 1.0000（虚假） |
| **v3: RAG + InfoNCE** | **可训练 retriever** | **1.0000** |

**v3 关键设计**：

1. **chunk 化编码**：每 8 个 token 一个 chunk，chunk 内 hidden 求 mean 作为 key
2. **排除尾部**：`chunk_and_encode(..., n_ignore_tail=3)` 不把 `[QUERY, bx, by]` 写进 memory，避免答案泄漏
3. **可训练 retriever**：`Retriever` 模块把 query 和 key 投到同一空间，用 InfoNCE 对齐
4. **注入位置**：检索回的 chunk 插在 QUERY 前（窗口 64 覆盖范围内）
5. **直接 CE loss**：只在 QUERY 位置算两个 token 的交叉熵

**最终结果**（500 步，T4 GPU，seed=42）：

| 序列长度 | hit@4 | acc_both |
|---|---:|---:|
| L=100 | 1.0000 | 1.0000 |
| L=500 | 1.0000 | 1.0000 |

**结论**：
- 软写 MemoryLayer 会把信息在 EMA 中稀释，长序列必崩
- **RAG 的检索必须可训练**：冻结 encoder + 硬 top-k，梯度到不了 encoder，检索永远瞎猜
- 一个 130 万参数的小 retriever 就能把 hit@4 从 0.05 拉到 1.0
- 记忆应该是**非参数化的外挂硬盘**，而不是内嵌在主干里的脑组织

**踩坑**：
- 未排除尾部导致信息泄漏，”成功“是假的
- `subprocess.run` 在 Colab 会缓存输出，看不到进度
- Kaggle 只有 `/kaggle/working/` 持久，`/root` 会清空
- Kaggle 必须手机验证才能用 GPU，且要手动打开 Internet 开关

---

## 仓库结构
 
mini-memory-for-AGI/
├── config.py
├── data/
│   ├── init.py
│   └── gridworld.py              # GridWorld Recall 数据生成
├── model/
│   ├── init.py
│   ├── mamba_block.py            # 并行扫描版 SSM
│   ├── attention_block.py        # 滑动窗口 + RoPE
│   └── hybrid.py                 # SSM + Attention 主干
├── memory/
│   ├── init.py
│   ├── bank.py                   # 非参数化记忆库
│   ├── rag.py                    # chunk 编码、检索、注入
│   └── retriever.py              # 可训练 query/key 对齐
├── legacy/                        # 失败的实验版本，仅作参考
│   ├── memory_layer.py
│   ├── hybrid_mem.py
│   └── train_mem.py
├── train_rag.py                   # 阶段 2 训练脚本
└── README.md

---

## 下一步：阶段 3 世界模型

在潜空间预测下一状态，为内在动机和行动闭环打基础。
 
 