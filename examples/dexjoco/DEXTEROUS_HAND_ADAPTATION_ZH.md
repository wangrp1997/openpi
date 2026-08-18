# DexJoCo π0.5 复现与灵巧手适配的区别

本文说明本分支的 `pi05_dexjoco_lora` 与 DexJoCo 原 π0.5 复现方案之间的共同点和差异。

## 定位

本分支不是逐项不变的 DexJoCo 原始复现，而是：

> DexJoCo π0.5 LoRA 训练配方 + 双 Allegro 灵巧手表示适配

训练超参数尽量与 DexJoCo 对齐，只修改状态表示和动作头初始化。当前改动具有明确的工程动机，
但在完成同评测种子的对照实验之前，不能宣称成功率已经优于 DexJoCo。

## 对齐的训练设置

| 项目 | DexJoCo 配方 | 本分支 |
| --- | --- | --- |
| 基础权重 | 官方 `pi05_base` | 官方 `pi05_base` |
| 微调方式 | PaliGemma 与 action expert LoRA | 相同 |
| 训练步数 | 60,000 | 60,000 |
| Batch size | 32 | 32 |
| Action horizon | 30 | 30 |
| Max token length | 250 | 250 |
| Warmup | 10,000 steps | 10,000 steps |
| Peak learning rate | `5e-5` | `5e-5` |
| Learning-rate decay | 训练期间基本不降 | 相同 |
| EMA | 关闭 | 关闭 |
| Checkpoint 间隔 | 10,000 steps | 10,000 steps |
| 动作定义 | 44D 绝对动作 | 相同 |

## 差异一：状态表示对齐

DexJoCo 数据中的状态与动作使用不同的旋转表示：

- 原始状态为 46D：左右末端分别使用 `xyz(3) + quaternion(4)`，再加双手关节。
- 动作为 44D：左右末端分别使用 `xyz(3) + rotvec(3)`，再加双手关节。

本分支在训练和推理中统一执行：

```text
state46 quaternion -> state44 rotvec
```

转换后的布局与 action44 一致：

```text
[right xyz3, right rotvec3, right hand16,
 left xyz3,  left rotvec3,  left hand16]
```

对应实现位于 `src/openpi/policies/dexjoco_policy.py`。

这样做的动机是减少模型隐式学习 quaternion 到 rotvec 映射的负担，并让状态与动作采用相同语义布局。
这是一项待验证的表示改动，而不是已经证明有效的结论。严格复现 DexJoCo 原始输入时，应保留 46D state。

## 差异二：Allegro 动作头重新初始化

官方 π0.5 checkpoint 的动作空间为 32D，DexJoCo 双臂动作为 44D。本分支先离线生成确定性的
44D checkpoint：

```bash
JAX_PLATFORMS=cpu uv run python scripts/convert_pi05_dexjoco_action_head.py
```

动作通道划分如下：

| 通道 | 含义 | 初始化方式 |
| --- | --- | --- |
| `0:6` | 右臂 `xyz + rotvec` | 保留官方权重 |
| `6:22` | 右 Allegro hand 16D | NNX Linear fresh init |
| `22:28` | 左臂 `xyz + rotvec` | 保留官方权重 |
| `28:44` | 左 Allegro hand 16D | NNX Linear fresh init |

转换使用固定随机种子，因此生成结果可复现。训练随后使用普通 `CheckpointWeightLoader` 加载：

```text
pi05_base_action_dim_44_hand_fresh/params
```

这里没有沿用官方 32D checkpoint 中落在手指索引上的权重，因为那些维度不具备 DexJoCo Allegro
关节的既定语义。手指 projection kernel 为 fresh init；Linear bias 默认初始化为零是预期行为。

## 为什么不直接使用普通 delta action

本分支保持 DexJoCo 的绝对动作，没有应用 OpenPI 的逐元素 `DeltaActions`。

原因是旋转向量的直接相减：

```text
target_rotvec - current_rotvec
```

一般不等于 SO(3) 上的正确相对旋转。若未来实验相对旋转动作，应先计算：

```text
R_delta = R_target * inverse(R_current)
```

再将 `R_delta` 转为 rotvec。手指关节也暂时保持数据集原始的绝对位置定义，以控制实验变量。

## 推荐的公平对照

要判断灵巧手适配是否真的更好，至少应训练以下两个模型：

1. **DexJoCo-compatible baseline**
   - 46D quaternion state。
   - 官方前 32 个动作通道全部保留。
   - 只初始化新增的 `32:44`。
   - 使用相同的 60k LoRA 配方。

2. **Dexterous-hand adaptation（本分支）**
   - 44D rotvec state。
   - 双臂位姿权重保留。
   - 两只 Allegro hand 通道 fresh init。
   - 其余训练设置完全相同。

两者应使用相同训练 seed、数据顺序、checkpoint 选择规则和独立评测 seeds。建议报告：

- 任务成功率；
- 完成插入的比例；
- time-to-success；
- 抓取丢失、接触失败和恢复失败的比例；
- 多个训练 seed 的均值与方差。

只有本分支在相同评测协议下稳定优于 baseline，才能称为“更适合灵巧手且性能更优”。
