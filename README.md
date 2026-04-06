# EaCP: 无标签下适应分布偏移的预测集

基于论文 [Adapting Prediction Sets to Distribution Shifts Without Labels](https://arxiv.org/abs/2406.01416) 的复现与扩展。

本项目是博士课程项目，在原始 EaCP（Entropy base-Adapted Conformal Prediction）方法上新增了三类改进，并在多个分布偏移数据集上进行了实验对比。

---

## 项目结构

```
EaCP/
├── main.py                      # 主入口：解析参数，运行实验
├── uncertainty_functions.py     # 不确定性度量函数（熵、Gini、Top-2 等）
├── utils.py                     # CP 校准、beta 更新策略（batch/online/adaptive/sliding）
├── generate_report.py           # 生成 PDF 实验报告
│
├── conformal/                   # Conformal Prediction 核心
│   ├── conformal_prediction.py  #   THR / RAPS 校准与预测
│   └── evaluation.py            #   Coverage 和 Set Size 评估
│
├── data/                        # 数据加载
│   ├── datasets.py              #   各数据集定义（IN1k, V2, A, R, C）
│   └── loader.py                #   DataLoader 工厂函数
│
├── models/                      # 模型加载
│   └── models.py                #   ResNet-50, ViT 等预训练模型
│
├── TTA/                         # 测试时自适应（Test-Time Adaptation）
│   ├── eata.py                  #   EATA：带样本过滤的熵最小化
│   └── tent.py                  #   Tent：基础熵最小化
│
├── run_remaining.sh             # 批量运行实验脚本
├── run_imagenet_c.py            # ImageNet-C 实验脚本
│
├── datasets/                    # 数据集（不上传）
│   ├── imagenet/val/            #   ImageNet-1k 验证集（50k 张，1000 类）
│   ├── imagenetv2/              #   ImageNet-V2
│   ├── imagenet-a/              #   ImageNet-A（对抗样本）
│   ├── imagenet-r/              #   ImageNet-R（多风格渲染）
│   └── imagenet-c/              #   ImageNet-C（corruption 偏移）
│       ├── contrast/1-5/
│       ├── brightness/1-5/
│       ├── gaussian_noise/1-5/
│       └── motion_blur/1-5/
│
├── inference_results/           # 校准数据（不上传）
│   └── IN1k/imagenet-resnet50.npz
│
└── results/                     # 实验结果
    ├── summary_table1.csv       #   Table 1 汇总
    ├── summary_table2_avg.csv   #   Table 2 汇总（平均）
    ├── summary_table2_detail.csv#   Table 2 详细（每个 severity）
    └── experiment_report.pdf    #   PDF 报告
```

---

## 使用的模型

| 模型 | 来源 | 用途 |
|------|------|------|
| ResNet-50 | `torchvision.models.resnet50(pretrained=True)` | 所有 ImageNet 实验的 base model |

模型权重由 PyTorch 自动下载至 `~/.cache/torch/hub/checkpoints/`，无需手动操作。

---

## 使用的数据集

| 数据集 | 大小 | 说明 |
|--------|------|------|
| ImageNet-1k 验证集 | 50k 张, 1000 类 | 校准集，生成 `imagenet-resnet50.npz` |
| ImageNet-V2 | 10k 张, 1000 类 | 自然分布偏移（重新采集） |
| ImageNet-R | 30k 张, 200 类 | 不同风格渲染（绘画、卡通等） |
| ImageNet-A | 7.5k 张, 200 类 | 对抗性自然样本（模型极易出错） |
| ImageNet-C | 50k×5 张/corruption | 人工 corruption（contrast, brightness, gaussian_noise, motion_blur × severity 1-5） |

---

## 方法说明

### 原始方法（论文复现）
- **SplitCP（`none`）**：标准 Split Conformal Prediction，不做任何适应
- **EaCP（`eacp`）**：EATA 测试时适应 + 熵驱动的 CP 阈值缩放

### 新增改动

#### 改动 1：替换不确定性度量
用 Gini impurity 和 Top-2 margin 替代熵来驱动 CP 阈值缩放。

| update 参数 | 不确定性函数 | 说明 |
|-------------|-------------|------|
| `eacp_gini` | `1 - Σp²` | Raw Gini，值域 [0,1)，β<1 导致 coverage 下降 |
| `eacp_top2` | `1 - (p1-p2)` | Raw Top-2 margin，同样 β<1 |
| `eacp_gini_norm` | `-log(1-gini)` | 归一化 Gini，映射到 [0,∞)，修复量级问题 |
| `eacp_top2_norm` | `-log(margin)` | 归一化 Top-2，接近原始效果且 set size 更小 |

代码位置：`uncertainty_functions.py`（标记 `# ===== NEW =====`）

#### 改动 2：自适应 beta 分位数选取
原始方法固定使用 `quantile(entropy, 1-α)` 计算 beta。新方法根据测试集熵分布的方差动态调整分位数水平。

| update 参数 | 策略 | 说明 |
|-------------|------|------|
| `eacp_adaptive` | 自适应分位数 | 偏移越大 → 分位数越高 → beta 越大 |
| `eacp_online` | 在线更新 | Pinball loss 逐步更新 beta |
| `eacp_sliding` | 滑动窗口 | 用最近 5 个 batch 的累积熵估计，更稳定 |

代码位置：`utils.py`（`update_beta_adaptive`, `SlidingWindowBeta`）

#### 改动 3：Tent TTA + 自适应 Scaling Factor
- 用 Tent（更简单的 TTA，无样本过滤）替代 EATA
- 自适应 scaling factor：根据偏移程度动态调整 `s`（原始固定 s=2）

| update 参数 | 说明 |
|-------------|------|
| `tent_ecp` | Tent + ECP |
| `tent_ecp_adaptive` | Tent + 自适应 beta |
| `eacp_adaptive_scaling` | EATA + ECP + 自适应 scaling factor |

代码位置：`main.py`（标记 `# ===== NEW =====`），`utils.py`（`compute_adaptive_scaling`）

#### 组合方法
| update 参数 | 说明 |
|-------------|------|
| `eacp_top2_norm_adaptive` | 归一化 Top-2 + 自适应 beta |

---

## 环境配置

```bash
conda create -n EaCP python=3.10 numpy pandas tqdm pillow -y
conda activate EaCP
pip install torch torchvision timm wilds matplotlib pyarrow
```

---

## 如何运行

### 1. 生成校准数据（首次运行需要）

用 ResNet-50 在 ImageNet-1k 验证集上跑推理，保存 softmax 输出：

```bash
python generate_calibration.py
# 输出: inference_results/IN1k/imagenet-resnet50.npz
```

### 2. 运行单个实验

```bash
python main.py \
  --dataset imagenet-v2 \
  --model resnet50 \
  --save-name my_experiment \
  --alpha 0.1 \
  --updates none eacp eacp_adaptive tent_ecp
```

### 3. 运行所有实验

```bash
# 跑所有数据集 × 所有方法（约 3 小时）
bash run_remaining.sh
```

### 4. 生成 PDF 报告

```bash
python generate_report.py
# 输出: results/experiment_report.pdf
```

---

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 无 | 数据集：`imagenet-v2`, `imagenet-r`, `imagenet-a`, `imagenet-c` |
| `--model` | `resnet50` | 模型 |
| `--alpha` | `0.1` | 目标错误率（coverage = 1 - alpha = 0.90） |
| `--scaling-factor` | `2` | 缩放因子 s，用于 `softmax × β^s` |
| `--updates` | `none tta ecp eacp naive` | 要运行的方法列表 |
| `--corruption` | `contrast` | ImageNet-C 的 corruption 类型 |
| `--severity` | `1` | ImageNet-C 的 severity 级别（1-5） |
| `--save-name` | 无 | 结果 CSV 文件名 |

---

## 如何查看结果

结果保存在 `results/` 目录下，每个 CSV 包含以下列：

| 列名 | 含义 |
|------|------|
| `update` | 方法名称 |
| `cal_acc` | 校准集准确率 |
| `cal_cov` | 校准集 coverage |
| `cal_size` | 校准集 set size |
| `ood_acc` | OOD 测试集准确率 |
| `ood_cov` | OOD 测试集 coverage（**越接近 0.9 越好**） |
| `ood_size` | OOD 测试集 set size（**越小越好**） |

汇总文件：
- `results/summary_table1.csv` — V2/R/A 三个数据集的 coverage 和 set size
- `results/summary_table2_avg.csv` — ImageNet-C 4 种 corruption 的平均结果
- `results/summary_table2_detail.csv` — ImageNet-C 每个 severity 的详细结果
- `results/experiment_report.pdf` — 完整 PDF 报告（含表格和图表）

---

## 新增代码标记

所有新增代码均用以下注释标记，便于与原始代码区分：

```python
# ===== NEW: 描述 =====
... 新增代码 ...
# ===== END NEW =====
```

---

## 致谢

本项目基于以下仓库：
- [EaCP 原始代码](https://github.com/uoguelph-mlrg/EaCP)
- [EATA](https://github.com/mr-eggplant/EATA)
- [Tent](https://github.com/DequanWang/tent)
- [Conformal Training](https://github.com/google-deepmind/conformal_training)
