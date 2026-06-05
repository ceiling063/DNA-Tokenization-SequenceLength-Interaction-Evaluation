# DNA-Tokenization-SequenceLength-Interaction-Evaluation
# DNA 序列物种分类中Tokenization 策略与序列⻓度的交互效应评估

## 项目简介

本项目为 GOODLab 考核项目，旨在系统评估不同 DNA tokenization 策略在物种分类任务中的性能，重点关注策略与序列长度的交互效应。

对比的策略包括：
- **重叠 k-mer**（stride=1）
- **Canonical k-mer**（重叠 k-mer + 反向互补归一化）
- **BPE**（Byte Pair Encoding，基于训练集学习合并规则）

序列长度段：
- **短 reads**：150–500 bp（模拟测序短片段）
- **长 contigs**：≥3000 bp（模拟组装后的长片段）

## 目录结构

## 环境配置

### 要求
- Python 3.10 或 3.11解释器
- PyCharm

### 安装依赖

## 数据准备
### 数据来源
从 NCBI RefSeq 数据库下载两个物种的参考基因组：
大肠杆菌（Escherichia coli）：GCF_000005845.2
枯草芽孢杆菌（Bacillus subtilis）：GCF_000009045.1
### 运行数据准备脚本
bash
python src/data/prepare_data.py
脚本功能：读取本地文件\随机切割短 reads（150-500 bp）和长 contigs（≥3000 bp）\按物种分层划分 train/val/test（比例 70/15/15）
保存为 CSV 文件到 data/processed/
### 数据统计
运行统计脚本生成图表：
bash
python src/data/stats.py
生成的图表保存在 results/figures/：
短 reads 长度分布直方图\长 contigs 长度分布直方图\各物种样本数量柱状图\序列长度箱线图

## Tokenization 实现
重叠 k-mer
参数：k = 3, 4, 5, 6
stride = 1
\Canonical k-mer
在重叠 k-mer 基础上，对每个 k-mer 及其反向互补取字典序较小者
\BPE
基于训练集学习合并规则
词表大小：200, 500, 1000
## 模型架构

本项目采用 **1D CNN** 作为神经网络架构，用于 DNA 序列的物种分类。

### 模型选择

通过对比实验，我们评估了两种神经网络架构：1D CNN 和 LSTM。实验结果表明，CNN 在准确率和训练效率上均优于 LSTM。

| 架构 | BPE + 短 reads 准确率 | 训练时间 (s) |
|------|----------------------|-------------|
| 1D CNN | 93.15% | 680 |
| LSTM | 49.33% | 672 |

基于以上结果，选择 1D CNN 作为最终模型架构。

### 模型结构

CNN 模型由以下层组成：

| 层 | 参数 | 输出形状 |
|---|------|---------|
| Embedding | 词表大小 × 64 | (batch, seq_len, 64) |
| Conv1D (1) | 64 → 128, kernel=3, padding=1 | (batch, 128, seq_len) |
| ReLU | - | (batch, 128, seq_len) |
| Conv1D (2) | 128 → 128, kernel=5, padding=2 | (batch, 128, seq_len) |
| ReLU | - | (batch, 128, seq_len) |
| Conv1D (3) | 128 → 128, kernel=7, padding=3 | (batch, 128, seq_len) |
| ReLU | - | (batch, 128, seq_len) |
| Global Avg Pool | 128 channels → 1 value each | (batch, 128) |
| Dropout | rate = 0.3 | (batch, 128) |
| Fully Connected | 128 → 2 | (batch, 2) |

**模型参数总量**：约 200,000 可训练参数

### 训练超参数

| 参数 | 值 |
|------|-----|
| Epochs | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Random Seed | 42 |

### 最终性能

| Tokenization | 长度 | 准确率 | F1 |
|-------------|------|--------|-----|
| BPE | 短 reads | 93.15% | 0.93 |
| BPE | 长 contigs | 84.21% | 0.84 |
##统计 Baseline
k-mer 频数 + TF-IDF + Logistic Regression / SVM
## 神经网络
1D CNN

Embedding dim = 64

3 个卷积块 + 全局平均池化 + 全连接层

参数量：约 200k

## Tokenization 实现

（你的三种策略描述...）

### 词表特性对比

| 策略 | 序列类型 | 词表大小 | 平均Token数 | 耗时(ms/seq) |
|------|---------|---------|-------------|--------------|
| 重叠 k-mer | 短 reads | 1024 | 320.7 | 0.08 |
| Canonical k-mer | 短 reads | 512 | 320.7 | 0.61 |
| BPE | 短 reads | 500 | 191.6 | 24.13 |
| 重叠 k-mer | 长 contigs | 1024 | 2996.0 | 0.66 |
| Canonical k-mer | 长 contigs | 512 | 2996.0 | 5.99 |
| BPE | 长 contigs | 500 | 1792.5 | 233.49 |

### 分类性能（CNN + BPE）

| 策略 | 长度段 | 准确率 | F1 |
|------|--------|--------|-----|
| BPE (vocab=500) | 短 reads | 93.15% | 0.93 |
| BPE (vocab=500) | 长 contigs | 84.21% | 0.84 |

## 主要发现

- **BPE + CNN** 在短 reads 上达到 **93.15%** 准确率，显著优于 k-mer 方法
- **Canonical k-mer** 词表大小比普通 k-mer 减少约 50%
- **BPE tokenization** 耗时较长（24ms/seq），但分类准确率最高
- 长序列推理时间显著长于短序列，符合预期
## 作者
姓名：[Cailing Chen]

学号/邮箱：[9116125030@email.ncu.edu.cn]

## AI 使用声明
本项目在以下环节使用了 AI 辅助（如 Deepseek）：

代码调试与优化（如数据准备脚本、BPE 实现）

实验报告撰写与润色

PyCharm 配置与 Git 操作指导
