"""
对比三种 Tokenization 策略
策略：重叠 k-mer (k=5)、Canonical k-mer (k=5)、BPE (vocab=500)
指标：词表大小、OOV率、平均token数、tokenization耗时、词表稀疏度(Gini系数)
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# 切换到项目根目录
os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.kmer import KmerTokenizer
from src.tokenization.canonical import CanonicalKmerTokenizer
from src.tokenization.bpe import BPETokenizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果文件夹
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

print("=" * 70)
print("三种 Tokenization 策略对比")
print("指标：词表大小 | OOV率 | 平均token数 | 耗时 | Gini系数(稀疏度)")
print("=" * 70)


# ========== 加载数据 ==========
print("\n[1] 加载数据...")
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_short = train_df[train_df['type'] == 'short']['sequence'].tolist()
test_short = test_df[test_df['type'] == 'short']['sequence'].tolist()
train_long = train_df[train_df['type'] == 'long']['sequence'].tolist()
test_long = test_df[test_df['type'] == 'long']['sequence'].tolist()

print(f"训练集短 reads: {len(train_short)} 条")
print(f"测试集短 reads: {len(test_short)} 条")
print(f"训练集长 contigs: {len(train_long)} 条")
print(f"测试集长 contigs: {len(test_long)} 条")


# ========== 计算 Gini 系数（衡量词表稀疏度）==========
def compute_gini_coefficient(tokenizer, sequences):
    """
    计算词表的 Gini 系数
    值越大 = 词表越稀疏（少数 token 出现很多次，多数 token 出现很少次）
    值越小 = 词表分布越均匀
    """
    # 统计所有 token 的频次
    all_tokens = []
    for seq in sequences[:300]:  # 用前300条统计
        tokens = tokenizer.tokenize(seq)
        all_tokens.extend(tokens)

    freq = Counter(all_tokens)
    freq_values = sorted(freq.values(), reverse=True)

    n = len(freq_values)
    if n == 0:
        return 0

    # Gini 系数公式
    indices = np.arange(1, n + 1)
    gini = (2 * np.sum(indices * freq_values)) / (n * np.sum(freq_values)) - (n + 1) / n

    return gini


# ========== 评估单个 tokenizer ==========
def evaluate_tokenizer(tokenizer, name, train_seqs, test_seqs, seq_type):
    """评估单个 tokenizer，返回所有指标"""
    print(f"  正在评估 {name} ({seq_type})...")

    # 1. 构建词表
    start = time.time()
    tokenizer.build_vocab(train_seqs)
    build_time = time.time() - start

    vocab_size = tokenizer.get_vocab_size()

    # 2. Tokenization 耗时（测前500条）
    n_test = min(500, len(test_seqs))
    test_sample = test_seqs[:n_test]

    start = time.time()
    token_lists = [tokenizer.tokenize(seq) for seq in test_sample]
    tok_time = (time.time() - start) / n_test * 1000  # ms/seq

    # 3. 平均 token 数
    avg_tokens = sum(len(tokens) for tokens in token_lists) / n_test

    # 4. OOV 率
    oov_rate = tokenizer.compute_oov_rate(test_seqs)

    # 5. Gini 系数（词表稀疏度）
    gini = compute_gini_coefficient(tokenizer, test_seqs)

    return {
        '策略': name,
        '序列类型': seq_type,
        '词表大小': vocab_size,
        'OOV率': oov_rate,
        '平均token数': avg_tokens,
        '耗时(ms/seq)': tok_time,
        'Gini系数': gini,
        '构建时间(s)': build_time
    }


# ========== 运行对比 ==========
print("\n[2] 运行对比实验...")

results = []

strategies = {
    '重叠 k-mer': (KmerTokenizer, {'k': 5}),
    'Canonical k-mer': (CanonicalKmerTokenizer, {'k': 5}),
    'BPE': (BPETokenizer, {'vocab_size': 500, 'min_frequency': 2})
}

for seq_type, train_seqs, test_seqs in [
    ('短 reads', train_short, test_short),
    ('长 contigs', train_long, test_long)
]:
    print(f"\n--- {seq_type} ---")
    for name, (TokenizerClass, params) in strategies.items():
        tokenizer = TokenizerClass(**params)
        result = evaluate_tokenizer(tokenizer, name, train_seqs, test_seqs, seq_type)
        results.append(result)


# ========== 打印结果表格 ==========
print("\n" + "=" * 70)
print("[3] 对比结果汇总")
print("=" * 70)

df_results = pd.DataFrame(results)

# 格式化输出
print("\n{:<16} {:<10} {:>10} {:>10} {:>12} {:>14} {:>12}".format(
    '策略', '序列类型', '词表大小', 'OOV率', '平均token数', '耗时(ms/seq)', 'Gini系数'
))
print("-" * 85)
for _, row in df_results.iterrows():
    print("{:<16} {:<10} {:>10} {:>9.2%} {:>12.1f} {:>14.2f} {:>12.3f}".format(
        row['策略'], row['序列类型'],
        row['词表大小'], row['OOV率'],
        row['平均token数'], row['耗时(ms/seq)'],
        row['Gini系数']
    ))

# 保存 CSV
df_results.to_csv("results/tables/tokenization_comparison.csv", index=False)
print("\n✅ 结果已保存到 results/tables/tokenization_comparison.csv")


# ========== 绘图 ==========
print("\n[4] 生成对比图表...")

short_data = df_results[df_results['序列类型'] == '短 reads']
long_data = df_results[df_results['序列类型'] == '长 contigs']
x = np.arange(len(short_data))
width = 0.35

# 图1：词表大小
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, short_data['词表大小'], width, label='短 reads', color='steelblue')
ax.bar(x + width/2, long_data['词表大小'], width, label='长 contigs', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(short_data['策略'])
ax.set_ylabel('词表大小')
ax.set_title('三种策略的词表大小对比')
ax.legend()
plt.savefig('results/figures/vocab_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 图2：OOV率
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, short_data['OOV率'], width, label='短 reads', color='steelblue')
ax.bar(x + width/2, long_data['OOV率'], width, label='长 contigs', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(short_data['策略'])
ax.set_ylabel('OOV 率')
ax.set_title('三种策略的 OOV 率对比')
ax.legend()
plt.savefig('results/figures/oov_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 图3：耗时
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, short_data['耗时(ms/seq)'], width, label='短 reads', color='steelblue')
ax.bar(x + width/2, long_data['耗时(ms/seq)'], width, label='长 contigs', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(short_data['策略'])
ax.set_ylabel('Tokenization 耗时 (ms/seq)')
ax.set_title('三种策略的耗时对比')
ax.legend()
plt.savefig('results/figures/time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 图4：Gini系数（稀疏度）
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, short_data['Gini系数'], width, label='短 reads', color='steelblue')
ax.bar(x + width/2, long_data['Gini系数'], width, label='长 contigs', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(short_data['策略'])
ax.set_ylabel('Gini 系数')
ax.set_title('三种策略的词表稀疏度对比（Gini系数越大越稀疏）')
ax.legend()
plt.savefig('results/figures/gini_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("✅ 对比完成！")
print("=" * 70)
print("\n生成的文件：")
print("  - results/tables/tokenization_comparison.csv")
print("  - results/figures/vocab_size_comparison.png")
print("  - results/figures/oov_rate_comparison.png")
print("  - results/figures/time_comparison.png")
print("  - results/figures/gini_comparison.png")