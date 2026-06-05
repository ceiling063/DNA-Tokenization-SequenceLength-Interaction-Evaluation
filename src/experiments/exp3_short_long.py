"""词表特性完整分析
包括：词表大小、OOV率、平均token数、耗时、稀疏度指标
"""

import time
import pandas as pd
import os
from collections import Counter

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.kmer import KmerTokenizer
from src.tokenization.canonical import CanonicalKmerTokenizer
from src.tokenization.bpe import BPETokenizer

# 加载数据
print("=" * 70)
print("词表特性完整分析")
print("=" * 70)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_short = train_df[train_df['type'] == 'short']['sequence'].tolist()
test_short = test_df[test_df['type'] == 'short']['sequence'].tolist()
train_long = train_df[train_df['type'] == 'long']['sequence'].tolist()
test_long = test_df[test_df['type'] == 'long']['sequence'].tolist()

# 定义策略
strategies = {
    '重叠 k-mer': (KmerTokenizer, {'k': 5}),
    'Canonical k-mer': (CanonicalKmerTokenizer, {'k': 5}),
    'BPE': (BPETokenizer, {'vocab_size': 500, 'min_frequency': 2})
}

results = []

for name, (TokenizerClass, params) in strategies.items():
    print(f"\n{'=' * 70}")
    print(f"策略: {name}")
    print(f"{'=' * 70}")

    for seq_type, train_seqs, test_seqs in [
        ('短 reads', train_short, test_short),
        ('长 contigs', train_long, test_long)
    ]:
        print(f"\n--- {seq_type} ---")

        # 1. 构建词表 + 计时
        tokenizer = TokenizerClass(**params)
        start = time.time()
        tokenizer.build_vocab(train_seqs)
        build_time = time.time() - start

        vocab_size = tokenizer.get_vocab_size()

        # 2. Tokenization 耗时
        n_test = min(500, len(test_seqs))
        test_sample = test_seqs[:n_test]

        start = time.time()
        token_lists = [tokenizer.tokenize(seq) for seq in test_sample]
        tok_time = (time.time() - start) / n_test * 1000

        # 3. 平均 token 数
        avg_tokens = sum(len(tokens) for tokens in token_lists) / n_test

        # 4. OOV 率
        oov_rate = tokenizer.compute_oov_rate(test_seqs)

        # 5. 词表稀疏度分析（只对训练集做）
        all_tokens = []
        for seq in train_seqs[:500]:
            tokens = tokenizer.tokenize(seq)
            all_tokens.extend(tokens)

        freq = Counter(all_tokens)
        freq_values = sorted(freq.values(), reverse=True)

        total_tokens = len(all_tokens)
        vocab_size_actual = len(freq_values)

        # 稀疏度指标
        avg_freq = total_tokens / vocab_size_actual if vocab_size_actual > 0 else 0
        low_freq_count = sum(1 for v in freq_values if v == 1)
        low_freq_ratio = low_freq_count / vocab_size_actual if vocab_size_actual > 0 else 0

        top_10_count = max(1, int(vocab_size_actual * 0.1))
        top_10_tokens = sum(freq_values[:top_10_count])
        top_10_coverage = top_10_tokens / total_tokens if total_tokens > 0 else 0

        # 稀疏度判断
        if low_freq_ratio > 0.5:
            sparsity_level = "高"
        elif low_freq_ratio > 0.3:
            sparsity_level = "中"
        else:
            sparsity_level = "低"

        # 打印结果
        print(f"  词表大小: {vocab_size}")
        print(f"  OOV率: {oov_rate:.2%}")
        print(f"  平均token数: {avg_tokens:.1f}")
        print(f"  耗时: {tok_time:.2f} ms/seq")
        print(f"  构建时间: {build_time:.2f} s")
        print(f"\n  --- 稀疏度指标 ---")
        print(f"  总token数: {total_tokens}")
        print(f"  平均出现次数: {avg_freq:.2f}")
        print(f"  出现1次的token比例: {low_freq_ratio:.2%}")
        print(f"  前10% token覆盖率: {top_10_coverage:.2%}")
        print(f"  稀疏度等级: {sparsity_level}")

        # 保存结果
        results.append({
            '策略': name,
            '序列类型': seq_type,
            '词表大小': vocab_size,
            'OOV率': f"{oov_rate:.2%}",
            '平均token数': f"{avg_tokens:.1f}",
            '耗时(ms/seq)': f"{tok_time:.2f}",
            '构建时间(s)': f"{build_time:.2f}",
            '总token数': total_tokens,
            '平均出现次数': f"{avg_freq:.2f}",
            '出现1次token比例': f"{low_freq_ratio:.2%}",
            '前10%token覆盖率': f"{top_10_coverage:.2%}",
            '稀疏度等级': sparsity_level
        })

# 保存结果
df_results = pd.DataFrame(results)
os.makedirs("results/tables", exist_ok=True)
df_results.to_csv("results/tables/exp3_analysis.csv", index=False)

# 打印汇总表格
print("\n" + "=" * 70)
print("汇总表格")
print("=" * 70)
print(df_results[['策略', '序列类型', '词表大小', 'OOV率', '平均token数', '耗时(ms/seq)', '稀疏度等级']].to_string(
    index=False))

print("\n✅ 结果已保存到 results/tables/exp3_analysis.csv")