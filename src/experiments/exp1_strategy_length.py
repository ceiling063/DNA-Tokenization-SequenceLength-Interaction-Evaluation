"""实验一：切分策略 × 长度交互
固定 k=5（BPE 固定词表大小=500）
对比 3 种策略在 short reads 与 long contigs 上的准确率、F1、效率
"""

import pandas as pd
import time
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.kmer import KmerTokenizer
from src.tokenization.canonical import CanonicalKmerTokenizer
from src.tokenization.bpe import BPETokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 加载数据
print("=" * 60)
print("实验一：切分策略 × 长度交互")
print("=" * 60)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 定义三种策略
strategies = {
    'Overlapping k-mer': KmerTokenizer,
    'Canonical k-mer': CanonicalKmerTokenizer,
    'BPE': BPETokenizer
}

# 策略参数
strategy_params = {
    'Overlapping k-mer': {'k': 5},
    'Canonical k-mer': {'k': 5},
    'BPE': {'vocab_size': 500, 'min_frequency': 2}
}

results = []

for length_type in ['short', 'long']:
    print(f"\n--- {length_type} reads ---")

    # 获取数据
    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_labels = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_labels = test_df[test_df['type'] == length_type]['label'].tolist()

    for name, TokenizerClass in strategies.items():
        print(f"\n  正在评估 {name}...")

        # 1. 构建 tokenizer 并计时
        params = strategy_params[name]
        tokenizer = TokenizerClass(**params)

        start = time.time()
        tokenizer.build_vocab(train_seqs)
        build_time = time.time() - start

        vocab_size = tokenizer.get_vocab_size()


        # 2. 将序列转换为 k-mer 字符串（供 TF-IDF 使用）
        def seq_to_kmer_string(seq):
            if name == 'BPE':
                tokens = tokenizer.tokenize(seq)
                return ' '.join(tokens)
            else:
                kmers = tokenizer.get_kmers(seq)
                return ' '.join(kmers)


        # 3. TF-IDF 向量化 + 计时
        start = time.time()
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform([seq_to_kmer_string(s) for s in train_seqs])
        X_test = vectorizer.transform([seq_to_kmer_string(s) for s in test_seqs])
        vectorize_time = time.time() - start

        # 4. 训练分类器 + 计时
        clf = LogisticRegression(max_iter=1000, random_state=42)

        start = time.time()
        clf.fit(X_train, train_labels)
        train_time = time.time() - start

        # 5. 预测 + 计时
        start = time.time()
        y_pred = clf.predict(X_test)
        infer_time_total = time.time() - start
        infer_time_per_sample = infer_time_total / len(test_seqs) * 1000

        # 6. 计算指标
        acc = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='macro')

        # 记录结果
        results.append({
            'Strategy': name,
            'Length': length_type,
            'Accuracy': acc,
            'F1_Score': f1,
            'Train_Time(s)': train_time,
            'Infer_Time(ms)': infer_time_per_sample,
            'Vocab_Size': vocab_size,
            'Build_Time(s)': build_time,
            'Vectorize_Time(s)': vectorize_time
        })

        print(f"    准确率: {acc:.4f}, F1: {f1:.4f}")
        print(f"    训练时间: {train_time:.2f}s, 推理时间: {infer_time_per_sample:.2f}ms")
        print(f"    词表大小: {vocab_size}")

# 保存结果
df_results = pd.DataFrame(results)
os.makedirs("results/tables", exist_ok=True)
df_results.to_csv("results/tables/exp1_strategy_length.csv", index=False)

# 打印汇总表格
print("\n" + "=" * 60)
print("实验一结果汇总")
print("=" * 60)
print(df_results[['Strategy', 'Length', 'Accuracy', 'F1_Score', 'Train_Time(s)', 'Infer_Time(ms)']].to_string(
    index=False))

print("\n✅ 结果已保存到 results/tables/exp1_strategy_length.csv")