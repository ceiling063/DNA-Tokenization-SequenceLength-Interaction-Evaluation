"""实验二：k 值敏感性分析
固定重叠 k-mer，对比 k=3,4,5,6 在短 reads 和长 contigs 上的性能
"""

import pandas as pd
import time
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.kmer import KmerTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 加载数据
print("=" * 60)
print("实验二：k 值敏感性分析 (重叠 k-mer)")
print("=" * 60)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# k 值列表
k_values = [3, 4, 5, 6]

results = []

for length_type in ['short', 'long']:
    print(f"\n--- {length_type} reads ---")

    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_labels = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_labels = test_df[test_df['type'] == length_type]['label'].tolist()

    for k in k_values:
        print(f"\n  正在评估 k={k}...")

        # 1. 构建 tokenizer
        tokenizer = KmerTokenizer(k=k)

        start = time.time()
        tokenizer.build_vocab(train_seqs)
        build_time = time.time() - start

        vocab_size = tokenizer.get_vocab_size()


        # 2. 转换为 k-mer 字符串
        def seq_to_kmer_string(seq):
            kmers = tokenizer.get_kmers(seq)
            return ' '.join(kmers)


        # 3. TF-IDF 向量化
        start = time.time()
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform([seq_to_kmer_string(s) for s in train_seqs])
        X_test = vectorizer.transform([seq_to_kmer_string(s) for s in test_seqs])
        vectorize_time = time.time() - start

        # 4. 训练分类器
        clf = LogisticRegression(max_iter=1000, random_state=42)

        start = time.time()
        clf.fit(X_train, train_labels)
        train_time = time.time() - start

        # 5. 预测
        start = time.time()
        y_pred = clf.predict(X_test)
        infer_time_total = time.time() - start
        infer_time_per_sample = infer_time_total / len(test_seqs) * 1000

        # 6. 计算指标
        acc = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='macro')

        results.append({
            'k': k,
            'Length': length_type,
            'Accuracy': acc,
            'F1_Score': f1,
            'Vocab_Size': vocab_size,
            'Train_Time(s)': train_time,
            'Infer_Time(ms)': infer_time_per_sample,
            'Build_Time(s)': build_time
        })

        print(f"    准确率: {acc:.4f}, F1: {f1:.4f}, 词表大小: {vocab_size}")

# 保存结果
df_results = pd.DataFrame(results)
os.makedirs("results/tables", exist_ok=True)
df_results.to_csv("results/tables/exp2_k_sensitivity.csv", index=False)

# 打印汇总表格
print("\n" + "=" * 60)
print("实验二结果汇总 (k 值敏感性)")
print("=" * 60)
print(df_results[['k', 'Length', 'Accuracy', 'F1_Score', 'Vocab_Size']].to_string(index=False))

print("\n✅ 结果已保存到 results/tables/exp2_k_sensitivity.csv")