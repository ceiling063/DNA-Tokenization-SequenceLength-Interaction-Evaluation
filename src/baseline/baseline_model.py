"""
统计 Baseline 模型
方法：k-mer 频数 + TF-IDF + Logistic Regression
作为神经网络对比的下限参照
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import sys
import os

# 添加项目根目录到路径（解决导入问题）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tokenization.kmer import KmerTokenizer


class BaselineModel:
    """
    统计 Baseline 分类器
    """

    def __init__(self, k=5, max_iter=1000):
        """
        初始化 Baseline 模型

        Parameters:
        -----------
        k : int
            k-mer 长度
        max_iter : int
            Logistic Regression 最大迭代次数
        """
        self.k = k
        self.kmer_tokenizer = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(max_iter=max_iter, random_state=42)
        self.is_fitted = False

    def _get_kmer_strings(self, sequences):
        """
        将序列列表转换为 k-mer 字符串列表（空格分隔，供 TF-IDF 使用）

        手动实现 k-mer 切分，然后用空格连接成字符串
        """
        kmer_strings = []

        for seq in sequences:
            # 使用手动实现的 k-mer 切分
            kmers = self.kmer_tokenizer.get_kmers(seq)
            # 用空格连接成字符串
            kmer_strings.append(' '.join(kmers))

        return kmer_strings

    def fit(self, train_sequences, train_labels):
        """
        训练 Baseline 模型

        Parameters:
        -----------
        train_sequences : list of str
            训练集序列
        train_labels : list or array
            训练集标签
        """
        print("=" * 50)
        print("训练 Baseline 模型")
        print("=" * 50)
        print(f"k-mer 长度: k={self.k}")
        print(f"训练集样本数: {len(train_sequences)}")

        # 1. 构建 k-mer tokenizer
        print("\n[1/4] 构建 k-mer 词表...")
        self.kmer_tokenizer = KmerTokenizer(k=self.k)
        self.kmer_tokenizer.build_vocab(train_sequences)
        print(f"词表大小: {self.kmer_tokenizer.get_vocab_size()}")

        # 2. 将序列转换为 k-mer 字符串（供 TF-IDF 使用）
        print("\n[2/4] 提取 k-mer 特征...")
        kmer_strings = self._get_kmer_strings(train_sequences)

        # 3. TF-IDF 向量化
        print("\n[3/4] 计算 TF-IDF...")
        X_train = self.tfidf_vectorizer.fit_transform(kmer_strings)
        print(f"特征维度: {X_train.shape[1]}")

        # 4. 训练分类器
        print("\n[4/4] 训练 Logistic Regression 分类器...")
        start_time = time.time()
        self.classifier.fit(X_train, train_labels)
        train_time = time.time() - start_time

        self.is_fitted = True

        print(f"\n✅ 训练完成！耗时: {train_time:.2f} 秒")

        return train_time

    def predict(self, sequences):
        """
        预测序列的物种标签
        """
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 训练模型")

        kmer_strings = self._get_kmer_strings(sequences)
        X = self.tfidf_vectorizer.transform(kmer_strings)
        return self.classifier.predict(X)

    def predict_proba(self, sequences):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 训练模型")

        kmer_strings = self._get_kmer_strings(sequences)
        X = self.tfidf_vectorizer.transform(kmer_strings)
        return self.classifier.predict_proba(X)

    def evaluate(self, test_sequences, test_labels):
        """
        评估模型性能

        Returns:
        --------
        dict
            包含 accuracy, f1, 推理时间等指标
        """
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 训练模型")

        # 测量推理时间
        start_time = time.time()
        y_pred = self.predict(test_sequences)
        infer_time_total = time.time() - start_time
        infer_time_per_sample = infer_time_total / len(test_sequences) * 1000  # 转换为毫秒

        # 计算指标
        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='macro')

        print("\n" + "=" * 50)
        print("Baseline 模型评估结果")
        print("=" * 50)
        print(f"测试集样本数: {len(test_sequences)}")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"F1 分数 (macro): {f1:.4f}")
        print(f"总推理时间: {infer_time_total:.4f} 秒")
        print(f"平均推理时间: {infer_time_per_sample:.2f} ms/条")
        print("\n详细分类报告:")
        print(classification_report(test_labels, y_pred))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'inference_time_total': infer_time_total,
            'inference_time_per_sample_ms': infer_time_per_sample,
            'predictions': y_pred
        }

    def get_vocab_size(self):
        """返回词表大小"""
        if self.kmer_tokenizer is not None:
            return self.kmer_tokenizer.get_vocab_size()
        return 0


# ========== 主程序：运行 Baseline 实验 ==========

if __name__ == "__main__":

    print("=" * 60)
    print("DNA 物种分类 - Baseline 模型实验")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")

    data_dir = "data/processed"
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    # 2. 准备序列和标签（先用短 reads 做实验）
    print("\n[2] 准备数据（短 reads）...")

    train_short = train_df[train_df['type'] == 'short']
    test_short = test_df[test_df['type'] == 'short']

    train_sequences = train_short['sequence'].tolist()
    train_labels = train_short['label'].tolist()
    test_sequences = test_short['sequence'].tolist()
    test_labels = test_short['label'].tolist()

    print(f"训练集短 reads: {len(train_sequences)} 条")
    print(f"测试集短 reads: {len(test_sequences)} 条")

    # 3. 训练 Baseline 模型（k=5）
    print("\n[3] 训练 Baseline 模型...")
    baseline = BaselineModel(k=5)
    train_time = baseline.fit(train_sequences, train_labels)

    # 4. 评估模型
    print("\n[4] 评估模型...")
    results = baseline.evaluate(test_sequences, test_labels)

    # 5. 保存结果
    print("\n[5] 保存结果...")
    results_df = pd.DataFrame([{
        'model': 'Baseline (k-mer + TF-IDF + LR)',
        'k': 5,
        'train_time_sec': train_time,
        'test_accuracy': results['accuracy'],
        'test_f1': results['f1_score'],
        'inference_time_ms_per_sample': results['inference_time_per_sample_ms'],
        'vocab_size': baseline.get_vocab_size()
    }])

    # 确保 results/tables 目录存在
    os.makedirs("results/tables", exist_ok=True)
    results_df.to_csv("results/tables/baseline_results.csv", index=False)
    print("结果已保存到 results/tables/baseline_results.csv")

    # 6. 对比不同 k 值（可选）
    print("\n[6] 可选：对比不同 k 值的效果...")
    print("-" * 40)

    for k in [3, 4, 5, 6]:
        print(f"\n训练 k={k} 的 Baseline...")
        baseline_k = BaselineModel(k=k)
        baseline_k.fit(train_sequences, train_labels)
        results_k = baseline_k.evaluate(test_sequences, test_labels)
        print(f"k={k}, 准确率: {results_k['accuracy']:.4f}, 词表大小: {baseline_k.get_vocab_size()}")

    print("\n" + "=" * 60)
    print("✅ Baseline 实验完成！")
    print("=" * 60)