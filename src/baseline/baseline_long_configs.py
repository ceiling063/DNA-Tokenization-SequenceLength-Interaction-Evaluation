import pandas as pd
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.baseline.baseline_model import BaselineModel

# 加载数据
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 只取长 contigs
train_long = train_df[train_df['type'] == 'long']
test_long = test_df[test_df['type'] == 'long']

train_seqs = train_long['sequence'].tolist()
train_labels = train_long['label'].tolist()
test_seqs = test_long['sequence'].tolist()
test_labels = test_long['label'].tolist()

print(f"训练集长 contigs: {len(train_seqs)} 条")
print(f"测试集长 contigs: {len(test_seqs)} 条")

# 训练和评估
baseline = BaselineModel(k=5)
baseline.fit(train_seqs, train_labels)
results = baseline.evaluate(test_seqs, test_labels)

print(f"\n✅ 长 contigs 结果：准确率 = {results['accuracy']:.4f}, F1 = {results['f1_score']:.4f}")