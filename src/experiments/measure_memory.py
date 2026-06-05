"""测量模型推理时的峰值内存"""

import tracemalloc
import torch
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.bpe import BPETokenizer
from src.models.cnn_model import DNA_CNN
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 加载数据
test_df = pd.read_csv("data/processed/test.csv")
test_seqs = test_df[test_df['type'] == 'short']['sequence'].tolist()
test_labels = test_df[test_df['type'] == 'short']['label'].tolist()

# 1. Baseline 内存测量
print("Baseline 峰值内存测量...")
tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
train_seqs = pd.read_csv("data/processed/train.csv")['sequence'].tolist()
tokenizer.build_vocab(train_seqs)

def seq_to_string(seq):
    return ' '.join(tokenizer.tokenize(seq))

vectorizer = TfidfVectorizer()
X_test = vectorizer.fit_transform([seq_to_string(s) for s in test_seqs])

tracemalloc.start()
clf = LogisticRegression(max_iter=1000)
clf.fit(X_test[:10], test_labels[:10])  # 小样本训练
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  Baseline 峰值内存: {peak / 1024 / 1024:.2f} MB")

# 2. CNN 内存测量
print("CNN 峰值内存测量...")
tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
tokenizer.build_vocab(train_seqs)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in sequences]
    return torch.stack(padded), torch.tensor(labels)

class SequenceDataset:
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.sequences[idx])
        ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

label_map = {l: i for i, l in enumerate(set(test_labels))}
test_labels = [label_map[l] for l in test_labels]
dataset = SequenceDataset(test_seqs, test_labels, tokenizer)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

model = DNA_CNN(vocab_size=tokenizer.get_vocab_size(), embed_dim=64)
model.eval()

tracemalloc.start()
with torch.no_grad():
    for x, y in loader:
        _ = model(x)
        break
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  CNN 峰值内存: {peak / 1024 / 1024:.2f} MB")