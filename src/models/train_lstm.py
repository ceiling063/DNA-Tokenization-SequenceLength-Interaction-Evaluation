"""训练 LSTM 并对比三种 tokenization 策略"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.kmer import KmerTokenizer
from src.tokenization.canonical import CanonicalKmerTokenizer
from src.tokenization.bpe import BPETokenizer
from src.models.lstm_model import DNA_LSTM

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
EMBED_DIM = 64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.sequences[idx])
        if self.max_len:
            tokens = tokens[:self.max_len]
        token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    """将同一批次的序列填充到相同长度"""
    sequences, labels = zip(*batch)

    # 找出本批次中最长的序列
    max_len = max(len(seq) for seq in sequences)

    # 填充所有序列到 max_len
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences), torch.tensor(labels)


def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    train_time = time.time() - start_time
    return train_time


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    infer_time = (time.time() - start) / total * 1000
    return correct / total, infer_time


# 主程序
print("=" * 60)
print("LSTM 对比实验")
print("=" * 60)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 把标签转成数字
all_labels = list(set(train_df['label'].tolist()))
label_map = {label: i for i, label in enumerate(all_labels)}

strategies = {
    'kmer': (KmerTokenizer, {'k': 5}, '重叠 k-mer'),
    'canonical': (CanonicalKmerTokenizer, {'k': 5}, 'Canonical k-mer'),
    'bpe': (BPETokenizer, {'vocab_size': 500, 'min_frequency': 2}, 'BPE')
}

results = []

for length_type in ['short', 'long']:
    print(f"\n--- {length_type} reads ---")

    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_labels_raw = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_labels_raw = test_df[test_df['type'] == length_type]['label'].tolist()

    train_labels = [label_map[l] for l in train_labels_raw]
    test_labels = [label_map[l] for l in test_labels_raw]

    for name, (TokenizerClass, params, display_name) in strategies.items():
        print(f"\n  训练 {display_name}...")

        tokenizer = TokenizerClass(**params)
        tokenizer.build_vocab(train_seqs)

        train_dataset = SequenceDataset(train_seqs, train_labels, tokenizer)
        test_dataset = SequenceDataset(test_seqs, test_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        model = DNA_LSTM(vocab_size=tokenizer.get_vocab_size(), embed_dim=EMBED_DIM).to(device)

        train_time = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)
        acc, infer_time = evaluate(model, test_loader)

        results.append({
            'Model': 'LSTM',
            'Strategy': display_name,
            'Length': length_type,
            'Accuracy': acc,
            'Train_Time(s)': train_time,
            'Infer_Time(ms)': infer_time,
            'Vocab_Size': tokenizer.get_vocab_size()
        })

        print(f"    准确率: {acc:.4f}, 训练时间: {train_time:.2f}s")

# 保存结果
df_results = pd.DataFrame(results)
os.makedirs("results/tables", exist_ok=True)
df_results.to_csv("results/tables/lstm_results.csv", index=False)
print("\n✅ LSTM 结果已保存到 results/tables/lstm_results.csv")
print(df_results[['Strategy', 'Length', 'Accuracy', 'Train_Time(s)']].to_string(index=False))