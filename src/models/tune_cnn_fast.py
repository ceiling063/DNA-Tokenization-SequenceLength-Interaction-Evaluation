"""CNN 超参数快速调优（减少轮数，快速出结果）"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.bpe import BPETokenizer
from src.models.cnn_model import DNA_CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences), torch.tensor(labels)


def train_and_evaluate(epochs, batch_size, embed_dim, train_loader, test_loader):
    model = DNA_CNN(vocab_size=tokenizer.get_vocab_size(), embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time

    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    infer_time = (time.time() - start) / total * 1000
    acc = correct / total

    return acc, train_time, infer_time


# 加载数据
print("=" * 60)
print("CNN 快速调优 (BPE + 短 reads)")
print("=" * 60)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_seqs = train_df[train_df['type'] == 'short']['sequence'].tolist()
train_labels_raw = train_df[train_df['type'] == 'short']['label'].tolist()
test_seqs = test_df[test_df['type'] == 'short']['sequence'].tolist()
test_labels_raw = test_df[test_df['type'] == 'short']['label'].tolist()

label_map = {label: i for i, label in enumerate(set(train_labels_raw))}
train_labels = [label_map[l] for l in train_labels_raw]
test_labels = [label_map[l] for l in test_labels_raw]

# 构建 BPE tokenizer
tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
tokenizer.build_vocab(train_seqs)

train_dataset = SequenceDataset(train_seqs, train_labels, tokenizer)
test_dataset = SequenceDataset(test_seqs, test_labels, tokenizer)

# 只跑 3 组关键实验，每组分母跑 10 轮
configs = [
    {'name': '原始 (30轮, batch32, dim64)', 'epochs': 10, 'batch_size': 32, 'embed_dim': 64},
    {'name': '优化1 (20轮, batch64, dim64)', 'epochs': 10, 'batch_size': 64, 'embed_dim': 64},
    {'name': '优化2 (20轮, batch64, dim32)', 'epochs': 10, 'batch_size': 64, 'embed_dim': 32},
]

results = []

for config in configs:
    print(f"\n--- {config['name']} ---")
    print(f"  EPOCHS={config['epochs']}, BATCH_SIZE={config['batch_size']}, EMBED_DIM={config['embed_dim']}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    acc, train_time, infer_time = train_and_evaluate(
        config['epochs'], config['batch_size'], config['embed_dim'],
        train_loader, test_loader
    )

    results.append({
        '配置': config['name'],
        'Epochs': config['epochs'],
        'Batch_Size': config['batch_size'],
        'Embed_Dim': config['embed_dim'],
        '准确率': f"{acc:.4f}",
        '训练时间(s)': f"{train_time:.1f}",
        '推理时间(ms)': f"{infer_time:.2f}"
    })

    print(f"  准确率: {acc:.4f}, 训练时间: {train_time:.1f}s, 推理时间: {infer_time:.2f}ms")

# 打印汇总
print("\n" + "=" * 60)
print("调优结果汇总（10轮快速测试）")
print("=" * 60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# 保存结果
os.makedirs("results/tables", exist_ok=True)
df_results.to_csv("results/tables/cnn_tuning_fast.csv", index=False)
print("\n✅ 结果已保存到 results/tables/cnn_tuning_fast.csv")
print("\n💡 提示：这是10轮的快速测试结果，最终可以用20轮重新训练最佳配置")