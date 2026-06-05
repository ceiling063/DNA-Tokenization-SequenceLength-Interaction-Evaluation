"""简化评估：对比 Baseline 和 CNN 的 F1 分数和 Tokenization 耗时
不需要重新训练模型，使用已有结果 + 测量 tokenization 耗时
"""

import time
import pandas as pd
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.bpe import BPETokenizer
from src.models.cnn_model import DNA_CNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset

print("=" * 60)
print("F1 分数 + Tokenization 耗时对比")
print("=" * 60)

# 加载数据
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 标签映射
all_labels = list(set(train_df['label'].tolist()))
label_map = {label: i for i, label in enumerate(all_labels)}

results = []


# ========== 辅助函数 ==========
def seq_to_string(seq, tokenizer):
    tokens = tokenizer.tokenize(seq)
    return ' '.join(tokens)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.sequences[idx])
        token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in sequences]
    return torch.stack(padded), torch.tensor(labels)


# ========== 测量 Tokenization 耗时 ==========
print("\n[1] Tokenization 耗时测量")
print("-" * 40)

tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
train_seqs_all = train_df['sequence'].tolist()
tokenizer.build_vocab(train_seqs_all)

tok_times = {}
for length_type in ['short', 'long']:
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    n_test = min(500, len(test_seqs))
    test_sample = test_seqs[:n_test]

    start = time.time()
    for seq in test_sample:
        tokenizer.tokenize(seq)
    tok_time = (time.time() - start) / n_test * 1000
    tok_times[length_type] = tok_time
    print(f"  {length_type} reads: {tok_time:.2f} ms/seq")

# ========== Baseline F1 ==========
print("\n[2] Baseline F1 分数")
print("-" * 40)

for length_type in ['short', 'long']:
    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_labels_raw = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_labels_raw = test_df[test_df['type'] == length_type]['label'].tolist()

    train_labels = [label_map[l] for l in train_labels_raw]
    test_labels = [label_map[l] for l in test_labels_raw]

    # 构建 tokenizer
    tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
    tokenizer.build_vocab(train_seqs)

    # TF-IDF + LR
    train_strings = [seq_to_string(s, tokenizer) for s in train_seqs]
    test_strings = [seq_to_string(s, tokenizer) for s in test_seqs]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_strings)
    X_test = vectorizer.transform(test_strings)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')

    print(f"  {length_type} reads:")
    print(f"    准确率: {acc:.4f}, F1: {f1:.4f}")

    results.append({
        'Model': 'Baseline',
        'Length': length_type,
        'F1': f1,
        'Tokenization_Time(ms)': tok_times[length_type]
    })

# ========== CNN F1（使用已训练好的模型或重新训练）==========
print("\n[3] CNN F1 分数")
print("-" * 40)

for length_type in ['short', 'long']:
    print(f"\n  训练 CNN ({length_type} reads)...")

    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_labels_raw = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_labels_raw = test_df[test_df['type'] == length_type]['label'].tolist()

    train_labels = [label_map[l] for l in train_labels_raw]
    test_labels = [label_map[l] for l in test_labels_raw]

    # 构建 tokenizer
    tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
    tokenizer.build_vocab(train_seqs)
    vocab_size = tokenizer.get_vocab_size()

    # 数据集
    train_dataset = SequenceDataset(train_seqs, train_labels, tokenizer)
    test_dataset = SequenceDataset(test_seqs, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DNA_CNN(vocab_size=vocab_size, embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练 30 轮
    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    all_preds = []
    all_labels_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels_list.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels_list, all_preds)
    f1 = f1_score(all_labels_list, all_preds, average='macro')

    print(f"    准确率: {acc:.4f}, F1: {f1:.4f}")

    results.append({
        'Model': 'CNN',
        'Length': length_type,
        'F1': f1,
        'Tokenization_Time(ms)': tok_times[length_type]
    })

# ========== 打印汇总 ==========
print("\n" + "=" * 60)
print("汇总结果")
print("=" * 60)

df = pd.DataFrame(results)
print(df[['Model', 'Length', 'F1', 'Tokenization_Time(ms)']].to_string(index=False))

# 保存
os.makedirs("results/tables", exist_ok=True)
df.to_csv("results/tables/f1_time_comparison.csv", index=False)
print("\n✅ 已保存到 results/tables/f1_time_comparison.csv")