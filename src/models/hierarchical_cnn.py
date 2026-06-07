"""层级分类：同时预测门（phylum）和种（species）"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import os

os.chdir(r"D:\github\DNA-Tokenization-SequenceLength-Interaction-Evaluation")

from src.tokenization.bpe import BPETokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
EMBED_DIM = 64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


class HierarchicalCNN(nn.Module):
    """层级分类 CNN：同时预测门和种"""

    def __init__(self, vocab_size, num_phylums=2, num_species=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.conv1 = nn.Conv1d(EMBED_DIM, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)

        # 两个输出头
        self.fc_phylum = nn.Linear(128, num_phylums)  # 预测门（2个类）
        self.fc_species = nn.Linear(128, num_species)  # 预测种（4个类）

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc_phylum(x), self.fc_species(x)


class SequenceDataset(Dataset):
    def __init__(self, sequences, species_labels, phylum_labels, tokenizer):
        self.sequences = sequences
        self.species_labels = species_labels
        self.phylum_labels = phylum_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.sequences[idx])
        token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(self.species_labels[idx], dtype=torch.long),
                torch.tensor(self.phylum_labels[idx], dtype=torch.long))


def collate_fn(batch):
    sequences, species_labels, phylum_labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        padded.append(padded_seq)
    return (torch.stack(padded),
            torch.tensor(species_labels),
            torch.tensor(phylum_labels))


def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_species = nn.CrossEntropyLoss()
    criterion_phylum = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, species_y, phylum_y in train_loader:
            x = x.to(device)
            species_y = species_y.to(device)
            phylum_y = phylum_y.to(device)

            optimizer.zero_grad()
            phylum_out, species_out = model(x)

            loss = criterion_species(species_out, species_y) + criterion_phylum(phylum_out, phylum_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return time.time() - start_time


def evaluate(model, loader):
    model.eval()
    correct_species = 0
    correct_phylum = 0
    correct_both = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for x, species_y, phylum_y in loader:
            x = x.to(device)
            species_y = species_y.to(device)
            phylum_y = phylum_y.to(device)

            phylum_out, species_out = model(x)

            species_pred = species_out.argmax(1)
            phylum_pred = phylum_out.argmax(1)

            correct_species += (species_pred == species_y).sum().item()
            correct_phylum += (phylum_pred == phylum_y).sum().item()
            correct_both += ((species_pred == species_y) & (phylum_pred == phylum_y)).sum().item()
            total += len(x)

    infer_time = (time.time() - start) / total * 1000

    return {
        'species_acc': correct_species / total,
        'phylum_acc': correct_phylum / total,
        'hierarchical_acc': correct_both / total,
        'infer_time_ms': infer_time
    }


# ========== 主程序 ==========
print("=" * 60)
print("层级分类：预测门 + 种")
print("=" * 60)

# 加载数据
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 定义标签映射
species_list = ['E_coli', 'B_subtilis', 'P_aeruginosa', 'S_aureus']
species_to_id = {s: i for i, s in enumerate(species_list)}

# 门映射（变形菌门=0，厚壁菌门=1）
phylum_to_id = {
    'E_coli': 0,  # 变形菌门
    'P_aeruginosa': 0,  # 变形菌门
    'B_subtilis': 1,  # 厚壁菌门
    'S_aureus': 1  # 厚壁菌门
}

for length_type in ['short', 'long']:
    print(f"\n{'=' * 60}")
    print(f"序列类型: {length_type} reads")
    print(f"{'=' * 60}")

    train_seqs = train_df[train_df['type'] == length_type]['sequence'].tolist()
    train_species = train_df[train_df['type'] == length_type]['label'].tolist()
    test_seqs = test_df[test_df['type'] == length_type]['sequence'].tolist()
    test_species = test_df[test_df['type'] == length_type]['label'].tolist()

    # 转换为标签 ID
    train_species_ids = [species_to_id[s] for s in train_species]
    train_phylum_ids = [phylum_to_id[s] for s in train_species]
    test_species_ids = [species_to_id[s] for s in test_species]
    test_phylum_ids = [phylum_to_id[s] for s in test_species]

    # 构建 tokenizer
    tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
    tokenizer.build_vocab(train_seqs)
    vocab_size = tokenizer.get_vocab_size()

    # 创建数据集
    train_dataset = SequenceDataset(train_seqs, train_species_ids, train_phylum_ids, tokenizer)
    test_dataset = SequenceDataset(test_seqs, test_species_ids, test_phylum_ids, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 创建模型
    model = HierarchicalCNN(vocab_size=vocab_size, num_phylums=2, num_species=4).to(device)

    # 训练
    print(f"\n训练中...")
    train_time = train_model(model, train_loader, EPOCHS, LEARNING_RATE)

    # 评估
    metrics = evaluate(model, test_loader)

    print(f"\n结果:")
    print(f"  训练时间: {train_time:.2f} s")
    print(f"  推理时间: {metrics['infer_time_ms']:.2f} ms/seq")
    print(f"  门准确率: {metrics['phylum_acc']:.4f} ({metrics['phylum_acc'] * 100:.2f}%)")
    print(f"  种准确率: {metrics['species_acc']:.4f} ({metrics['species_acc'] * 100:.2f}%)")
    print(f"  层级准确率: {metrics['hierarchical_acc']:.4f} ({metrics['hierarchical_acc'] * 100:.2f}%)")