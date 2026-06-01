"""
数据统计与可视化脚本
功能：读取已生成的 CSV 文件，输出统计信息和图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/figures"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# 读取数据
train_df = pd.read_csv(f"{DATA_PROCESSED_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_PROCESSED_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_PROCESSED_DIR}/test.csv")

# 合并完整数据集（用于统计）
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 添加序列长度列
df['length'] = df['sequence'].apply(len)

print("========== 数据统计 ==========")
print(f"总样本数: {len(df)}")
print(f"训练集: {len(train_df)}")
print(f"验证集: {len(val_df)}")
print(f"测试集: {len(test_df)}")
print("\n各类型样本分布:")
print(df.groupby(['label', 'type']).size())
print("\n序列长度统计:")
print(df.groupby('type')['length'].describe())

# ========== 图1：短 reads 长度分布直方图 ==========
plt.figure(figsize=(10, 5))
short_df = df[df['type'] == 'short']
plt.hist(short_df['length'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('序列长度 (bp)')
plt.ylabel('频次')
plt.title('短 reads 长度分布 (150-500 bp)')
plt.axvline(short_df['length'].mean(), color='red', linestyle='--', label=f'均值: {short_df["length"].mean():.0f} bp')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/short_reads_length_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# ========== 图2：长 contigs 长度分布直方图 ==========
plt.figure(figsize=(10, 5))
long_df = df[df['type'] == 'long']
plt.hist(long_df['length'], bins=20, edgecolor='black', alpha=0.7, color='darkgreen')
plt.xlabel('序列长度 (bp)')
plt.ylabel('频次')
plt.title('长 contigs 长度分布 (≥3000 bp)')
plt.axvline(long_df['length'].mean(), color='red', linestyle='--', label=f'均值: {long_df["length"].mean():.0f} bp')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/long_contigs_length_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# ========== 图3：各物种样本数量柱状图 ==========
plt.figure(figsize=(8, 5))
sample_counts = df.groupby(['label', 'type']).size().unstack()
sample_counts.plot(kind='bar', color=['steelblue', 'darkgreen'], edgecolor='black', ax=plt.gca())
plt.xlabel('物种')
plt.ylabel('样本数量')
plt.title('各物种样本分布')
plt.legend(title='序列类型')
plt.xticks(rotation=0)
plt.savefig(f"{RESULTS_DIR}/sample_counts.png", dpi=150, bbox_inches='tight')
plt.show()

# ========== 图4：序列长度箱线图（可选） ==========
plt.figure(figsize=(8, 5))
sns.boxplot(x='type', y='length', data=df, palette=['steelblue', 'darkgreen'])
plt.xlabel('序列类型')
plt.ylabel('序列长度 (bp)')
plt.title('序列长度分布箱线图')
plt.savefig(f"{RESULTS_DIR}/length_boxplot.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n图表已保存到 {RESULTS_DIR}/")