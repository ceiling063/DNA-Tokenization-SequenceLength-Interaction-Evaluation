"""
DNA 序列物种分类 - 数据准备脚本（手动下载版）
功能：读取本地基因组文件、切割短reads和长contigs、划分训练/验证/测试集
"""

import os
import random
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

# ========== 配置参数 ==========
SPECIES = {
    "E_coli": {
        "name": "Escherichia coli",
        "refseq_url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz",
        "filename": "GCF_000005845.2_ASM584v2_genomic.fna"
    },
    "B_subtilis": {
        "name": "Bacillus subtilis",
        "refseq_url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.fna.gz",
        "filename": "GCF_000009045.1_ASM904v1_genomic.fna"
    },
    "P_aeruginosa": {
        "name": "Pseudomonas aeruginosa",
        "refseq_url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1/GCF_000006765.1_ASM676v1_genomic.fna.gz",
        "filename": "GCF_000006765.1_ASM676v1_genomic.fna"
    },
    "S_aureus": {
        "name": "Staphylococcus aureus",
        "refseq_url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/013/425/GCF_000013425.1_ASM1342v1/GCF_000013425.1_ASM1342v1_genomic.fna.gz",
        "filename": "GCF_000013425.1_ASM1342v1_genomic.fna"
    }
}
SHORT_READ_LEN = (150, 500)      # 短 reads 长度范围
LONG_CONTIG_LEN = 3000           # 长 contigs 最小长度

# 每个物种的样本数量（先小后大）
N_SHORT_PER_SPECIES = 500         # 短 reads 每个物种条数（测试用5，正式改500）
N_LONG_PER_SPECIES = 50           # 长 contigs 每个物种条数（测试用2，正式改50）

# 输出路径
DATA_RAW_DIR = "../../data/raw"
DATA_PROCESSED_DIR = "../../data/processed"
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


# ========== 加载本地基因组 ==========
def load_genome(fasta_path):
    """读取本地 .fna 文件，返回合并后的完整基因组序列"""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    full_seq = "N".join(str(rec.seq) for rec in records)
    return full_seq


# ========== 切割序列 ==========
def extract_short_reads(genome, n_samples, length_range):
    """随机提取短 reads (150-500 bp)"""
    reads = []
    genome_len = len(genome)
    length_min, length_max = length_range

    for _ in range(n_samples):
        read_len = random.randint(length_min, length_max)
        if genome_len > read_len:
            start = random.randint(0, genome_len - read_len)
        else:
            start = 0
            read_len = genome_len
        reads.append(genome[start:start + read_len])
    return reads


def extract_long_contigs(genome, n_samples, min_length):
    """随机提取长片段 (≥3000 bp)"""
    contigs = []
    genome_len = len(genome)

    for _ in range(n_samples):
        max_start = genome_len - min_length
        if max_start <= 0:
            contigs.append(genome)
        else:
            start = random.randint(0, max_start)
            contigs.append(genome[start:start + min_length])
    return contigs


# ========== 主流程 ==========
def main():
    # 检查文件是否存在
    for species_id, info in SPECIES.items():
        fasta_path = os.path.join(DATA_RAW_DIR, info["filename"])
        if not os.path.exists(fasta_path):
            print(f"❌ 错误：找不到文件 {fasta_path}")
            print(f"   请手动下载 {info['name']} 的基因组文件，放到 {DATA_RAW_DIR}/ 文件夹")
            return

    all_short_reads = []
    all_long_contigs = []

    for species_id, info in SPECIES.items():
        print(f"\n处理物种: {info['name']} ({species_id})")

        fasta_path = os.path.join(DATA_RAW_DIR, info["filename"])
        genome_seq = load_genome(fasta_path)
        print(f"基因组总长度: {len(genome_seq)} bp")

        # 提取短 reads
        short_reads = extract_short_reads(genome_seq, N_SHORT_PER_SPECIES, SHORT_READ_LEN)
        for seq in short_reads:
            all_short_reads.append({"sequence": seq, "label": species_id, "type": "short"})

        # 提取长 contigs
        long_contigs = extract_long_contigs(genome_seq, N_LONG_PER_SPECIES, LONG_CONTIG_LEN)
        for seq in long_contigs:
            all_long_contigs.append({"sequence": seq, "label": species_id, "type": "long"})

    # 合并数据
    all_data = all_short_reads + all_long_contigs
    df = pd.DataFrame(all_data)

    # 划分 train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    # 保存
    train_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"), index=False)

    # 打印统计信息
    print("\n========== 数据统计 ==========")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    print("\n各物种样本分布:")
    print(df.groupby(["label", "type"]).size())
    print("\n数据准备完成！")


if __name__ == "__main__":
    main()