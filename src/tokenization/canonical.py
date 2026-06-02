"""
Canonical k-mer Tokenization
策略2：重叠 k-mer + 反向互补归一化
手动实现：k-mer 切分、反向互补、取 canonical 形式、词表构建、频率统计
"""

from collections import Counter
import numpy as np


class CanonicalKmerTokenizer:
    """
    Canonical k-mer tokenizer
    将每个 k-mer 转换为其 canonical 形式（与原 k-mer 和其反向互补中较小的那个）
    这样互补链会被视为同一个 token，词表大小约为普通 k-mer 的一半
    """

    def __init__(self, k=5):
        """
        初始化 tokenizer

        Parameters:
        -----------
        k : int
            k-mer 的长度
        """
        self.k = k
        self.vocab = {}  # canonical k-mer -> 索引
        self.idx_to_kmer = []  # 索引 -> canonical k-mer
        self.is_fitted = False

    def reverse_complement(self, kmer):
        """
        计算 DNA k-mer 的反向互补

        手动实现：
        1. 反向：将字符串倒序
        2. 互补：A<->T, C<->G

        Example:
        --------
        >>> tokenizer.reverse_complement("ATCG")
        'CGAT'
        """
        # 互补映射表
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

        # 反向并互补
        rc = ''.join(complement[base] for base in reversed(kmer))
        return rc

    def canonical(self, kmer):
        """
        返回 k-mer 的 canonical 形式

        Canonical = min(kmer, reverse_complement(kmer))
        按字典序比较，取较小的那个


        """
        rc = self.reverse_complement(kmer)
        # 取字典序较小的
        return min(kmer, rc)

    def get_kmers(self, sequence):
        """
        从一条序列中提取所有重叠 k-mer，并转换为 canonical 形式

        """
        kmers = []
        seq_len = len(sequence)

        if seq_len < self.k:
            return kmers

        for i in range(seq_len - self.k + 1):
            kmer = sequence[i:i + self.k]
            canonical_kmer = self.canonical(kmer)
            kmers.append(canonical_kmer)

        return kmers

    def build_vocab(self, sequences):
        """
        从训练集序列构建词表

        手动实现：
        1. 提取所有序列的 canonical k-mer
        2. 统计频率
        3. 为每个唯一的 canonical k-mer 分配索引

        Parameters:
        -----------
        sequences : list of str
            训练集的 DNA 序列列表
        """
        kmer_counter = Counter()

        for seq in sequences:
            kmers = self.get_kmers(seq)
            kmer_counter.update(kmers)

        # 按频率降序排序（高频的在前）
        sorted_kmers = sorted(kmer_counter.items(), key=lambda x: x[1], reverse=True)

        self.vocab = {}
        self.idx_to_kmer = []

        for idx, (kmer, _) in enumerate(sorted_kmers):
            self.vocab[kmer] = idx
            self.idx_to_kmer.append(kmer)

        self.is_fitted = True

        print(f"Canonical 词表构建完成：k={self.k}, 词表大小={len(self.vocab)}")
        return self.vocab

    def tokenize(self, sequence):
        """
        将单条序列转换为 canonical token 索引列表
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        kmers = self.get_kmers(sequence)
        token_ids = [self.vocab.get(kmer, -1) for kmer in kmers]
        return token_ids

    def tokenize_batch(self, sequences):
        """批量转换"""
        return [self.tokenize(seq) for seq in sequences]

    def get_kmer_counts(self, sequence):
        """
        统计单条序列中各 canonical k-mer 的出现次数
        """
        kmers = self.get_kmers(sequence)
        return dict(Counter(kmers))

    def get_count_vector(self, sequence, normalize=False):
        """
        将序列转换为频数向量（词袋模型）
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        count_vector = np.zeros(len(self.vocab), dtype=np.float32)
        kmer_counts = self.get_kmer_counts(sequence)

        for kmer, count in kmer_counts.items():
            if kmer in self.vocab:
                idx = self.vocab[kmer]
                count_vector[idx] = count

        if normalize:
            total = np.sum(count_vector)
            if total > 0:
                count_vector = count_vector / total

        return count_vector

    def get_vocab_size(self):
        return len(self.vocab)

    def compute_oov_rate(self, sequences):
        """
        计算测试集的 OOV (Out-Of-Vocabulary) 率
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        total_kmers = 0
        oov_kmers = 0

        for seq in sequences:
            kmers = self.get_kmers(seq)
            total_kmers += len(kmers)

            for kmer in kmers:
                if kmer not in self.vocab:
                    oov_kmers += 1

        if total_kmers == 0:
            return 0.0

        return oov_kmers / total_kmers


# ========== 辅助函数（方便直接调用） ==========

def reverse_complement_simple(kmer):
    """简单函数：计算反向互补"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(kmer))


def canonical_kmer_simple(kmer):
    """简单函数：返回 canonical 形式"""
    rc = reverse_complement_simple(kmer)
    return min(kmer, rc)


def get_canonical_kmers(sequence, k=5):
    """
    简单函数：从序列提取 canonical k-mer
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers.append(canonical_kmer_simple(kmer))
    return kmers


# ========== 对比测试：普通 k-mer vs Canonical k-mer ==========

def compare_with_regular():
    """
    对比普通 k-mer 和 Canonical k-mer 的词表大小
    """
    from src.tokenization.kmer import KmerTokenizer

    test_sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "ATATATATATAT",
        "CGATCGATCGAT",
        "TTTTAAAAAAA"
    ]

    print("=" * 60)
    print("对比测试：普通 k-mer vs Canonical k-mer")
    print("=" * 60)

    for k in [3, 4, 5]:
        # 普通 k-mer
        regular_tokenizer = KmerTokenizer(k=k)
        regular_tokenizer.build_vocab(test_sequences)

        # Canonical k-mer
        canonical_tokenizer = CanonicalKmerTokenizer(k=k)
        canonical_tokenizer.build_vocab(test_sequences)

        reduction = 100 * (1 - canonical_tokenizer.get_vocab_size() / regular_tokenizer.get_vocab_size())

        print(f"\nk={k}:")
        print(f"  普通 k-mer 词表大小: {regular_tokenizer.get_vocab_size()}")
        print(f"  Canonical k-mer 词表大小: {canonical_tokenizer.get_vocab_size()}")
        print(f"  词表缩减: {reduction:.1f}%")


# ========== 测试代码 ==========

if __name__ == "__main__":

    print("=" * 50)
    print("测试 1：反向互补和 Canonical 函数")
    print("=" * 50)

    test_kmers = ["ATCG", "CGAT", "AAAA", "TTTT", "GATC"]

    for kmer in test_kmers:
        rc = reverse_complement_simple(kmer)
        can = canonical_kmer_simple(kmer)
        print(f"{kmer} -> 反向互补: {rc} -> Canonical: {can}")

    print("\n" + "=" * 50)
    print("测试 2：CanonicalKmerTokenizer 完整流程")
    print("=" * 50)

    train_sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "ATATATATATAT"
    ]

    # 初始化
    tokenizer = CanonicalKmerTokenizer(k=5)

    # 构建词表
    vocab = tokenizer.build_vocab(train_sequences)
    print(f"词表: {vocab}")

    # tokenize
    test_seq = "ATCGATCGATCG"
    tokens = tokenizer.tokenize(test_seq)
    print(f"序列: {test_seq}")
    print(f"Canonical token 索引: {tokens}")

    # 频数向量
    count_vec = tokenizer.get_count_vector(test_seq)
    print(f"频数向量维度: {len(count_vec)}")
    print(f"非零元素: {np.sum(count_vec > 0)}")

    # 验证 canonical 效果：正向和反向序列应该得到相同的 token
    print("\n" + "=" * 50)
    print("测试 3：验证 canonical 特性（正向和反向序列应该相同）")
    print("=" * 50)

    seq_forward = "ATCGATCG"
    seq_reverse = reverse_complement_simple(seq_forward)  # 反向互补

    print(f"正向序列: {seq_forward}")
    print(f"反向互补序列: {seq_reverse}")

    tokens_forward = tokenizer.tokenize(seq_forward)
    tokens_reverse = tokenizer.tokenize(seq_reverse)

    print(f"正向 token: {tokens_forward}")
    print(f"反向 token: {tokens_reverse}")

    if tokens_forward == tokens_reverse:
        print("✅ 验证通过：正向和反向序列得到相同的 token 序列！")
    else:
        print("❌ 验证失败")

    # OOV 率
    print("\n" + "=" * 50)
    print("测试 4：OOV 率")
    print("=" * 50)

    oov_rate = tokenizer.compute_oov_rate(train_sequences)
    print(f"OOV 率 (训练集上): {oov_rate:.4f}")

    new_sequences = ["TTTTTTTTTTTT", "CCCCCCCCCCC"]
    oov_rate_new = tokenizer.compute_oov_rate(new_sequences)
    print(f"OOV 率 (新序列): {oov_rate_new:.4f}")

    # 对比测试
    print("\n" + "=" * 50)
    print("测试 5：词表大小对比（普通 vs Canonical）")
    print("=" * 50)
    compare_with_regular()

    print("\n" + "=" * 50)
    print("测试 6：不同 k 值的 Canonical 词表大小")
    print("=" * 50)

    for k in [3, 4, 5, 6]:
        tokenizer_k = CanonicalKmerTokenizer(k=k)
        tokenizer_k.build_vocab(train_sequences)
        print(f"k={k}, 词表大小={tokenizer_k.get_vocab_size()}")