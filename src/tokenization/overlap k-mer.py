"""
重叠 k-mer Tokenization
策略1：stride=1 的滑动窗口切分
手动实现：k-mer 切分、词表构建、频率统计
"""

from collections import Counter
import numpy as np

class KmerTokenizer:
    """
    重叠 k-mer tokenizer
    手动实现，不依赖任何高级库
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
        self.vocab = {}  # 词表：k-mer -> 索引
        self.idx_to_kmer = []  # 索引 -> k-mer
        self.is_fitted = False

    def get_kmers(self, sequence):
        """
        从一条序列中提取所有重叠 k-mer

        手动实现：滑动窗口，stride=1

        Example:
        --------
        ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
        """
        kmers = []
        seq_len = len(sequence)

        # 如果序列长度小于 k，无法提取完整的 k-mer
        if seq_len < self.k:
            return kmers

        # 滑动窗口，步长为 1
        for i in range(seq_len - self.k + 1):
            kmer = sequence[i:i + self.k]
            kmers.append(kmer)

        return kmers

    def build_vocab(self, sequences):
        """
        从训练集序列构建词表

        手动实现：
        1. 统计所有 k-mer 的出现次数
        2. 为每个唯一的 k-mer 分配一个整数索引

        Parameters:
        -----------
        sequences : list of str
            训练集的 DNA 序列列表
        """
        # 统计所有 k-mer 的频率
        kmer_counter = Counter()

        for seq in sequences:
            kmers = self.get_kmers(seq)
            kmer_counter.update(kmers)

        # 构建词表（按频率降序排序，高频的在前）
        # 也可以按字母顺序，这里按频率排序让常见 k-mer 有更小的索引
        sorted_kmers = sorted(kmer_counter.items(), key=lambda x: x[1], reverse=True)

        self.vocab = {}
        self.idx_to_kmer = []

        for idx, (kmer, _) in enumerate(sorted_kmers):
            self.vocab[kmer] = idx
            self.idx_to_kmer.append(kmer)

        self.is_fitted = True

        print(f"词表构建完成：k={self.k}, 词表大小={len(self.vocab)}")
        return self.vocab

    def tokenize(self, sequence):
        """
        将单条序列转换为 token 索引列表

        Parameters:
        -----------
        sequence : str
            DNA 序列

        Returns:
        --------
        list of int
            token 索引列表
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        kmers = self.get_kmers(sequence)
        # 对于不在词表中的 k-mer（理论上所有 k-mer 都应该在词表中）
        # 但为了安全，用 -1 表示未知 token
        token_ids = [self.vocab.get(kmer, -1) for kmer in kmers]

        return token_ids

    def tokenize_batch(self, sequences):
        """
        将多条序列批量转换为 token 索引列表

        Parameters:
        -----------
        sequences : list of str
            DNA 序列列表

        Returns:
        --------
        list of list of int
            token 索引列表的列表
        """
        return [self.tokenize(seq) for seq in sequences]

    def get_kmer_counts(self, sequence):
        """
        统计单条序列中各 k-mer 的出现次数

        手动实现频率统计

        Parameters:
        -----------
        sequence : str
            DNA 序列

        Returns:
        --------
        dict
            k-mer -> 出现次数
        """
        kmers = self.get_kmers(sequence)
        return dict(Counter(kmers))

    def get_count_vector(self, sequence, normalize=False):
        """
        将序列转换为频数向量（词袋模型）

        手动实现：按词表顺序构建频数向量

        Parameters:
        -----------
        sequence : str
            DNA 序列
        normalize : bool
            是否归一化为频率（除以总 k-mer 数）

        Returns:
        --------
        np.ndarray
            频数向量，长度 = 词表大小
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        # 初始化全零向量
        count_vector = np.zeros(len(self.vocab), dtype=np.float32)

        # 统计 k-mer 出现次数
        kmer_counts = self.get_kmer_counts(sequence)

        # 填充向量
        for kmer, count in kmer_counts.items():
            if kmer in self.vocab:
                idx = self.vocab[kmer]
                count_vector[idx] = count

        # 可选：归一化
        if normalize:
            total = np.sum(count_vector)
            if total > 0:
                count_vector = count_vector / total

        return count_vector

    def get_vocab_size(self):
        """返回词表大小"""
        return len(self.vocab)

    def get_kmer_frequency(self):
        """
        返回训练集中各 k-mer 的频率（用于分析词表特性）

        Returns:
        --------
        dict
            k-mer -> 频率
        """
        if not self.is_fitted:
            return {}

        # 这里返回的是构建词表时统计的频率
        # 如果要重新统计，需要传入训练集
        return self.kmer_frequency if hasattr(self, 'kmer_frequency') else {}

    def compute_oov_rate(self, sequences):
        """
        计算测试集的 OOV (Out-Of-Vocabulary) 率

        OOV 率 = 不在词表中的 k-mer 数量 / 所有 k-mer 数量
        这是衡量词表稀疏度的关键指标

        Parameters:
        -----------
        sequences : list of str
            测试集序列列表

        Returns:
        --------
        float
            OOV 率（0-1 之间）
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

def get_kmers_simple(sequence, k=5):
    """
    简单函数：从单条序列中提取所有重叠 k-mer

    这是最基础的实现，用于快速测试

    Example:
    --------
    >>> get_kmers_simple("ATCGATCG", k=3)
    ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers


def count_kmers(sequence, k=5):
    """
    统计单条序列中各 k-mer 的出现次数

    Returns:
    --------
    dict
        k-mer -> 出现次数
    """
    kmers = get_kmers_simple(sequence, k)
    return dict(Counter(kmers))


def build_vocab_from_sequences(sequences, k=5):
    """
    从多条序列构建词表

    Returns:
    --------
    dict
        k-mer -> 索引
    """
    all_kmers = []
    for seq in sequences:
        all_kmers.extend(get_kmers_simple(seq, k))

    unique_kmers = list(dict.fromkeys(all_kmers))  # 保持顺序去重

    vocab = {kmer: idx for idx, kmer in enumerate(unique_kmers)}
    return vocab


# ========== 测试代码 ==========

if __name__ == "__main__":
    # 测试数据
    test_sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "ATATATATATAT"
    ]

    print("=" * 50)
    print("测试 1：简单 k-mer 提取")
    print("=" * 50)
    seq = "ATCGATCG"
    kmers = get_kmers_simple(seq, k=3)
    print(f"序列: {seq}")
    print(f"k=3 的 k-mer: {kmers}")
    print(f"数量: {len(kmers)}")

    print("\n" + "=" * 50)
    print("测试 2：KmerTokenizer 完整流程")
    print("=" * 50)

    # 初始化
    tokenizer = KmerTokenizer(k=5)

    # 构建词表
    vocab = tokenizer.build_vocab(test_sequences)
    print(f"词表: {vocab}")

    # tokenize
    tokens = tokenizer.tokenize(test_sequences[0])
    print(f"序列: {test_sequences[0]}")
    print(f"Token 索引: {tokens}")

    # 频数向量
    count_vec = tokenizer.get_count_vector(test_sequences[0])
    print(f"频数向量维度: {len(count_vec)}")
    print(f"非零元素: {np.sum(count_vec > 0)}")

    # OOV 率
    oov_rate = tokenizer.compute_oov_rate(test_sequences)
    print(f"OOV 率 (训练集上): {oov_rate:.4f}")

    # 对新序列测试 OOV
    new_seq = ["TTTTTTTTTTTT"]
    oov_rate_new = tokenizer.compute_oov_rate(new_seq)
    print(f"OOV 率 (新序列): {oov_rate_new:.4f}")

    print("\n" + "=" * 50)
    print("测试 3：不同 k 值的词表大小")
    print("=" * 50)
    for k in [3, 4, 5, 6]:
        tokenizer_k = KmerTokenizer(k=k)
        tokenizer_k.build_vocab(test_sequences)
        print(f"k={k}, 词表大小={tokenizer_k.get_vocab_size()}")