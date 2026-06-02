"""
BPE (Byte Pair Encoding) Tokenization
策略3：基于训练集学习合并规则
手动实现：统计相邻对频次、合并、构建词表
禁止直接调用 tokenizers 库
"""

from collections import Counter, defaultdict
import numpy as np
import re


class BPETokenizer:
    """
    手动实现 BPE tokenizer

    核心算法：
    1. 初始化：将每个字符作为一个 token（用空格分隔）
    2. 统计所有相邻 token 对的频次
    3. 找到频次最高的对，合并成一个新 token
    4. 重复步骤 2-3，直到达到预设的词表大小或无法继续合并
    """

    def __init__(self, vocab_size=500, min_frequency=2):
        """
        初始化 BPE tokenizer

        Parameters:
        -----------
        vocab_size : int
            最终词表大小（包括基础字符）
        min_frequency : int
            合并的最小频次阈值（低于此值的对不合并）
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges = {}  # 合并规则：{(a, b): ab}
        self.vocab = {}  # 最终词表：token -> 索引
        self.idx_to_token = []  # 索引 -> token
        self.is_fitted = False

    def _get_stats(self, word_splits):
        """
        统计相邻 token 对的频次

        手动实现：遍历所有词的 token 序列，统计相邻对的出现次数

        Parameters:
        -----------
        word_splits : list of list of str
            每个词被切分成的 token 列表

        Returns:
        --------
        Counter
            {(token1, token2): 频次}
        """
        pair_counts = Counter()

        for splits in word_splits:
            # 遍历相邻的 token 对
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                pair_counts[pair] += 1

        return pair_counts

    def _merge_pair(self, pair, word_splits):
        """
        合并一个 token 对

        手动实现：将序列中所有相邻的 (a, b) 合并成 ab

        Parameters:
        -----------
        pair : tuple
            要合并的 token 对 (a, b)
        word_splits : list of list of str
            当前所有词的 token 序列

        Returns:
        --------
        list of list of str
            合并后的新 token 序列
        """
        a, b = pair
        new_token = a + b
        new_word_splits = []

        for splits in word_splits:
            new_splits = []
            i = 0
            while i < len(splits):
                # 如果当前位置是 a，且下一个是 b，则合并
                if i < len(splits) - 1 and splits[i] == a and splits[i + 1] == b:
                    new_splits.append(new_token)
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            new_word_splits.append(new_splits)

        return new_word_splits

    def _word_to_splits(self, word):
        """
        将词初始化为字符级别的 token 列表

        """
        return list(word)

    def _train_bpe(self, sequences):
        """
        训练 BPE 合并规则

        手动实现核心 BPE 算法：
        1. 将每条序列初始化为单个字符
        2. 重复合并频次最高的相邻对
        3. 记录每次合并的规则

        Parameters:
        -----------
        sequences : list of str
            训练集的 DNA 序列列表
        """
        # 初始化：每个字符作为一个 token
        word_splits = [self._word_to_splits(seq) for seq in sequences]

        # 初始词表：所有出现的字符
        initial_vocab = set()
        for splits in word_splits:
            initial_vocab.update(splits)

        print(f"初始词表大小（基础字符）: {len(initial_vocab)}")
        print(f"目标词表大小: {self.vocab_size}")

        num_merges = self.vocab_size - len(initial_vocab)
        if num_merges <= 0:
            print("词表大小小于等于基础字符数，无需合并")
            self.merges = {}
            return

        print(f"计划进行 {num_merges} 次合并")

        # 逐步合并
        for merge_idx in range(num_merges):
            # 1. 统计所有相邻对的频次
            pair_counts = self._get_stats(word_splits)

            if not pair_counts:
                print(f"第 {merge_idx + 1} 次合并：无可合并的对，提前终止")
                break

            # 2. 找到频次最高的对
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best_pair]

            if best_count < self.min_frequency:
                print(f"最高频次 {best_count} 低于阈值 {self.min_frequency}，停止合并")
                break

            # 3. 记录合并规则
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token

            # 4. 执行合并
            word_splits = self._merge_pair(best_pair, word_splits)

            if (merge_idx + 1) % 50 == 0:
                print(f"已完成 {merge_idx + 1}/{num_merges} 次合并，当前词表大小: {len(initial_vocab) + merge_idx + 1}")

        print(f"训练完成，共进行 {len(self.merges)} 次合并")

    def build_vocab(self, sequences):
        """
        基于训练集构建 BPE 词表

        Parameters:
        -----------
        sequences : list of str
            训练集的 DNA 序列列表
        """
        # 训练 BPE 合并规则
        self._train_bpe(sequences)

        # 构建最终词表
        # 初始词表：所有基础字符（A, T, C, G）
        base_tokens = {'A', 'T', 'C', 'G'}

        # 添加合并产生的所有 token
        merged_tokens = set(self.merges.values())

        all_tokens = base_tokens.union(merged_tokens)

        # 按 token 长度降序排序（长的优先，避免切分冲突）
        sorted_tokens = sorted(all_tokens, key=lambda x: (-len(x), x))

        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.idx_to_token = sorted_tokens

        self.is_fitted = True

        print(f"BPE 词表构建完成：词表大小={len(self.vocab)}")
        return self.vocab

    def _apply_merges(self, word):
        """
        对单个词应用所有合并规则

        手动实现：按规则逐次合并

        Parameters:
        -----------
        word : str
            输入的 DNA 序列

        Returns:
        --------
        list of str
            BPE token 列表
        """
        # 初始化为字符列表
        tokens = list(word)

        # 按规则长度排序：先应用长规则的合并（避免冲突）
        # 将 merges 按 token 长度降序排序
        sorted_merges = sorted(self.merges.items(), key=lambda x: -len(x[1]))

        # 应用合并规则
        for (a, b), merged in sorted_merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def tokenize(self, sequence):
        """
        将单条序列转换为 BPE token 列表

        Parameters:
        -----------
        sequence : str
            DNA 序列

        Returns:
        --------
        list of str
            BPE token 列表
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        # 应用合并规则
        tokens = self._apply_merges(sequence)

        return tokens

    def tokenize_to_ids(self, sequence):
        """
        将单条序列转换为 token 索引列表
        """
        tokens = self.tokenize(sequence)
        return [self.vocab.get(token, self.vocab.get('?', -1)) for token in tokens]

    def tokenize_batch(self, sequences):
        """批量转换为 token 列表"""
        return [self.tokenize(seq) for seq in sequences]

    def get_count_vector(self, sequence, normalize=False):
        """
        将序列转换为频数向量（词袋模型）
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        tokens = self.tokenize(sequence)
        token_counts = Counter(tokens)

        count_vector = np.zeros(len(self.vocab), dtype=np.float32)
        for token, count in token_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
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
        计算测试集的 OOV 率

        OOV = token 不在词表中（正常情况下 BPE 应该没有 OOV，因为所有字符都在基础词表里）
        """
        if not self.is_fitted:
            raise ValueError("请先调用 build_vocab() 构建词表")

        total_tokens = 0
        oov_tokens = 0

        for seq in sequences:
            tokens = self.tokenize(seq)
            total_tokens += len(tokens)

            for token in tokens:
                if token not in self.vocab:
                    oov_tokens += 1

        if total_tokens == 0:
            return 0.0

        return oov_tokens / total_tokens

    def get_merge_stats(self):
        """
        返回合并统计信息
        """
        return {
            'num_merges': len(self.merges),
            'merges': self.merges
        }


# ========== 辅助函数 ==========

def simple_bpe_merge(sequences, num_merges=100):
    """
    简化版 BPE：直接返回合并后的结果（用于快速测试）

    Parameters:
    -----------
    sequences : list of str
        训练序列
    num_merges : int
        合并次数

    Returns:
    --------
    dict
        合并规则
    """
    tokenizer = BPETokenizer(vocab_size=4 + num_merges)
    tokenizer.build_vocab(sequences)
    return tokenizer.merges


# ========== 测试代码 ==========

if __name__ == "__main__":

    print("=" * 60)
    print("BPE Tokenizer 测试")
    print("=" * 60)

    # 测试数据
    train_sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "ATATATATATAT",
        "CGATCGATCGAT",
        "TTTTAAAAAAA",
        "ATGCATGCATGC"
    ]

    print("\n" + "=" * 50)
    print("测试 1：BPE 训练")
    print("=" * 50)

    # 初始化 BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=20, min_frequency=2)

    # 构建词表
    vocab = tokenizer.build_vocab(train_sequences)

    print(f"\n最终词表大小: {tokenizer.get_vocab_size()}")
    print(f"合并规则数量: {len(tokenizer.merges)}")
    print("\n部分合并规则示例:")
    for i, ((a, b), merged) in enumerate(list(tokenizer.merges.items())[:10]):
        print(f"  {i + 1}: ({a}, {b}) -> {merged}")

    print("\n" + "=" * 50)
    print("测试 2：Tokenization")
    print("=" * 50)

    test_seq = "ATCGATCG"
    tokens = tokenizer.tokenize(test_seq)
    print(f"序列: {test_seq}")
    print(f"BPE tokens: {tokens}")
    print(f"Token IDs: {tokenizer.tokenize_to_ids(test_seq)}")

    print("\n" + "=" * 50)
    print("测试 3：不同词表大小的影响")
    print("=" * 50)

    for vocab_size in [10, 20, 30, 50]:
        tok = BPETokenizer(vocab_size=vocab_size, min_frequency=1)
        tok.build_vocab(train_sequences)
        print(f"词表大小 {vocab_size} -> 实际词表: {tok.get_vocab_size()}, 合并次数: {len(tok.merges)}")

    print("\n" + "=" * 50)
    print("测试 4：频数向量")
    print("=" * 50)

    count_vec = tokenizer.get_count_vector(test_seq)
    print(f"频数向量维度: {len(count_vec)}")
    print(f"非零元素个数: {np.sum(count_vec > 0)}")
    print(f"向量前20维: {count_vec[:20]}")

    print("\n" + "=" * 50)
    print("测试 5：OOV 率（BPE 通常没有 OOV，因为基础字符全覆盖）")
    print("=" * 50)

    oov_train = tokenizer.compute_oov_rate(train_sequences)
    print(f"训练集 OOV 率: {oov_train:.4f}")

    new_sequences = ["NNNNNNN", "ATGCATGC"]
    oov_new = tokenizer.compute_oov_rate(new_sequences)
    print(f"新序列 OOV 率: {oov_new:.4f}")

    print("\n" + "=" * 50)
    print("测试 6：与普通 k-mer 的对比")
    print("=" * 50)

    from src.tokenization.kmer import KmerTokenizer

    kmer_tok = KmerTokenizer(k=5)
    kmer_tok.build_vocab(train_sequences)

    print(f"普通 k-mer (k=5) 词表大小: {kmer_tok.get_vocab_size()}")
    print(f"BPE (vocab_size=20) 词表大小: {tokenizer.get_vocab_size()}")
    print(f"BPE 词表缩减: {100 * (1 - tokenizer.get_vocab_size() / kmer_tok.get_vocab_size()):.1f}%")