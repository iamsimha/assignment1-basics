import os
import regex as re
import pickle
import os
import time
from typing import IO, Any, BinaryIO
from collections import defaultdict
from multiprocessing import Pool
from tqdm.auto import tqdm
from contextlib import contextmanager

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

@contextmanager
def timer(name="Block"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.4f} seconds")

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        """
        Tokenizer class for encoding and decoding
        Args:
            vocab: Dict[int, tuple(bytes)]
            merges: List[(tuple(bytes), tuple(bytes))]
            special_tokens: str
        """
        self.rev_vocab = {v:k for k, v in vocab.items()}
        self.vocab = vocab
        self.merges = merges
        special_tokens = sorted(special_tokens, key=lambda x: -len(x)) if special_tokens else special_tokens
        self.special_tok_pat = re.compile( "(" + "|".join([re.escape(tok) for tok in special_tokens]) + ")") if special_tokens else None
        self.special_tok_set = set([tok.encode("utf-8") for tok in special_tokens]) if special_tokens else set()
        # We have to keep track of which merges appear first, therefore we maintain the order.
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

    def _find_best_pair(self, pre_token):
        best_pair = None
        for i in range(len(pre_token) - 1):
            a, b = pre_token[i], pre_token[i+1]
            if (a, b) in self.rank:
                if best_pair is None:
                    best_pair = (self.rank[(a, b)], a, b)
                elif self.rank[(a, b)] < best_pair[0]:
                    best_pair = (self.rank[(a, b)], a, b)
        if best_pair:
            # dont return rank
            return best_pair[1], best_pair[2]
        return None

    def encode_with_pairs(self, pre_token):
        while True:
            best_pair = self._find_best_pair(pre_token)
            if best_pair is None:
                return pre_token
            a, b = best_pair
            pre_token = get_new_key(pre_token, best_pair)
        return pre_token

    def encode(self, s):
        # split `s` into pretokens
        list_s: List[str] = self.special_tok_pat.split(s) if self.special_tok_pat else [s]
        # split based on the pattern
        pre_tokens: List[str] = []
        for ss in list_s:
            if ss.encode("utf-8") in self.special_tok_set:
                pre_tokens.append(ss.encode("utf-8"))
            else:
                for m in PAT.finditer(ss):
                    w = tuple([bytes([b]) for b in m.group().encode("utf-8")])
                    pre_tokens.append(w)

        token_ids = []
        for pre_token in pre_tokens:
            if pre_token in self.special_tok_set:
                token_ids.extend([self.rev_vocab[pre_token]])
            else:
                new_key = pre_token
                # for merge_pair in self.merges:
                #     new_key = get_new_key(new_key, merge_pair)
                new_key = self.encode_with_pairs(new_key)

                token_ids.extend([self.rev_vocab[t] for t in new_key])

        return token_ids

    def decode(self, token_ids):
        return b"".join([self.vocab[x] for x in token_ids]).decode("utf-8", errors="ignore")

    def encode_iterable(self, f):
        for line in f:
            yield from self.encode(line)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def initialize_count_dict(chunk, special_token_pattern):
    counts = defaultdict(int)
    sub_chunks = special_token_pattern.split(chunk) if special_token_pattern else [chunk]
    for sub_chunk in sub_chunks:
        for m in PAT.finditer(sub_chunk):
            w = m.group()
            byte_arr = tuple([bytes([b]) for b in w.encode("utf-8")])
            counts[byte_arr] += 1
    return counts


def create_byte_pair_count(count_dict):
    byte_pair_count_dict = defaultdict(int)
    byte_pair_to_token = defaultdict(set)
    for key, cnt in count_dict.items():
        for ind1, ind2 in zip(key, key[1:]):
            byte_pair_count_dict[(ind1, ind2)] += cnt
            byte_pair_to_token[(ind1, ind2)].add(key)
    return byte_pair_count_dict, byte_pair_to_token


def get_most_frequent_pair(count_dict):
    max_count = max(count_dict.values())
    return max(
        (k for k, v in count_dict.items() if v == max_count),
        key=lambda x: x  # lexicographically larger wins on tie
    )


def get_new_key(old_key, merge_pair):
    i = 0
    n = len(old_key)
    ind1, ind2 = merge_pair
    new_pair = merge_pair[0] + merge_pair[1]
    new_key = []
    while i < n:
        if i < n - 1 and old_key[i] == ind1 and old_key[i+1] == ind2:
            new_key.append(new_pair)
            i += 2
        else:
            new_key.append(old_key[i])
            i += 1
    assert len(new_key) > 0
    return tuple(new_key)


def merge_optimised(token_dict, merge_pair, byte_count, byte_pair_to_token):
    """ This procedure updates `token_dict` by adding the inserting the merge_pair.
    This also updates byte_count and byte_pair_to_token with the new counts and tokens
    after the merge.

    Args:
        token_dict: A dictionary with pre-tokens as keys and counts as values.
        merge_pair: Pair of tokens that has to be merged
        byte_count: A dictionary mapping byte-pairs to their counts
        byte_pair_to_token: A dictionary mapping byte-pairs to a list of pre-tokens that contain
                    the byte-pair
    """
    keys_to_delete = []
    update_dict = defaultdict(int)
    ind1, ind2 = merge_pair
    new_pair = merge_pair[0] + merge_pair[1]
    keys_to_change = byte_pair_to_token[(ind1, ind2)].copy()
    for old_key in keys_to_change:
        cnt = token_dict.pop(old_key)
        new_key = get_new_key(old_key, merge_pair)
        token_dict[new_key] += cnt
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i+1]
            byte_count[(left, right)] -= cnt
            byte_pair_to_token[(left, right)].discard(old_key)
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i+1]
            byte_count[(left, right)] += cnt
            byte_pair_to_token[(left, right)].add(new_key)
    byte_pair_to_token[(ind1, ind2)] = set()
    byte_count.pop(merge_pair, None)


def pre_tokenize(input_path, num_processes, special_tokens):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes,
                                           b"<|endoftext|>")
        boundary_points = [(start, end) for (start, end) in zip(boundaries[:-1], boundaries[1:])]
        special_tok_pat = re.compile("|".join([re.escape(tok) for tok in special_tokens])) if special_tokens else None
        chunked_count_arr = []
        pool = Pool(num_processes)
        for point in boundary_points:
            start, end = point
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunked_count_arr.append(pool.apply_async(initialize_count_dict, (chunk, special_tok_pat)))
        pool.close()
        pool.join()

        token_dict = defaultdict(int)
        for chunked_count in chunked_count_arr:
            chunked_count = chunked_count.get()
            for key, value in chunked_count.items():
                token_dict[key] = token_dict.get(key, 0) + value

        return token_dict


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    initial_tokens = [token.encode("UTF-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: tok for i, tok in enumerate(initial_tokens)}
    merges = []
    with timer("Pre tokenisation"):
        token_dict = pre_tokenize(input_path, os.cpu_count() // 2, special_tokens)

    new_idx = len(vocab)

    with timer("Initial Byte Pair Count"):
        byte_pair_count_dict, byte_pair_to_token = create_byte_pair_count(token_dict)

    num_merges = vocab_size - len(vocab)

    for _ in tqdm(range(num_merges)):

        most_frequent_pair = get_most_frequent_pair(byte_pair_count_dict)

        merges.append((most_frequent_pair[0], most_frequent_pair[1]))
        vocab[new_idx] = most_frequent_pair[0] + most_frequent_pair[1]

        merge_optimised(token_dict, most_frequent_pair, byte_pair_count_dict, byte_pair_to_token)

        new_idx += 1

    if os.path.exists("output"):
        with open("output/vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open("output/merges.pkl", "wb") as f:
            pickle.dump(merges, f)
    return (vocab, merges)


def train_bpe_tinystories(split="validation"):
    if split == "validation":
        train_bpe("../tests/fixtures/tinystories_validation.txt", vocab_size=10000, special_tokens=['<|endoftext|>'])
    elif split == "train":
        train_bpe("../tests/fixtures/tinystories_train.txt", vocab_size=10000, special_tokens=['<|endoftext|>'])
    else:
        raise ValueError(f"Invalid split: {split}")


def get_tokenization_stats(tokenizer, list_s):
    """
    Returns average tokenizer compression ratio and tokenizer throughput.

    Args:
        tokenizer: Tokenizer to be tested (must have .encode(str) -> List[int]).
        list_s: List of strings over which stats are averaged.

    Returns:
        compression_ratio (bytes/token), throughput (tokens/sec).
    """
    total_bytes = 0
    total_tokens = 0
    total_time = 0.0

    for s in list_s:
        b = s.encode("utf-8")
        start = time.perf_counter()
        token_ids = tokenizer.encode(s)
        end = time.perf_counter()

        if not token_ids:   # avoid division by zero
            continue

        total_bytes += len(b)
        total_tokens += len(token_ids)
        total_time += (end - start)

    compression_ratio = total_bytes / total_tokens if total_tokens else 0
    throughput = total_tokens / total_time if total_time else 0

    return compression_ratio, throughput


def get_tokenzation_stats1(tokenizer, list_s):
    """This function return average tokenizer compression ratio and tokenizer throughput
    Args:
        tokenizer: Tokenizer to be tested
        list_s: List of strings over which this stats are averaged.
    Returns:
        compression ratio(bytes/token) and tokenizer throughput(tokens/second)
    """
    compression_ratio = []
    for s in list_s:
        # compression ratio = #bytes/#tokens
        token_ids = tokenizer.encode(s)
        elapsed_time = end - start
        compression_ratio.append(len(s.encode("utf-8"))/len(token_ids))
        tokenizer_throughput.append(len(token_ids)/elapsed_time)
    return sum(compression_ratio) / len(compression_ratio), sum(tokenizer_throughput) / len(tokenizer_throughput)

def load_dataset_sample(doc_path, num_documents, delimiter):
    docs = []
    with open(doc_path) as f:
        doc = ""
        for line in f:
            doc += line
            if delimiter in line:
                docs.append(doc)
                if len(docs) >= num_documents:
                    break
    return docs

def load_tokenizer(vocab_path):
    with open(os.path.join(vocab_path, "merges.pkl"), "rb") as f:
        merges = pickle.load(f)
    with open(os.path.join(vocab_path, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    return Tokenizer(vocab, merges, ["<|endoftext|>"])

def measure_compression_ratio(doc_path, vocab_path, num_documents):
    tokenizer = load_tokenizer(vocab_path)
    docs = load_dataset_sample(doc_path, num_documents, "<|endoftext|>")
    comp_ratio, tok_throughput = get_tokenization_stats(tokenizer, docs)
    print(f"Compression ratio = {comp_ratio:0.2f}, Tokenizaton throughput = {tok_throughput:0.2f}")



if __name__ == "__main__":
    measure_compression_ratio("../tests/fixtures/tinystories_train.txt",
        "../tests/fixtures/tokenizers/tinystories/", 10)

    # train_bpe("../tests/fixtures/openwebtext.txt", vocab_size=32000, special_tokens=['<|endoftext|>'])