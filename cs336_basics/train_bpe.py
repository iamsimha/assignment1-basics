import os
import regex as re
import multiprocessing
from typing import IO, Any, BinaryIO
from collections import defaultdict
from multiprocessing import Pool

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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

def intialize_count_dict(chunk, special_token_pattern):
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
    for key, cnt in count_dict.items():
        for ind1, ind2 in zip(key, key[1:]):
            byte_pair_count_dict[(ind1, ind2)] += cnt
    return byte_pair_count_dict

def get_most_frequent_pair(count_dict):
    max_count = max(count_dict.values())
    max_val = [k for k, v in count_dict.items() if v == max_count]
    
    return max(
        (k for k, v in count_dict.items() if v == max_count),
        key=lambda x: x  # lexicographically larger wins on tie
    ), max_count


def merge_optimised(token_dict, merge_pair, new_idx, byte_count, vocab):
    keys_to_delete = []
    update_dict = defaultdict(int)
    ind1, ind2 = merge_pair
    new_pair = merge_pair[0] + merge_pair[1]
    for key, cnt in token_dict.items():
        modified = False
        i = 0
        n = len(key)
        new_key = []
        while i < n:
            if i < n - 1 and key[i] == ind1 and key[i+1] == ind2:
                new_key.append(new_pair)
                left_neigh = key[i - 1] if i > 0 else None
                right_neigh = key[i + 2] if i < n - 2 else None
                byte_count[(ind1, ind2)] -= cnt
                if left_neigh is not None:
                    byte_count[(left_neigh, key[i])] -= cnt
                    byte_count[(left_neigh, new_pair)] += cnt
                if right_neigh is not None:
                    byte_count[(key[i+1], right_neigh)] -= cnt
                    byte_count[(new_pair, right_neigh)] += cnt
                i += 2
                modified = True
            else:
                new_key.append(key[i])
                i += 1
        
        if modified:
           update_dict[tuple(new_key)] += cnt
           keys_to_delete.append(key)
    
    for k in keys_to_delete:
        token_dict.pop(k, None)
    # del byte_count[(ind1, ind2)]
    token_dict.update(update_dict)


def get_token_counts(input_path, num_processes, special_tokens):
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
            chunked_count_arr.append(pool.apply_async(intialize_count_dict, (chunk, special_tok_pat)))
        pool.close()
        pool.join()

        token_dict = {}
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
    token_dict = get_token_counts(input_path, 4, special_tokens)
    new_idx = len(vocab)

    byte_pair_count_dict = create_byte_pair_count(token_dict)

    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        most_frequent_pair, cnt = get_most_frequent_pair(byte_pair_count_dict)
        merges.append((most_frequent_pair[0], most_frequent_pair[1]))
        vocab[new_idx] = most_frequent_pair[0] + most_frequent_pair[1]
        merge_optimised(token_dict, most_frequent_pair, new_idx, byte_pair_count_dict, vocab)
        new_idx += 1
    return (vocab, merges)

if __name__ == "__main__":
    print(
        
        train_bpe("../tests/fixtures/simple.txt", vocab_size=270, special_tokens=['<|endoftext|>'])

    )