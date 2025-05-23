import os
import re
from collections import Counter

import regex

_REGEX_MODULE_TO_USE = regex
GPT2_SPLIT_PATTERN_COMPILED = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def train_bpe_tokenizer(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    train a BPE tokenizer on the input text corpus.
    :param input_path:
    :param vocab_size:
    :param special_tokens:
    :param kwargs:
    :return: vocab, merges
    """

    # --- 0. Initial setups ---
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    current_vocab_byte_values = set(vocab.values())

    # Transform special tokens to bytes and add them to vocab
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes in vocab.values():
            pass
        vocab[next_token_id] = token_bytes
        current_vocab_byte_values.add(token_bytes)
        next_token_id += 1

    # 1. Chunking the data based on special tokens
    with open(input_path, "r", encoding="utf-8") as f:
        text_corpus = f.read()

    # Create regex pattern for special tokens
    corpus_chunk = None
    if not special_tokens:
        corpus_chunk = [text_corpus]
    else:
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = "|".join(escaped_special_tokens)
        text_chunks = re.split(f"({split_pattern})", text_corpus)

        corpus_chunk = []
        temp_chunk = ""
        for part in text_chunks:
            if part in special_tokens:
                if temp_chunk:
                    corpus_chunk.append(temp_chunk)
                    temp_chunk = ""
                corpus_chunk.append(part)
            else:
                temp_chunk += part
        if temp_chunk:
            corpus_chunk.append(temp_chunk)

    # --- 2. Pre-tokenization ---
    bpe_input_word_bytes_list: list[bytes] = []

    for text_chunk in corpus_chunk:
        if not text_chunk:
            continue

        pre_tokens_as_string = GPT2_SPLIT_PATTERN_COMPILED.findall(text_chunk)
        for pre_token_str in pre_tokens_as_string:
            bpe_input_word_bytes_list.append(pre_token_str.encode("utf-8"))

    # --- 3. BPE Tokenizer Training ---
    merges: list[tuple[bytes, bytes]] = []

    # Transform the byte sequences into a list of lists of bytes
    current_word_token_sequences: list[list[bytes]] = []
    for word_bytes in bpe_input_word_bytes_list:
        token_sequence_for_word = [bytes([b]) for b in word_bytes]
        current_word_token_sequences.append(token_sequence_for_word)

    num_merges_needed = vocab_size - len(vocab)
    num_merges_needed = 0 if num_merges_needed < 0 else num_merges_needed

    for i in range(num_merges_needed):
        pair_stats = _get_pair_stats(current_word_token_sequences)
        if not pair_stats:
            break

        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        merges.append(best_pair)
        new_token_bytes = best_pair[0] + best_pair[1]

        # Add the new token to the vocab if it doesn't already exist
        if new_token_bytes not in current_vocab_byte_values:
            vocab[next_token_id] = new_token_bytes
            current_vocab_byte_values.add(new_token_bytes)
            next_token_id += 1

        # Merge the best pair in the current word token sequences
        current_word_token_sequences = _merge_pair_in_sequences(
            current_word_token_sequences, best_pair, new_token_bytes
        )

        if len(vocab) > vocab_size:
            break

    return vocab, merges


def _get_pair_stats(
        word_token_sequences: list[list[bytes]],
) -> Counter[tuple[bytes, bytes]]:
    """
    Calculate the frequency of each pair of adjacent tokens in the word token sequences.
    :param word_token_sequences: list of lists of bytes
    :return: Counter of pairs of bytes
    """
    pair_counts = Counter()
    for token_sequence in word_token_sequences:
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i], token_sequence[i + 1])
            pair_counts[pair] += 1
    return pair_counts


def _merge_pair_in_sequences(
        word_token_sequences: list[list[bytes]],
        pair_to_merge: tuple[bytes, bytes],
        new_token: bytes
) -> list[list[bytes]]:
    """
    Merge the specified pair of tokens in the word token sequences.
    :param word_token_sequences: list of lists of bytes
    :param pair_to_merge: tuple of bytes to merge
    :param new_token: bytes representing the new token
    :return: list of lists of bytes with the merged tokens
    """
    new_word_token_sequences = []
    for token_sequence in word_token_sequences:
        new_sequence = []
        i = 0
        while i < len(token_sequence):
            if i < len(token_sequence) - 1 and \
                    token_sequence[i] == pair_to_merge[0] and \
                    token_sequence[i + 1] == pair_to_merge[1]:
                new_sequence.append(new_token)
                i += 2  # Skip the next token as it is merged
            else:
                new_sequence.append(token_sequence[i])
                i += 1
        new_word_token_sequences.append(new_sequence)
    return new_word_token_sequences
