import regex as re
from collections import defaultdict, deque
from itertools import count

PRE_TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pairs(tokens: dict[tuple[bytes, ...], int]):
    pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for t, freq in tokens.items():
        for id1, id2 in zip(t, t[1:]):
            pairs[(id1, id2)] += freq
    return pairs


def merge(words: dict[tuple[bytes, ...], int], best: tuple[bytes, bytes]):
    new_tokens = defaultdict(int)
    for word, freq in words.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                new_word.append(best[0] + best[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_tokens[tuple(new_word)] += freq
    return new_tokens


def total_seq_len(tokens: dict[tuple[int, ...], int]):
    return sum([len(w) * freq for w, freq in tokens.items()])


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        data = f.read()
    corpus = re.split("|".join(map(re.escape, special_tokens)), data)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    tokens: dict[tuple[bytes, ...], int] = defaultdict(int)  # list[bytes] -> freq

    for context in corpus:
        # validate
        for t in special_tokens:
            assert t not in context

        for word in re.finditer(PRE_TOKENIZE_PATTERN, context):
            word = word.group().encode("utf-8")
            tokens[tuple(bytes([b]) for b in word)] += 1

    # print(total_seq_len(tokens))
    n_merges = vocab_size - len(vocab) - len(special_tokens)
    for i in range(n_merges):
        pairs = get_pairs(tokens)
        best = max(pairs, key=lambda x: (pairs.get(x), x))
        # print(f"best pair={best}, hits={pairs[best]}")
        vocab[len(vocab)] = best[0] + best[1]
        merges.append(best)
        tokens = merge(tokens, best)
        # print(total_seq_len(tokens))

    for special in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = special.encode('utf-8')

    return vocab, merges


if __name__ == "__main__":
    # train_bpe("./data/TinyStoriesV2-GPT4-train.txt", vocab_size=256 + 1 + 30, special_tokens=["<|endoftext|>"])
    train_bpe("./tests/fixtures/corpus.en", vocab_size=256 + 1 + 218, special_tokens=["<|endoftext|>"])
