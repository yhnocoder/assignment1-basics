from functools import partial

import regex as re
from dataclasses import dataclass, field
from cs336_basics.pretokenization import find_chunk_boundaries
from multiprocessing import Pool


PRE_TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
num_processes = 8
strict = True

Token = bytes
Word = bytes
Pair = tuple[Token, Token]


@dataclass(slots=True)
class TokenItem:
    freq: int
    tokens: list[Token]


@dataclass(slots=True)
class PairItem:
    freq: int = 0
    words: set[Word] = field(default_factory=set)


WordToTokens = dict[Word, TokenItem]
Pairs = dict[Pair, PairItem]


def pretokenize(boundary: tuple[int, int], file: str, special_tokens: list[str]):
    start, end = boundary
    print(f"boundary: {start=}, {end=}")
    with open(file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    corpus = re.split("|".join(map(re.escape, special_tokens)), chunk)
    pairs = initial_tokens_and_pairs(corpus)
    return pairs


def count_sub_freq_of_w(word: bytes, sub: bytes):
    return sum(word.startswith(sub, i) for i in range(len(word) - len(sub) + 1))


def initial_tokens_and_pairs(corpus: list[str]):
    wts: WordToTokens = {}
    for context in corpus:
        for w in re.finditer(PRE_TOKENIZE_PATTERN, context):
            w = w.group().encode("utf-8")
            if w in wts:
                wts[w].freq += 1
            else:
                ts = [bytes([b]) for b in w]
                wts[w] = TokenItem(1, ts)

    pairs: Pairs = {}
    for w, t in wts.items():
        for i in range(len(w) - 1):
            p = (t.tokens[i], t.tokens[i + 1])
            if p in pairs:
                pairs[p].freq += t.freq
                pairs[p].words.add(w)
            else:
                pairs[p] = PairItem(t.freq, set([w]))

    if strict:
        for p, item in pairs.items():
            freq = sum([wts[w].freq * count_sub_freq_of_w(w, p[0] + p[1]) for w in item.words])
            assert freq == item.freq, f"mismatch freq: {p=}, {freq=}, {item.freq=}"

    return wts, pairs


def combine_tokens_and_pairs(parts: list[tuple[WordToTokens, Pairs]]):
    wts: WordToTokens = {}
    pairs: Pairs = {}
    for _wts, _pairs in parts:
        for w in _wts:
            if w in wts:
                wts[w].freq += _wts[w].freq
                assert wts[w].tokens == _wts[w].tokens, f"{w=}, {wts[w]=}, {_wts[w]=}"
            else:
                wts[w] = _wts[w]

        for p in _pairs:
            if p in pairs:
                pairs[p].freq += _pairs[p].freq
                pairs[p].words = pairs[p].words | _pairs[p].words
            else:
                pairs[p] = _pairs[p]
    return wts, pairs


def merge(wts: WordToTokens, pairs: Pairs, best: Pair):
    best_item = pairs.pop(best)
    tb = best[0] + best[1]
    to_add: Pairs = {}
    to_del = set()

    for raw_word in best_item.words:
        token_item = wts[raw_word]
        tokens = token_item.tokens
        seen_ids = set()

        # remove stage
        i = 0
        while i < len(tokens) - 1:
            t1, t2 = tokens[i], tokens[i + 1]
            if t1 == best[0] and t2 == best[1]:
                if i > 0 and (i - 1) not in seen_ids:
                    t0 = tokens[i - 1]
                    p = (t0, t1)
                    seen_ids.add(i - 1)
                    if p != best:
                        pairs[p].freq -= token_item.freq
                        if pairs[p].freq == 0:
                            to_del.add(p)
                if i < len(tokens) - 2:  # 不需要加 seen_ids 检查，因为我们是从左往右遍历的
                    t3 = tokens[i + 2]
                    p = (t2, t3)
                    seen_ids.add(i + 1)
                    if p != best:
                        pairs[p].freq -= token_item.freq
                        if pairs[p].freq == 0:
                            to_del.add(p)
                i += 2
            else:
                i += 1

        # merge stage
        i = 0
        while i < len(tokens) - 1:
            t1, t2 = tokens[i], tokens[i + 1]
            if t1 == best[0] and t2 == best[1]:
                tokens[i : i + 2] = [tb]
            i += 1

        # add stage
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            p = (t1, t2)
            if p not in pairs:
                if p not in to_add:
                    to_add[p] = PairItem(token_item.freq, set([raw_word]))
                else:
                    to_add[p].freq += token_item.freq
                    to_add[p].words.add(raw_word)

    for p, v in to_add.items():
        pairs[p] = v
    for p in to_del:
        del pairs[p]

    return wts, pairs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    if num_processes == 1:
        wts, pairs = pretokenize(boundaries, file=input_path, special_tokens=special_tokens)
    else:
        with Pool(num_processes) as pool:
            _pretokenize = partial(pretokenize, file=input_path, special_tokens=special_tokens)
            parts = pool.map(_pretokenize, list(zip(boundaries, boundaries[1:])))
        wts, pairs = combine_tokens_and_pairs(parts)

    vocab: dict[int, Token] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []

    n_vocab = len(vocab)
    n_merges = vocab_size - n_vocab - len(special_tokens)
    print(f"Start to do merging with {vocab_size=}, {n_merges=}...")
    for i in range(n_merges):
        if i % 100 == 0 and strict:
            print(f"Working on {i + 1}/{n_merges} ...")
        best = max(pairs, key=lambda x: (pairs.get(x).freq, x))
        if i >= 0 and i < 42 and strict:
            print(f"{i=} best pair={best}, hits={pairs[best].freq}")
        vocab[n_vocab] = best[0] + best[1]
        n_vocab += 1
        merges.append(best)
        wts, pairs = merge(wts, pairs, best)

    for special in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = special.encode("utf-8")

    return vocab, merges


if __name__ == "__main__":
    train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", vocab_size=256 + 1 + 1500, special_tokens=["<|endoftext|>"])
    # train_bpe("./data/TinyStoriesV2-GPT4-train.txt", vocab_size=256 + 1 + 10000, special_tokens=["<|endoftext|>"])
    # train_bpe("./tests/fixtures/corpus.en", vocab_size=256 + 1 + 1000, special_tokens=["<|endoftext|>"])
    # train_bpe("./tests/fixtures/example.txt", vocab_size=256 + 1 + 10, special_tokens=["<|endoftext|>"])
