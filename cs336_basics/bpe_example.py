corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest aaaa
"""


vocab = {bytes([i]): i + 1 for i in range(256)}
vocab[b"<|endoftext|>"] = 0


def pre_tokenize(corpus: str) -> dict[tuple[bytes, ...], int]:
    tokens = [x.encode("utf-8") for x in corpus.split()]
    counts = {}
    for t in tokens:
        t = tuple(bytes([b]) for b in t)
        if t not in counts:
            counts[t] = 0
        counts[t] += 1
    return counts


corpus_counts = pre_tokenize(corpus)


def count_pairs(corpus_counts: dict[tuple[bytes, ...], int]):
    pairs = {}
    for tokens, freq in corpus_counts.items():
        for i in range(len(tokens) - 1):
            p = (tokens[i], tokens[i + 1])
            if p not in pairs:
                pairs[p] = 0
            pairs[p] += freq
    return pairs


def merge(corpus_counts: dict[tuple[bytes, ...], int], best: tuple[bytes, bytes]):
    new_corpus_counts = {}
    for tokens, freq in corpus_counts.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i == len(tokens) - 1:
                new_tokens.append(tokens[i])
                i += 1
                continue
            t = (tokens[i], tokens[i + 1])
            if t == best:
                new_tokens.append(t[0] + t[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_corpus_counts[tuple(new_tokens)] = freq
    return new_corpus_counts


for _ in range(14):
    print("--------")
    pairs = count_pairs(corpus_counts)
    if not pairs:
        print("pairs is empty, nothing to merge, exit")
        break
    best = max(pairs, key=lambda k: (pairs.get(k), k))
    corpus_counts = merge(corpus_counts, best)
    vocab[best[0] + best[1]] = len(vocab)

    _counts = {b",".join(k): v for k, v in corpus_counts.items()}
    print(f"best={best}, freq={pairs[best]} {_counts=}")

# print(vocab)
