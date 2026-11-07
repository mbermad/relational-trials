"""Text embedders for PyTorch Frame's text columns.

* ``glove``: the 300-d averaged GloVe model RelBench's reference GNN uses (needs
  ``sentence-transformers`` and an internet connection the first time).
* ``hash``: a dependency-free hashed bag-of-words, for offline tests and low-RAM runs.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


def _clean(sentences: List) -> List[str]:
    out = []
    for s in sentences:
        if s is None or (isinstance(s, float) and s != s):
            out.append("")
        else:
            out.append(str(s))
    return out


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(_clean(sentences), convert_to_tensor=True).cpu()


class HashTextEmbedding:
    def __init__(self, dim: int = 64):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def __call__(self, sentences: List[str]) -> Tensor:
        X = self.vec.transform(_clean(sentences))
        return torch.tensor(X.toarray(), dtype=torch.float32)


def make_text_embedder(kind: str, device: Optional[str] = None):
    if kind == "glove":
        return GloveTextEmbedding(device=device)
    if kind == "hash":
        return HashTextEmbedding()
    raise ValueError(f"unknown text embedder {kind!r} (use 'glove' or 'hash')")
