from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import zlib

import numpy as np

from refmark_train.synthetic import Example, tokenize


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    bigram_hash_buckets: int = 0
    subword_hash_buckets: int = 0
    subword_min_n: int = 3
    subword_max_n: int = 4

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_count: int = 1,
        *,
        bigram_hash_buckets: int = 256,
        subword_hash_buckets: int = 256,
        subword_min_n: int = 3,
        subword_max_n: int = 4,
    ) -> "Vocab":
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        id_to_token = ["<pad>", "<unk>"]
        for token in sorted(counts):
            if counts[token] >= min_count:
                id_to_token.append(token)
        id_to_token.extend(f"<bg:{idx}>" for idx in range(bigram_hash_buckets))
        id_to_token.extend(f"<sw:{idx}>" for idx in range(subword_hash_buckets))
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            bigram_hash_buckets=bigram_hash_buckets,
            subword_hash_buckets=subword_hash_buckets,
            subword_min_n=subword_min_n,
            subword_max_n=subword_max_n,
        )

    @staticmethod
    def _stable_bucket(text: str, buckets: int) -> int:
        if buckets <= 0:
            return 0
        return zlib.crc32(text.encode("utf-8")) % buckets

    def encode(self, text: str) -> list[int]:
        tokens = tokenize(text)
        unk = self.token_to_id["<unk>"]
        encoded = [self.token_to_id.get(token, unk) for token in tokens]
        if self.bigram_hash_buckets > 0:
            for left, right in zip(tokens, tokens[1:], strict=False):
                bucket = self._stable_bucket(f"{left}|{right}", self.bigram_hash_buckets)
                encoded.append(self.token_to_id[f"<bg:{bucket}>"])
        if self.subword_hash_buckets > 0:
            for token in tokens:
                wrapped = f"^{token}$"
                max_n = min(self.subword_max_n, len(wrapped))
                for n in range(self.subword_min_n, max_n + 1):
                    for start in range(0, len(wrapped) - n + 1):
                        piece = wrapped[start : start + n]
                        bucket = self._stable_bucket(piece, self.subword_hash_buckets)
                        encoded.append(self.token_to_id[f"<sw:{bucket}>"])
        return encoded


def build_label_index(examples: Iterable[Example]) -> dict[str, int]:
    labels = sorted({example.refmark for example in examples})
    return {label: idx for idx, label in enumerate(labels)}


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(logits, axis=1)[:, -k:]
    hits = np.any(topk == labels[:, None], axis=1)
    return float(hits.mean())


class KeywordOverlapBaseline:
    """A tiny non-neural baseline that scores label-specific token overlap."""

    def __init__(self) -> None:
        self.label_token_scores: dict[int, dict[int, float]] = {}

    def fit(self, encoded_texts: list[list[int]], labels: np.ndarray) -> None:
        per_label_counts: dict[int, dict[int, int]] = {}
        for token_ids, label in zip(encoded_texts, labels, strict=True):
            label_counts = per_label_counts.setdefault(int(label), {})
            for token_id in token_ids:
                label_counts[token_id] = label_counts.get(token_id, 0) + 1
        self.label_token_scores = {}
        for label, counts in per_label_counts.items():
            total = sum(counts.values()) or 1
            self.label_token_scores[label] = {
                token_id: count / total for token_id, count in counts.items()
            }

    def logits(self, encoded_texts: list[list[int]], num_labels: int) -> np.ndarray:
        logits = np.zeros((len(encoded_texts), num_labels), dtype=np.float32)
        for row, token_ids in enumerate(encoded_texts):
            token_set = set(token_ids)
            for label in range(num_labels):
                score_map = self.label_token_scores.get(label, {})
                logits[row, label] = float(sum(score_map.get(token_id, 0.0) for token_id in token_set))
        return logits


class TfidfAnchorRetriever:
    """A simple TF-IDF retriever over anchor texts."""

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.idf: np.ndarray | None = None
        self.anchor_matrix: np.ndarray | None = None
        self.anchor_refmarks: list[str] = []

    @staticmethod
    def _counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        x = np.zeros((len(encoded_texts), vocab_size), dtype=np.float32)
        for row, token_ids in enumerate(encoded_texts):
            for token_id in token_ids:
                x[row, token_id] += 1.0
        return x

    def fit(self, anchor_texts: list[str], anchor_refmarks: list[str]) -> None:
        encoded = [self.vocab.encode(text) for text in anchor_texts]
        counts = self._counts_matrix(encoded, len(self.vocab.id_to_token))
        doc_freq = (counts > 0).sum(axis=0)
        num_docs = max(len(anchor_texts), 1)
        self.idf = np.log((1.0 + num_docs) / (1.0 + doc_freq)) + 1.0
        weighted = counts * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.anchor_matrix = weighted / norms
        self.anchor_refmarks = list(anchor_refmarks)

    def logits(self, query_texts: list[str], label_index: dict[str, int]) -> np.ndarray:
        if self.idf is None or self.anchor_matrix is None:
            raise RuntimeError("retriever must be fit before scoring")
        encoded = [self.vocab.encode(text) for text in query_texts]
        counts = self._counts_matrix(encoded, len(self.vocab.id_to_token))
        weighted = counts * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        query_matrix = weighted / norms
        scores = query_matrix @ self.anchor_matrix.T
        logits = np.full((len(query_texts), len(label_index)), -1e9, dtype=np.float32)
        for anchor_idx, refmark in enumerate(self.anchor_refmarks):
            label = label_index.get(refmark)
            if label is not None:
                logits[:, label] = scores[:, anchor_idx]
        return logits


class BM25AnchorRetriever:
    """A small Okapi BM25 retriever over anchor texts."""

    def __init__(self, vocab: Vocab, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.vocab = vocab
        self.k1 = k1
        self.b = b
        self.idf: np.ndarray | None = None
        self.anchor_counts: np.ndarray | None = None
        self.anchor_lengths: np.ndarray | None = None
        self.avg_anchor_length: float = 1.0
        self.anchor_refmarks: list[str] = []

    @staticmethod
    def _counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        return TfidfAnchorRetriever._counts_matrix(encoded_texts, vocab_size)

    def fit(self, anchor_texts: list[str], anchor_refmarks: list[str]) -> None:
        encoded = [self.vocab.encode(text) for text in anchor_texts]
        counts = self._counts_matrix(encoded, len(self.vocab.id_to_token))
        doc_freq = (counts > 0).sum(axis=0)
        num_docs = max(len(anchor_texts), 1)
        self.idf = np.log(((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0).astype(np.float32)
        self.anchor_counts = counts
        self.anchor_lengths = counts.sum(axis=1).astype(np.float32)
        self.avg_anchor_length = float(self.anchor_lengths.mean()) if len(self.anchor_lengths) else 1.0
        if self.avg_anchor_length <= 0.0:
            self.avg_anchor_length = 1.0
        self.anchor_refmarks = list(anchor_refmarks)

    def logits(self, query_texts: list[str], label_index: dict[str, int]) -> np.ndarray:
        if self.idf is None or self.anchor_counts is None or self.anchor_lengths is None:
            raise RuntimeError("retriever must be fit before scoring")
        encoded_queries = [self.vocab.encode(text) for text in query_texts]
        scores = np.zeros((len(query_texts), len(self.anchor_refmarks)), dtype=np.float32)
        for row, token_ids in enumerate(encoded_queries):
            if not token_ids:
                continue
            token_ids = list(set(token_ids))
            for token_id in token_ids:
                tf = self.anchor_counts[:, token_id]
                if np.all(tf == 0.0):
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * (self.anchor_lengths / self.avg_anchor_length))
                numer = tf * (self.k1 + 1.0)
                denom = tf + norm
                scores[row] += self.idf[token_id] * (numer / np.maximum(denom, 1e-6))
        logits = np.full((len(query_texts), len(label_index)), -1e9, dtype=np.float32)
        for anchor_idx, refmark in enumerate(self.anchor_refmarks):
            label = label_index.get(refmark)
            if label is not None:
                logits[:, label] = scores[:, anchor_idx]
        return logits


class TinyBoWClassifier:
    """A tiny trainable softmax classifier over bag-of-words counts."""

    def __init__(self, vocab_size: int, num_labels: int, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, (vocab_size, num_labels)).astype(np.float32)
        self.b = np.zeros(num_labels, dtype=np.float32)

    @staticmethod
    def _counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        x = np.zeros((len(encoded_texts), vocab_size), dtype=np.float32)
        for row, token_ids in enumerate(encoded_texts):
            for token_id in token_ids:
                x[row, token_id] += 1.0
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return x / row_sums

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def train(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        indices = np.arange(len(x))

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x[batch_idx]
                yb = labels[batch_idx]
                logits = xb @ self.w + self.b
                logits = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-9).mean()
                epoch_loss += float(loss)
                steps += 1

                grad_logits = probs
                grad_logits[np.arange(len(yb)), yb] -= 1.0
                grad_logits /= len(yb)
                grad_w = xb.T @ grad_logits + weight_decay * self.w
                grad_b = grad_logits.sum(axis=0)

                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b

            losses.append(epoch_loss / max(steps, 1))
        return losses


class TinyEmbeddingClassifier:
    """Embedding bag + tanh hidden layer + linear classifier trained in NumPy."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 24,
        hidden_dim: int = 48,
        seed: int = 13,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.embedding = rng.normal(0.0, scale, (vocab_size, embedding_dim)).astype(np.float32)
        self.w1 = rng.normal(0.0, scale, (embedding_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale, (hidden_dim, num_labels)).astype(np.float32)
        self.b2 = np.zeros(num_labels, dtype=np.float32)

    def _embed_batch(self, batch: list[list[int]]) -> np.ndarray:
        output = np.zeros((len(batch), self.embedding.shape[1]), dtype=np.float32)
        for row, token_ids in enumerate(batch):
            if not token_ids:
                continue
            output[row] = self.embedding[token_ids].mean(axis=0)
        return output

    def logits(self, batch: list[list[int]]) -> np.ndarray:
        embedded = self._embed_batch(batch)
        hidden = np.tanh(embedded @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def train(
        self,
        encoded_texts: list[list[int]],
        labels: np.ndarray,
        *,
        epochs: int = 60,
        learning_rate: float = 0.25,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        indices = np.arange(len(encoded_texts))

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = [encoded_texts[int(i)] for i in batch_idx]
                y = labels[batch_idx]

                embedded = self._embed_batch(batch)
                hidden_pre = embedded @ self.w1 + self.b1
                hidden = np.tanh(hidden_pre)
                logits = hidden @ self.w2 + self.b2

                logits = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                loss = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()
                epoch_loss += float(loss)
                steps += 1

                grad_logits = probs
                grad_logits[np.arange(len(y)), y] -= 1.0
                grad_logits /= len(y)

                grad_w2 = hidden.T @ grad_logits + weight_decay * self.w2
                grad_b2 = grad_logits.sum(axis=0)

                grad_hidden = grad_logits @ self.w2.T
                grad_hidden_pre = grad_hidden * (1.0 - hidden * hidden)

                grad_w1 = embedded.T @ grad_hidden_pre + weight_decay * self.w1
                grad_b1 = grad_hidden_pre.sum(axis=0)
                grad_embedded = grad_hidden_pre @ self.w1.T

                grad_embedding = np.zeros_like(self.embedding)
                for row, token_ids in enumerate(batch):
                    if not token_ids:
                        continue
                    share = grad_embedded[row] / len(token_ids)
                    for token_id in token_ids:
                        grad_embedding[token_id] += share
                grad_embedding += weight_decay * self.embedding

                self.w2 -= learning_rate * grad_w2
                self.b2 -= learning_rate * grad_b2
                self.w1 -= learning_rate * grad_w1
                self.b1 -= learning_rate * grad_b1
                self.embedding -= learning_rate * grad_embedding

            losses.append(epoch_loss / max(steps, 1))
        return losses


class TorchBoWClassifier:
    """PyTorch version of the tiny bag-of-words classifier for optional DirectML use."""

    def __init__(self, vocab_size: int, num_labels: int, device: str) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.linear = torch.nn.Linear(vocab_size, num_labels).to(self.device)

    @staticmethod
    def counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        return TinyBoWClassifier._counts_matrix(encoded_texts, vocab_size)

    def logits(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            xb = self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
            return self.linear(xb).detach().cpu().numpy()

    def train(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = torch.optim.AdamW(
            self.linear.parameters(),
            lr=learning_rate * 0.1,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                yb = y_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                logits = self.linear(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses


class TorchBoWMLPClassifier:
    """A stronger bag-of-words MLP classifier for real-corpus direct refmark prediction."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        device: str,
        *,
        hidden_dim: int = 512,
        hidden2_dim: int = 0,
        dropout: float = 0.2,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(vocab_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        if hidden2_dim > 0:
            layers.extend(
                [
                    torch.nn.Linear(hidden_dim, hidden2_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden2_dim, num_labels),
                ]
            )
        else:
            layers.append(torch.nn.Linear(hidden_dim, num_labels))
        self.network = torch.nn.Sequential(*layers).to(self.device)

    @staticmethod
    def counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        return TinyBoWClassifier._counts_matrix(encoded_texts, vocab_size)

    def logits(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            xb = self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
            self.network.eval()
            return self.network(xb).detach().cpu().numpy()

    def train(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.network.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                yb = y_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses

    def train_soft(
        self,
        x: np.ndarray,
        soft_targets: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(soft_targets, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.network.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                tb = target_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(xb)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = -(tb * log_probs).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses

    def train_hybrid(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        soft_targets: np.ndarray,
        *,
        exact_weight: float = 0.7,
        soft_weight: float = 0.3,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(soft_targets, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.network.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                yb = y_tensor[batch_idx]
                tb = target_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(xb)
                exact_loss = torch.nn.functional.cross_entropy(logits, yb)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                soft_loss = -(tb * log_probs).sum(dim=1).mean()
                loss = exact_weight * exact_loss + soft_weight * soft_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses


class TorchBoWStartEndPredictor:
    """A deterministic start/end anchor predictor over anchor indices."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        device: str,
        *,
        hidden_dim: int = 512,
        hidden2_dim: int = 0,
        dropout: float = 0.2,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(vocab_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        if hidden2_dim > 0:
            layers.extend(
                [
                    torch.nn.Linear(hidden_dim, hidden2_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            trunk_out = hidden2_dim
        else:
            trunk_out = hidden_dim
        self.trunk = torch.nn.Sequential(*layers).to(self.device)
        self.start_head = torch.nn.Linear(trunk_out, num_labels).to(self.device)
        self.end_head = torch.nn.Linear(trunk_out, num_labels).to(self.device)

    @staticmethod
    def counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        return TinyBoWClassifier._counts_matrix(encoded_texts, vocab_size)

    def logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            xb = self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
            self.trunk.eval()
            hidden = self.trunk(xb)
            start_logits = self.start_head(hidden).detach().cpu().numpy()
            end_logits = self.end_head(hidden).detach().cpu().numpy()
            return start_logits, end_logits

    def train(
        self,
        x: np.ndarray,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        start_tensor = torch.tensor(start_labels, dtype=torch.long, device=self.device)
        end_tensor = torch.tensor(end_labels, dtype=torch.long, device=self.device)
        params = list(self.trunk.parameters()) + list(self.start_head.parameters()) + list(self.end_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.trunk.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                sb = start_tensor[batch_idx]
                eb = end_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                hidden = self.trunk(xb)
                start_logits = self.start_head(hidden)
                end_logits = self.end_head(hidden)
                start_loss = torch.nn.functional.cross_entropy(start_logits, sb)
                end_loss = torch.nn.functional.cross_entropy(end_logits, eb)
                loss = 0.5 * (start_loss + end_loss)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses

    def train_soft_ranges(
        self,
        x: np.ndarray,
        start_targets: np.ndarray,
        end_targets: np.ndarray,
        *,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        start_tensor = torch.tensor(start_targets, dtype=torch.float32, device=self.device)
        end_tensor = torch.tensor(end_targets, dtype=torch.float32, device=self.device)
        params = list(self.trunk.parameters()) + list(self.start_head.parameters()) + list(self.end_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.trunk.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                sb = start_tensor[batch_idx]
                eb = end_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                hidden = self.trunk(xb)
                start_logits = self.start_head(hidden)
                end_logits = self.end_head(hidden)
                start_log_probs = torch.nn.functional.log_softmax(start_logits, dim=1)
                end_log_probs = torch.nn.functional.log_softmax(end_logits, dim=1)
                start_loss = -(sb * start_log_probs).sum(dim=1).mean()
                end_loss = -(eb * end_log_probs).sum(dim=1).mean()
                loss = 0.5 * (start_loss + end_loss)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses

    def train_hybrid_ranges(
        self,
        x: np.ndarray,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
        start_targets: np.ndarray,
        end_targets: np.ndarray,
        *,
        exact_weight: float = 0.7,
        soft_weight: float = 0.2,
        width_weight: float = 0.1,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        start_label_tensor = torch.tensor(start_labels, dtype=torch.long, device=self.device)
        end_label_tensor = torch.tensor(end_labels, dtype=torch.long, device=self.device)
        start_target_tensor = torch.tensor(start_targets, dtype=torch.float32, device=self.device)
        end_target_tensor = torch.tensor(end_targets, dtype=torch.float32, device=self.device)
        params = list(self.trunk.parameters()) + list(self.start_head.parameters()) + list(self.end_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)
        label_positions = torch.arange(start_targets.shape[1], dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.trunk.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                sl = start_label_tensor[batch_idx]
                el = end_label_tensor[batch_idx]
                st = start_target_tensor[batch_idx]
                et = end_target_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                hidden = self.trunk(xb)
                start_logits = self.start_head(hidden)
                end_logits = self.end_head(hidden)

                exact_start = torch.nn.functional.cross_entropy(start_logits, sl)
                exact_end = torch.nn.functional.cross_entropy(end_logits, el)
                exact_loss = 0.5 * (exact_start + exact_end)

                start_log_probs = torch.nn.functional.log_softmax(start_logits, dim=1)
                end_log_probs = torch.nn.functional.log_softmax(end_logits, dim=1)
                soft_start = -(st * start_log_probs).sum(dim=1).mean()
                soft_end = -(et * end_log_probs).sum(dim=1).mean()
                soft_loss = 0.5 * (soft_start + soft_end)

                start_probs = torch.nn.functional.softmax(start_logits, dim=1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=1)
                expected_start = (start_probs * label_positions).sum(dim=1)
                expected_end = (end_probs * label_positions).sum(dim=1)
                width = torch.relu(expected_end - expected_start)
                width_loss = (width ** 2).mean()

                loss = exact_weight * exact_loss + soft_weight * soft_loss + width_weight * width_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses


class TorchBoWCenterWidthPredictor:
    """A deterministic center/width anchor predictor over anchor indices."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        width_bins: list[int],
        device: str,
        *,
        hidden_dim: int = 512,
        hidden2_dim: int = 0,
        dropout: float = 0.2,
        count_cap: int = 8,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.width_bins = [max(int(width), 1) for width in width_bins]
        self.count_cap = max(int(count_cap), 1)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(vocab_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        if hidden2_dim > 0:
            layers.extend(
                [
                    torch.nn.Linear(hidden_dim, hidden2_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            trunk_out = hidden2_dim
        else:
            trunk_out = hidden_dim
        self.trunk = torch.nn.Sequential(*layers).to(self.device)
        self.center_head = torch.nn.Linear(trunk_out, num_labels).to(self.device)
        self.width_head = torch.nn.Linear(trunk_out, len(self.width_bins)).to(self.device)
        self.count_head = torch.nn.Linear(trunk_out, self.count_cap).to(self.device)

    @staticmethod
    def counts_matrix(encoded_texts: list[list[int]], vocab_size: int) -> np.ndarray:
        return TinyBoWClassifier._counts_matrix(encoded_texts, vocab_size)

    def logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            xb = self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
            self.trunk.eval()
            hidden = self.trunk(xb)
            center_logits = self.center_head(hidden).detach().cpu().numpy()
            width_logits = self.width_head(hidden).detach().cpu().numpy()
            return center_logits, width_logits

    def logits_with_count(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            xb = self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
            self.trunk.eval()
            hidden = self.trunk(xb)
            center_logits = self.center_head(hidden).detach().cpu().numpy()
            width_logits = self.width_head(hidden).detach().cpu().numpy()
            count_logits = self.count_head(hidden).detach().cpu().numpy()
            return center_logits, width_logits, count_logits

    def train_hybrid(
        self,
        x: np.ndarray,
        center_labels: np.ndarray,
        width_labels: np.ndarray,
        center_targets: np.ndarray,
        width_targets: np.ndarray,
        *,
        count_labels: np.ndarray | None = None,
        exact_weight: float = 0.7,
        soft_weight: float = 0.2,
        width_weight: float = 0.1,
        count_weight: float = 0.0,
        epochs: int = 120,
        learning_rate: float = 1.0,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 13,
    ) -> list[float]:
        torch = self.torch
        torch.manual_seed(seed)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        center_label_tensor = torch.tensor(center_labels, dtype=torch.long, device=self.device)
        width_label_tensor = torch.tensor(width_labels, dtype=torch.long, device=self.device)
        center_target_tensor = torch.tensor(center_targets, dtype=torch.float32, device=self.device)
        width_target_tensor = torch.tensor(width_targets, dtype=torch.float32, device=self.device)
        count_label_tensor = None if count_labels is None else torch.tensor(count_labels, dtype=torch.long, device=self.device)
        params = list(self.trunk.parameters()) + list(self.center_head.parameters()) + list(self.width_head.parameters()) + list(self.count_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate * 0.02,
            weight_decay=weight_decay,
        )
        losses: list[float] = []
        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)
        width_values = torch.tensor(self.width_bins, dtype=torch.float32, device=self.device)
        gold_width_values = (width_target_tensor * width_values.unsqueeze(0)).sum(dim=1)

        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            steps = 0
            self.trunk.train()
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_tensor[batch_idx]
                cl = center_label_tensor[batch_idx]
                wl = width_label_tensor[batch_idx]
                ct = center_target_tensor[batch_idx]
                wt = width_target_tensor[batch_idx]
                gw = gold_width_values[batch_idx]
                count_batch = None if count_label_tensor is None else count_label_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                hidden = self.trunk(xb)
                center_logits = self.center_head(hidden)
                width_logits = self.width_head(hidden)
                count_logits = self.count_head(hidden)

                exact_center = torch.nn.functional.cross_entropy(center_logits, cl)
                exact_width = torch.nn.functional.cross_entropy(width_logits, wl)
                exact_loss = 0.5 * (exact_center + exact_width)

                center_log_probs = torch.nn.functional.log_softmax(center_logits, dim=1)
                width_log_probs = torch.nn.functional.log_softmax(width_logits, dim=1)
                soft_center = -(ct * center_log_probs).sum(dim=1).mean()
                soft_width = -(wt * width_log_probs).sum(dim=1).mean()
                soft_loss = 0.5 * (soft_center + soft_width)

                width_probs = torch.nn.functional.softmax(width_logits, dim=1)
                expected_width = (width_probs * width_values.unsqueeze(0)).sum(dim=1)
                width_loss = ((torch.log1p(expected_width) - torch.log1p(gw)) ** 2).mean()

                loss = exact_weight * exact_loss + soft_weight * soft_loss + width_weight * width_loss
                if count_batch is not None and count_weight > 0.0:
                    loss = loss + count_weight * torch.nn.functional.cross_entropy(count_logits, count_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
        return losses


def encode_examples(examples: Iterable[Example], vocab: Vocab, label_index: dict[str, int]) -> tuple[list[list[int]], np.ndarray]:
    encoded = [vocab.encode(example.question) for example in examples]
    labels = np.array([label_index[example.refmark] for example in examples], dtype=np.int64)
    return encoded, labels


def summarize_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy(predictions, labels),
        "top3_accuracy": topk_accuracy(logits, labels, k=min(3, logits.shape[1])),
    }
