"""Cosine / L2 comparison of one ref-vs-ik embedding pair."""
import base64

import numpy as np


class ImageEmbeddingSequence:
    """One image's embedding sequence: base64 f32 rows decoded to [n_tokens, n_embd].

    Compare two sequences row-by-row with cosim()/rel_l2(); each op asserts the
    two operands share a shape, so a mismatch raises instead of silently
    broadcasting.
    """

    def __init__(self, rows):
        self._e = self._decode(rows)

    @staticmethod
    def _decode(rows) -> np.ndarray:
        return np.stack(
            [np.frombuffer(base64.b64decode(r), dtype="<f4") for r in rows]
        ).astype(np.float64)

    def shape(self) -> tuple:
        """[n_tokens, n_embd] of this sequence."""
        return self._e.shape

    def cosim(self, other) -> np.ndarray:
        """Per-row cosine similarity vs other; both-zero rows score 1.0, one-zero rows 0.0."""
        assert self._e.shape == other._e.shape, (
            f"shape mismatch: {self._e.shape} vs {other._e.shape}"
        )
        a, b = self._e, other._e
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        dot = (a * b).sum(axis=1)
        denom = na * nb
        cosim = np.zeros(dot.shape, dtype=np.float64)
        both_present = denom > 0.0
        cosim[both_present] = dot[both_present] / denom[both_present]
        cosim[(na == 0.0) & (nb == 0.0)] = 1.0
        return cosim

    def rel_l2(self, other) -> np.ndarray:
        """Per-row relative L2 error ||self-other|| / ||self||; zero-self rows: 0.0 if other also zero, else inf."""
        assert self._e.shape == other._e.shape, (
            f"shape mismatch: {self._e.shape} vs {other._e.shape}"
        )
        a, b = self._e, other._e
        ref_norm = np.linalg.norm(a, axis=1)
        diff = np.linalg.norm(a - b, axis=1)
        out = np.zeros(diff.shape, dtype=np.float64)
        present = ref_norm > 0.0
        out[present] = diff[present] / ref_norm[present]
        out[(~present) & (diff > 0.0)] = np.inf
        return out
