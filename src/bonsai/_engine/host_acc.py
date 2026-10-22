"""float64 host accumulators of a parent's message (amendment A8).

A parent with many children (the root during the greedy pass: N children) keeps W = Σ_c w_c and
WX = Σ_c w_c x_c as float64 NumPy G-vectors, updated once per merge.
Same-weight rule by recomputation (A21): the M2 engine computes each weight on the host in float64,
w_c = 1/(v_c + t_c) from the stored compute-dtype message v_c (widened exactly) and the float64 t_c
(``greedy.NodeTable.weights64``). The same expression on the same stored values gives the same w_c
every time, so the weight subtracted is the one added, without device → host weight transfers, and
removing a child leaves exactly the sum over the other children (up to float64 rounding).
The child count n is tracked too: for a 2-child parent W_R = W − w_1 − w_2 is rounding noise whose
size depends on the update history, so ``rest`` guards on the exact count, not on a threshold.
"""

from dataclasses import dataclass

import numpy as np


def _f64(a) -> np.ndarray:
    return np.asarray(a).astype(np.float64)  # exact for float32 input


@dataclass
class ParentAcc:
    """W (G,) and WX (G,) in float64; n children."""

    W: np.ndarray
    WX: np.ndarray
    n: int

    @classmethod
    def from_children(cls, w, x) -> "ParentAcc":
        """From the child weights w (C, G) and child means x (C, G), widened to float64."""
        w, x = _f64(w), _f64(x)
        return cls(w.sum(axis=0), (w * x).sum(axis=0), len(w))

    def add(self, w, x) -> None:
        self.W += _f64(w)
        self.WX += _f64(w) * _f64(x)
        self.n += 1

    def remove(self, w, x) -> None:
        self.W -= _f64(w)
        self.WX -= _f64(w) * _f64(x)
        self.n -= 1

    def message(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, W) of the parent, float64."""
        return self.WX / self.W, self.W.copy()

    def rest(self, w1, x1, w2, x2) -> tuple[np.ndarray, np.ndarray]:
        """(x_R, W_R) of the parent without children 1 and 2, for P pairs ((P, G) inputs), float64.

        "Root minus first", then minus second, as upstream (TH:2281-2282, 5372-5374). Raises
        ValueError for a parent with fewer than 3 children (no rest; ``_ref.pair.pair_score`` too).
        """
        if self.n < 3:
            raise ValueError(f"rest: the parent needs at least 3 children, has {self.n}")
        w1, x1, w2, x2 = _f64(w1), _f64(x1), _f64(w2), _f64(x2)
        WR = (self.W - w1) - w2
        return ((self.WX - w1 * x1) - w2 * x2) / WR, WR
