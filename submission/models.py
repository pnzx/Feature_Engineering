"""Small, deterministic NumPy estimators used by the submission experiment."""
from dataclasses import dataclass
import numpy as np


class LogisticRegression:
    """L2-penalized logistic regression using damped Newton steps (C=1)."""
    def fit(self, x, y):
        # Diagonal preconditioning only. Transform the penalty as well so the
        # objective in the supplied input units is mathematically unchanged.
        self.scale = np.maximum(np.sqrt((x*x).mean(axis=0)), 1e-8)
        z = np.column_stack([np.ones(len(x)), x / self.scale])
        penalty = np.r_[0., 1 / self.scale**2]
        w = np.zeros(z.shape[1])

        def objective(v):
            t = z @ v
            return (np.logaddexp(0, t) - y*t).sum() + .5*np.dot(penalty, v*v)

        for _ in range(80):
            t = z @ w
            p = np.exp(-np.logaddexp(0, -t))
            grad = z.T @ (p-y) + penalty*w
            hessian = z.T @ ((p*(1-p))[:, None]*z) + np.diag(penalty)
            step = np.linalg.solve(hessian, grad)
            factor = 1.
            loss = objective(w)
            while factor > 1e-7 and objective(w-factor*step) > loss:
                factor *= .5
            w -= factor*step
            if np.max(np.abs(factor*step)) < 1e-6:
                break
        self.weights = w
        return self

    def predict_proba(self, x):
        t = self.weights[0] + (x/self.scale) @ self.weights[1:]
        return np.exp(-np.logaddexp(0, -t))


@dataclass
class Node:
    probability: float
    column: int = -1
    cut: int = -1
    left: object = None
    right: object = None


class RandomForest:
    """Bootstrap CART forest with a histogram approximation to split search."""
    def __init__(self, seed=42, n_trees=48, depth=10, leaf=6, sample=6000):
        self.seed, self.n_trees, self.depth = seed, n_trees, depth
        self.leaf, self.sample = leaf, sample

    def _bin(self, x):
        return np.column_stack([np.searchsorted(c, x[:, j], side="right")
                                for j, c in enumerate(self.cuts)]).astype(np.uint8)

    def fit(self, x, y):
        self.cuts = []
        for col in x.T:
            values = np.unique(col)
            if len(values) <= 32:
                cuts = values[1:]
            else:
                cuts = np.unique(np.quantile(col, np.arange(1, 32)/32))
                cuts = cuts[cuts > values[0]]
            self.cuts.append(cuts)
        binned = self._bin(x)
        rng = np.random.default_rng(self.seed)
        mtry = max(1, int(np.sqrt(x.shape[1])))
        self.importance = np.zeros(x.shape[1])

        def grow(rows, level):
            target = y[rows]
            total = len(rows)
            positives = target.sum()
            node = Node(positives/total)
            if level == self.depth or total < 2*self.leaf or positives in (0, total):
                return node
            best = 0.
            for j in rng.choice(x.shape[1], mtry, replace=False):
                codes = binned[rows, j]
                n = np.bincount(codes, minlength=32).cumsum()[:-1]
                s = np.bincount(codes, weights=target, minlength=32).cumsum()[:-1]
                valid = (n >= self.leaf) & (total-n >= self.leaf)
                idx = np.flatnonzero(valid)
                if idx.size == 0:
                    continue
                gains = 2*(s[idx]**2/n[idx] + (positives-s[idx])**2/(total-n[idx])
                           - positives**2/total)
                k = gains.argmax()
                if gains[k] > best + 1e-9:
                    best = gains[k]
                    node.column, node.cut = j, idx[k]
            if node.column >= 0:
                self.importance[node.column] += best
                left = binned[rows, node.column] <= node.cut
                node.left = grow(rows[left], level+1)
                node.right = grow(rows[~left], level+1)
            return node

        size = min(len(y), self.sample)
        self.trees = [grow(rng.integers(len(y), size=size), 0) for _ in range(self.n_trees)]
        self.importance /= max(self.importance.sum(), 1.)
        return self

    def predict_proba(self, x):
        binned = self._bin(x)
        scores = np.zeros(len(x))
        def visit(node, rows):
            if node.column < 0:
                scores[rows] += node.probability
                return
            left = binned[rows, node.column] <= node.cut
            visit(node.left, rows[left])
            visit(node.right, rows[~left])
        for tree in self.trees:
            visit(tree, np.arange(len(x)))
        return scores/len(self.trees)
