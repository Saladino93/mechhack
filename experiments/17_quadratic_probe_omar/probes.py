"""Pleshkov 2026 polynomial-quadratic probe.

Architecture (intentionally simple, all closed-form, no SGD):

    X_pca  = PCA(n_components=d).fit_transform(activations)
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_pca)
    Ridge  = Ridge(alpha=...).fit(X_poly, y)

For binary classification we use Ridge regression on the {0,1} target and treat
the raw output as a score for AUC. This matches Pleshkov's recipe and avoids
the numerical issues of trying to maximise log-likelihood on (potentially
ill-conditioned) polynomial features. ``predict_proba`` exposes a clipped &
normalised version for use with sklearn-style downstream code, but the
*ordering* (the only thing AUC cares about) is unchanged.

The class is a single sklearn-style estimator wrapping
``StandardScaler -> PCA -> PolynomialFeatures -> Ridge``. PCA is fitted on the
training fold only — no test-set leakage.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class QuadraticProbe:
    """PCA + degree-2 polynomial features + Ridge regression.

    Parameters
    ----------
    d_pca : int
        Number of PCA components to keep. Quadratic feature count is
        ``d*(d+3)/2`` (no bias): d=16 -> 152, d=32 -> 560.
    alpha : float
        Ridge L2 regularization strength.
    standardize : bool
        Whether to z-score features before PCA. Useful when activation scales
        vary across coordinates; recommended in Pleshkov 2026.
    random_state : int
        Random state for PCA's randomized solver.
    """

    def __init__(self, d_pca: int = 16, alpha: float = 1.0,
                 standardize: bool = True, random_state: int = 0):
        self.d_pca = int(d_pca)
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.random_state = int(random_state)
        self._pipe: Pipeline | None = None

    # -- internal -----------------------------------------------------------
    def _build_pipe(self) -> Pipeline:
        steps = []
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=self.d_pca,
                                 random_state=self.random_state)))
        steps.append(("poly", PolynomialFeatures(degree=2,
                                                 include_bias=False,
                                                 interaction_only=False)))
        steps.append(("ridge", Ridge(alpha=self.alpha,
                                     random_state=self.random_state)))
        return Pipeline(steps)

    # -- sklearn API --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuadraticProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # Defensive: NaN/Inf in X would silently corrupt PCA.
        if not np.isfinite(X).all():
            raise ValueError("QuadraticProbe.fit: X contains NaN or Inf.")
        self._pipe = self._build_pipe()
        self._pipe.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("QuadraticProbe.decision_function called before fit().")
        X = np.asarray(X, dtype=np.float32)
        return self._pipe.predict(X).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n, 2)`` array; column 1 is the positive-class score.

        Ridge regression on a {0,1} target is *not* a probability — we clip to
        [0, 1] and return ``[1-p, p]``. The ordering used for AUC is
        unaffected by this clipping.
        """
        s = self.decision_function(X)
        p1 = np.clip(s, 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0.5).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return ROC AUC (higher is better) — overrides sklearn default of R²."""
        s = self.decision_function(X)
        return float(roc_auc_score(np.asarray(y), s))

    def get_params(self, deep: bool = True) -> dict:
        return {"d_pca": self.d_pca, "alpha": self.alpha,
                "standardize": self.standardize,
                "random_state": self.random_state}

    def set_params(self, **params) -> "QuadraticProbe":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def n_quadratic_features(d: int) -> int:
    """Number of features after PolynomialFeatures(degree=2, include_bias=False).

    Formula: ``d + d*(d+1)/2``  (linear terms + upper-triangular pairs incl. squares).
    """
    return d + d * (d + 1) // 2
