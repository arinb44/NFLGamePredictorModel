"""Model definitions. Each factory returns a fresh, unfitted estimator."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic(C: float = 0.01) -> Pipeline:
    """Standardize features, then L2-regularized logistic regression.

    Standardizing puts every feature on the same scale (mean 0, std 1), so the
    penalty treats them equally and the coefficients are comparable.
    Smaller C = stronger penalty = coefficients pulled harder toward 0, which
    helps when many features are correlated (they are here).
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])
