"""Model definitions. Each factory returns a fresh, unfitted estimator."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic(C: float = 0.01, symmetric: bool = True) -> Pipeline:
    """Scale features, then L2-regularized logistic regression.

    Scaling puts every feature on the same scale (std 1), so the penalty
    treats them equally and the coefficients are comparable.
    Smaller C = stronger penalty = coefficients pulled harder toward 0, which
    helps when many features are correlated (they are here).

    symmetric=True drops the intercept (and doesn't center features). Every
    feature is a home-minus-away difference, so two identical teams at a
    neutral site get exactly 50%, and home advantage has to come from the
    home_field feature. With an intercept, the model gave the listed "home"
    team a phantom edge at neutral sites. It was also slightly better on the
    tuning seasons (0.6154 vs 0.6156 log loss).
    """
    return Pipeline([
        ("scale", StandardScaler(with_mean=not symmetric)),
        ("clf", LogisticRegression(C=C, fit_intercept=not symmetric, max_iter=2000)),
    ])


def random_forest(n_estimators: int = 500, max_depth=None, min_samples_leaf: int = 20,
                  max_features: float = 0.3, calibrate: bool = True, seed: int = 0):
    """Random forest: many decorrelated trees, probabilities averaged.

    min_samples_leaf keeps leaves large (NFL data is noisy), and max_features
    limits how many features each split can look at, so trees disagree more.
    Forest probabilities are squashed toward 0.5, so by default they are
    recalibrated with a sigmoid fitted by cross-validation on the training data.
    """
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, max_features=max_features,
                                n_jobs=-1, random_state=seed)
    return _calibrated(rf) if calibrate else rf


def gradient_boosting(learning_rate: float = 0.03, max_iter: int = 300, max_leaf_nodes: int = 8,
                      min_samples_leaf: int = 50, l2_regularization: float = 1.0,
                      calibrate: bool = False, seed: int = 0):
    """Histogram gradient boosting: shallow trees added one at a time, each
    fixing the previous ones' errors. A low learning rate, few leaves and large
    leaves keep it from memorizing noise."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter,
                                        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                        l2_regularization=l2_regularization, random_state=seed)
    return _calibrated(gb) if calibrate else gb


def _calibrated(model):
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(model, method="sigmoid", cv=5)


class Ensemble:
    """Average the probabilities of several models, each on its own feature list."""

    def __init__(self, members):
        self.members = members  # list of (make_model, features, weight)

    def fit(self, X, y):
        self.fitted_ = [(m().fit(X[f], y), f, w) for m, f, w in self.members]
        return self

    def predict_proba(self, X):
        import numpy as np
        total = sum(w for _, _, w in self.fitted_)
        p = sum(w * m.predict_proba(X[f])[:, 1] for m, f, w in self.fitted_) / total
        return np.column_stack([1 - p, p])
