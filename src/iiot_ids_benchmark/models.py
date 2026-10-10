"""Three pipelines with attack-oriented continuous scores."""

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def make_models(seed=42, rf_trees=100, if_trees=100, n_jobs=1):
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            C=1.0,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "isolation_forest": IsolationForest(
            n_estimators=if_trees,
            max_samples="auto",
            contamination="auto",
            n_jobs=n_jobs,
            random_state=seed,
        ),
    }
    return {
        name: Pipeline(
            [("preprocess", make_preprocessor(name == "logistic_regression")), ("model", model)]
        )
        for name, model in estimators.items()
    }


def fit_model(name, pipeline, X_train, y_train):
    if name == "isolation_forest":
        normal = y_train == 0
        if not normal.any():
            raise ValueError("Isolation Forest requires normal training observations")
        pipeline.fit(X_train.loc[normal])
        return int(normal.sum())
    pipeline.fit(X_train, y_train)
    return len(X_train)


def attack_scores(name, pipeline, X):
    if name == "isolation_forest":
        # sklearn score_samples is larger for normal points: explicitly negate it.
        return -pipeline.score_samples(X)
    if name == "logistic_regression":
        return pipeline.decision_function(X)
    attack_index = list(pipeline.named_steps["model"].classes_).index(1)
    return pipeline.predict_proba(X)[:, attack_index]
