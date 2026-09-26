import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

AVAILABLE_MODELS = {
    'RandomForest': lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
    'LogisticRegression': lambda: LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': lambda: GradientBoostingClassifier(random_state=42),
}


def compare_models(X_train, y_train, cv=5, scoring='f1'):
    """
    Compares several models using stratified cross-validation (instead of
    relying on a single train/test split, which can give misleading
    results due to a lucky/unlucky split).

    Returns a DataFrame sorted by mean score, to make it easy to pick the
    final model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, builder in AVAILABLE_MODELS.items():
        model = builder()
        scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )
        results.append({
            'model': name,
            f'{scoring}_mean': scores.mean(),
            f'{scoring}_std': scores.std(),
        })
        print(f"{name}: {scoring} = {scores.mean():.4f} (+/- {scores.std():.4f})")

    results_df = pd.DataFrame(results).sort_values(
        f'{scoring}_mean', ascending=False
    ).reset_index(drop=True)

    return results_df


def train_model(X_train, y_train, model_type='RandomForest'):
    """
    Trains the final model (already chosen via compare_models) on the
    full training set.
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_type}' not recognized. Options: {list(AVAILABLE_MODELS)}"
        )
    model = AVAILABLE_MODELS[model_type]()
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model, feature_names):
    """
    Extracts the importance of each feature (only for tree-based models,
    like RandomForest and GradientBoosting). Helps interpret which sensor
    readings contribute the most to predicting failures.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError(
            f"Model {type(model).__name__} does not expose feature_importances_"
        )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df
