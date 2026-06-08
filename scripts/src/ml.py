from src.features import ADDITIONAL_FEATURES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import numpy as np
from src.consts import composition_labels, TARGET, MODEL_USE_EARLY_STOPPING

FEATURES_TO_TRAIN_MODEL = composition_labels + ADDITIONAL_FEATURES


def make_model(seed=100, iterations=5000):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        subsample=0.8,
        random_seed=seed,
        verbose=False,
    )


def evaluate_predictions(y_true, pred):
    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = float(r2_score(y_true, pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_cb_model(kkr_data, predict_df=None, seed=100, valid_size=0.2, elements=None):
    """
    Train CatBoost for active learning / BO.

    Procedure:
    1. Split known data into train/validation
    2. Fit model with early stopping on validation
    3. Get best_iteration
    4. Retrain final model on ALL known data using best_iteration
    5. Use final model for candidate prediction

    Returns
    -------
    final_model, metrics, y_pred
    """
    df_ = pd.DataFrame(kkr_data)
    feature_cols = (elements if elements is not None else composition_labels) + ADDITIONAL_FEATURES
    data = df_[feature_cols + [TARGET]].copy()
    data = data.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    X = data[feature_cols]
    y = data[TARGET]

    if len(data) < 20:
        raise ValueError(f"Too little data to train reliably: n={len(data)}")

    # split only for early stopping / monitoring
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=seed
    )

    # stage 1: tune effective number of trees
    model = make_model(seed=seed, iterations=5000)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False,
    )

    best_iteration = int(model.get_best_iteration())
    if best_iteration <= 0:
        best_iteration = 5000

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_metrics = evaluate_predictions(y_train, train_pred)
    valid_metrics = evaluate_predictions(y_valid, valid_pred)

    if MODEL_USE_EARLY_STOPPING:
        # use the early-stopping model directly — preserves genuine per-model variance
        # so ensemble σ reflects real uncertainty rather than collapsing to near-zero
        predict_model = model
    else:
        # retrain on ALL subsample data at the tuned iteration count
        # (collapses train R²→1 and kills ensemble diversity — not recommended)
        predict_model = make_model(seed=seed, iterations=best_iteration)
        predict_model.fit(X, y, verbose=False)

    metrics = {
        "n_rows_total": int(len(data)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "feature_cols": feature_cols,
        "train": train_metrics,
        "validation": valid_metrics,
        "best_iteration": int(best_iteration),
        "use_early_stopping": MODEL_USE_EARLY_STOPPING,
    }

    y_pred = None
    if predict_df is not None:
        X_pred = predict_df[feature_cols]
        y_pred = predict_model.predict(X_pred)

    return predict_model, metrics, y_pred