from src.features import ADDITIONAL_FEATURES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import numpy as np
from src.consts import composition_labels

FEATURES_TO_TRAIN_MODEL = composition_labels + ADDITIONAL_FEATURES
TARGET = 'Tc_mu0.2'

def train_cb_model(kkr_data, predict_df=None, seed=100):
    df_ = pd.DataFrame(kkr_data)
    data = df_[FEATURES_TO_TRAIN_MODEL + [TARGET]].copy()

    # Drop rows with missing values in features/target
    data = data.dropna(subset=FEATURES_TO_TRAIN_MODEL + [TARGET]).reset_index(drop=True)

    X = data[FEATURES_TO_TRAIN_MODEL]
    y = data[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=seed
    )
    # 0.1765 of 85% ≈ 15%, so final split is ~70/15/15

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        subsample=0.8,
        random_seed=42,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False,
    )

    def evaluate_split(name, X_part, y_part):
        pred = model.predict(X_part)
        mse = float(mean_absolute_error(y_part, pred))
        metrics = {
            "MAE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_part, pred)),
        }
        return pred, metrics

    train_pred, train_metrics = evaluate_split("Train", X_train, y_train)
    valid_pred, valid_metrics = evaluate_split("Validation", X_valid, y_valid)
    test_pred, test_metrics = evaluate_split("Test", X_test, y_test)

    metrics = {
        "n_rows_total": int(len(data)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "feature_cols": FEATURES_TO_TRAIN_MODEL,
        "train": train_metrics,
        "validation": valid_metrics,
        "test": test_metrics,
        "best_iteration": int(model.get_best_iteration()),
    }
    test_results = X_test.copy()
    test_results["Tc_true"] = y_test.values
    test_results["Tc_pred"] = test_pred

    # predict on predict_df
    y_pred = None
    if predict_df is not None:
        X_pred = predict_df[FEATURES_TO_TRAIN_MODEL]
        y_pred = model.predict(X_pred)
    return model, metrics, y_pred