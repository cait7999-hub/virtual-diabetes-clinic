import os
import json
import joblib
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import precision_score, recall_score, root_mean_squared_error

np.random.seed(42)


# ==========================================
# 1. Load Diabetes Dataset
# ==========================================

data = load_diabetes(as_frame=True)

X = data.data
y = data.target

# Same train/test split as v0.1
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 2. Baseline Model (v0.1)
# ==========================================

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_rmse = root_mean_squared_error(
    y_test,
    baseline_predictions
)

print(f"Baseline LinearRegression RMSE: {baseline_rmse:.2f}")


# ==========================================
# 3. Improved Models (v0.2)
# ==========================================

models = {
    "Ridge": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge())
        ]),
        {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0]
        }
    ),

    "RandomForest": (
        RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 3]
        }
    )
}


# ==========================================
# 4. Cross-validation
# ==========================================

baseline_cv_rmse = -cross_val_score(
    baseline,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
).mean()

best_name = "LinearRegression"
best_model = baseline
best_cv_rmse = baseline_cv_rmse
best_params = {}

print(f"LinearRegression CV RMSE: {baseline_cv_rmse:.2f}")

for name, (estimator, params) in models.items():

    search = GridSearchCV(
        estimator,
        params,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    search.fit(X_train, y_train)

    cv_rmse = -search.best_score_

    print(f"{name} CV RMSE: {cv_rmse:.2f}")
    print(f"{name} Best parameters: {search.best_params_}")

    if cv_rmse < best_cv_rmse:
        best_cv_rmse = cv_rmse
        best_name = name
        best_model = search.best_estimator_
        best_params = search.best_params_


# ==========================================
# 5. Evaluate Selected Model
# ==========================================

final_predictions = best_model.predict(X_test)

final_rmse = root_mean_squared_error(
    y_test,
    final_predictions
)

rmse_delta = final_rmse - baseline_rmse

print("\n===== Model Comparison =====")

print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Selected model: {best_name}")
print(f"v0.2 RMSE: {final_rmse:.2f}")
print(f"RMSE delta: {rmse_delta:+.2f}")

if rmse_delta < 0:
    print("Result: v0.2 improved performance")
elif rmse_delta > 0:
    print("Result: v0.2 did not improve test RMSE")
else:
    print("Result: Same test RMSE as baseline")


# ======================================
# 6. Save Model Based on Version
# ======================================

model_version = os.getenv("MODEL_VERSION", "v0.2")

if model_version == "v0.1":
    model_to_save = baseline
elif model_version == "v0.2":
    model_to_save = best_model
else:
    raise ValueError(f"Unsupported model version: {model_version}")

Path("models").mkdir(exist_ok=True)

joblib.dump(model_to_save, "models/model.joblib")

print(f"Model saved for {model_version}!")

# ==========================================
# 7. High-risk flag evaluation (threshold = 75th percentile of y_train)
# ==========================================

threshold = float(np.percentile(y_train, 75))
y_true_flag = (y_test >= threshold).astype(int)


def flag_metrics(preds):
    flag = (preds >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true_flag, flag, zero_division=0)),
        "recall": float(recall_score(y_true_flag, flag, zero_division=0)),
    }


saved_preds = model_to_save.predict(X_test)

metrics = {
    "model_version": model_version,
    "model": "LinearRegression" if model_version == "v0.1" else best_name,
    "test_rmse": float(root_mean_squared_error(y_test, saved_preds)),
    "high_risk_threshold": threshold,
    "high_risk_flag": flag_metrics(saved_preds),
    "comparison": {
        "v0.1": {
            "test_rmse": float(baseline_rmse),
            "cv_rmse": float(baseline_cv_rmse),
            **flag_metrics(baseline_predictions),
        },
        "v0.2": {
            "model": best_name,
            "best_params": best_params,
            "test_rmse": float(final_rmse),
            "cv_rmse": float(best_cv_rmse),
            **flag_metrics(final_predictions),
        },
        "rmse_delta": float(rmse_delta),
    },
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("Metrics saved successfully!")
