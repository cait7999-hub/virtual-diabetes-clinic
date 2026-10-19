import json
import joblib
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Load data and use the same split as v0.1
data = load_diabetes(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# v0.1 baseline
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
baseline.fit(X_train, y_train)

baseline_rmse = root_mean_squared_error(
    y_test, baseline.predict(X_test)
)
print(f"Baseline RMSE: {baseline_rmse:.2f}")

# v0.2 candidates
candidates = {
    "Ridge": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge())
        ]),
        {"regressor__alpha": [0.1, 1, 10, 100]}
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 3]
        }
    )
}

# Select using training-set cross-validation
baseline_cv = -cross_val_score(
    baseline, X_train, y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
).mean()

best_name = "LinearRegression"
best_model = baseline
best_cv = baseline_cv
best_params = {}

print(f"LinearRegression CV RMSE: {baseline_cv:.2f}")

for name, (estimator, params) in candidates.items():
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
    print(f"{name} Best params: {search.best_params_}")

    if cv_rmse < best_cv:
        best_cv = cv_rmse
        best_name = name
        best_model = search.best_estimator_
        best_params = search.best_params_

# Final held-out evaluation
final_rmse = root_mean_squared_error(
    y_test, best_model.predict(X_test)
)
delta = final_rmse - baseline_rmse

print(f"Selected model: {best_name}")
print(f"v0.2 RMSE: {final_rmse:.2f}")
print(f"RMSE delta: {delta:+.2f}")

# Save model and metrics
Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/model.joblib")

metrics = {
    "baseline_rmse": float(baseline_rmse),
    "selected_model": best_name,
    "v02_rmse": float(final_rmse),
    "rmse_delta": float(delta),
    "best_params": best_params
}

Path("models/metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)

print("Model and metrics saved successfully!")
