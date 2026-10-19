/ CHANGELOG

/ v0.2 - Model Improvement

/Changes
- Added Ridge Regression and RandomForestRegressor as candidate models.
- Used 5-fold cross-validation for model selection.
- Selected Ridge Regression as the best-performing model based on cross-validation.
- Maintained the same train/test split as v0.1 (`random_state=42`).
- Saved the selected model to `models/model.joblib`.
- Saved evaluation results to `models/metrics.json`.

/Performance Comparison

| Metric | v0.1 | v0.2 |
|---|---|---|
| Model | LinearRegression | Ridge Regression |
| Test RMSE | 53.85 | 53.63 |
| RMSE change | Baseline | -0.23 (approximately 0.4% improvement) |

/Results

The v0.2 Ridge Regression model achieved a test RMSE of 53.63, compared with 53.85 for the v0.1 LinearRegression baseline.

The lower RMSE indicates a small improvement in prediction accuracy on the held-out test set.

### Conclusion

Iteration 2 successfully introduced alternative regression models and improved the baseline performance. The service continues to return a continuous disease progression score rather than a binary high-risk classification.

No probability calibration or precision/recall evaluation was required because no binary risk flag was introduced.
