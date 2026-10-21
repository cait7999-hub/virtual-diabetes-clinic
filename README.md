# Virtual Diabetes Clinic: MLOps Pipeline

## 1. Project Overview

This project was developed as part of an MLOps assignment focused on building and deploying a machine learning service for a virtual diabetes clinic.

In the proposed scenario, healthcare professionals regularly review patient information to identify individuals who may require additional follow-up. Reviewing large numbers of patient records manually can be time-consuming.

Our goal was to develop a machine learning pipeline that predicts diabetes disease progression and produces a continuous score that could help prioritize patient follow-ups.

For this assignment, we used the publicly available Diabetes regression dataset from scikit-learn as a substitute for de-identified electronic health records.

The project includes model training, evaluation, an API for generating predictions, Docker containerization, automated testing, and CI/CD workflows using GitHub Actions.

This is an educational project and is not intended for clinical use.

## 2. Dataset

We used the Diabetes regression dataset provided by scikit-learn.

The dataset contains 442 observations and 10 numerical features, including age, sex, body mass index (BMI), blood pressure, and six additional measurements.

The target variable represents a quantitative measure of disease progression.

For this assignment, higher predicted values are interpreted as indicating greater disease progression.

The dataset can be loaded using:

```python
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)

X = data.data
y = data.target
```

We divided the dataset into training (80%) and testing (20%) subsets using a fixed random state of 42 to ensure consistent comparisons between models.

## 3. Model Development

We developed the project in two iterations.

### Version 0.1: Baseline Model

For the first iteration, we implemented a simple regression pipeline using StandardScaler and LinearRegression.

StandardScaler was used to standardize the input features before fitting the regression model.

The baseline model achieved a test RMSE of 53.85.

This provided a reference point for evaluating improvements in the second iteration.

### Version 0.2: Model Improvement

In the second iteration, we explored alternative regression models to determine whether predictive performance could be improved.

We evaluated Ridge Regression and Random Forest Regression.

We used five-fold cross-validation and GridSearchCV to compare models and tune their hyperparameters.

Ridge Regression was selected because it achieved the lowest cross-validation RMSE among the evaluated models.

The selected Ridge model achieved a test RMSE of 53.63.

In addition to the continuous prediction, version 0.2 introduces a high-risk flag based on the 75th percentile of the training target distribution. Predictions at or above this threshold are flagged as high-risk, and the flag's precision and recall are recorded in `models/metrics.json` to support the prioritization of patient follow-ups.

### Performance Comparison

| Metric | v0.1 | v0.2 |
|---|---|---|
| Model | Linear Regression | Ridge Regression |
| Test RMSE | 53.85 | 53.63 |
| Cross-validation RMSE | 55.97 | 55.86 |
| Difference in test RMSE | Baseline | -0.23 |

The second iteration achieved a small improvement of approximately 0.4% in test RMSE.

Although the improvement was modest, the results demonstrate how model selection, regularization, and hyperparameter tuning can be incorporated into an automated machine learning workflow.

A more detailed comparison is available in [CHANGELOG.md](CHANGELOG.md).

## 4. Project Structure

The main files in the repository are:

- `train.py`: Loads the dataset, trains the regression models, evaluates their performance, and saves the selected model.
- `main.py`: Implements the FastAPI application and prediction endpoints.
- `tests/test_api.py`: Contains automated API acceptance tests.
- `tests/test_unit.py`: Contains unit tests for the API endpoints, run with pytest.
- `Dockerfile`: Defines how the application is packaged and executed in a Docker container.
- `requirements.txt`: Lists the Python dependencies.
- `requirements-dev.txt`: Lists the development and testing dependencies.
- `.github/workflows/ci.yml`: Automates linting, training, testing, and Docker builds.
- `.github/workflows/release.yml`: Defines the Docker release and publishing process.
- `CHANGELOG.md`: Documents the changes and performance differences between model versions.

## 5. Installation and Setup

To run the project locally, first clone the GitHub repository:

```bash
git clone https://github.com/cait7999-hub/virtual-diabetes-clinic.git
cd virtual-diabetes-clinic
```

Create and activate a Python virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project uses Python 3.11 and pinned dependencies to support reproducibility.

## 6. Training the Models

The training script supports both model versions through the MODEL_VERSION environment variable.

To train and save the baseline Linear Regression model:

```bash
MODEL_VERSION=v0.1 python train.py
```

To train and save the improved model:

```bash
MODEL_VERSION=v0.2 python train.py
```

After training, the following files are generated:

- `models/model.joblib`: The trained model used by the API.
- `models/metrics.json`: The evaluation results and model comparison metrics.

The model training process uses fixed random seeds and a consistent train/test split to make the results reproducible.

## 7. API Implementation

We developed a REST API using FastAPI.

The API allows users to submit patient feature values and receive a predicted disease progression score.

Two endpoints are provided.

### Health Endpoint

`GET /health`

This endpoint confirms that the application is running and identifies the model version currently being used.

Example request:

```bash
curl http://localhost:80/health
```

Example response for v0.1:

```json
{
  "status": "ok",
  "model_version": "v0.1"
}
```

The v0.2 release returns `"model_version": "v0.2"`.

### Prediction Endpoint

`POST /predict`

This endpoint accepts the ten numerical features from the Diabetes dataset and returns a continuous prediction.

Example request:

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001
  }'
```

Example response:

```json
{
  "prediction": 150.0
}
```

The numerical prediction shown above is illustrative. The actual value depends on the trained model.

### Input Validation

The API uses Pydantic to validate incoming requests.

All ten features are required and must contain valid numerical values.

If a required feature is missing or invalid, the API returns HTTP status code 422 with a JSON error response.

For example, submitting a request without `s6` produces an error identifying that the required field is missing.

Validation errors follow FastAPI's standard structure, which includes a `detail` field describing the invalid input.

The API also rejects unexpected additional features.

### Error response structure

Invalid input returns HTTP **422** with FastAPI's standard JSON body. Example (missing `s6`):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
                "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}
    }
  ]
}
```

- `loc` – where the problem is (`body` + field name)
- `type` – `missing`, `float_parsing` (not a number) or `extra_forbidden` (unknown field)
- `msg` – human-readable explanation

## 8. Docker Containerization

We used Docker to package the application and its dependencies into a self-contained environment.

The Dockerfile uses a Python 3.11 slim image and trains the selected model during the image-building process.

This ensures the trained model is included in the final container, allowing it to run without additional setup.

### Build Version 0.1

```bash
docker build --build-arg MODEL_VERSION=v0.1 -t diabetes-api:v0.1 .
```

Run the container:

```bash
docker run --rm -p 80:80 diabetes-api:v0.1
```

### Build Version 0.2

```bash
docker build --build-arg MODEL_VERSION=v0.2 -t diabetes-api:v0.2 .
```

Run the container:

```bash
docker run --rm -p 80:80 diabetes-api:v0.2
```

Only one container should be run on port 80 at a time.

The application is configured to listen on port 80, as required by the assignment.

## 9. Continuous Integration and Deployment

We implemented GitHub Actions workflows to automate key stages of the machine learning development process.

### Continuous Integration

The CI workflow is defined in:

`.github/workflows/ci.yml`

It runs automatically when changes are pushed to the repository or when a pull request is opened.

The workflow performs the following tasks:

1. Checks out the repository.
2. Sets up Python and installs the dependencies.
3. Runs a linting check with ruff.
4. Runs the model training script.
5. Runs the pytest unit tests.
6. Saves the trained model and evaluation metrics as artifacts.
7. Builds the Docker image.
8. Starts the application inside a Docker container.
9. Executes automated API acceptance tests.

This helps identify problems before changes are released.

### Docker Release Workflow

The release workflow is defined in:

`.github/workflows/release.yml`

It is configured to run when version tags beginning with `v` are pushed to GitHub.

The workflow is designed to build the appropriate Docker image, perform container smoke tests, publish the image to GitHub Container Registry (GHCR), and create a GitHub Release with the metrics attached.

The release workflow must complete successfully before a version can be considered published.

## 10. Automated Testing

The project has two levels of automated testing.

Unit tests in `tests/test_unit.py` run directly against the FastAPI application with pytest and verify that:

- The `/health` endpoint returns HTTP 200 and the expected model version.
- The `/predict` endpoint accepts valid patient data and returns a numerical prediction.
- Requests with missing required features return HTTP 422 and a JSON validation error.
- Requests with non-numeric feature values return HTTP 422.
- Requests with unexpected additional fields are rejected with HTTP 422.

API acceptance tests in `tests/test_api.py` run against the application inside a running Docker container and verify the same behavior end-to-end.

These tests are executed within the GitHub Actions CI pipeline.

To run the unit tests locally:

```bash
pip install -r requirements-dev.txt
ruff check .
MODEL_VERSION=v0.2 python train.py
pytest
```

To run the API acceptance tests locally, first build the Docker image and start the application:

```bash
docker build --build-arg MODEL_VERSION=v0.2 -t diabetes-api:v0.2 .
docker run -d --name diabetes-test -p 8080:80 diabetes-api:v0.2
```

Once the API is running, execute:

```bash
MODEL_VERSION=v0.2 python tests/test_api.py
```

To stop and remove the test container:

```bash
docker rm -f diabetes-test
```

## 11. Reproducibility

Reproducibility was an important consideration throughout the project.

We used a consistent train/test split, fixed random seeds, pinned dependencies, and automated model training.

The same training script can reproduce either model version using the MODEL_VERSION environment variable.

GitHub Actions also provides a record of workflow executions, allowing training results and build outcomes to be reviewed.

## 12. Limitations

This project uses a publicly available research dataset rather than actual patient records.

The predicted progression score should therefore be viewed as a demonstration of the machine learning workflow rather than a clinically validated risk estimate.

Additionally, the improvement between the baseline and updated model was relatively small.

Version 0.2 adds a high-risk flag based on the 75th percentile of the training target distribution, together with precision and recall for that flag in `models/metrics.json`. However, the threshold was derived from the same dataset rather than clinically validated guidelines, so the flag is intended for demonstration purposes only.

The application is intended for educational use only.

## 13. Contributors

This project was developed collaboratively as part of the MLOps assignment.

Individual contributions and changes can be reviewed through the GitHub commit history.
