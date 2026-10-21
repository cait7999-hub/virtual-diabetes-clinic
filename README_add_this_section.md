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

### Run tests locally

```bash
pip install -r requirements-dev.txt
ruff check .
MODEL_VERSION=v0.2 python train.py
pytest
```
