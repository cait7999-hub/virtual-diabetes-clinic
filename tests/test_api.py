import json
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8080"


def post_prediction(data):
    request = urllib.request.Request(
        BASE_URL + "/predict",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=5)


def main():
    # Test /health
    with urllib.request.urlopen(BASE_URL + "/health") as response:
        health = json.load(response)
        assert response.status == 200
        assert health["status"] == "ok"
        assert health["model_version"] == "v0.2"

    # Example patient
    patient = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }

    # Test /predict
    with post_prediction(patient) as response:
        result = json.load(response)
        assert response.status == 200
        assert isinstance(result["prediction"], (int, float))

    # Test missing feature
    del patient["s6"]

    try:
        post_prediction(patient)
        raise AssertionError("Expected HTTP 422")
    except urllib.error.HTTPError as error:
        assert error.code == 422
        details = json.load(error)
        assert "s6" in str(details)

    print("All API acceptance tests passed!")


if __name__ == "__main__":
    main()
