from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Zeno's ML model API!"}


def test_api_locally_get_post_ok():
    # Given payload from the cURL command
    payload = {
        "age": [19, 37, 81],
        "capital-gain": [0, 0, 1887],
        "capital-loss": [0, 1567, 76],
        "education": ["Some-college", "HS-grad", "Some-college"],
        "education-num": [10, 9, 10],
        "fnlgt": [212800, 185556, 182898],
        "hours-per-week": [40, 35, 40],
        "marital-status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse"],
        "native-country": ["United-States", "Italy", "United-States"],
        "occupation": ["Sales", "Craft-repair", "Adm-clerical"],
        "race": ["White", "White", "Asian"],
        "relationship": ["Own-child", "Husband", "Other-relative"],
        "salary": ["<=50K", "<=50K", ">50K"],
        "sex": ["Female", "Male", "Female"],
        "workclass": ["Private", "Private", "Federal-gov"],
    }

    # Make POST request to the endpoint
    r = client.post("/infer/", json=payload)
    assert r.status_code == 200
    assert r.json() == {"predictions": "[0 1 0]"}


def test_api_locally_get_post_not_ok():
    # Given payload from the cURL command
    payload = {"age": [19, 37, 81]}

    # Make POST request to the endpoint
    r = client.post("/infer/", json=payload)
    assert r.status_code != 200
