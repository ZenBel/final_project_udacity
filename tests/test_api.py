from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Zeno's ML model API!"}


def test_api_locally_get_post_ok_0():
    # Given payload from the cURL command
    payload = {
        "age": [19],
        "capital-gain": [0],
        "capital-loss": [0],
        "education": ["Some-college"],
        "education-num": [10],
        "fnlgt": [212800],
        "hours-per-week": [40],
        "marital-status": ["Never-married"],
        "native-country": ["United-States"],
        "occupation": ["Sales"],
        "race": ["White"],
        "relationship": ["Own-child"],
        "salary": ["<=50K"],
        "sex": ["Female"],
        "workclass": ["Private"],
    }

    # Make POST request to the endpoint
    r = client.post("/infer/", json=payload)
    assert r.status_code == 200
    assert r.json() == {"predictions": "[0]"}


def test_api_locally_get_post_ok_1():
    # Given payload from the cURL command
    payload = {
        "age": [37],
        "capital-gain": [0],
        "capital-loss": [1567],
        "education": ["HS-grad"],
        "education-num": [9],
        "fnlgt": [185556],
        "hours-per-week": [35],
        "marital-status": ["Married-civ-spouse"],
        "native-country": ["Italy"],
        "occupation": ["Craft-repair"],
        "race": ["White"],
        "relationship": ["Husband"],
        "salary": ["<=50K"],
        "sex": ["Male"],
        "workclass": ["Private"],
    }

    # Make POST request to the endpoint
    r = client.post("/infer/", json=payload)
    assert r.status_code == 200
    assert r.json() == {"predictions": "[1]"}


def test_api_locally_get_post_not_ok():
    # Given payload from the cURL command
    payload = {"age": [19, 37, 81]}

    # Make POST request to the endpoint
    r = client.post("/infer/", json=payload)
    assert r.status_code != 200
