import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Define the API endpoint URL
url = "https://final-project-udacity.onrender.com/infer/"  # Replace this with the actual URL of your API endpoint

# Define the payload data
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

# Send POST request to the endpoint
response = requests.post(url, json=payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    print("Request successful!")
    print("Response:")
    print(response.json())
else:
    print("Request failed with status code:", response.status_code)
