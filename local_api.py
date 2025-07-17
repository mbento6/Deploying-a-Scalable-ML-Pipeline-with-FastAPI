import json

import requests

# Base URL of the FastAPI
url = "http://127.0.0.1:8000"

# Send GET request to the root endpoint
r = requests.get(url)

# Print status code and welcome message
print("GET Request:")
print(f"Status Code: {r.status_code}")
print(f"Response: {r.json()}")
print("\n" + "="*40 + "\n")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education_num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


# Send POST request to /data/
r = requests.post(f"{url}/data/", json=data)

# Print status code and result
print("POST Request:")
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()}")
