import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict' #http://127.0.0.1:9696
response = requests.post(url, json=ride)
print(response.json())