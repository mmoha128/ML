import requests
import urllib3

url = 'http://127.0.0.1:5000/predict'
data = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023809,
    "Population": 322,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post(url, json=data)
print(response.json())


