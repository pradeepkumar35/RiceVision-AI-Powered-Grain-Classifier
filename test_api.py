import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('grain.jpg', 'rb')}
response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response Text:", response.text)  # Print raw response
try:
    print("Response JSON:", response.json())  # Try parsing JSON
except ValueError:
    print("Response is not valid JSON")
