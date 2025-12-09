import requests
import random
api_url="http://localhost:8000/predict" # Replace with your actual API endpoint
def generate_random_input():
    seq = []
    for _ in range(24):
        timestep = [
            random.uniform(280, 310),   # temp
            random.uniform(0, 50),     # rain_1h
            random.uniform(0, 20),     # snow_1h
            random.uniform(0, 100),    # clouds_all
            random.randint(0, 23),     # hour
            random.randint(0, 6),      # dayofweek
            random.randint(1, 12)      # month
        ]
        seq.append(timestep)
    return {"seq": seq}
print(generate_random_input())
print("Sending test request to the API...")


response = requests.post(api_url, json=generate_random_input())
if response.status_code == 200:
    print("Response from API:")
    print(response.json()) 
else:
    print(f"Failed to get a valid response. Status code: {response.status_code}")
    print(response.text)