import requests

url = "http://127.0.0.1:5000/analyze"
files = {
    'input_file': open('/Users/gaungalif/Documents/ADA/MC3/Intonation/input_2.wav', 'rb'),
    'reference_file': open('/Users/gaungalif/Documents/ADA/MC3/Intonation/reference.wav', 'rb')
}

response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print("Response from server:")
    print(result)
else:
    print(f"Failed to get response. Status code: {response.status_code}")
    print(response.text)
