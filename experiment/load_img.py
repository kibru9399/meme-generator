from PIL import Image
import requests
from io import BytesIO

url = "https://upload.wikimedia.org/wikipedia/commons/5/5f/Nyc_Skyline_%28138268933%29.jpeg"
headers = {
    "User-Agent": "Mozilla/5.0"
}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img.show()
else:
    print(f"Failed to fetch image: {response.status_code}")
