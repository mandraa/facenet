import requests
import base64
import json

# Baca file gambar
with open("FOTO ANDRA 2.jpg", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Kirim ke endpoint process-image
response = requests.post(
    'http://127.0.0.1:5000/process-image',
    json={"image": image_data}
)

# Cetak hasil
print("Status Code:", response.status_code)
print("Response:", response.json())