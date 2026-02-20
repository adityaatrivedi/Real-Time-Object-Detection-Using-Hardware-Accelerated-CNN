import os
import requests

# Create folder
img_dir = "calibration_images"
os.makedirs(img_dir, exist_ok=True)

# Image URLs (Standard YOLO samples)
urls = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/e4/Cars_in_traffic_in_Auckland_Gand_Street_4096x2730.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/60/Willys_Jeep_Truck.jpg"
]

print("Downloading images...")
for i, url in enumerate(urls):
    try:
        response = requests.get(url, timeout=10)
        with open(f"{img_dir}/image_{i}.jpg", "wb") as f:
            f.write(response.content)
        print(f"Downloaded image_{i}.jpg")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("Done! Images saved in 'calibration_images/'")