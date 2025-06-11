import requests

# Updated URL with raw download enabled
url = "https://github.com/opencv/opencv_zoo/blob/main/models/midas/midas_small.onnx?raw=true"
output_file = "midas_small.onnx"

print("Downloading MiDaS small model...")

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed: midas_small.onnx")
else:
    print(f"Failed to download. Status code: {response.status_code}")
