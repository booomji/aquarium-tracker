from flask import Flask, jsonify
from flask_cors import CORS
import threading
import webbrowser
import http.server
import socketserver
import cv2
import numpy as np
import onnxruntime as ort

# === Flask App ===
app = Flask(__name__, static_folder='static')
CORS(app)

# === Load MiDaS ONNX model ===
model_path = "models/midas_v21_small_256.onnx"
#session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])#old cpu execution provider

input_name = session.get_inputs()[0].name
print("[INFO] Loaded MiDaS model using ONNX Runtime.")

def estimate_depth(frame):
    # Resize and normalize
    resized = cv2.resize(frame, (256, 256))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

    # Run model
    depth_map = session.run(None, {input_name: img})[0]
    depth_map = np.squeeze(depth_map)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    return depth_map

# === Face Tracking ===
face_coords = {"x": 0.0, "y": 0.0}

def track_face():
    global face_coords
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        flip_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            fx = (x + w/2) / frame.shape[1]
            fy = (y + h/2) / frame.shape[0]

            depth_map = estimate_depth(flip_frame)
            face_depth = float(depth_map[int(y + h / 2), int(x + w / 2)])
            norm_depth = np.clip((face_depth - depth_map.min()) / (depth_map.max() - depth_map.min()), 0.0, 1.0)

            face_coords = {
                "x": fx,
                "y": fy,
                "depth": norm_depth
            }

        depth_colored = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = depth_colored.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_MAGMA)

        overlay = cv2.addWeighted(flip_frame, 0.6, depth_colored, 0.4, 0)
        cv2.imshow("Face + Depth Overlay", overlay)

        #cv2.imshow("Face Tracking", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

""" @app.route("/position")
def position():
    return jsonify(face_coords) """

@app.route("/position")
def position():
    serializable_coords = {k: float(v) for k, v in face_coords.items()}
    return jsonify(serializable_coords)


# === Frontend Server ===
def start_frontend():
    PORT = 8000
    Handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory="static", **kwargs)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving frontend at http://localhost:{PORT}/index.html")
        httpd.serve_forever()

# === Main Entrypoint ===
if __name__ == "__main__":
    threading.Thread(target=track_face, daemon=True).start()
    threading.Thread(target=lambda: app.run(debug=False, port=5000), daemon=True).start()
    threading.Thread(target=start_frontend, daemon=True).start()
    webbrowser.open("http://localhost:8000/index.html")
    input("Press Enter to exit...\n")
