import onnxruntime as ort

model_path = "models/midas_v21_small_256.onnx"

print("Available providers:", ort.get_available_providers())
session = ort.InferenceSession("midas_v21_small_256.onnx", providers=["CUDAExecutionProvider"])
print("Using CUDA provider!")
