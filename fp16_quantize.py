import onnx
from onnxconverter_common import float16

model = onnx.load("arcface.onnx")
model_fp16 = float16.convert_float_to_float16(model)

onnx.save(model_fp16, "arcface_fp16.onnx")
print("FP16 model saved as arcface_fp16.onnx")
