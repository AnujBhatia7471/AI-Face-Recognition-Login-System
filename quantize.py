from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "arcface.onnx"
model_int8 = "arcface_int8.onnx"

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

print("Quantized model saved as arcface_int8.onnx")
