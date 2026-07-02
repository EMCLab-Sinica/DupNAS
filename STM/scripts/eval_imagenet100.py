import argparse
import numbers

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from PIL import Image


DATASET = "clane9/imagenet-100"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def image_size(input_shape):
    if len(input_shape) == 4:
        height, width = input_shape[2], input_shape[3]
        if isinstance(height, numbers.Integral) and height == width:
            return int(height)
    raise ValueError(f"Cannot infer image size from input shape: {input_shape}")


def preprocess(image, size):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((image - MEAN) / STD)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file")
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx_file, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    size = image_size(model_input.shape)
    dataset = load_dataset(DATASET, split="validation")

    correct = total = 0
    for item in dataset:
        pred = session.run(None, {model_input.name: preprocess(item["image"], size)})[0].argmax()
        correct += int(pred == item["label"])
        total += 1

    print(f"accuracy: {correct / total:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()
