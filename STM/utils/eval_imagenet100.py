import argparse

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from PIL import Image


DATASET = "clane9/imagenet-100"
IMAGE_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = (np.asarray(image).astype(np.float32) / 255.0 - MEAN) / STD
    return image.transpose(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file")
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx_file, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dataset = load_dataset(DATASET, split="validation")

    correct = total = 0
    for item in dataset:
        pred = session.run(None, {input_name: preprocess(item["image"])})[0].argmax()
        correct += int(pred == item["label"])
        total += 1

    print(f"accuracy: {correct / total:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()
