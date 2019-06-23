#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Benchmark the processing time of the classify inference.

    Copyright (c) 2019 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import statistics
import argparse

from edgetpu.classification.engine import ClassificationEngine
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", help="File path of Tflite model.", required=True)
    parser.add_argument(
        "--image", help="File path of the image to be recognized.", required=True
    )
    parser.add_argument(
        "--num", help="Number of inference executions.", default=100, type=int
    )
    args = parser.parse_args()

    # Initialize engine.
    engine1 = ClassificationEngine(args.model1)

    # Run inference.
    inference_time1 = []

    for i in range(num + 1):
        img = Image.open(args.image)
        result1 = engine1.ClassifyWithImage(img, top_k=3)

        if i > 0:
            # Get Inference time.
            inference_time1 = engine1.get_inference_time()
            inference_time1.append(inference_time1)

    # Print avg
    print("Model inference time avg: {0:.4f}".format(statistics.mean(inference_time1)))


if __name__ == "__main__":
    main()

