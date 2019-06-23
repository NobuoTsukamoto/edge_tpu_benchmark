#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Benchmark the processing time of the first classify inference.

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
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument(
        "--image", help="File path of the image to be recognized.", required=True
    )
    parser.add_argument(
        "--num", help="Number of inference executions.", default=10, type=int
    )
    args = parser.parse_args()

    inference_time = []
    img = Image.open(args.image)

    for i in range(args.num):
        # Initialize engine.
        engine = ClassificationEngine(args.model)

        # Run inference.
        result1 = engine.ClassifyWithImage(img, top_k=3)

        # Get Inference time.
        inference_time.append(engine.get_inference_time())

        # delete Engine
        del engine

    # Print avg
    print("Model inference time avg: {0:.4f}".format(statistics.mean(inference_time)))


if __name__ == "__main__":
    main()
