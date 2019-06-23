#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Benchmark the processing time of the classify multi model first time inference.

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
    parser.add_argument("--model1", help="File path of Tflite model 1.", required=True)
    parser.add_argument("--model2", help="File path of Tflite model 2.", required=True)
    parser.add_argument(
        "--image", help="File path of the image to be recognized.", required=True
    )
    parser.add_argument(
        "--num", help="Number of inference executions.", default=10, type=int
    )
    args = parser.parse_args()

    inference_time1 = []
    inference_time2 = []
    img = Image.open(args.image)

    for i in range(args.num):
        # Initialize engine.
        engine1 = ClassificationEngine(args.model1)
        engine2 = ClassificationEngine(args.model2)

        # Run inference.
        result1 = engine1.ClassifyWithImage(img, top_k=3)
        result2 = engine2.ClassifyWithImage(img, top_k=3)

        # Get Inference time.
        inference_time1.append(engine1.get_inference_time())
        inference_time2.append(engine2.get_inference_time())

        # delete Engine
        del engine1
        del engine2

    # Print avg
    print("Model1 inference time avg: {0:.4f}".format(statistics.mean(inference_time1)))
    print("Model2 inference time avg: {0:.4f}".format(statistics.mean(inference_time2)))


if __name__ == "__main__":
    main()

