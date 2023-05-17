import argparse
import os
from pathlib import Path
import time
import cv2
import numpy as np
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
import torch
import yaml
from SuperPointFrontend import SuperPointFrontend
from PointTracker import PointTracker, myjet
from VideoStreamer import VideoStreamer


def get_args():
    parser = argparse.ArgumentParser(description="HRL")
    parser.add_argument("--model-path",
                        default="",
                        help='name of pretrained model (default: "")' )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    ONNX_PATH = Path("./tmp.onnx")
    device = torch.device('cpu')

    # Load the network in inference mode.
    net = SuperPointNet_gauss2()
    checkpoint = torch.load(f"{args.model_path}", map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    dummy_input = torch.randn(1, 1, 640, 480)
    torch.onnx.export(
        net,
        dummy_input,
        f"{ONNX_PATH}",
        input_names=["image"],
        output_names=["semi", "desc"],
        dynamic_axes={
            # dict value: manually named axes
            "image": {2: "height",
                      3: "width"},
            "semi": {2: "height",
                     3: "width"},
            "desc": {2: "height",
                     3: "width"}
        },
        verbose=True
    )
