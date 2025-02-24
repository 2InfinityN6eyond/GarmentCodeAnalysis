import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from glob import glob
import math
from pprint import pprint
import torch

import pygarment as pyg
import numpy as np
import matplotlib.pyplot as plt
import json
import svgpathtools
from svgpathtools import Path, Line
from matplotlib.colors import CenteredNorm
import time

import plotly.graph_objects as go
import trimesh
import pickle
from PIL import Image
import random
import cv2
from tqdm import tqdm

def adaptive_simplify_line(points, angle_threshold=5.0):
    """
    Simplifies a 2D point line adaptively based on curvature.

    Args:
        points: List of 2D points (N x 2 array).
        angle_threshold: Minimum angle (degrees) to retain a point in straight sections.

    Returns:
        Simplified list of points.
    """
    points = np.array(points)  # Convert to NumPy array
    simplified = [points[0]]  # Always keep the first point

    for i in range(1, len(points) - 1):
        # Get three consecutive points
        p1, p2, p3 = points[i - 1], points[i], points[i + 1]

        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Calculate angle between vectors (in degrees)
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure numerical stability
        angle = np.degrees(np.arccos(cosine_angle))  # Convert to degrees

        # Retain point if angle exceeds threshold (curved region)
        if angle > angle_threshold:
            simplified.append(p2)

    simplified.append(points[-1])  # Always keep the last point
    return np.array(simplified)

render_props = {
    "resolution": [800, 800],
    "sides": ["front", "right", "back", "left"],
    "front_camera_location": [0, 0.97, 4.15],
    "uv_texture": {
        "seam_width": 0.5,
        "dpi": 1500,
        "fabric_grain_texture_path": "./assets/img/fabric_texture.png",
        "fabric_grain_resolution": 5
    }
}

PROJECT_ROOT_PATH = "/media/hjp/05aba9a7-0e74-4e54-9bc9-5f11b9c4c757/GarmentCodeData/"
GARMENT_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, "GarmentCodeData_v2")
BODY_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, "body_mesh")
MEAN_ALL_BODY_PATH = os.path.join(PROJECT_ROOT_PATH, "neutral_body/mean_all.obj")

default_body_mesh = trimesh.load(MEAN_ALL_BODY_PATH)

# BODY_TYPE = "random_body"
BODY_TYPE = "default_body"

garment_path_list = []
for path in  sorted(glob(os.path.join(GARMENT_ROOT_PATH, "*", BODY_TYPE, "*"))):
    if not os.path.isdir(path):
        continue
    dir_content = os.listdir(path)
    garmetn_id = os.path.basename(path)
    
    if (
        "combined_image_seam_info.json" in dir_content
    ) and (
        "combined_image_seam_info_1000_norm.json" in dir_content
    ) and (
        "combined_image.png" in dir_content
    ):
        garment_path_list.append(path)
        
        
        
with tqdm(garment_path_list) as pbar:
    for garment_path in garment_path_list:
        garment_id = os.path.basename(garment_path)
        pbar.set_description(f"Processing {garment_id}")

        SPEC_FILE_PATH = os.path.join(garment_path, f"{garment_id}_specification.json")
        pattern = pyg.pattern.wrappers.VisPattern(SPEC_FILE_PATH)

        panel_svg_path_dict = {
            panel_name : pattern._draw_a_panel(
                panel_name, apply_transform=False, fill=True
            )
            for panel_name in pattern.panel_order()
        }
        stitch_dict = {
            i : v for i, v in enumerate(pattern.pattern['stitches'])
        }

        rendered_image_list = list(map(
            lambda x : Image.open(os.path.join(garment_path, f"{garment_id}_render_{x}.png")),
            render_props["sides"]
        ))


        with open(os.path.join(garment_path, "combined_image_seam_info.json"), "r") as f:
            combined_seam_dict = json.load(f)
        with open(os.path.join(garment_path, "combined_image_seam_info_1000_norm.json"), "r") as f:
            combined_seam_dict_1000_norm = json.load(f)

        final_image = Image.open(
            os.path.join(garment_path, "combined_image.png")
        )
        vis_image = Image.open(
            os.path.join(garment_path, "combined_image_with_seam.png")
        )
        vis_image_2 = Image.open(
            os.path.join(garment_path, "combined_image_seam_only.png")
        )


        combined_seam_dict_norm_trimed_dict = {}
        for side, stitch_dict in combined_seam_dict_1000_norm.items():
            combined_seam_dict_norm_trimed_dict[side] = {}
            for stitch_idx, vertices in stitch_dict.items():
                combined_seam_dict_norm_trimed_dict[side][int(stitch_idx)] = adaptive_simplify_line(vertices, angle_threshold=5.0)


        stitch_idx_map = {}
        florence_input_prototype = []
        for idx, (side, stitch_dict) in enumerate(combined_seam_dict_norm_trimed_dict.items()):
            
            florence_input_prototype.append(f"<side_{idx}>")
            
            for stitch_idx, vertices in stitch_dict.items():
                if stitch_idx not in stitch_idx_map:
                    stitch_idx_map[stitch_idx] = len(stitch_idx_map)
                florence_input_prototype.extend(
                    [
                        f"<stitch_{stitch_idx_map[stitch_idx]}>",
                        *list(map(
                            lambda x : f"<loc_{x}>",
                            vertices.flatten()
                        ))
                    ]
                )
                
        with open(os.path.join(
            garment_path,
            "florence_input_prototype.txt"
        ), "w") as f:
            json.dump(florence_input_prototype, f, indent=4)
            
        pbar.update(1)