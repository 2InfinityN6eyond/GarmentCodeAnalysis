import os, sys
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
            

def is_point_inside_path(path: Path, point: complex) -> bool:
    """
    Determine if a point is inside the path using ray casting.
    """
    ray_end = point + 1000 + 0j  # Cast ray in positive x direction
    intersections = 0
    
    for segment in path:
        ray = Line(point, ray_end)
        crossings = segment.intersect(ray)
        # crossings returns list of (t1, t2) tuples
        # t1 is parameter for first curve, t2 for second curve
        # We only care about t1 values between 0 and 1
        intersections += sum(1 for t1, t2 in crossings if 0 <= t1 <= 1)
    
    return intersections % 2 == 1

def compute_signed_distance_grid(
    path: svgpathtools.Path,
    n_samples: int = 1000,
    grid_size: int = 200,
    SDF_DOMAIN_SIZE: float = 2,
    ZOOM_OUT_FACTOR: float = 1.2,
    K_NEIGHBORS: int = 5
) -> torch.Tensor:
    """
    Compute signed distance field for an SVG path.
    args :
        path : svgpathtools.Path
            unnormalized closed path
        n_samples : int, number of samples along the path

        img_size : int, size of the grid
    """
    edge_lengths = [segment.length() for segment in path]
    total_length = sum(edge_lengths)
    
    # Distribute points proportionally to edge lengths
    points_per_edge = [
        max(int(n_samples * length / total_length), 10)  # Ensure minimum 10 points per edge
        for length in edge_lengths
    ]
    
    edge_points = []
    for edge_idx, segment in enumerate(path):
        t_vals = torch.linspace(0, 1, points_per_edge[edge_idx])
        edge_samples = torch.tensor([
            [segment.point(t.item()).real, segment.point(t.item()).imag]
            for t in t_vals
        ])
        edge_points.append(edge_samples)
    
    # Combine all points and create edge index mapping
    boundary_points = torch.cat(edge_points, dim=0)
    edge_indices = torch.cat([
        torch.full((points_per_edge[i],), i, dtype=torch.int64)
        for i in range(len(path))
    ])
    
    # get bounding box of the path
    xmin, xmax, ymin, ymax = path.bbox()
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    x_scale = xmax - xmin
    y_scale = ymax - ymin
    
    scale_factor = max(x_scale, y_scale) * ZOOM_OUT_FACTOR / SDF_DOMAIN_SIZE
    
    boundary_points_normalized = (
        boundary_points - torch.tensor([x_center, y_center])
    ) / scale_factor
     
    # get grid of evaluation points
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, grid_size),
        torch.linspace(-1, 1, grid_size)
    ), dim=-1).reshape(-1, 2)[:, [1, 0]].float()
    
    # Compute unsigned distances
    coords_expanded = coords.unsqueeze(1)
    points_expanded = boundary_points_normalized.unsqueeze(0)
    distances = torch.norm(coords_expanded - points_expanded, dim=2)
    unsigned_distances, min_indices = torch.min(distances, dim=1)
    
    k_distances, k_indices = torch.topk(
        distances, k=K_NEIGHBORS, dim=1, largest=False
    )
    k_edge_indices = edge_indices[k_indices]
    weights = 1.0 / (k_distances + 1e-6)

    n_edges = len(path)

    one_hot = torch.zeros(
        k_edge_indices.shape[0], K_NEIGHBORS, n_edges
    )
    one_hot.scatter_(2, k_edge_indices.unsqueeze(-1), 1)
    
    # Apply weights to votes
    weighted_votes = one_hot * weights.unsqueeze(-1)
    
    # Sum votes for each edge
    edge_votes = weighted_votes.sum(dim=1)
    
    # Get edge with maximum votes
    closest_edge_indices = edge_votes.argmax(dim=1)
    
    # closest_edge_indices = edge_indices[min_indices]
    
    # For sign computation, transform coords back to original space
    signs = torch.tensor([
        -1 if is_point_inside_path(
            path, complex(x.item(), y.item())
        ) else 1
        for x, y in coords * scale_factor + torch.tensor([x_center, y_center])
    ])
    signed_distances = unsigned_distances * signs
    
    return (
        signed_distances.reshape(grid_size, grid_size),
        closest_edge_indices.reshape(grid_size, grid_size),
        scale_factor
    )


def make_signed_distance_data(
    DATAPOINT_PATH: str,
    GRID_SIZE: int = 256,
    N_SAMPLES: int = 4000,
    SDF_DOMAIN_SIZE: float = 2,
    ZOOM_OUT_FACTOR: float = 1.1,
    K_NEIGHBORS: int = 5,
) :
    GARMENT_ID = os.path.basename(DATAPOINT_PATH)
    SPEC_FILE_PATH = os.path.join(DATAPOINT_PATH, f"{GARMENT_ID}_specification.json")
    pattern = pyg.pattern.wrappers.VisPattern(SPEC_FILE_PATH)
    
    panel_name_list = pattern.panel_order()
    
    panel_svg_path_dict = {
        panel_name : pattern._draw_a_panel(
            panel_name, apply_transform=False, fill=True
        )
        for panel_name in panel_name_list
    }
    stitch_dict = {
        i : v for i, v in enumerate(pattern.pattern['stitches'])
    }
    
    
    signed_distance_data_dict = {
        "grid_size" : GRID_SIZE,
        "n_samples" : N_SAMPLES,
        "panel_name_list" : panel_name_list,
        "sdf_domain_size" : SDF_DOMAIN_SIZE,
        "zoom_out_factor" : ZOOM_OUT_FACTOR,
        "k_neighbors" : K_NEIGHBORS,
        "scale_factor_list" : [],
    }
    
    signed_distance_grid_list = []
    edge_indices_grid_list = []
    
    for panel_name in panel_name_list:
        panel_svg_path = panel_svg_path_dict[panel_name][0]
        sd_grid, edge_indices_grid, scale_factor = compute_signed_distance_grid(
            panel_svg_path,
            n_samples=N_SAMPLES,
            grid_size=GRID_SIZE,
            ZOOM_OUT_FACTOR=ZOOM_OUT_FACTOR,
            K_NEIGHBORS=K_NEIGHBORS
        )
        signed_distance_data_dict["scale_factor_list"].append(scale_factor)
        signed_distance_grid_list.append(sd_grid)
        edge_indices_grid_list.append(edge_indices_grid)
        
    signed_distance_grid_array = np.array(signed_distance_grid_list)
    edge_indices_grid_array = np.array(edge_indices_grid_list)
    

    SIGNED_DISTANCE_METADATA_PATH = os.path.join(
        DATAPOINT_PATH, "signed_distance_metadata.json"
    )
    with open(SIGNED_DISTANCE_METADATA_PATH, "w") as f:
        json.dump(signed_distance_data_dict, f, indent=4)
        
    SIGNED_DISTANCE_GRID_PATH = os.path.join(
        DATAPOINT_PATH, "signed_distance_grid.npy"
    )
    np.save(SIGNED_DISTANCE_GRID_PATH, signed_distance_grid_array)

    CLOSEST_EDGE_INDICES_GRID_PATH = os.path.join(
        DATAPOINT_PATH, "closest_edge_indices_grid.npy"
    )
    np.save(CLOSEST_EDGE_INDICES_GRID_PATH, edge_indices_grid_array)
    
    
    
if __name__ == "__main__":
    
    from tqdm import tqdm
    
    print("processing under ", sys.argv[1])
    
    params = {
        "GRID_SIZE": 256,
        "N_SAMPLES": 4000,
        "ZOOM_OUT_FACTOR": 1.1,
        "K_NEIGHBORS": 5,
    }
    
       
    datapoint_paths = sorted(list(filter(
        lambda x: (
            os.path.isdir(x)
        ) and (
            "rand" in os.path.basename(x)
        ) and (
            f"{os.path.basename(x)}_specification.json" in os.listdir(x)
        ),
        glob(f"{sys.argv[1]}/default_body/*")
    )))
    
    print(len(datapoint_paths))
    
    
    for datapoint_path in tqdm(datapoint_paths):
        make_signed_distance_data(
            datapoint_path,
            **params
        )
        
    