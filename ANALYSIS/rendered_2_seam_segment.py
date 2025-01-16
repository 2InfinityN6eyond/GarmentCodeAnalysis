import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import sys
import yaml
import json
from pprint import pprint
from glob import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import pyrender

import pygarment as pyg
import open3d as o3d
import trimesh
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm
from PIL import Image
import PIL



MIN_CONSEC_VERT_TO_BE_SEGMENT = 2
# Connect Disconnected vertices, if length of disconnected vertices <= MIN_CONSEC_VERT_TO_DISCONNECT
MIN_CONSEC_VERT_TO_DISCONNECT = 2
    
    
def v_id_map(vertices): 
    v_map = [None] * len(vertices) 
    v_map[0] = 0 
    for i in range(1, len(vertices)): 
        if all(vertices[i - 1] == vertices[i]): 
            v_map[i] = v_map[i-1]   
        else: 
            v_map[i] = v_map[i-1] + 1 
    return v_map


render_props = {
    "resolution": [800, 800],
    "sides": ["front", "back", "right", "left"],
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


BODY_TYPE = "default_body"


NUM = 3
if len(sys.argv) > 1 :
    NUM = int(sys.argv[1])

STT = NUM
END = NUM + 3


JOIN_PATH_LIST = [
    f"garments_5000_{i}" for i in range(STT, END)
]

print(JOIN_PATH_LIST)

garment_path_list = []
for join_path in JOIN_PATH_LIST:
    garment_path_list.extend(
        sorted(list(filter(
            os.path.isdir,
            glob(os.path.join(GARMENT_ROOT_PATH, join_path, BODY_TYPE, "*"))
        )))
    )
    
# print(os.path.dirname(os.path.dirname(garment_path_list[0])))
print(garment_path_list[0])
print(garment_path_list[-1])


for garment_path in tqdm(garment_path_list):

    try :
        garment_id = os.path.basename(garment_path)
        
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
        
        with open(os.path.join(garment_path, f"{garment_id}_sim_segmentation.txt"), "r") as f:
            segmentation = list(map(
                lambda x : x.strip(),
                f.readlines()
            ))
        
        with open(os.path.join(garment_path, f"{garment_id}_projected_vertex_pose.json"), "r") as f:
            projected_vertex_pose_dict = json.load(f)

        with open(os.path.join(garment_path, f"{garment_id}_vertex_visibility_mask.json"), "r") as f:
            vertex_visibility_mask_dict = json.load(f)

        simulated_garment_mesh = trimesh.load_mesh(
            os.path.join(garment_path, f"{garment_id}_sim.ply"),
            process=False
        )
        simulated_garment_mesh.vertices = simulated_garment_mesh.vertices / 100

        idx_convert_map = np.array(v_id_map(simulated_garment_mesh.vertices))

        stitch_vertex_mask_dict = {}
        stitch_vertex_extended_mask_dict = {}
        for k in stitch_dict.keys():
            raw_mask = list(map(
                lambda x : True if f"stitch_{k}" in x.split(",") else False,
                segmentation
            ))
            base_mask = np.array(list(map(
                lambda idx : True if raw_mask[idx] else False,
                idx_convert_map
            )))
            stitch_vertex_mask_dict[k] = base_mask
            

        full_vertices = np.array(simulated_garment_mesh.vertices)
        full_edges = np.array(simulated_garment_mesh.edges)
        full_faces = np.array(simulated_garment_mesh.faces)

        filtered2full_idx_map = {}
        for idx, filtered_idx in enumerate(idx_convert_map):
            if filtered_idx in filtered2full_idx_map :
                filtered2full_idx_map[filtered_idx].append(idx)
            else:
                filtered2full_idx_map[filtered_idx] = [idx]
                
        filtered_idx_list = []
        filtered_vertices = []
        for full_idx, filtered_idx in enumerate(idx_convert_map):
            if filtered_idx in filtered_idx_list:
                continue
            filtered_idx_list.append(filtered_idx)
            filtered_vertices.append(full_vertices[full_idx])
        filtered_vertices = np.array(filtered_vertices)


        filtered_edges = []
        for orig_edge in full_edges:
            stt = idx_convert_map[orig_edge[0]]
            end = idx_convert_map[orig_edge[1]]
            filtered_edges.append([stt, end])
        filtered_edges = np.array(filtered_edges)

        filtered_faces = []
        for idx, orig_face in enumerate(full_faces):
            
            v1, v2, v3 = sorted([
                idx_convert_map[orig_face[0]],
                idx_convert_map[orig_face[1]],
                idx_convert_map[orig_face[2]]
            ])
            filtered_faces.append([v1, v2, v3])
        filtered_faces = np.array(filtered_faces)

        filtered_mesh = trimesh.Trimesh(
            vertices=filtered_vertices,
            edges=filtered_edges,
            faces=filtered_faces,
            process=False
        )


        filtered_stitch_vertex_mask_dict = {}
        for stitch_idx in stitch_vertex_mask_dict.keys():
            filtered_stitch_vertex_mask_dict[stitch_idx] = np.array(
                [False] * len(filtered_vertices)
            )
            for orig_vert_idx, val in enumerate(stitch_vertex_mask_dict[stitch_idx]):
                if val:
                    filtered_vert_idx = idx_convert_map[orig_vert_idx]
                    filtered_stitch_vertex_mask_dict[stitch_idx][filtered_vert_idx] = True


        fltrd_vis_vert_mask_dict = {}
        fltrd_proj_vert_pos_dict = {}

        idx_convert_map_arr = np.array(idx_convert_map)
        for side in vertex_visibility_mask_dict.keys():
            
            filtrd_idx_list = []
            
            fltrd_vis_mask = []
            fltrd_proj_vert_pos = []
            
            for orig_idx, fltrd_idx in enumerate(idx_convert_map_arr):
                if fltrd_idx in filtrd_idx_list:
                    continue
                filtrd_idx_list.append(fltrd_idx)
                fltrd_vis_mask.append(vertex_visibility_mask_dict[side][orig_idx])
                fltrd_proj_vert_pos.append(projected_vertex_pose_dict[side][orig_idx])

            fltrd_vis_vert_mask_dict[side] = fltrd_vis_mask
            fltrd_proj_vert_pos_dict[side] = fltrd_proj_vert_pos
            
        vis_sim_segment_pos_dict = {}
        for side in fltrd_vis_vert_mask_dict.keys():
            for stch_idx in filtered_stitch_vertex_mask_dict.keys():
                fltrd_stch_mask = filtered_stitch_vertex_mask_dict[stch_idx]
                fltrd_vis_mask = fltrd_vis_vert_mask_dict[side]
                
                mask = fltrd_stch_mask & fltrd_vis_mask
                    
        # Find vertices that belong to seam line is easier done in filtered mesh
        # first get idx of stitch vertices, and reorder them to construct a connected path

        fltrd_seam_line_dict = {}
        for fltrd_stch_idx in filtered_stitch_vertex_mask_dict.keys():
            fltrd_stch_vert_map = filtered_stitch_vertex_mask_dict[fltrd_stch_idx]
            fltrd_stch_vert_idx_arr = np.where(fltrd_stch_vert_map)[0]
            
            adj_dict = {}
            for v1, v2 in filtered_edges:
                if v1 in fltrd_stch_vert_idx_arr and v2 in fltrd_stch_vert_idx_arr:
                    if v1 not in adj_dict: adj_dict[v1] = set()
                    if v2 not in adj_dict: adj_dict[v2] = set()
                    adj_dict[v1].add(v2)
                    adj_dict[v2].add(v1)
            
            endpoints = [
                v for v in fltrd_stch_vert_idx_arr if len(adj_dict.get(v, set())) == 1
            ]
            if len(endpoints) != 2:
                
                print("stitch idx", fltrd_stch_idx)
                print(fltrd_stch_vert_idx_arr)
                print(f"Warning: Found {len(endpoints)} endpoints, expected 2. Path may not be linear.")
                continue
            
            seam_vert_idx_list = [endpoints[0]]
            while len(seam_vert_idx_list) < len(fltrd_stch_vert_idx_arr):
                current_vert = seam_vert_idx_list[-1]
                neighbors = adj_dict[current_vert]
                next_vert = next((v for v in neighbors if v not in seam_vert_idx_list), None)
                if next_vert is None:
                    break
                seam_vert_idx_list.append(next_vert)
            fltrd_seam_line_dict[fltrd_stch_idx] = seam_vert_idx_list    

        fltrd_vis_seam_line_dict = {}
        for side in fltrd_vis_vert_mask_dict.keys() :
            fltrd_vis_seam_line_dict[side] = {}
            vis_mask = np.array(fltrd_vis_vert_mask_dict[side])
            for stch_idx in fltrd_seam_line_dict.keys():
                
                fltrd_vis_seam_line_dict[side][stch_idx] = {}
                
                seam_vert_idx_list = fltrd_seam_line_dict[stch_idx]
                
                vis_list = []
                for seam_vert_idx in seam_vert_idx_list :
                    vis_list.append(vis_mask[seam_vert_idx])
                fltrd_vis_seam_line_dict[side][stch_idx]["raw_idx_list"] = seam_vert_idx_list
                fltrd_vis_seam_line_dict[side][stch_idx]["raw_vis_mask"] = vis_list
                vis_list = np.array(vis_list)
                # If length of disconnection between visible seam vertices
                # is less then MIN_CONSEC_VERT_TO_DISCONNECT,
                # consider the disconnection is connected
                # (which change invisible seam vertices to visible)
                idx = 0
                while idx < len(vis_list) :
                    if idx >= len(vis_list) - 2 :
                        break
                    while not (vis_list[idx] == True and vis_list[idx + 1] == False) :
                        if idx >= len(vis_list) - 2 :
                            break
                        idx += 1
                    if idx >= len(vis_list) - 2 :
                        break
                    window_end_idx = min(idx + MIN_CONSEC_VERT_TO_DISCONNECT + 1, len(vis_list) - 1)
                    for rid in range(window_end_idx, idx, -1) :
                        if vis_list[rid] == True :
                            vis_list[idx:rid] = True
                            break
                    idx = rid
                    
                    
                # If length of connection between visible seam vertices
                # is less then MIN_CONSEC_VERT_TO_BE_SEGMENT,
                # consider the connection is not a segment
                # (which change visible seam vertices to invisible)
                idx = 0
                while idx < len(vis_list) :
                    if vis_list[idx] != True :
                        idx += 1
                        continue
                    idx2 = idx + 1
                    while idx2 < len(vis_list) and vis_list[idx2] == True :
                        idx2 += 1
                    if idx2 - idx < MIN_CONSEC_VERT_TO_BE_SEGMENT :
                        vis_list[idx:idx2] = False
                    idx = idx2
                    
                line_segment_idx_list = []
                line_segment_pos_list = []
                if True in vis_list :
                    idx = vis_list.tolist().index(True)
                    while idx < len(vis_list) :
                        line_segment_idx = []
                        line_segment_pos = []
                        while idx < len(vis_list) and vis_list[idx] == True :
                            line_segment_idx.append(
                                seam_vert_idx_list[idx]
                            )
                            line_segment_pos.append(
                                fltrd_proj_vert_pos_dict[side][seam_vert_idx_list[idx]]
                            )
                            idx += 1
                        if len(line_segment_idx) > 0 :
                            line_segment_idx_list.append(line_segment_idx)
                            line_segment_pos_list.append(line_segment_pos)
                        idx += 1
                else :
                    line_segment_idx_list.append([])
                    
                fltrd_vis_seam_line_dict[side][stch_idx]["segment_idx_list"] = line_segment_idx_list
                fltrd_vis_seam_line_dict[side][stch_idx]["segment_pos_list"] = line_segment_pos_list
                
        ## Save
        with open(os.path.join(garment_path, "fltrd_vis_seam_line_dict.pkl"),"wb") as f:
            pickle.dump(fltrd_vis_seam_line_dict, f)


    except Exception as e:
        print(e)

