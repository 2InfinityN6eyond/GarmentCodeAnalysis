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


import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1'  # Add this line
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '410'  # Add this line



import os
import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    
def rotate_matrix_y(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def rotate_matrix_x(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def create_lights(scene, intensity=30.0):
    light_positions = [
        np.array([1.60614, 1.5341, 1.23701]),
        np.array([1.31844, 1.92831, -2.52238]),
        np.array([-2.80522, 1.2594, 2.34624]),
        np.array([0.160261, 1.81789, 3.52215]),
        np.array([-2.65752, 1.41194, -1.26328])
    ]
    light_colors = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

    # Add lights to the scene
    for i in range(5):
        light = pyrender.PointLight(color=light_colors[i], intensity=intensity)
        light_pose = np.eye(4)
        light_pose[:3, 3] = light_positions[i]
        scene.add(light, pose=light_pose)



def create_camera(pyrender, pyrender_body_mesh, scene, side, camera_location=None):

    # Create a camera
    y_fov = np.pi / 6. 
    camera = pyrender.PerspectiveCamera(yfov=y_fov)
    

    if camera_location is None:
        # Evaluate w.r.t. body

        fov = 50  # Set your desired field of view in degrees 

        # # Calculate the bounding box center of the mesh
        bounding_box_center = pyrender_body_mesh.bounds.mean(axis=0)

        # Calculate the diagonal length of the bounding box
        diagonal_length = np.linalg.norm(pyrender_body_mesh.bounds[1] - pyrender_body_mesh.bounds[0])

        # Calculate the distance of the camera from the object based on the diagonal length
        distance = 1.5 * diagonal_length / (2 * np.tan(np.radians(fov / 2)))

        camera_location = bounding_box_center
        camera_location[-1] += distance

    # Calculate the camera pose
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_location[0]],
        [0.0, 1.0, 0.0, camera_location[1]],
        [0.0, 0.0, 1.0, camera_location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    camera_pose = rotate_matrix_x(camera_pose, -15)
    camera_pose = rotate_matrix_y(camera_pose, 20)
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)
    elif side == 'right':
        camera_pose = rotate_matrix_y(camera_pose, -90)
    elif side == 'left':
        camera_pose = rotate_matrix_y(camera_pose, 90)

    # Set camera's pose in the scene
    scene.add(camera, pose=camera_pose)

    
    
    
    
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


NUM = 24
STT = NUM
END = NUM + 6
JOIN_PATH_LIST = [
    f"garments_5000_{i}" for i in range(STT, END)
]

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

    garment_id = os.path.basename(garment_path)

    # SPEC_FILE_PATH = os.path.join(garment_path, f"{garment_id}_specification.json")

    # pattern = pyg.pattern.wrappers.VisPattern(SPEC_FILE_PATH)

    # panel_svg_path_dict = {
    #     panel_name : pattern._draw_a_panel(
    #         panel_name, apply_transform=False, fill=True
    #     )
    #     for panel_name in pattern.panel_order()
    # }
    # stitch_dict = {
    #     i : v for i, v in enumerate(pattern.pattern['stitches'])
    # }
    
    tex_image = PIL.Image.open(os.path.join(garment_path, f"{garment_id}_texture.png"))
    
    # with open(os.path.join(garment_path, f"{garment_id}_sim_segmentation.txt"), "r") as f:
    #     segmentation = f.readlines()
    
    
    simulated_garment_mesh = trimesh.load_mesh(
        os.path.join(garment_path, f"{garment_id}_sim.ply"),
        process=False
    )
    simulated_garment_mesh.vertices = simulated_garment_mesh.vertices / 100

    texture = trimesh.visual.TextureVisuals(
        simulated_garment_mesh.visual.uv,
        image=tex_image
    )
    simulated_garment_mesh.visual = texture

    idx_convert_map = v_id_map(simulated_garment_mesh.vertices)

    # ready pyrender meshes
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.0, 0.0, 0.0, 1.0),  # RGB color, Alpha
        metallicFactor=0.658,  # Range: [0.0, 1.0]
        roughnessFactor=0.5  # Range: [0.0, 1.0]
    )
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(default_body_mesh, material=body_material)





    material = simulated_garment_mesh.visual.material.to_pbr()
    material.baseColorFactor = [1., 1., 1., 1.]
    material.doubleSided = True  # color both face sides  

    white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
    white_back.paste(material.baseColorTexture)
    material.baseColorTexture = white_back.convert('RGB')  

    simulated_garment_mesh.visual.material = material

    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(
        simulated_garment_mesh, smooth=True
    ) 



    rendered_image_dict = {}
    depth_image_dict = {}
    projected_vertex_pose_dict = {}
    vertex_visibility_mask_dict = {}

    for side in render_props["sides"]:
        scene = pyrender.Scene(bg_color=(1., 1., 1., 0.))
        
        scene.add(pyrender_garm_mesh)
        scene.add(pyrender_body_mesh)

        create_camera(
            pyrender,
            pyrender_body_mesh, scene, side,
            camera_location=render_props["front_camera_location"]
        )
        create_lights(scene, intensity=80.)
        
        camera_node = list(filter(
            lambda x : x.camera is not None,
            scene.get_nodes()
        ))[-1]
        
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_props["resolution"][0],
            viewport_height=render_props["resolution"][1]
        )
        
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        color, depth = renderer.render(scene, flags=flags)
        
        rendered_image_dict[side] = Image.fromarray(color)
        depth_image_dict[side] = depth
        
        view_matrix = np.linalg.inv(scene.get_pose(camera_node))
        proj_matrix = camera_node.camera.get_projection_matrix(*render_props["resolution"])
        
        vertices_homog = np.hstack([simulated_garment_mesh.vertices, np.ones((simulated_garment_mesh.vertices.shape[0], 1))])
        
        view_proj = proj_matrix @ view_matrix
        projected = vertices_homog @ view_proj.T
        
        z_coords = projected[:, 2].copy()
        projected = projected[:, :3] / projected[:, 3:4]
        
        pixel_coords = np.zeros((projected.shape[0], 2))
        pixel_coords[:, 0] = (projected[:, 0] + 1.0) * render_props["resolution"][0] / 2.0
        pixel_coords[:, 1] = render_props["resolution"][1] - (projected[:, 1] + 1.0) * render_props["resolution"][1] / 2.0
        
        px = np.clip(pixel_coords[:, 0].astype(int), 0, render_props["resolution"][0] - 1)
        py = np.clip(pixel_coords[:, 1].astype(int), 0, render_props["resolution"][1] - 1)
        
        visibility_mask = (z_coords > 0) & \
                    (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < render_props["resolution"][0]) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < render_props["resolution"][1]) & \
                    (z_coords + 0.08 < depth[py, px]) # | (z_coords - 0.1 > depth[py, px])
        
        projected_vertex_pose_dict[side] = pixel_coords.tolist()
        vertex_visibility_mask_dict[side] = visibility_mask.tolist()
        
        renderer.delete()



    for side, depth_image in depth_image_dict.items():
        np.save(os.path.join(garment_path, f"{garment_id}_depth_{side}.npy"), depth_image)


    with open(os.path.join(garment_path, f"{garment_id}_projected_vertex_pose.json"), "w") as f:
        json.dump( projected_vertex_pose_dict, f)

    with open(os.path.join(garment_path, f"{garment_id}_vertex_visibility_mask.json"), "w") as f:
        json.dump(vertex_visibility_mask_dict, f)
