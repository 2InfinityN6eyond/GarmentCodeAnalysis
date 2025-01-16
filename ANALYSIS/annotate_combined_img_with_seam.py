import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from glob import glob

import pygarment as pyg
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from PIL import Image
import cv2

from tqdm import tqdm

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

X_MIN_LIST = [150, 200, 150, 200]
WIDTH_LIST = [500, 400, 500, 400]

Y_MIN_LIST = [100, 100, 100, 100]
HEIGHT_LIST = [700, 700, 700, 700]

X_OFFSET_LIST = [0, 500, 0, 500]
Y_OFFSET_LIST = [0, 0, 700, 700]



PROJECT_ROOT_PATH = "/media/hjp/05aba9a7-0e74-4e54-9bc9-5f11b9c4c757/GarmentCodeData/"
GARMENT_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, "GarmentCodeData_v2")
BODY_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, "body_mesh")
MEAN_ALL_BODY_PATH = os.path.join(PROJECT_ROOT_PATH, "neutral_body/mean_all.obj")
BODY_TYPE = "default_body"

garment_path_list = sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_0", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_3", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_4", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_5", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_6", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_7", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_8", BODY_TYPE, "*"))
))) + sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_9", BODY_TYPE, "*"))
)))

garment_path_list = sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "garments_5000_3*", BODY_TYPE, "*"))
)))


with tqdm(total=len(garment_path_list)) as pbar:
    for garment_path in garment_path_list:
        garment_id = os.path.basename(garment_path)
        
        pbar.set_description(f"{garment_id}")

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

        final_image = np.vstack([
            np.hstack([
                np.array(rendered_image_list[0])[Y_MIN_LIST[0]:Y_MIN_LIST[0] + HEIGHT_LIST[0], X_MIN_LIST[0]:X_MIN_LIST[0] + WIDTH_LIST[0]],
                np.array(rendered_image_list[1])[Y_MIN_LIST[1]:Y_MIN_LIST[1] + HEIGHT_LIST[1], X_MIN_LIST[1]:X_MIN_LIST[1] + WIDTH_LIST[1]]
            ]),
            np.hstack([
                np.array(rendered_image_list[2])[Y_MIN_LIST[2]:Y_MIN_LIST[2] + HEIGHT_LIST[2], X_MIN_LIST[2]:X_MIN_LIST[2] + WIDTH_LIST[2]], 
                np.array(rendered_image_list[3])[Y_MIN_LIST[3]:Y_MIN_LIST[3] + HEIGHT_LIST[3], X_MIN_LIST[3]:X_MIN_LIST[3] + WIDTH_LIST[3]]
            ])
        ])
        vis_image = final_image.copy()
        vis_image_2 = np.ones_like(final_image, dtype=np.uint8) * 255

        VISIBLE_SEAM_VERTEX_POSITION_DICT_PATH = os.path.join(
            garment_path, "visible_seam_vertex_position_dict.pkl"
        )
        visible_seam_vertices_dict = pickle.load(open(VISIBLE_SEAM_VERTEX_POSITION_DICT_PATH, "rb"))
        unique_stitch_indices = list(visible_seam_vertices_dict.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_stitch_indices)))
        unique_color_list = (colors[:, [2, 1, 0, 3]] * 255).astype(np.uint8).tolist()  # [:, :3] removes alpha, [:, ::-1] converts RGB to BGR;

        final_seam_vertices_dict = {}
        final_seam_vertices_dict_1000_norm = {}
        for side_idx, side in enumerate(render_props["sides"]):
            
            bbox_width = WIDTH_LIST[side_idx]
            bbox_height = HEIGHT_LIST[side_idx]
            
            x_offset = X_OFFSET_LIST[side_idx]
            y_offset = Y_OFFSET_LIST[side_idx]
            
            x_min = X_MIN_LIST[side_idx]
            y_min = Y_MIN_LIST[side_idx]
            
            final_seam_vertices_dict[side] = {}
            final_seam_vertices_dict_1000_norm[side] = {}
            
            for stitch_idx, visible_info in visible_seam_vertices_dict.items():
                if visible_info[side] is not None and len(visible_info[side]) > 0:
            
                    pixel_coords = visible_info[side] - np.array([x_min, y_min]) + np.array([x_offset, y_offset])
                    
                    final_seam_vertices_dict[side][stitch_idx] = pixel_coords.astype(np.int32).tolist()
                    final_seam_vertices_dict_1000_norm[side][stitch_idx] = np.around(
                        pixel_coords / np.array([final_image.shape[1], final_image.shape[0]]) * 1000,
                        decimals=0
                    ).astype(np.int32).tolist()
                    

                    cv2.putText(
                        vis_image,
                        str(stitch_idx),
                        org=np.array(pixel_coords).mean(axis=0).astype(np.int32),  # Offset text slightly from point
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=unique_color_list[stitch_idx],
                        thickness=1
                    )
                    
                    cv2.putText(
                        vis_image_2,
                        str(stitch_idx),
                        org=np.array(pixel_coords).mean(axis=0).astype(np.int32),  # Offset text slightly from point
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=unique_color_list[stitch_idx],
                        thickness=1
                    )
                    
                    # Draw circles and text for each point
                    for point in pixel_coords:
                        x, y = point.astype(np.int32)
                        # Draw circle
                        cv2.circle(
                            vis_image, 
                            center=(x, y),
                            radius=1,
                            color=unique_color_list[stitch_idx],
                            thickness=-1  # Filled circle
                        )

                    for point in pixel_coords:
                        x, y = point.astype(np.int32)
                        cv2.circle(
                            vis_image_2, 
                            center=(x, y),
                            radius=1,
                            color=unique_color_list[stitch_idx],
                            thickness=-1  # Filled circle
                        )

        # SAVE
        with open(os.path.join(garment_path, "combined_image_seam_info.json"), "w") as f:
            json.dump(final_seam_vertices_dict, f, indent=4)
        with open(os.path.join(garment_path, "combined_image_seam_info_1000_norm.json"), "w") as f:
            json.dump(final_seam_vertices_dict_1000_norm, f, indent=4)

        Image.fromarray(final_image).save(
            os.path.join(garment_path, "combined_image.png")
        )
        Image.fromarray(vis_image).save(
            os.path.join(garment_path, "combined_image_with_seam.png")
        )
        Image.fromarray(vis_image_2).save(
            os.path.join(garment_path, "combined_image_seam_only.png")
        )

        pbar.update(1)