import os 
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import time
import random
from pprint import pprint
from glob import glob
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import pandas as pd

from env_constants import DATASET_ROOT, PYGARMENT_ROOT


sys.path.append(PYGARMENT_ROOT)
import pygarment as pyg


if __name__ == "__main__" :
    # See Every Panel Name.
    # For each Panel Name, gather N_SAMPLE_PER_PANEL sample panels.
    
    METADATA_PATH = "garment_code_data_meta_df.csv"

    garment_code_data_df = pd.read_csv(
        os.path.join(PYGARMENT_ROOT, METADATA_PATH)
    )
    MAX_PANEL_COUNT = max(garment_code_data_df["panel_count"])
    
    print("Total Garment Count : ", len(garment_code_data_df))
    print("Max Panel Count : ", MAX_PANEL_COUNT)
    
    SAVE_DIR = "DATAs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Save Dir : ", os.path.abspath(SAVE_DIR))


    unique_panel_name_per_panel_count_dict = {}
    for panel_count in range(1, MAX_PANEL_COUNT + 1) :
        unique_panel_name_per_panel_count_dict[panel_count] = set()
        garment_path_list = garment_code_data_df[
            garment_code_data_df["panel_count"] <= panel_count
        ]["garment_path"].tolist()
        
        print(f"Processing Panel Count : {panel_count}, Garment Count : {len(garment_path_list)}")
        
        for garment_path_base in tqdm(garment_path_list) :
            garment_path = os.path.join(DATASET_ROOT, "GarmentCodeData_v2", garment_path_base)
            garment_id = os.path.basename(garment_path)

            SPEC_FILE_PATH = os.path.join(garment_path, f"{garment_id}_specification.json")
            pattern = pyg.pattern.wrappers.VisPattern(SPEC_FILE_PATH)

            # Get Garment Blueprint
            panel_svg_path_dict = {
                panel_name : pattern._draw_a_panel(
                    panel_name, apply_transform=False, fill=True
                )
                for panel_name in pattern.panel_order()
            }
            # stitch_dict = {
            #     i : v for i, v in enumerate(pattern.pattern['stitches'])
            # }

            for panel_name in panel_svg_path_dict.keys() :
                if panel_name not in unique_panel_name_per_panel_count_dict[panel_count] :
                    unique_panel_name_per_panel_count_dict[panel_count].add(panel_name)
                
        unique_panel_name_per_panel_count_dict[panel_count] = list(unique_panel_name_per_panel_count_dict[panel_count])

    pprint(unique_panel_name_per_panel_count_dict)
    
    
    # Save Unique Panel Name Per Panel Count
    with open(os.path.join(SAVE_DIR, "unique_panel_name_per_panel_count_dict.json"), "w") as f :
        json.dump(unique_panel_name_per_panel_count_dict, f)
