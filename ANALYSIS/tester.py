import os, sys
from glob import glob

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from env_constants import DATASET_ROOT, PYGARMENT_ROOT


GARMENT_ROOT_PATH = os.path.join(DATASET_ROOT, "GarmentCodeData_v2")
BODY_TYPE = "default_body"

print(GARMENT_ROOT_PATH)


subdir_list = sorted(os.listdir(GARMENT_ROOT_PATH))

count = 0
for subdir in subdir_list:
    garment_path_list = sorted(list(filter(
        os.path.isdir,
        glob(os.path.join(GARMENT_ROOT_PATH, subdir, BODY_TYPE, "*"))
    )))
    print(subdir, len(garment_path_list))
    count += len(garment_path_list)

print(count)


garment_path_list = sorted(list(filter(
    os.path.isdir,
    glob(os.path.join(GARMENT_ROOT_PATH, "*", BODY_TYPE, "*"))
)))

print(len(garment_path_list))
