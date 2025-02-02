import os, sys, socket
from glob import glob
hostname = socket.gethostname()
if hostname == "hjpui-MacBookPro.local":
    # PROJECT_ROOT    = "/Users/hjp/HJP/KUAICV/VTO/florence-tailor"
    DATASET_ROOT    = "/Users/hjp/HJP/KUAICV/VTO/DATASET/PoC59"
    PYGARMENT_ROOT  = "/Users/hjp/HJP/KUAICV/VTO/GarmentCodeAnalysis"
elif hostname == "epyc64": # A6000 Ada X 4
    # PROJECT_ROOT    = "/data/HJP/VTO2025/florence-tailor"
    # DATASET_ROOT    = "/home/hjp/VTO2025/GarmentCodeData"
    # DATASET_ROOT    = "/home/hjp/VTO2025/DATASET/PoC"
    DATASET_ROOT    = "/home/hjp/VTO2025/DATASET/PoC59"
    PYGARMENT_ROOT  = "/data/HJP/VTO2025/REFERENCES/3D_VTO/GarmentCode"
elif hostname == "server" : # 4090 X 4
    # PROJECT_ROOT    = "/media/hjp/db6095ca-a560-4c3a-90ad-b667ec189671/florence-tailor"
    DATASET_ROOT    = "/media/hjp/05aba9a7-0e74-4e54-9bc9-5f11b9c4c757/GarmentCodeData/GarmentCodeData_v2"
    PYGARMENT_ROOT  = "/media/hjp/db6095ca-a560-4c3a-90ad-b667ec189671/REFERENCES/3D_VTO/GarmentCode/GarmentCode"
elif hostname == "miracle" : # A6000 X 4
    # PROJECT_ROOT    = "/home/hjp/VTO/florence-tailor"
    # DATASET_ROOT    = "/ssd/HJP/CodeData"
    DATASET_ROOT    = "/ssd/HJP/CodeData2/GarmentCodeData_v2"
    PYGARMENT_ROOT  = "/home/hjp/VTO/GarmentCodeAnalysis"    
elif hostname == "hjp-MS-7D42" : # 3090 X 1
    image_path_list = sorted(glob(
        "/media/hjp/db6095ca-a560-4c3a-90ad-b667ec189671/REFERENCES/VLM/MM/data/Screenshots_accum/*"
    ))
    # PROJECT_ROOT    = "/media/hjp/FAAC278CAC27430D/HJP/KUAICV/VTO/florence-tailor"
    DATASET_ROOT    = "/media/hjp/efef19d3-9b92-453c-ba04-c205f7233cab/VTO_DATASET/PoC59"
    PYGARMENT_ROOT  = "/media/hjp/FAAC278CAC27430D/HJP/KUAICV/VTO/REFERENCES/GarmentCodeAnalysis"
elif hostname == "gpu-1" : # H100 X 8
    # PROJECT_ROOT    = "/data/hjp/VTO2025/florence-tailor"
    DATASET_ROOT    = "/data/hjp/VTO2025/DATASETs/690432"
    PYGARMENT_ROOT  = "/data/hjp/VTO2025/GarmentCodeAnalysis"