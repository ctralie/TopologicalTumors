import nibabel as nib
import numpy as np 
import pandas as pd
import os
import glob


def load_dictionary(metadata_path):
    """
    Load the metadata CSV into a dictionary
    
    """
    df = pd.read_csv(metadata_path)
    metadata_dictionary = {}
    for _, row in df.iterrows():
        patient_id = row["ID"]
        patient_id = "M-0".join(patient_id.split("M-")) #File paths have an extra 0 in ID
        dx = row["Final pathologic diagnosis (WHO 2021)"]
        gbm = 0
        if "Glioblastoma" in dx:
            gbm = 1
        metadata_dictionary[patient_id] = gbm
    return metadata_dictionary
