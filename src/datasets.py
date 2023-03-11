import torch
from torch.utils.data import Dataset
from persim import PersistenceImager
import numpy as np
import pickle
import glob
import os
import pandas as pd


class PImageTumorDataset(Dataset):
    """
    A dataset of layers of persistence images of tumors
    """
    def __init__(self, data_dir, metadata_path, imgr, filtration_type, seed=0, is_training=True, perc=0.9, layers=np.array([])):
        """
        Parameters
        ----------
        data_dir: str
            Path to directory containing pickle files with persistence diagrams
        metadata_path: str
            Path to metadata
        imgr: PersistenceImager
            An object for creating persistence images
        filtration_type: str
            Type of filtration to process (alpha, cubical)
        seed: int
            A seed for randomly permuting into the train/test dataset
        is_training: bool
            Whether this is the training or test set
        perc: float
            Use this percentage of the dataset for training
        layers: int
            If > 0, take only these persistence diagrams
        """
        ## Step 1: Filter out everything except glioblastoma
        files = glob.glob("{}/*.pkl".format(data_dir))
        files = sorted(files)
        to_keep = []
        metadata = pd.read_csv(metadata_path)
        for f in files:
            if not "cache" in f:
                ID = f.split("/")[-1].split(".pkl")[0]
                s1, s2 = ID.split("-0")
                ID = "{}-{}".format(s1, s2)
                row = metadata.loc[metadata['ID'] == ID]
                if "Glioblastoma" in row["Final pathologic diagnosis (WHO 2021)"].to_numpy()[0]:
                    to_keep.append(f)
        files = to_keep
        
        ## Step 2: Shuffle files to create dataset
        N = int(perc*len(files))
        np.random.seed(seed)
        files = [files[i] for i in np.random.permutation(len(files))]
        if is_training:
            self.files = files[0:N]
        else:
            self.files = files[N::]
        self.imgr = imgr
        self.filtration_type = filtration_type
        self.layers = layers

    def __len__(self):
        return len(self.files)

    def get_cache_path(self, idx):
        """
        Return the path to a file where the persistence images on this
        data with the chosen parameters is stored
        
        Parameters
        ----------
        idx: int
            Index of data object
        
        Returns
        -------
        string: Path to cache file
        """
        imgr = self.imgr
        name = "{}_cache".format(self.files[idx])
        name += "_Filtration{}".format(self.filtration_type)
        name += "_PixelSize{:.3f}".format(imgr.pixel_size)
        name += "_BirthRange{:.3f}to".format(imgr.birth_range[0])
        name += "{:.3f}".format(imgr.birth_range[1])
        name += "_PersRange{:.3f}to".format(imgr.pers_range[0])
        name += "{:.3f}".format(imgr.pers_range[1])
        name += "_Sigma{:.3f}".format(imgr.kernel_params["sigma"])
        name += "_n{:.3f}".format(imgr.weight_params["n"])
        return name+".pkl"
    
    def __getitem__(self, idx):
        cache_path = self.get_cache_path(idx)
        filename = self.files[idx]
        images = []
        OS = 0
        if not os.path.exists(cache_path):
            res = pickle.load(open(filename, "rb"))
            imgr = self.imgr
            img = imgr.transform([]) # Dummy image to figure out the resolution
            n_pds = 3*len(res["Edema_{}_PDs".format(self.filtration_type)])
            images = torch.zeros((1, n_pds, img.shape[0], img.shape[1]))
            i = 0
            for region in ["Edema", "Main Tumor", "Necrotic"]:
                print(".", end="", flush=True)
                for PD in res["{}_{}_PDs".format(region, self.filtration_type)]:
                    image = imgr.transform(PD)
                    images[0, i, :, :] = torch.from_numpy(image)
                    i += 1
            pickle.dump({"images":images, "OS":res["OS"]}, open(cache_path, "wb"))
        res = pickle.load(open(cache_path, "rb"))
        images = res["images"]
        if self.layers.size > 0:
            images = images[:, self.layers, :, :]
        return images, float(res["OS"] > 365) # Alive for at least one year

if __name__ == '__main__':
    data_dir = "../preprocessed"
    metadata_path = "../Data/UCSF-PDGM-metadata_v2.csv"
    imgr = PersistenceImager(pixel_size=0.1, birth_range=(-1,1), pers_range=(0, 2), 
                              kernel_params={'sigma':0.1}, weight_params={'n':1.5})
    filtration_type = "cubical"

    dataset = PImageTumorDataset(data_dir, metadata_path, imgr, filtration_type)
    for i in range(len(dataset)):
        images, alive = dataset[i]
        print(i, end="", flush=True)
