import os, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

def pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')

class PairIndexDataset(Dataset):
    """Rows: path_sketch, path_image, label. Paths are relative to data_root."""
    def __init__(self, data_root, csv_file, transform_sketch=None, transform_image=None, loader=pil_loader):
        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, csv_file))
        required = {'path_sketch','path_image','label'}
        assert required.issubset(self.df.columns), f"CSV must contain columns: {required}"
        self.t_sk = transform_sketch
        self.t_im = transform_image
        self.loader = loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sk_path = os.path.join(self.data_root, r['path_sketch'])
        im_path = os.path.join(self.data_root, r['path_image'])
        label = int(r['label'])
        sk = self.loader(sk_path)
        im = self.loader(im_path)
        if self.t_sk: sk = self.t_sk(sk)
        if self.t_im: im = self.t_im(im)
        return sk, im, label

class Collate:
    def __call__(self, batch):
        sk, im, lb = zip(*batch)
        return torch.stack(sk,0), torch.stack(im,0), torch.tensor(lb, dtype=torch.long)
