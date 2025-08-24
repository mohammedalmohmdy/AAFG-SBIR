import os, torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from ..data.datasets import PairIndexDataset, Collate
from ..models.aafg_sbir import AAFG_SBiR
from ..utils.checkpoint import load_ckpt
from tqdm import tqdm

def default_transforms(img_size=224):
    t = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    return t, t

def compute_map_prec_at_k(sk_emb, im_emb, labels, ks=(100,200)):
    sim = sk_emb @ im_emb.T  # (Ns, Ni)
    ranks = np.argsort(-sim, axis=1)
    rel = (labels[:,None] == labels[None,:]).astype(np.int32)  # (Ns,Ni)
    APs = []
    P_at = {k: [] for k in ks}
    for i in range(sim.shape[0]):
        order = ranks[i]
        y = rel[i, order]
        if y.sum() == 0:
            APs.append(0.0)
        else:
            cumsum = np.cumsum(y)
            idx = np.where(y==1)[0]
            precisions = cumsum[idx] / (idx+1)
            APs.append(precisions.mean())
        for k in ks:
            topk = y[:k]
            P_at[k].append(topk.mean() if len(topk)>0 else 0.0)
    mAP = float(np.mean(APs))
    P_at_k = {k: float(np.mean(v)) for k,v in P_at.items()}
    return mAP, P_at_k

def evaluate(data_root, test_csv, ckpt, batch_size=64, num_workers=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    t_sk, t_im = default_transforms()
    ds = PairIndexDataset(data_root, test_csv, t_sk, t_im)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=Collate())

    model = AAFG_SBiR()
    load_ckpt(ckpt, model, map_location=device)
    model.to(device)
    model.eval()

    sk_embs, im_embs, lbs = [], [], []
    with torch.no_grad():
        for sk, im, y in tqdm(loader, desc='Embed'):
            sk, im = sk.to(device), im.to(device)
            zs, zi = model(sk, im)
            sk_embs.append(zs.cpu().numpy())
            im_embs.append(zi.cpu().numpy())
            lbs.append(y.numpy())
    sk_emb = np.concatenate(sk_embs,0)
    im_emb = np.concatenate(im_embs,0)
    labels = np.concatenate(lbs,0)

    mAP, P_at_k = compute_map_prec_at_k(sk_emb, im_emb, labels)
    print({'mAP': mAP, **{f'P@{k}': v for k,v in P_at_k.items()}})
    return mAP, P_at_k
