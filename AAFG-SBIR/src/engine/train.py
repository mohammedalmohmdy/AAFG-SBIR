import os, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ..utils.seed import set_seed
from ..utils.logger import Logger
from ..utils.checkpoint import save_ckpt
from ..data.datasets import PairIndexDataset, Collate
from ..models.aafg_sbir import AAFG_SBiR
from ..losses.triplet import BatchHardTripletLoss
from tqdm import tqdm

def default_transforms(img_size=224):
    t_sk = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    t_im = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t_sk, t_im

def train(
    data_root, train_csv, val_csv, log_dir='runs/exp', ckpt_dir='checkpoints',
    epochs=60, batch_size=32, lr=1e-4, margin=0.2, embed_dim=512, attn_reduction=1,
    seed=42, amp=False, num_workers=4, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_dir)

    t_sk, t_im = default_transforms()
    train_ds = PairIndexDataset(data_root, train_csv, t_sk, t_im)
    val_ds   = PairIndexDataset(data_root, val_csv,   t_sk, t_im)
    collate = Collate()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

    model = AAFG_SBiR(embed_dim=embed_dim, attn_reduction=attn_reduction).to(device)
    criterion = BatchHardTripletLoss(margin=margin)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and torch.cuda.is_available()))

    best_val = -1.0
    global_step = 0

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}")
        run_loss = 0.0
        for sk, im, y in pbar:
            sk, im, y = sk.to(device, non_blocking=True), im.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available())):
                zs, zi = model(sk, im)
                emb = torch.cat([zs, zi], dim=0)
                lbl = torch.cat([y, y], dim=0)
                loss = criterion(emb, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            logger.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = run_loss / max(1, len(train_loader))
        logger.log_kv(epoch, 'train', {'loss': avg_loss})

        # light validation proxy: average matched cosine similarity
        model.eval()
        sims, count = 0.0, 0
        with torch.no_grad():
            for sk, im, y in tqdm(val_loader, desc='Validate'):
                sk, im = sk.to(device), im.to(device)
                zs, zi = model(sk, im)
                sims += (zs * zi).sum(dim=1).mean().item()
                count += 1
        val_score = sims / max(1, count)
        logger.add_scalar('val/avg_pair_similarity', val_score, epoch)
        logger.log_kv(epoch, 'val', {'avg_pair_similarity': val_score})

        # checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
        save_ckpt(ckpt_path, epoch, model, optimizer, scaler, best_metric=val_score)
        if val_score > best_val:
            best_val = val_score
            save_ckpt(os.path.join(ckpt_dir, 'best.pt'), epoch, model, optimizer, scaler, best_metric=best_val)

    logger.close()
