import argparse
from src.engine.train import train

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--log_dir', default='runs/exp')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--margin', type=float, default=0.2)
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--attn_reduction', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()

    train(
        data_root=args.data_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        embed_dim=args.embed_dim,
        attn_reduction=args.attn_reduction,
        seed=args.seed,
        amp=args.amp,
    )
