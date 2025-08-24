import argparse
from src.engine.eval import evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    evaluate(
        data_root=args.data_root,
        test_csv=args.test_csv,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
    )
