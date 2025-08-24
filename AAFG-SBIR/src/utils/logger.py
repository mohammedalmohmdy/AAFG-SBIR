import csv, os, time
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['time','epoch','split','metric','value'])
        self._csv_file.flush()

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_kv(self, epoch, split, d: dict):
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        for k,v in d.items():
            self._csv.writerow([now, epoch, split, k, v])
        self._csv_file.flush()

    def close(self):
        self.writer.close()
        self._csv_file.close()
