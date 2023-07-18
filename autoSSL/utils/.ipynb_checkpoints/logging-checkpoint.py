from pytorch_lightning.loggers import CSVLogger
import os
import csv
 

class ContinuousCSVLogger(CSVLogger):
    def __init__(self, save_dir, name="default"):
        super().__init__(save_dir, name, version=None)

    def log_metrics(self, metrics, step=None):
        if step is None:
            raise ValueError("Step cannot be None. Please provide a step number.")

        for metric, value in metrics.items():
            # Create a file for each unique metric name
            filepath = os.path.join(self.save_dir, f"{self.name}_{metric}.csv")

            # Check if the file exists. If not, write the header
            if not os.path.isfile(filepath):
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['step', metric])
                    writer.writeheader()

            # Write the metrics
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['step', metric])
                writer.writerow({'step': step, metric: value})

from pytorch_lightning.callbacks import ModelCheckpoint


def ck_callback(checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(
       monitor='train_loss',
       dirpath=checkpoint_dir,
       filename='checkpoints-{epoch:02d}-{train_loss:.2f}',
       save_top_k=3,
       mode='min',)
    return checkpoint_callback