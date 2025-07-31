# path: scripts/train.py

import sys, os
from datetime import datetime
import argparse, yaml
from pathlib import Path

# make sure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import NpySegmentationDataset as Dataset
from models import get_model
from engines.trainer import Trainer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))

    # timestamped output folder per arch
    base = Path(cfg['train']['save_dir']) / cfg['model']['arch']
    stamp = datetime.now().strftime("%m-%d-%Y %H-%M")
    outp = base / stamp
    outp.mkdir(parents=True, exist_ok=True)
    cfg['train']['save_dir'] = str(outp)

    # datasets
    train_ds = Dataset(cfg, 'train')
    val_ds   = Dataset(cfg, 'val')
    test_ds  = Dataset(cfg, 'test') if cfg['train'].get('evaluate_test') else None

    # build model by name
    model = get_model(
        cfg['model']['arch'],
        n_channels=3,
        n_classes= cfg['model']['n_classes'],
        **cfg['model'].get('params', {})
    )

    # train & eval
    Trainer(model, cfg).train(train_ds, val_ds, test_ds)
