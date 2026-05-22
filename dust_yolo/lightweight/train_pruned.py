import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILE = Path(__file__).resolve()
import argparse
import time
import yaml
import numpy as np
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.cuda import amp

import val
from utils.general import (LOGGER, colorstr, check_dataset, check_file,
                           check_img_size, check_suffix, check_yaml,
                           init_seeds, fitness, strip_optimizer,
                           increment_path, print_args)
from utils.torch_utils import ModelEMA, select_device, de_parallel
from utils.loss import ComputeLoss
from utils.datasets import create_dataloader


def labels_to_class_weights(labels, nc=80):
    from utils.general import labels_to_class_weights as _l2cw
    return _l2cw(labels, nc)


class EarlyStopping:
    def __init__(self, patience=50):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch = epoch
        stop = (epoch - self.best_epoch) >= self.patience
        if stop:
            LOGGER.info(f'EarlyStopping patience {self.patience} reached, stopping at epoch {epoch}')
        return stop


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - np.cos(x * np.pi / steps)) / 2) * (y2 - y1) + y1


def train_pruned(hyp, opt, device):
    save_dir = Path(opt.save_dir)
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = opt.weights

    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])
    names = data_dict['names']

    check_suffix(weights, '.pt')
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].float().to(device)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    if not hasattr(model, 'names') or model.names is None:
        model.names = names
    if not hasattr(model, 'nc') or model.nc is None:
        model.nc = nc

    del ckpt

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs

    lr0 = hyp['lr0'] * opt.lr_scale

    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=lr0, betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g0, lr=lr0, momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    del g0, g1, g2

    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model)

    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size, gs, False,
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
        rank=-1, workers=opt.workers, image_weights=False, quad=False,
        prefix=colorstr('train: '))
    nb = len(train_loader)

    val_loader = create_dataloader(
        val_path, imgsz, batch_size * 2, gs, False,
        hyp=hyp, cache=None, rect=True, rank=-1,
        workers=opt.workers, pad=0.5,
        prefix=colorstr('val: '))[0]

    nl = model.model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = 0.0
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    last_opt_step = -1
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=True)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)

    for epoch in range(epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = tqdm(enumerate(train_loader), total=nb)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(
                        ni, xi,
                        [hyp['warmup_bias_lr'] if j == 2 else 0.0,
                         x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi,
                            [hyp['warmup_momentum'], hyp['momentum']])

            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))

            scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        results, maps, _ = val.run(
            data_dict,
            batch_size=batch_size * 2,
            imgsz=imgsz,
            model=ema.ema,
            dataloader=val_loader,
            save_dir=save_dir,
            plots=False,
            compute_loss=compute_loss)

        fi = float(fitness(np.array(results).reshape(1, -1)))
        if fi > best_fitness:
            best_fitness = fi

        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'model': deepcopy(de_parallel(model)).half(),
            'ema': deepcopy(ema.ema).half(),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict(),
            'names': names,
            'date': datetime.now().isoformat()
        }
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        del ckpt

        LOGGER.info(
            f'Epoch {epoch}: '
            f'mAP@0.5={float(results[2]):.4f}, mAP@0.5:0.95={float(results[3]):.4f}, '
            f'fitness={fi:.4f}, best={best_fitness:.4f}, '
            f'lr={lr[0]:.6f}')

        if stopper(epoch=epoch, fitness=fi):
            break

    for f in last, best:
        if f.exists():
            strip_optimizer(f)

    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='./data/VisDrone.yaml')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.VisDrone.yaml')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1536)
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--cache', type=str, nargs='?', const='ram')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='')
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='pruned')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr-scale', type=float, default=0.1)
    return parser.parse_args()


def main():
    opt = parse_opt()
    print_args(FILE.stem, opt)

    opt.data = check_file(opt.data)
    opt.hyp = check_yaml(opt.hyp)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    init_seeds(57)

    train_pruned(opt.hyp, opt, device)


if __name__ == "__main__":
    main()
