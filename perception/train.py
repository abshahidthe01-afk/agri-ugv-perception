"""
Train a crop/weed segmentation network on PhenoBench.

    python -m perception.train --data <PhenoBench folder> --out runs/<name> [--epochs 100]

Every epoch goes once through the training pictures in random squares (data.py), with
cross-entropy weighted per class (the PhenoBench baseline's weights: weeds cover only about
0.5 % of the pixels). AdamW; the learning rate rises over the first epoch and then falls
linearly to zero; the new decoder learns 10 times faster than the pretrained encoder (as in
SegFormer). Every --val-every epochs the whole validation pictures are scored (IoU of soil,
crop and weed, partial plants counted as whole ones, as the official evaluation does).
Writes <out>/log.csv (one row per epoch), <out>/last.pt (to continue with --resume) and
<out>/best.pt (the best validation mIoU so far).
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from perception.data import PhenoBench
from perception.metrics import confusion, summary
from perception.models import build, MODELS, parameter_groups, save

BASELINE_WEIGHTS = (1.47, 5.06, 10.02)    # PhenoBench baselines, config_erfnet.yaml


def schedule(warmup, total):
    """Learning-rate factor per step: linear rise over 'warmup' steps, then linear fall to 0."""
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))
    return factor


def evaluate(model, loader, device, amp):
    """Return (confusion matrix, seconds per picture) over a loader of whole pictures."""
    model.eval()
    matrix, start, count = np.zeros((3, 3), np.int64), time.perf_counter(), 0
    with torch.no_grad():
        for pictures, labels, _ in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(pictures.to(device, non_blocking=True))
            matrix += confusion(labels.numpy(), logits.argmax(dim=1).cpu().numpy())
            count += len(pictures)
    model.train()
    return matrix, (time.perf_counter() - start) / max(count, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--data', required=True, type=Path, help='PhenoBench folder')
    parser.add_argument('--out', required=True, type=Path, help='run folder to write')
    parser.add_argument('--model', default='segformer_b0', choices=MODELS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--crop', type=int, default=512, help='side of the training squares')
    parser.add_argument('--lr', type=float, default=6e-5, help="encoder's learning rate")
    parser.add_argument('--head-factor', type=float, default=10.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--class-weights', type=float, nargs=3, default=BASELINE_WEIGHTS)
    parser.add_argument('--val-every', type=int, default=5, help='epochs between scorings')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path, help='last.pt of a run to continue')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='start from random weights (tests; no download)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--limit-train', type=int, default=0, help='only N pictures (tests)')
    parser.add_argument('--limit-val', type=int, default=0, help='only N pictures (tests)')
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    amp = device.type == 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=2))

    train_set = PhenoBench(args.data, 'train', crop=args.crop, seed=args.seed,
                           limit=args.limit_train)
    val_set = PhenoBench(args.data, 'val', limit=args.limit_val)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=len(train_set) >= args.batch, pin_memory=amp,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=args.workers,
                                             pin_memory=amp)

    model = build(args.model, pretrained=not args.no_pretrained).to(device)
    optimiser = torch.optim.AdamW(parameter_groups(model, args.lr, args.head_factor),
                                  weight_decay=args.weight_decay)
    steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser,
                                                  schedule(len(loader), steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).to(device))
    first, best = 0, -math.inf
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        first, best = state['epoch'] + 1, state['best']
        print(f'Continuing {args.resume} after epoch {state["epoch"]}')

    log_path = args.out / 'log.csv'
    columns = ['epoch', 'loss', 'lr', 'seconds', 'soil', 'crop', 'weed', 'mIoU', 'gpu_MB']
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(columns)
    parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{args.model}: {parameters:.2f} M parameters on {device}; {len(train_set)} training '
          f'pictures in squares of {args.crop} px, {len(loader)} steps per epoch; '
          f'{len(val_set)} validation pictures')

    for epoch in range(first, args.epochs):
        train_set.epoch = epoch
        start, losses = time.perf_counter(), []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for pictures, labels, _ in loader:
            pictures = pictures.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(pictures), labels)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'lr': scheduler.get_last_lr()[0],
               'seconds': time.perf_counter() - start,
               'gpu_MB': (torch.cuda.max_memory_allocated() / 2**20
                          if device.type == 'cuda' else 0.0)}
        last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_every == 0 or last:
            matrix, per_picture = evaluate(model, val_loader, device, amp)
            scores = summary(matrix)
            row.update(scores)
            if scores['mIoU'] > best:
                best = scores['mIoU']
                save(args.out / 'best.pt', model, args.model, epoch, scores, settings)
            print(f'epoch {epoch + 1:3d}: loss {row["loss"]:.4f}, {row["seconds"]:.0f} s, '
                  f'val soil {scores["soil"]:.2f} crop {scores["crop"]:.2f} weed '
                  f'{scores["weed"]:.2f} mIoU {scores["mIoU"]:.2f} '
                  f'({1000 * per_picture:.0f} ms per picture), best {best:.2f}', flush=True)
        else:
            print(f'epoch {epoch + 1:3d}: loss {row["loss"]:.4f}, {row["seconds"]:.0f} s, '
                  f'peak GPU memory {row["gpu_MB"]:.0f} MB', flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(c, '') for c in columns])
        torch.save({'model': args.model, 'state_dict': model.state_dict(), 'epoch': epoch,
                    'best': best, 'optimiser': optimiser.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'settings': settings}, args.out / 'last.pt')
    print(f'Done: best validation mIoU {best:.2f} in {args.out / "best.pt"}')


if __name__ == '__main__':
    main()
