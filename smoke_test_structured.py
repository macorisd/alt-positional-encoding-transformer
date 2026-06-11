#!/usr/bin/env python3
"""Run one real forward/backward step for an alt-positional structured dataset."""

from __future__ import annotations

import argparse
import torch
from torch import nn

import train_k_fold_cross_validation as trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--wave", default="sinusoid")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    trainer.batch_size = args.batch_size
    if torch.cuda.is_available():
        trainer.device = torch.device("cuda:0")

    loader, train_data, _, _ = trainer.prepare_data_loaders(dataset_name=args.dataset, dataset_root=args.dataset_root)
    folds = trainer.create_k_fold_splits(train_data, k=10)
    train_fold, val_fold = trainer.get_fold_data(folds, args.fold - 1)
    train_iter, _, _ = loader.make_iter(train_fold, val_fold, val_fold, batch_size=args.batch_size, device=trainer.device)

    model = trainer.Transformer(
        src_pad_idx=loader.source.vocab["<pad>"],
        trg_pad_idx=loader.target.vocab["<pad>"],
        trg_sos_idx=loader.target.vocab["<sos>"],
        d_model=trainer.d_model,
        enc_voc_size=len(loader.source.vocab),
        dec_voc_size=len(loader.target.vocab),
        max_len=trainer.max_len,
        ffn_hidden=trainer.ffn_hidden,
        n_head=trainer.n_heads,
        n_layers=trainer.n_layers,
        drop_prob=trainer.drop_prob,
        device=trainer.device,
        periodic_func=args.wave,
    ).to(trainer.device)
    model.apply(trainer.initialize_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=loader.source.vocab["<pad>"])
    src, trg = next(iter(train_iter))
    output = model(src, trg[:, :-1])
    loss = criterion(output.contiguous().view(-1, output.shape[-1]), trg[:, 1:].contiguous().view(-1))
    loss.backward()
    torch.cuda.synchronize() if trainer.device.type == "cuda" else None

    print(
        "ALT smoke OK: "
        f"dataset={args.dataset} wave={args.wave} fold={args.fold} "
        f"batch={args.batch_size} loss={loss.item():.6f} device={trainer.device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
