"""Weight-average several checkpoints from the SAME run (poor-man's SWA).

Motivation: this project's val_ap is noisy epoch-to-epoch (the combo run sat at
0.497-0.505 across ep15-ep19 with no clear winner), and val_ap on 2019 has
repeatedly failed to predict 2020 test performance -- `trimmed` won on val_ap
by 4% and LOST on real test. Averaging the weights of several near-equal
late-training checkpoints is the cheapest known variance reducer: no retraining,
one forward pass to evaluate, and it typically lands at or above the best
individual member.

Safe here because the architecture has NO normalization layers carrying running
statistics (only nn.LayerNorm, which is purely functional over the feature dim)
-- so no BatchNorm re-estimation pass is needed after averaging. Verified by
grep over seasfire/backbones/conv_lstm.py before writing this.

Only average checkpoints from ONE run with identical architecture/hparams;
averaging across runs (different random inits) does not work -- the weights are
not in the same loss basin.

Usage:
    CKPTS="a.ckpt:b.ckpt:c.ckpt" OUT=avg.ckpt python average_checkpoints.py
"""
import os
import sys
from collections import OrderedDict

import torch

CKPTS = [p for p in os.environ["CKPTS"].split(":") if p.strip()]
OUT = os.environ.get("OUT", "checkpoints/averaged.ckpt")


def main():
    if len(CKPTS) < 2:
        sys.exit(f"[error] need >=2 checkpoints to average, got {len(CKPTS)}")

    print(f"[info] averaging {len(CKPTS)} checkpoints:")
    for c in CKPTS:
        print(f"        {os.path.basename(c)}")

    base = torch.load(CKPTS[0], map_location="cpu", weights_only=False)
    if "state_dict" not in base:
        sys.exit("[error] no 'state_dict' in checkpoint -- is this a Lightning ckpt?")

    ref_keys = set(base["state_dict"].keys())
    ref_hp = base.get("hyper_parameters", {})

    acc = OrderedDict()
    n_float, n_copied = 0, 0
    for k, v in base["state_dict"].items():
        if v.is_floating_point():
            acc[k] = v.detach().clone().to(torch.float64)
            n_float += 1
        else:
            # ints/bools (e.g. buffers like pos_weight bookkeeping) can't be
            # meaningfully averaged -- carry the first checkpoint's value.
            acc[k] = v.detach().clone()
            n_copied += 1

    for path in CKPTS[1:]:
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        if set(sd.keys()) != ref_keys:
            sys.exit(f"[error] key mismatch in {path} -- checkpoints are not from the same architecture")
        for k, v in sd.items():
            if acc[k].is_floating_point():
                if acc[k].shape != v.shape:
                    sys.exit(f"[error] shape mismatch for {k} in {path}: {acc[k].shape} vs {v.shape}")
                acc[k] += v.detach().to(torch.float64)

    n = float(len(CKPTS))
    out_sd = OrderedDict()
    for k, v in acc.items():
        if v.is_floating_point():
            out_sd[k] = (v / n).to(base["state_dict"][k].dtype)
        else:
            out_sd[k] = v

    # Keep the full Lightning structure so load_from_checkpoint() works
    # unchanged downstream (fit_recalibration.py / operational_stats_dual.py
    # both go through ConvLSTMLitDual.load_from_checkpoint).
    base["state_dict"] = out_sd
    base["averaged_from"] = CKPTS
    # An averaged model does not correspond to any single epoch; leaving the
    # original epoch/global_step would be misleading if anything resumed from it.
    base["epoch"] = -1
    base["global_step"] = -1
    for k in ("optimizer_states", "lr_schedulers", "callbacks"):
        base.pop(k, None)   # meaningless post-average, and they bloat the file

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    torch.save(base, OUT)

    print(f"[info] {n_float} float tensors averaged, {n_copied} non-float carried from first ckpt")
    if ref_hp:
        keys = ("patch_size", "fuse", "hidden_dim", "kernel_size", "dilation")
        shown = {k: ref_hp[k] for k in keys if k in ref_hp}
        if shown:
            print(f"[info] hparams preserved: {shown}")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
