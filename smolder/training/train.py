"""Train SMOLDER (dual-branch ConvLSTM, next-3-day fire occurrence).

Every setting is read from environment variables; configs/smolder.env holds
the exact values of the released model, verified against the hyperparameters
stored in its checkpoint. Typical use, from the repository root:

    set -a; source configs/smolder.env; set +a
    SMOLDER_DATA=/path/to/cubes python -m smolder.training.train

Data: train 2015-2018, validation 2019 (checkpoint selection and early
stopping on validation average precision). The 2020 test year is never read.
Outputs go to $SMOLDER_RUNS/<RUN_TAG>/job_<SLURM_JOB_ID or "local">/.
"""
from pathlib import Path

import os, sys

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
from smolder.data.io import CHANNEL_STATS, resolve
from smolder.data.zarr_dual_datamodule import DualDataModule, SLOW_CHANNELS, FAST_CHANNELS

MAX_EPOCHS = 25

SLOW_DAYS = 144      # inside the measured 130-150 saturation band; divides by 8
SLOW_BIN = 8         # native cadence of LAI (8-day composites)
FAST_DAYS = 14

# Setting pos_weight to the raw neg:pos ratio (~1900) saturates the output from
# epoch 0; 100 annealed to 20 is used instead (see configs/smolder.env).
POS_WEIGHT = 100.0
SAMPLES_PER_EPOCH = 2000


def main():
    # Cube names are resolved against $SMOLDER_DATA (see smolder/data/io.py).
    train_paths = [f"cube_daily_smgrid_{y}.zarr" for y in (2015, 2016, 2017, 2018)]
    val_paths = ["cube_daily_smgrid_2019.zarr"]
    stats_path = str(CHANNEL_STATS)
    if not Path(stats_path).exists():
        raise FileNotFoundError(f"{stats_path} missing; run compute_channel_stats.py first")

    pos_weight = float(os.environ.get("POS_WEIGHT", POS_WEIGHT))
    samples_per_epoch = int(os.environ.get("SAMPLES_PER_EPOCH", SAMPLES_PER_EPOCH))
    max_epochs = int(os.environ.get("MAX_EPOCHS", MAX_EPOCHS))
    # Early stopping cannot end training before MIN_EPOCHS (0 = off, the
    # released recipe). The positive weight is annealed over the first 8
    # epochs and validation AP often peaks early in that phase and dips after
    # it; with patience 6 a run can then stop at epoch 9-13, before the
    # climb that follows (seen in the 2026-10 distance-weight sweep).
    min_epochs = int(os.environ.get("MIN_EPOCHS", 0))
    slow_days = int(os.environ.get("SLOW_DAYS", SLOW_DAYS))
    fast_days = int(os.environ.get("FAST_DAYS", FAST_DAYS))

    # Multi-seed noise-floor check (patch-size sweep): seeds model init, dataloader
    # shuffling, AND the train sampler's own RNG (train_seed below). val_ds seed
    # stays fixed at 999 in zarr_dual_datamodule.py so the val set never changes.
    seed = int(os.environ.get("SEED", 123))
    pl.seed_everything(seed, workers=True)

    # Patch-size sweep: min_pos_pixels is scaled with patch AREA so every patch
    # size samples the same fire DENSITY (20px / 256^2 = 0.0305%), not the same
    # absolute pixel count -- otherwise a 128px patch would face a 4x harder bar
    # and a 384px patch a 2.25x easier one, confounding patch size with sampling
    # difficulty. Baseline (256, 20px) is preserved exactly when PATCH_SIZE=256.
    BASE_PATCH, BASE_MIN_POS = 256, 20
    patch_size = int(os.environ.get("PATCH_SIZE", 384))
    min_pos_pixels = max(1, round(BASE_MIN_POS * (patch_size / BASE_PATCH) ** 2))

    # ---- pos_weight annealing (None/None = old constant behaviour) ----
    pw_start = os.environ.get("POS_WEIGHT_START")
    pw_end = os.environ.get("POS_WEIGHT_END")
    pw_anneal_epochs = int(os.environ.get("POS_WEIGHT_ANNEAL_EPOCHS", 8))
    pos_weight_start = float(pw_start) if pw_start else None
    pos_weight_end = float(pw_end) if pw_end else None

    # ---- isolation-weighted loss (0.0 = disabled, old behaviour) ----
    isolation_gamma = float(os.environ.get("ISOLATION_GAMMA", 0.0))
    isolation_kernel = int(os.environ.get("ISOLATION_KERNEL", 9))

    # ---- weight on fire far from past fire (0.0 = disabled, old behaviour) ----
    past_fire_weight_a = float(os.environ.get("PAST_FIRE_WEIGHT_A", 0.0))
    past_fire_weight_scale_px = float(os.environ.get("PAST_FIRE_WEIGHT_SCALE_PX", 10.0))
    past_fire_dist_store = os.environ.get("PAST_FIRE_DIST_STORE", "fire_dist30_continental.zarr")
    if os.environ.get("SLOW_WINDOW_LEGACY", "0") == "1":
        print("[warn] SLOW_WINDOW_LEGACY=1: slow bins reach past the issue day (leak)")

    print(f"[info] SMOLDER training: slow {SLOW_CHANNELS} {slow_days}d/{SLOW_BIN}d-bins, "
          f"fast {FAST_CHANNELS} {fast_days}d | pos_weight={pos_weight} "
          f"samples_per_epoch={samples_per_epoch} max_epochs={max_epochs} "
          f"patch_size={patch_size} min_pos_pixels={min_pos_pixels} "
          f"(density {100*min_pos_pixels/patch_size**2:.4f}%) | "
          f"pos_weight_anneal={pos_weight_start}->{pos_weight_end} over {pw_anneal_epochs}ep | "
          f"isolation_gamma={isolation_gamma} kernel={isolation_kernel} | "
          f"past_fire_weight_a={past_fire_weight_a} scale={past_fire_weight_scale_px}px")

    # Pre-binned slow cube (build_slow_cube.py). Without it the datamodule falls
    # back to aggregating 144 raw days per sample: correct, but 22x slower
    # (2.30s vs 0.103s/sample) because it pulls 1079MB to keep 14MB.
    slow_cube = "cube_slow_8day.zarr"
    if not resolve(slow_cube).exists():
        raise FileNotFoundError(f"{slow_cube} not found in cwd or $SMOLDER_DATA "
                                "(build it with python -m smolder.data.build_slow_cube)")
    for p in train_paths + val_paths:
        if not resolve(p).exists():
            raise FileNotFoundError(f"{p} not found in cwd or $SMOLDER_DATA")

    dm = DualDataModule(
        train_paths=train_paths,
        val_paths=val_paths,
        stats_path=stats_path,
        slow_cube_path=slow_cube,
        y_key="y_fire_3d",
        valid_key="y_fire_3d_valid",
        slow_days=slow_days,
        slow_bin=SLOW_BIN,
        fast_days=fast_days,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch,
        batch_size=int(os.environ.get("BATCH_SIZE", 2)),
        num_workers=int(os.environ.get("NUM_WORKERS", 8)),
        min_pos_pixels=min_pos_pixels,
        train_seed=seed,
        fire_history=os.environ.get("FIRE_HISTORY","0")=="1",
        fire_history_lags=(tuple(int(x) for x in os.environ["FIRE_HISTORY_LAGS"].replace(",",":").split(":") if x)
                           if os.environ.get("FIRE_HISTORY_LAGS") else None),
        fire_history_distance=os.environ.get("FIRE_HISTORY_DISTANCE","0")=="1",
        use_lightning=os.environ.get("USE_LIGHTNING","0")=="1",
        use_elevation=os.environ.get("USE_ELEVATION","0")=="1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT","0")=="1",
        use_fuel_age=os.environ.get("USE_FUEL_AGE","0")=="1",
        fuel_age_lookback=int(os.environ.get("FUEL_AGE_LOOKBACK", 250)),
        use_wind_dir=os.environ.get("USE_WIND_DIR","0")=="1",
        use_ffdi=os.environ.get("USE_FFDI","0")=="1",
        use_fmc=os.environ.get("USE_FMC","0")=="1",
        fmc_store=os.environ.get("FMC_STORE","fmc_weekly_ff.zarr"),
        augment=os.environ.get("AUGMENT","0")=="1",
        fire_history_dropout_prob=float(os.environ.get("FIRE_HISTORY_DROPOUT_PROB", 0.0)),
        new_fire_frac=float(os.environ.get("NEW_FIRE_FRAC", 0.0)),
        min_new_fire_pixels=int(os.environ.get("MIN_NEW_FIRE_PIXELS", 1)),
        past_fire_dist_store=past_fire_dist_store if past_fire_weight_a > 0 else None,
        use_vpd_anomaly=os.environ.get("USE_VPD_ANOMALY", "0") == "1",
        vpd_source=os.environ.get("VPD_SOURCE", "montes"),
        perfect_forecast=os.environ.get("PERFECT_FORECAST", "0") == "1",
        use_frp=os.environ.get("USE_FRP", "0") == "1",
        use_barra_uv=os.environ.get("USE_BARRA_UV", "0") == "1",
        slow_veg=os.environ.get("SLOW_VEG", "lai"),
        use_fast_ndvi=os.environ.get("USE_FAST_NDVI", "0") == "1",
    )

    dm.setup("fit")
    b0 = next(iter(dm.train_dataloader()))
    c_slow = int(b0["x_slow"].shape[-1])
    c_fast = int(b0["x_fast"].shape[-1])
    c_static = int(b0["x_static"].shape[-1])
    use_static_head = os.environ.get("USE_STATIC_HEAD", "0") == "1"
    print(f"[info] x_slow={tuple(b0['x_slow'].shape)} x_fast={tuple(b0['x_fast'].shape)} "
          f"x_static={tuple(b0['x_static'].shape)} use_static_head={use_static_head} "
          f"(predictors + agb + landmask + sin/cos doy)")

    model = ConvLSTMLitDual(
        input_dim_slow=c_slow,
        input_dim_fast=c_fast,
        emb_dim_kg=4,
        emb_dim_lc=6,          # landcover as an input embedding, not a split
        lc_class_idx=None,     # train on all pixels
        hidden_layers=int(os.environ.get("HIDDEN_LAYERS", 1)),
        hidden_dim=64,
        # Whole-network spatial receptive field is only ~2*k-1 px (one ConvLSTM
        # cell + the classification head's conv, both same-padded, no spatial
        # reach from the per-pixel cross-attention fusion) -- 9px at the 5x5
        # default, independent of patch_size. A bigger kernel (or more layers)
        # tests whether that's a real bottleneck for local terrain context
        # (slope/aspect), orthogonal to the patch-size effect (which is about
        # un-truncating the fire-distance features, not CNN receptive field).
        kernel_size=(int(os.environ.get("KERNEL_SIZE", 5)), int(os.environ.get("KERNEL_SIZE", 5))),
        dilation=int(os.environ.get("DILATION", 1)),
        lr=3e-4,
        weight_decay=float(os.environ.get("WEIGHT_DECAY", 1e-2)),
        seq_len=fast_days,     # deep supervision runs on the fast (daily) axis
        use_recency_weights=False,
        pos_weight=pos_weight,
        loss_schedule="bce_dice_bce_focal",
        dice_end_epoch=int(os.environ.get("DICE_END_EPOCH", 3)),
        # Every run this session (9+ seeds/configs) peaks at ep6-7 then drops hard
        # the epoch the loss switches to focal (ep8 default) -- e.g. 256px
        # seed789 ep7=0.213 -> ep8=0.081, 384px seed123 ep7=0.376 -> ep8=0.177.
        # Push this past max_epochs to test staying on BCE throughout.
        focal_start_epoch=int(os.environ.get("FOCAL_START_EPOCH", 8)),
        switch_epoch=3,
        dice_weight=0.3,
        focal_alpha=0.75,
        focal_gamma=1.0,
        topk_fracs=(0.15, 0.05, 0.01),
        aux_loss_weight=0.3,
        cosine_t_max=int(os.environ.get("COSINE_T_MAX", max_epochs)),
        fuse=os.environ.get("FUSE", "cross_attn"),   # FireSenseNet-style fusion
        attn_heads=int(os.environ.get("ATTN_HEADS", 4)),
        soft_pos=float(os.environ.get("SOFT_POS", 0.9)),
        soft_neg=float(os.environ.get("SOFT_NEG", 0.02)),
        pos_weight_start=pos_weight_start,
        pos_weight_end=pos_weight_end,
        pos_weight_anneal_epochs=pw_anneal_epochs,
        isolation_gamma=isolation_gamma,
        isolation_kernel=isolation_kernel,
        past_fire_weight_a=past_fire_weight_a,
        past_fire_weight_scale_px=past_fire_weight_scale_px,
        # OHEM: keep all positives + only the hardest OHEM_FRAC of negatives.
        # 0.0 (default) = disabled, byte-identical to the pre-OHEM loss.
        ohem_frac=float(os.environ.get("OHEM_FRAC", 0.0)),
        ohem_min_negatives=int(os.environ.get("OHEM_MIN_NEG", 256)),
        ohem_rescale_negatives=os.environ.get("OHEM_RESCALE", "1") == "1",
        static_dim=(c_static if use_static_head else 0),
    )

    jobid = os.environ.get("SLURM_JOB_ID", "local")
    run_tag = os.environ.get("RUN_TAG", "smolder")
    run_dir = Path(os.environ.get("SMOLDER_RUNS", "runs")) / run_tag / f"job_{jobid}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    print(f"[info] CSVLogger -> {run_dir}/lightning_logs")

    callbacks = [
        # Select on AP -- see the note in train_convlstm_30day_v2_unified.py.
        ModelCheckpoint(
            monitor="val_ap", mode="max", save_top_k=3,
            filename="best-{epoch}-{val_ap:.4f}", save_last=True,
        ),
        # Tracked separately from val_ap: save_top_k on val_ap alone has already
        # pruned away the single best new-fire epoch of a run once (newfiresample30
        # ep7, val_ap_newfire=0.1059, ratio 24.7% -- the best new-fire score of the
        # whole session -- got silently deleted once a later epoch scored higher on
        # plain val_ap). This callback keeps the best-by-new-fire checkpoint around
        # too so it can still be checked against real 2020 test data later.
        ModelCheckpoint(
            monitor="val_ap_newfire", mode="max", save_top_k=1,
            filename="bestnewfire-{epoch}-{val_ap_newfire:.4f}",
        ),
        EarlyStopping(monitor="val_ap", mode="max", patience=6),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    torch.backends.cudnn.benchmark = True        # fixed input sizes: let cuDNN pick the fastest conv algorithms
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=int(os.environ.get("ACCUM", 4)),
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        default_root_dir=str(run_dir),
        logger=csv_logger,
        callbacks=callbacks,
    )

    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip() or None
    if resume_ckpt:
        print(f"[info] resuming from checkpoint: {resume_ckpt}")
    trainer.fit(model, dm, ckpt_path=resume_ckpt)

    trainer.save_checkpoint(str(run_dir / "final.ckpt"))
    print(f"[done] wrote {run_dir / 'final.ckpt'}")


if __name__ == "__main__":
    main()
