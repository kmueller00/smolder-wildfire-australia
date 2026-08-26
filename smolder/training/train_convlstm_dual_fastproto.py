"""Dual-branch training: variable-specific temporal encoders, unified over landcover.

What differs from train_convlstm_30day_v2_unified.py (the comparison baseline):
ONLY the temporal encoding. Same cubes, same normalization, same loss schedule,
same pos_weight=350, same samples_per_epoch=2000, same deep supervision, same
cosine LR, landcover still an input embedding (emb_dim_lc=6) rather than a split.

The change:
  slow branch  LAI, SM, PPT    144 days @ 8-day bins  -> 18 steps
  fast branch  VPD, LST, WIND   14 days @ daily       -> 14 steps

Grounded in lagged_skill_extended_agg.csv (6 years, patched cubes, lags 0-180):
LAI peaks at lag 130 (AUC 0.779), SM at 150 (0.653), PPT at 150 (0.585), while
VPD peaks at lag 0 (0.572) and LST at 10 (0.559), both collapsing to ~0.45 by
90 days. The production 30-day window sees SM at ~0.52 of an available 0.65.

NDVI is dropped: it correlates 0.795 with LAI and adds nothing for the target
(LAI alone 0.771 AUC, NDVI+LAI 0.770, all three 0.765). AGB is kept as a static
-- it is the least redundant of the three and costs no timestep.

FAST PROTOTYPE variant: cheap single-run testbed for two ideas before spending
real GPU hours on a full generation --
  1. pos_weight annealing (POS_WEIGHT_START/END/ANNEAL_EPOCHS): pos_weight=100
     buys ranking/recall at the cost of calibration (raw sigmoid compressed into
     ~0.65-0.75 almost everywhere, fixed post-hoc via fit_recalibration.py).
     Annealing it down over training is the training-time alternative.
  2. isolation-weighted loss (ISOLATION_GAMMA/KERNEL): permutation importance
     showed this model's #1 feature is distance-to-recent-fire with weather
     near zero -- it has learned "fire near fire" and has ~no signal for
     isolated new ignitions (new-fire lift is below random under 0.2%).
     Upweights loss on positives that are spatially isolated in the CURRENT
     target, forcing gradient attention onto exactly that blind spot.
Uses patch_size=128 (fastest, already the best patch-size-sweep result) with
reduced samples_per_epoch/max_epochs for a cheap single-seed prototype -- NOT
a publishable comparison, just "is this worth a real run".

Run tag: dual_fastproto
"""
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualDataModule, SLOW_CHANNELS, FAST_CHANNELS

SCRIPT_DIR = Path(__file__).resolve().parent
MAX_EPOCHS = 12      # fast prototype: patience=6 stopped every prior run by ~ep13-14 anyway

SLOW_DAYS = 144      # inside the measured 130-150 saturation band; divides by 8
SLOW_BIN = 8         # native cadence of LAI (8-day composites)
FAST_DAYS = 14

# Match the split and unified runs exactly. Do NOT set this from the neg:pos
# ratio -- that was tried (job 1747766, pos_weight=1905) and saturated the model
# (val_p_mean 0.28 vs a 0.0005 base rate), freezing the score from epoch 0.
POS_WEIGHT = 100.0   # matches dual_fh_attn (job 1755283), the production checkpoint
SAMPLES_PER_EPOCH = 800   # fast prototype: 1/2.5 of the production 2000


def main():
    train_paths = [str(SCRIPT_DIR / f"cube_daily_smgrid_{y}.zarr") for y in (2015, 2016, 2017, 2018)]
    val_paths = [str(SCRIPT_DIR / "cube_daily_smgrid_2019.zarr")]
    stats_path = str(SCRIPT_DIR / "channel_stats_2015_2018.json")
    if not Path(stats_path).exists():
        raise FileNotFoundError(f"{stats_path} missing — run compute_channel_stats.py first")

    pos_weight = float(os.environ.get("POS_WEIGHT", POS_WEIGHT))
    samples_per_epoch = int(os.environ.get("SAMPLES_PER_EPOCH", SAMPLES_PER_EPOCH))
    max_epochs = int(os.environ.get("MAX_EPOCHS", MAX_EPOCHS))
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
    patch_size = int(os.environ.get("PATCH_SIZE", 128))   # fast prototype default: 128px
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

    print(f"[info] FAST PROTOTYPE dual run: slow {SLOW_CHANNELS} {slow_days}d/{SLOW_BIN}d-bins, "
          f"fast {FAST_CHANNELS} {fast_days}d | pos_weight={pos_weight} "
          f"samples_per_epoch={samples_per_epoch} max_epochs={max_epochs} "
          f"patch_size={patch_size} min_pos_pixels={min_pos_pixels} "
          f"(density {100*min_pos_pixels/patch_size**2:.4f}%) | "
          f"pos_weight_anneal={pos_weight_start}->{pos_weight_end} over {pw_anneal_epochs}ep | "
          f"isolation_gamma={isolation_gamma} kernel={isolation_kernel}")

    # Pre-binned slow cube (build_slow_cube.py). Without it the datamodule falls
    # back to aggregating 144 raw days per sample: correct, but 22x slower
    # (2.30s vs 0.103s/sample) because it pulls 1079MB to keep 14MB.
    slow_cube = str(SCRIPT_DIR / "cube_slow_8day.zarr")
    if not Path(slow_cube).exists():
        raise FileNotFoundError(f"{slow_cube} missing — run build_slow_cube.py first")

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
        num_workers=8,
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
        augment=os.environ.get("AUGMENT","0")=="1",
        fire_history_dropout_prob=float(os.environ.get("FIRE_HISTORY_DROPOUT_PROB", 0.0)),
        new_fire_frac=float(os.environ.get("NEW_FIRE_FRAC", 0.0)),
        min_new_fire_pixels=int(os.environ.get("MIN_NEW_FIRE_PIXELS", 1)),
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
        cosine_t_max=max_epochs,
        fuse=os.environ.get("FUSE", "cross_attn"),   # FireSenseNet-style fusion
        attn_heads=int(os.environ.get("ATTN_HEADS", 4)),
        soft_pos=float(os.environ.get("SOFT_POS", 0.9)),
        soft_neg=float(os.environ.get("SOFT_NEG", 0.02)),
        pos_weight_start=pos_weight_start,
        pos_weight_end=pos_weight_end,
        pos_weight_anneal_epochs=pw_anneal_epochs,
        isolation_gamma=isolation_gamma,
        isolation_kernel=isolation_kernel,
        # OHEM: keep all positives + only the hardest OHEM_FRAC of negatives.
        # 0.0 (default) = disabled, byte-identical to the pre-OHEM loss.
        ohem_frac=float(os.environ.get("OHEM_FRAC", 0.0)),
        ohem_min_negatives=int(os.environ.get("OHEM_MIN_NEG", 256)),
        ohem_rescale_negatives=os.environ.get("OHEM_RESCALE", "1") == "1",
        static_dim=(c_static if use_static_head else 0),
    )

    jobid = os.environ.get("SLURM_JOB_ID", "local")
    run_tag = os.environ.get("RUN_TAG", "dual_fastproto")
    run_dir = Path("/home/saturn/gwgi/gwgi107h/wildfire_data/checkpoints") / run_tag / f"job_{jobid}"
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

    trainer = pl.Trainer(
        max_epochs=max_epochs,
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

    trainer.save_checkpoint(f"checkpoints/last_dual_fastproto_{jobid}.ckpt")
    print(f"[done] wrote checkpoints/last_dual_fastproto_{jobid}.ckpt")


if __name__ == "__main__":
    main()
