import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from smolder.models.backbones.conv_lstm import ConvLSTMSeg

logger = logging.getLogger(__name__)


class ConvLSTMLit(pl.LightningModule):
    """PyTorch-Lightning wrapper around ConvLSTMSeg for wildfire risk segmentation.

    Expected batch from datamodule:
      x:     (B, T, H, W, C) float32/float16
      x_cat: (B, n_cat, H, W) or (B, H, W, n_cat) int ; channel 0 = landcover, channel 1 = Koppen-Geiger
      y:     one of:
               (B, 1, T, H, W) uint8/bool/float
               (B, T, H, W)
               (B, H, W)

    Loss schedules (epoch-based) for extreme class imbalance:
      - dice_to_focal:      BCE+Dice warmup -> focal
      - dice_to_bce:        BCE+Dice warmup -> BCE
      - bce_only:           BCE throughout
      - bce_to_focal:       BCE (optionally weighted) warmup -> focal
      - bce_dice_bce_focal: 3-stage
            epochs [0 .. dice_end_epoch-1]              : BCE + Dice
            epochs [dice_end_epoch .. focal_start_epoch-1] : BCE only (weighted)
            epochs [focal_start_epoch ..]               : focal

    Per-landcover-class training:
      Set lc_class_idx to a specific integer to train only on pixels where
      x_cat[:, 0] == lc_class_idx.  All other pixels are masked from the loss
      and metrics.  At inference, run one model per class and combine with
      ConvLSTMLit.combine_lc_predictions().

      For a single unified model trained on all landcover classes together,
      leave lc_class_idx=None and set emb_dim_lc > 0 so the model receives
      landcover as an input embedding instead of using it to mask the loss.
      emb_dim_lc=0 (default) keeps prior behaviour unchanged; no lc_emb
      parameter is created and existing per-class checkpoints load as before.
    """

    def __init__(
        self,
        input_dim_grid_nodes: int = 7,
        hidden_layers: int = 1,
        hidden_dim: int = 128,
        kernel_size: Union[Tuple[int, int], List[int], int] = (5, 5),
        num_classes: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        target_name: str = "fire",
        use_last_timestep: bool = True,
        pr_auc_every: int = 1,
        pr_auc_thresholds: int = 200,

        # ---- temporal configuration ----
        seq_len: int = 8,

        # ---- loss base (BCE) ----
        pos_weight: Optional[float] = None,

        # ---- loss schedule ----
        loss_schedule: str = "dice_to_focal",
        switch_epoch: int = 3,
        dice_weight: float = 0.5,
        focal_alpha: float = 0.35,
        focal_gamma: float = 1.0,

        # ---- 3-stage schedule knobs ----
        dice_end_epoch: int = 2,
        focal_start_epoch: int = 6,

        # ---- evaluation metrics ----
        topk_fracs: Tuple[float, float] = (0.05, 0.01),

        # ---- categorical embeddings ----
        emb_dim_kg: int = 4,
        emb_dim_lc: int = 0,  # 0 = no landcover embedding (per-class training default)
        cat_num_embeddings: int = 256,

        # ---- per-landcover-class training ----
        lc_class_idx: Optional[int] = None,  # None = all pixels; int = mask to this LC class

        # ---- recency weights ----
        use_recency_weights: bool = False,
        recency_init: str = "linear",

        # ---- soft labels ----
        # Remap hard 0/1 targets to soft targets: background -> soft_neg,
        # fire -> soft_pos. Softening the huge background class stops the loss
        # from driving every non-fire logit to -inf and reduces overconfidence
        # under extreme imbalance; a standard trick in next-day-fire work
        # (bg -> 0.01-0.03, fire -> 0.8-0.99). 0/1 disables it.
        soft_pos: float = 1.0,
        soft_neg: float = 0.0,

        # ---- pos_weight annealing ----
        # pos_weight=100 buys ranking/recall under extreme imbalance at the cost
        # of calibration: the loss-optimal output for a pixel with true
        # probability p is w*p/(w*p+(1-p)), which for w=100 pushes even low-p
        # pixels toward ~0.5+. Annealing w down over training lets the model
        # spend early epochs on "don't ignore the rare class", then relax toward
        # a better-calibrated optimum in later epochs. None/None (default)
        # preserves the old constant-pos_weight behaviour exactly.
        pos_weight_start: Optional[float] = None,
        pos_weight_end: Optional[float] = None,
        pos_weight_anneal_epochs: int = 10,

        # ---- isolation-weighted loss ----
        # Permutation importance showed this model's #1 feature by far is
        # distance-to-recent-fire, with every weather channel near zero -- i.e.
        # it has learned "fire spreads near fire" and has ~no signal for a
        # genuinely NEW ignition with no nearby recent fire (measured: new-fire
        # lift is BELOW random under a 0.2% budget). isolation_gamma upweights
        # the loss on positive pixels that are spatially isolated from other
        # fire in the current target, forcing gradient attention onto exactly
        # the cases the model currently has no reason to learn from.
        # 0.0 (default) disables it, preserving old loss values exactly.
        isolation_gamma: float = 0.0,
        isolation_kernel: int = 9,

        # ---- online hard example mining (OHEM) ----
        # The evaluation metric is a RANKING one (AP, and lift at top-0.5%-1%),
        # but BCE spends most of its gradient budget on the ~99.8% of pixels
        # that are trivially-easy background (2020 test base rate: 0.175%).
        # OHEM keeps every positive but only the hardest `ohem_frac` of the
        # negatives, so gradient goes to the negatives that actually compete
        # with fire pixels for the top-k slots -- exactly the pixels that decide
        # the metric. 0.0 (default) disables it and preserves the previous loss
        # values EXACTLY (verified: identical fast path).
        #
        # ohem_rescale_negatives renormalizes the kept negatives so they carry
        # the same TOTAL LOSS MASS all negatives did, keeping the loss value
        # (and so train_loss curves) directly comparable to non-OHEM runs --
        # measured ratio 1.0000 at every frac. This matters because this
        # training has proven very sensitive to loss-balance shifts (see the
        # focal-switch crash and the pos_weight-annealing findings during
        # development), so ohem_frac should not silently act as a second
        # pos_weight knob.
        #
        # Caveat, measured not assumed: loss mass and GRADIENT mass cannot both
        # be preserved. At frac=0.1 the pos:neg gradient ratio still moves
        # 0.134 -> 0.232 (~1.7x toward positives) because gradient concentrates
        # on the retained hard negatives. That is far milder than the naive
        # n_neg/k scaling (which lands at ~3.1x the original loss mass), but it
        # is not zero -- treat ohem_frac as mostly-orthogonal to pos_weight,
        # not perfectly so.
        ohem_frac: float = 0.0,
        ohem_min_negatives: int = 256,
        ohem_rescale_negatives: bool = True,

        # ---- weight on fire far from PAST fire ----
        # Fire pixels get w = 1 + a * min(d / scale, 1), d = distance (px, about
        # km) to the nearest fire of the last 32 days (batch["past_dist"]),
        # multiplied with every other weight. Background keeps weight 1.
        # a = 0 (default) builds no weight at all: the loss is unchanged.
        past_fire_weight_a: float = 0.0,
        past_fire_weight_scale_px: float = 10.0,
    ):
        super().__init__()

        self.cat_num_embeddings = int(cat_num_embeddings)
        self.kg_emb = nn.Embedding(self.cat_num_embeddings, int(emb_dim_kg))
        self.lc_emb = nn.Embedding(self.cat_num_embeddings, int(emb_dim_lc)) if int(emb_dim_lc) > 0 else None
        self.save_hyperparameters()

        # --- learnable recency weighting over the time dimension ---
        self.use_recency_weights = bool(use_recency_weights)
        if self.use_recency_weights:
            seq_len_cfg = int(seq_len)
            init = torch.linspace(-3, 3, steps=seq_len_cfg) if recency_init == "linear" else torch.zeros(seq_len_cfg)
            self.time_logits = torch.nn.Parameter(init)
        else:
            self.time_logits = None

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_classes = int(num_classes)
        self.target_name = str(target_name)
        self.use_last_timestep = bool(use_last_timestep)

        # Loss schedule config
        self.loss_schedule = str(loss_schedule)
        self.switch_epoch = int(switch_epoch)
        self.dice_weight = float(dice_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.dice_end_epoch = int(dice_end_epoch)
        self.focal_start_epoch = int(focal_start_epoch)

        if self.loss_schedule == "bce_dice_bce_focal":
            if not (0 <= self.dice_end_epoch <= self.focal_start_epoch):
                raise ValueError(
                    f"Expected dice_end_epoch <= focal_start_epoch, "
                    f"got {self.dice_end_epoch} and {self.focal_start_epoch}"
                )

        self.topk_fracs = tuple(float(x) for x in topk_fracs)

        # Base BCE criterion. If pos_weight annealing is configured, start the
        # buffer at pos_weight_start regardless of the constant `pos_weight` arg
        # -- on_train_epoch_start updates it in place every epoch.
        self.pos_weight_start = pos_weight_start
        self.pos_weight_end = pos_weight_end
        self.pos_weight_anneal_epochs = int(pos_weight_anneal_epochs)
        pw_init = pos_weight
        if pos_weight_start is not None and pos_weight_end is not None:
            pw_init = pos_weight_start
            logger.info("pos_weight ANNEALING %s -> %s over %d epochs",
                        pos_weight_start, pos_weight_end, self.pos_weight_anneal_epochs)
        if pw_init is not None:
            pw = torch.tensor(float(pw_init))
            self._bce = nn.BCEWithLogitsLoss(pos_weight=pw)
            logger.info("BCEWithLogitsLoss pos_weight=%s (target=%s)", pw_init, self.target_name)
        else:
            self._bce = nn.BCEWithLogitsLoss()
            logger.info("BCEWithLogitsLoss (no pos_weight) (target=%s)", self.target_name)

        self.isolation_gamma = float(isolation_gamma)
        self.isolation_kernel = int(isolation_kernel)
        self.past_fire_weight_a = float(past_fire_weight_a)
        self.past_fire_weight_scale_px = float(past_fire_weight_scale_px)

        self.ohem_frac = float(ohem_frac)
        self.ohem_min_negatives = int(ohem_min_negatives)
        self.ohem_rescale_negatives = bool(ohem_rescale_negatives)
        if self.ohem_frac > 0.0:
            logger.info("OHEM enabled: keeping hardest %.1f%% of negatives "
                        "(min %d), rescale_negatives=%s",
                        100.0 * self.ohem_frac, self.ohem_min_negatives,
                        self.ohem_rescale_negatives)

        # Backbone: continuous features + kg embedding + optional lc embedding
        effective_input_dim = int(input_dim_grid_nodes) + int(emb_dim_kg) + int(emb_dim_lc)
        self.model = ConvLSTMSeg(
            input_dim=int(effective_input_dim),
            num_layers=int(hidden_layers),
            hidden_dim=int(hidden_dim),
            kernel_size=kernel_size,
            num_classes=int(num_classes),
        )

    # ------------------------------------------------------------------
    # Landcover mask
    # ------------------------------------------------------------------

    def _get_lc_mask(
        self,
        x_cat: Optional[torch.Tensor],
        H: int,
        W: int,
        land_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return (B, H, W) bool mask of pixels the loss should see, or None.

        Combines two things:
          - the landmask, when the batch provides one. ~46% of a sampled 256x256
            patch is ocean, always labelled y=0. Training on it means half the
            negative pool is guaranteed filler, and pos_weight ends up
            compensating for ocean rather than for real class imbalance.
          - the per-landcover-class filter, when lc_class_idx is set.
        """
        lc_class_idx = self.hparams.get("lc_class_idx")

        base = None
        if land_mask is not None:
            base = land_mask.reshape(-1, H, W) > 0.5

        if lc_class_idx is None or x_cat is None:
            return base

        xc = x_cat
        # Collapse time dim if present
        if xc.ndim == 5:
            xc = xc[:, 0]
        # (B,H,W,n_cat) -> (B,n_cat,H,W)
        if xc.ndim == 4 and xc.shape[-1] in (1, 2, 3, 4) and xc.shape[1] not in (1, 2, 3, 4):
            xc = xc.permute(0, 3, 1, 2).contiguous()
        # (B,H,W) with no channel dim -> (B,1,H,W)
        if xc.ndim == 3:
            xc = xc.unsqueeze(1)

        lc = xc[:, 0].long()                   # (B, H, W)
        # Support single int or tuple of ints (grouped LC classes)
        if isinstance(lc_class_idx, (list, tuple)):
            mask = torch.zeros_like(lc, dtype=torch.bool)
            for c in lc_class_idx:
                mask |= (lc == int(c))
        else:
            mask = lc == int(lc_class_idx)      # (B, H, W) bool
        return mask if base is None else (mask & base)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Accepts either:
          - (B, T, H, W, C)  channels-last from dataloader
          - (B, T, C, H, W)  channels-first after _prepare_x()
        """
        if x is None:
            raise ValueError("x is None")
        if x.ndim != 5:
            raise ValueError(f"Expected x to be 5D. Got {tuple(x.shape)}")

        C_expected = int(self.hparams.get("input_dim_grid_nodes", x.shape[-1]))

        # Channels-last (B,T,H,W,C) -> channels-first (B,T,C,H,W)
        if x.shape[-1] == C_expected:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        elif x.shape[2] == C_expected:
            x = x.contiguous()
        else:
            raise ValueError(
                f"Cannot infer channel dimension. "
                f"Got x.shape={tuple(x.shape)}, expected C={C_expected} "
                f"either at dim -1 (B,T,H,W,C) or dim 2 (B,T,C,H,W)."
            )

        # Learnable recency weights, applied on channels-first (B,T,C,H,W).
        # BUGFIX: previously this lived only in the channels-last branch above,
        # but training always passes channels-first via _prepare_x(), so the
        # weights were never applied and never received gradients.
        if getattr(self, "time_logits", None) is not None:
            T_in = x.shape[1]
            logits = self.time_logits
            if logits.numel() != T_in:
                logits = F.interpolate(
                    logits.view(1, 1, -1), size=T_in, mode="linear", align_corners=False
                ).view(-1)
            w = torch.softmax(logits, dim=0)    # (T_in,)
            x = x * w.view(1, T_in, 1, 1, 1)

        B, T = int(x.shape[0]), int(x.shape[1])
        Hx, Wx = int(x.shape[3]), int(x.shape[4])

        # Koppen-Geiger + optional landcover embedding
        kg_e = None
        lc_e = None
        if x_cat is not None:
            xc = x_cat
            if xc.ndim == 5:
                xc = xc[:, 0]
            if xc.ndim == 4 and xc.shape[-1] in (1, 2, 3, 4) and xc.shape[1] not in (1, 2, 3, 4):
                xc = xc.permute(0, 3, 1, 2).contiguous()
            elif xc.ndim == 3:
                xc = xc.unsqueeze(0)

            if xc.ndim != 4:
                raise ValueError(f"Expected x_cat to be 4D (B,n_cat,H,W) after normalization. Got {tuple(xc.shape)}")

            # Spatial alignment check
            if (int(xc.shape[2]), int(xc.shape[3])) == (Wx, Hx):
                xc = xc.transpose(2, 3).contiguous()
            elif (int(xc.shape[2]), int(xc.shape[3])) != (Hx, Wx):
                raise ValueError(
                    f"x_cat spatial shape {(int(xc.shape[2]), int(xc.shape[3]))} "
                    f"does not match x spatial shape {(Hx, Wx)}"
                )

            xc = xc.clamp(0, self.cat_num_embeddings - 1).long()

            if self.lc_emb is not None and xc.shape[1] >= 1:
                lc = xc[:, 0]                                       # (B,H,W)
                lc_e = self.lc_emb(lc)                              # (B,H,W,emb_dim_lc)
                lc_e = lc_e.unsqueeze(1).expand(B, T, -1, -1, -1)  # (B,T,H,W,emb_dim_lc)

            if xc.shape[1] >= 2:
                kg = xc[:, 1]                                       # (B,H,W)
                kg_e = self.kg_emb(kg)                              # (B,H,W,emb_dim_kg)
                kg_e = kg_e.unsqueeze(1).expand(B, T, -1, -1, -1)  # (B,T,H,W,emb_dim_kg)

        if kg_e is None:
            kg_e = torch.zeros(
                (B, T, Hx, Wx, int(self.hparams["emb_dim_kg"])),
                device=x.device, dtype=x.dtype,
            )
        if self.lc_emb is not None and lc_e is None:
            lc_e = torch.zeros(
                (B, T, Hx, Wx, int(self.hparams["emb_dim_lc"])),
                device=x.device, dtype=x.dtype,
            )

        embs = [kg_e] if self.lc_emb is None else [lc_e, kg_e]
        emb = torch.cat(embs, dim=-1).permute(0, 1, 4, 2, 3).contiguous()  # (B,T,E,H,W)
        x = torch.cat([x, emb], dim=2)                      # (B,T,C+E,H,W)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model(x)

    # ------------------------------------------------------------------
    # Prepare helpers
    # ------------------------------------------------------------------

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            raise ValueError("x is None")
        if x.ndim != 5:
            raise ValueError(f"x must be 5D. Got {tuple(x.shape)}")
        x = x.permute(0, 1, 4, 2, 3).contiguous()          # (B,T,C,H,W)
        x = x.to(dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -1e6, 1e6)
        return x

    def _prepare_y(self, y: torch.Tensor) -> torch.Tensor:
        if y is None:
            raise ValueError("y is None")
        if y.ndim == 5:
            y = y[:, 0, -1, :, :]
        elif y.ndim == 4:
            y = y[:, -1, :, :]
        elif y.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported y ndim={y.ndim} shape={tuple(y.shape)}")
        y = y.to(dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.clamp(y, 0.0, 1.0)
        return y

    def _select_logits_map(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 5:
            logits = logits[:, 0, -1, :, :]
        elif logits.ndim == 4:
            logits = logits[:, 0, :, :] if logits.shape[1] == 1 else logits[:, -1, :, :]
        elif logits.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected logits shape={tuple(logits.shape)}")
        if logits.ndim != 3:
            raise ValueError(f"Logits not reduced to (B,H,W). Got {tuple(logits.shape)}")
        return logits

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Soft dice.

        WARNING on shape sensitivity: this reduces per-image when given (B, N)
        but over the whole batch as a single blob when given a flat 1D tensor.
        Those are different objectives (verified: 0.625 vs 0.474 on identical
        data), so a caller that flattens for masking silently trains against a
        different loss than one that does not. Prefer passing (B, N) plus a
        `weights` mask over pre-flattening; see _compute_loss.
        """
        probs = torch.sigmoid(logits)
        if probs.ndim > 1:
            # Spatial case (B, H, W) or (B, N) -> per-image dice, then mean
            probs   = probs.reshape(probs.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            if weights is not None:
                w = weights.reshape(weights.shape[0], -1).to(probs.dtype)
                probs   = probs * w
                targets = targets * w
            intersection = (probs * targets).sum(dim=1)
            denom = probs.sum(dim=1) + targets.sum(dim=1)
            dice = (2.0 * intersection + eps) / (denom + eps)
            return 1.0 - dice.mean()
        else:
            # 1D flat (already-masked pixels); treat all as one "image"
            if weights is not None:
                w = weights.reshape(-1).to(probs.dtype)
                probs   = probs * w
                targets = targets * w
            intersection = (probs * targets).sum()
            denom = probs.sum() + targets.sum()
            return 1.0 - (2.0 * intersection + eps) / (denom + eps)

    def _sigmoid_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        # Carry pos_weight into focal too. Without this the loss silently drops
        # from a 350x positive weighting (BCE phase) to alpha=0.75 the moment
        # focal_start_epoch hits -- a huge discontinuity mid-training on a 0.05%
        # positive rate. alpha and pos_weight are complementary here: alpha
        # rebalances pos/neg, pos_weight restores the magnitude the BCE phase
        # was tuned against.
        pw = getattr(self._bce, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=pw.to(logits.device) if pw is not None else None,
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        modulating = (1.0 - p_t).pow(gamma)
        if alpha is not None:
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * modulating * bce
        else:
            loss = modulating * bce
        if weight is not None:
            loss = loss * weight
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    def _ohem_reduce(
        self,
        per_px: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce a per-pixel loss keeping ALL positives + the hardest
        `ohem_frac` of negatives.

        `targets` may be soft (soft_pos/soft_neg), so positives are taken as
        >0.5 rather than ==1; that is correct for hard labels and for any
        sane soft-label setting (soft_neg well below 0.5, soft_pos well above).

        With ohem_rescale_negatives the surviving negatives are scaled by
        n_neg/k, so the negative term keeps the same total mass it had under a
        plain mean and ohem_frac stays orthogonal to pos_weight -- see the
        constructor docstring.
        """
        flat_loss = per_px.reshape(-1)
        flat_y    = targets.reshape(-1)
        pos = flat_y > 0.5
        neg = ~pos

        n_neg = int(neg.sum())
        n_all = int(flat_loss.numel())
        if n_neg == 0:
            return flat_loss.mean()

        k = int(round(self.ohem_frac * n_neg))
        k = max(k, min(self.ohem_min_negatives, n_neg))
        k = min(k, n_neg)
        if k >= n_neg:
            # keeping everything -- identical to a plain mean
            return flat_loss.mean()

        neg_losses = flat_loss[neg]
        topk_neg, _ = torch.topk(neg_losses, k=k, largest=True, sorted=False)

        neg_sum = topk_neg.sum()
        if self.ohem_rescale_negatives:
            # TRUE mass preservation: rescale so the kept negatives carry the
            # same total as ALL negatives did. Scaling by n_neg/k instead would
            # over-inflate badly -- the hardest 10% already hold ~31% of the
            # negative mass, so an n_neg/k factor lands at ~3.1x the original
            # (measured). The ratio is detached so it acts as a constant
            # multiplier and does not itself carry gradient.
            factor = (neg_losses.sum() / neg_sum.clamp_min(1e-12)).detach()
            neg_sum = neg_sum * factor

        pos_sum = flat_loss[pos].sum()
        # Divide by the ORIGINAL pixel count so the loss stays on the same
        # scale as the non-OHEM path (comparable train_loss curves across runs).
        return (pos_sum + neg_sum) / float(n_all)

    def _weighted_bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same criterion as self._bce, but with an optional per-pixel weight
        multiplier (isolation weighting) alongside the existing pos_weight,
        and optional OHEM negative mining."""
        pw = getattr(self._bce, "pos_weight", None)
        pw = pw.to(logits.device) if pw is not None else None

        if self.ohem_frac <= 0.0:
            # Unchanged fast path -- byte-identical to the previous behaviour.
            if weight is None:
                return self._bce(logits, targets)
            return F.binary_cross_entropy_with_logits(
                logits, targets, weight=weight, pos_weight=pw,
            )

        per_px = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, pos_weight=pw, reduction="none",
        )
        return self._ohem_reduce(per_px, targets)

    def _isolation_weight(self, y_full: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-pixel loss weight upweighting POSITIVE pixels that are spatially
        isolated from other fire in this batch's target, graded across THREE
        scales (isolation_kernel and its 3x/9x widths) instead of one fixed
        radius. A single k=9 box-filter DENSITY saturates at its max weight
        for any pixel with no other fire within ~4px, so it cannot distinguish
        "isolated by 20px" from "isolated by 200px" -- the actual new-fire
        lift curve (2026-07-26) peaks at top-0.5% and falls off on both sides,
        i.e. isolation strength is graded, not binary. Naively averaging raw
        density across scales doesn't fix this either: dividing by an 81x81
        window's 6561 pixels makes even a nearby cluster's contribution too
        small to move the average -- tested directly, the weight difference
        between a truly-isolated pixel and one with fire 40px away came out
        under 0.001. Using per-scale PRESENCE (is there any OTHER fire pixel
        within this radius at all, via a pooled count with the center pixel's
        own contribution subtracted off) instead of raw density avoids that
        dilution and gives four clean, well-separated tiers: fire present even
        at the smallest scale -> weight 1 (not isolated); present only at the
        widest -> intermediate; absent at all three tested scales -> the max
        weight 1+isolation_gamma. Pure PyTorch (avg_pool2d), no CPU/scipy
        roundtrip, safe for per-step training cost.
        Background pixels always keep weight 1. Returns None if isolation_gamma
        <= 0 so callers fall back to the unweighted path unchanged.
        """
        gamma = float(getattr(self, "isolation_gamma", 0.0))
        if gamma <= 0.0:
            return None
        yf = y_full.to(dtype=torch.float32)
        squeeze_back = yf.dim() == 3
        y4 = yf.unsqueeze(1) if squeeze_back else yf
        k0 = int(getattr(self, "isolation_kernel", 9))
        other_presences = []
        for scale in (1, 3, 9):
            k = k0 * scale
            if k % 2 == 0:
                k += 1
            local_count = F.avg_pool2d(y4, kernel_size=k, stride=1, padding=k // 2) * (k * k)
            other_count = (local_count - y4).clamp(min=0.0)
            other_presences.append((other_count > 0.5).float())
        # fraction of tested scales where NO other fire is visible: 1.0 = fully
        # isolated out to the widest scale, 0.0 = other fire even at the tightest.
        isolation_frac = 1.0 - torch.stack(other_presences, dim=0).mean(dim=0)
        if squeeze_back:
            isolation_frac = isolation_frac.squeeze(1)
        iso = 1.0 + gamma * isolation_frac
        weight = torch.where(yf > 0.5, iso, torch.ones_like(iso))
        return weight.detach()

    def _past_fire_weight(self, y: torch.Tensor, dist: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """w = 1 + a * min(d / scale, 1) on fire pixels, 1 elsewhere; None if
        a <= 0 or no distance is given (the loss is then unchanged)."""
        a = float(getattr(self, "past_fire_weight_a", 0.0))
        if a <= 0.0 or dist is None:
            return None
        far = torch.clamp(dist.to(torch.float32) / float(self.past_fire_weight_scale_px), max=1.0)
        w = 1.0 + a * far
        return torch.where(y.to(torch.float32) > 0.5, w, torch.ones_like(w)).detach()

    @staticmethod
    def _mask_select(t: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None or mask is None:
            return t
        return t[mask]

    def _apply_mask(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten logits and y to 1D using a (B,H,W) bool mask.
        Returns originals unchanged when mask is None.
        """
        if mask is None:
            return logits, y
        return logits[mask], y[mask]

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        extra_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        comps: Dict[str, torch.Tensor] = {}

        # Apply LC mask; flatten to (N_pixels,) for the target class only
        logits_m, y_m = self._apply_mask(logits, y, mask)
        # Isolation weight computed on the FULL (pre-mask, pre-soft-label) target
        # so local density reflects real neighbouring fire, then flattened with
        # the same mask as everything else. None when isolation_gamma<=0.
        # extra_weight (same shape as y, e.g. _past_fire_weight) multiplies in.
        weight = self._isolation_weight(y)
        if extra_weight is not None:
            weight = extra_weight if weight is None else weight * extra_weight
        weight_m = self._mask_select(weight, mask)

        if logits_m.numel() == 0:
            # No pixels for this LC class in the batch; return a graph-connected zero
            # so DDP can still all-reduce gradients (avoids undefined-gradient crash)
            zero = (logits * 0.0).mean()
            comps["loss_phase"] = torch.tensor(-1.0, device=logits.device)
            comps["lc_pixels"]  = torch.tensor(0.0, device=logits.device)
            return zero, comps

        if mask is not None:
            comps["lc_pixels"] = torch.tensor(float(logits_m.numel()), device=logits.device)

        # Soft labels for the probabilistic losses (BCE/focal). Dice keeps the
        # hard target since it is a set-overlap measure. y_hard stays 0/1 for
        # metrics; y_m becomes the soft target the loss regresses toward.
        y_hard = y_m
        sp = float(self.hparams.get("soft_pos", 1.0))
        sn = float(self.hparams.get("soft_neg", 0.0))
        if sp != 1.0 or sn != 0.0:
            y_m = y_hard * sp + (1.0 - y_hard) * sn

        ep = int(self.current_epoch)

        # ---- 3-stage: BCE+Dice -> BCE -> focal ----
        if self.loss_schedule == "bce_dice_bce_focal":
            if ep < self.dice_end_epoch:
                bce  = self._weighted_bce(logits_m, y_m, weight_m)
                dice = self._dice_loss(logits_m, y_hard, weights=weight_m)
                loss = (1.0 - self.dice_weight) * bce + self.dice_weight * dice
                comps["loss_bce"]   = bce.detach()
                comps["loss_dice"]  = dice.detach()
                comps["loss_phase"] = torch.tensor(0.0, device=logits.device)
                return loss, comps
            if ep < self.focal_start_epoch:
                bce = self._weighted_bce(logits_m, y_m, weight_m)
                comps["loss_bce"]   = bce.detach()
                comps["loss_phase"] = torch.tensor(1.0, device=logits.device)
                return bce, comps
            focal = self._sigmoid_focal_loss(logits_m, y_m, alpha=self.focal_alpha, gamma=self.focal_gamma, weight=weight_m)
            comps["loss_focal"] = focal.detach()
            comps["loss_phase"] = torch.tensor(2.0, device=logits.device)
            return focal, comps

        # ---- existing schedules ----
        if self.loss_schedule in ("dice_to_bce", "dice_to_focal") and ep < self.switch_epoch:
            bce  = self._weighted_bce(logits_m, y_m, weight_m)
            dice = self._dice_loss(logits_m, y_hard, weights=weight_m)
            loss = (1.0 - self.dice_weight) * bce + self.dice_weight * dice
            comps["loss_bce"]   = bce.detach()
            comps["loss_dice"]  = dice.detach()
            comps["loss_phase"] = torch.tensor(0.0, device=logits.device)
            return loss, comps

        if self.loss_schedule == "bce_to_focal" and ep < self.switch_epoch:
            bce = self._weighted_bce(logits_m, y_m, weight_m)
            comps["loss_bce"]   = bce.detach()
            comps["loss_phase"] = torch.tensor(0.0, device=logits.device)
            return bce, comps

        if self.loss_schedule in ("dice_to_bce", "bce_only"):
            bce = self._weighted_bce(logits_m, y_m, weight_m)
            comps["loss_bce"]   = bce.detach()
            comps["loss_phase"] = torch.tensor(1.0, device=logits.device)
            return bce, comps

        if self.loss_schedule in ("dice_to_focal", "bce_to_focal"):
            focal = self._sigmoid_focal_loss(logits_m, y_m, alpha=self.focal_alpha, gamma=self.focal_gamma, weight=weight_m)
            comps["loss_focal"] = focal.detach()
            comps["loss_phase"] = torch.tensor(1.0, device=logits.device)
            return focal, comps

        raise ValueError(f"Unknown loss_schedule={self.loss_schedule}")

    # ------------------------------------------------------------------
    # Metrics: top-k area recall / precision
    # ------------------------------------------------------------------

    @staticmethod
    def _f1(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return torch.where((p + r) > 0, 2 * p * r / (p + r), torch.zeros_like(p))

    @torch.no_grad()
    def _topk_metrics(self, probs: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = probs.shape[0]
        flat_p = probs.reshape(B, -1)
        flat_y = y.reshape(B, -1)

        out: Dict[str, List[torch.Tensor]] = {}
        for frac in self.topk_fracs:
            k = max(1, int(round(frac * flat_p.shape[1])))
            _, topk_idx = torch.topk(flat_p, k=k, dim=1, largest=True, sorted=False)
            y_topk    = torch.gather(flat_y, 1, topk_idx)
            tp        = y_topk.sum(dim=1)
            total_pos = flat_y.sum(dim=1)
            recall    = torch.where(total_pos > 0, tp / total_pos, torch.zeros_like(tp))
            precision = tp / float(k)
            out.setdefault("recall",    []).append(recall)
            out.setdefault("precision", []).append(precision)
            out.setdefault("k",         []).append(torch.full_like(tp, float(k)))

        metrics: Dict[str, torch.Tensor] = {}
        for i, frac in enumerate(self.topk_fracs):
            tag = f"{frac*100:.1f}".replace(".", "p")
            metrics[f"top{tag}_k"]         = out["k"][i].float().mean()
            metrics[f"top{tag}_recall"]    = out["recall"][i].float().mean()
            metrics[f"top{tag}_precision"] = out["precision"][i].float().mean()
        return metrics

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Update self._bce.pos_weight in place if annealing is configured.
        No-op (old constant-pos_weight behaviour) when pos_weight_start/end
        were not both set at construction time."""
        ps, pe = self.pos_weight_start, self.pos_weight_end
        if ps is None or pe is None:
            return
        ep = int(self.current_epoch)
        frac = min(1.0, ep / max(1, self.pos_weight_anneal_epochs))
        pw_now = float(ps) + (float(pe) - float(ps)) * frac
        with torch.no_grad():
            self._bce.pos_weight.fill_(pw_now)
        self.log("pos_weight_now", pw_now, prog_bar=True, on_step=False, on_epoch=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x     = self._prepare_x(batch.get("x"))
        y     = self._prepare_y(batch.get("y"))
        x_cat = batch.get("x_cat")

        logits = self._select_logits_map(self(x, x_cat=x_cat))
        mask   = self._get_lc_mask(x_cat, y.shape[-2], y.shape[-1], land_mask=batch.get("mask"))
        loss, comps = self._compute_loss(logits, y, mask=mask)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in comps.items():
            self.log(f"train_{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            train_pos_rate        = (y.sum(dim=(1, 2)) > 0).float().mean()
            train_pos_pixels_mean = y.sum(dim=(1, 2)).float().mean()
            self.log("train_y_mean",          y.mean(),              on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_pos_rate",        train_pos_rate,        on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_pos_pixels_mean", train_pos_pixels_mean, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_p_mean",          probs.mean(),          on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def _val_logits_and_target(self, batch: Dict[str, Any]):
        """Return (logits, y, x_cat) for validation.

        Subclasses with a different input contract (e.g. the dual-branch model,
        which takes x_slow/x_fast instead of a single x) override this hook so
        they inherit the metric logging below unchanged -- top-k conditional F1
        must be computed identically across runs to stay comparable.
        """
        x     = self._prepare_x(batch.get("x"))
        y     = self._prepare_y(batch.get("y"))
        x_cat = batch.get("x_cat")
        logits = self._select_logits_map(self(x, x_cat=x_cat))
        return logits, y, x_cat

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, y, x_cat = self._val_logits_and_target(batch)

        mask   = self._get_lc_mask(x_cat, y.shape[-2], y.shape[-1], land_mask=batch.get("mask"))
        loss, comps = self._compute_loss(logits, y, mask=mask)

        B = int(y.shape[0])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B, sync_dist=True)
        for k, v in comps.items():
            self.log(f"val_{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=B, sync_dist=True)

        with torch.no_grad():
            probs = torch.sigmoid(logits)

            # Exclude masked-out pixels (ocean, and non-target landcover) from
            # EVERY metric below.
            #
            # This is not cosmetic. y_fire_3d labels ocean as y=1, and ~46% of a
            # sampled patch is ocean, so an unmasked top-k ranks over the sea and
            # counts it as a true positive -- a model that simply outputs 1 over
            # water scores well. Measured on a 51%-ocean patch: 32,881 of 32,884
            # "positives" were ocean. Every score logged before 2026-07-15
            # (unified 0.3193, shrubland 0.2587, fig1/fig4) is inflated by this.
            #
            # Zero the labels so ocean cannot be a true positive, and push the
            # probabilities to -inf so torch.topk never selects those pixels.
            if mask is not None:
                keep = mask.to(dtype=torch.bool)
                y = torch.where(keep, y, torch.zeros_like(y))
                probs = torch.where(keep, probs, torch.full_like(probs, -float("inf")))

            val_pos_rate        = (y.sum(dim=(1, 2)) > 0).float().mean()
            val_pos_pixels_mean = y.sum(dim=(1, 2)).float().mean()
            self.log("val_last_pos_rate",        val_pos_rate,        on_step=False, on_epoch=True, prog_bar=True,  batch_size=B, sync_dist=True)
            self.log("val_last_pos_pixels_mean", val_pos_pixels_mean, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            self.log("val_y_mean",               y.mean(),            on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            self.log("val_pos_rate",             val_pos_rate,        on_step=False, on_epoch=True, prog_bar=True,  batch_size=B, sync_dist=True)
            self.log("val_pos_pixels_mean",      val_pos_pixels_mean, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            # mean over kept pixels only (probs is -inf elsewhere)
            keep = torch.isfinite(probs)
            p_finite = probs[keep]
            self.log("val_p_mean", p_finite.mean() if p_finite.numel() else torch.zeros((), device=probs.device),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

            # pool kept (prob,label) for epoch-level ROC-AUC / AP. y is already 0
            # on masked pixels, and probs is -inf there, so `keep` selects exactly
            # the land / target-class pixels the metric should see.
            if getattr(self, "_val_p_buf", None) is not None:
                cur = sum(t.numel() for t in self._val_p_buf)
                if cur < self._val_buf_cap:
                    self._val_p_buf.append(p_finite.detach().float().cpu())
                    self._val_y_buf.append(y[keep].detach().float().cpu())

            # New-fire AP: same pooled-AP machinery as val_ap above, but the
            # label is 1 only for fire pixels NOT within ~3px of the fire_hist
            # t-3 mask (batch["y"]'s 4th-from-last step) -- i.e. genuinely new
            # ignitions, not persistence/growth of an already-known fire. This
            # is the metric that actually matters operationally (flagging a
            # pixel next to an already-burning fire is the easy case) and the
            # one operational_stats_dual.py measures per-patch as "liftNEW" --
            # tracking it every epoch means checkpoint selection is no longer
            # blind to it (val_ap is dominated by the easy persistence cases).
            y_raw_seq = batch.get("y")
            if (y_raw_seq is not None and y_raw_seq.dim() == 4 and y_raw_seq.shape[1] >= 4
                    and getattr(self, "_val_newfire_p_buf", None) is not None):
                recent = (y_raw_seq[:, -4] > 0).float().unsqueeze(1).to(probs.device)
                known = F.max_pool2d(recent, kernel_size=7, stride=1, padding=3) > 0.5
                new_fire = (y > 0) & (~known.squeeze(1)) & keep
                cur_nf = sum(t.numel() for t in self._val_newfire_p_buf)
                if cur_nf < self._val_buf_cap:
                    self._val_newfire_p_buf.append(p_finite.detach().float().cpu())
                    self._val_newfire_y_buf.append(new_fire[keep].detach().float().cpu())

            # Unconditional top-k (full spatial map, no LC filter)
            tk = self._topk_metrics(probs, y)
            for k, v in tk.items():
                self.log(f"val_{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=B, sync_dist=True)

            # Conditional top-k (only samples that contain at least one positive pixel)
            pos_mask = (y.sum(dim=(1, 2)) > 0)
            pos_n    = int(pos_mask.sum().item())
            self.log("val_cond_n",    torch.tensor(float(pos_n), device=y.device), on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            self.log("val_cond_frac", pos_mask.float().mean(),                     on_step=False, on_epoch=True, prog_bar=True,  batch_size=B, sync_dist=True)

            if pos_n > 0:
                probs_pos = probs[pos_mask]
                y_pos     = y[pos_mask]
                flat_p    = probs_pos.reshape(pos_n, -1)
                flat_y    = y_pos.reshape(pos_n, -1)
                n_pix     = int(flat_p.shape[1])

                for i, frac in enumerate(self.topk_fracs):
                    k = max(1, int(round(frac * n_pix)))
                    _, topk_idx = torch.topk(flat_p, k=k, dim=1, largest=True, sorted=False)
                    y_topk    = torch.gather(flat_y, 1, topk_idx)
                    tp        = y_topk.sum(dim=1)
                    total_pos = flat_y.sum(dim=1)
                    rec  = torch.where(total_pos > 0, tp / total_pos, torch.zeros_like(tp))
                    prec = tp / float(k)
                    self._val_cond_sum_precision[i] = self._val_cond_sum_precision[i] + prec.float().sum()
                    self._val_cond_sum_recall[i]    = self._val_cond_sum_recall[i]    + rec.float().sum()

                self._val_cond_count += pos_n

        return loss

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def _log_time_weights(self, prefix: str = "val") -> None:
        if getattr(self, "time_logits", None) is None:
            return
        w = torch.softmax(self.time_logits.detach(), dim=0)
        for i in range(int(w.numel())):
            self.log(f"{prefix}_time_w_t{i:02d}", w[i], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if int(w.numel()) >= 3:
            self.log(f"{prefix}_time_w_recent3", w[-3:].sum(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_time_w_old3",    w[:3].sum(),  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        self._val_cond_count         = 0
        self._val_cond_sum_precision = [torch.tensor(0.0, device=self.device) for _ in self.topk_fracs]
        self._val_cond_sum_recall    = [torch.tensor(0.0, device=self.device) for _ in self.topk_fracs]
        # pooled masked (prob,label) for threshold-free ranking metrics (ROC-AUC,
        # AP). These are what the literature selects on and where the signal is
        # actually visible -- the conditional top-k composite sits near the noise
        # floor even when the model ranks fire at AUC ~0.73, so selecting on it
        # kept near-random checkpoints. Subsample-capped to bound memory.
        self._val_p_buf: List[torch.Tensor] = []
        self._val_y_buf: List[torch.Tensor] = []
        self._val_buf_cap = 4_000_000
        # pooled (prob, new-fire label) for val_ap_newfire -- see validation_step.
        self._val_newfire_p_buf: List[torch.Tensor] = []
        self._val_newfire_y_buf: List[torch.Tensor] = []

    def on_train_epoch_end(self) -> None:
        self._log_time_weights(prefix="train")

    def on_validation_epoch_end(self) -> None:
        n = int(getattr(self, "_val_cond_count", 0))
        if n > 0:
            means_p = [s / float(n) for s in self._val_cond_sum_precision]
            means_r = [s / float(n) for s in self._val_cond_sum_recall]

            for i, frac in enumerate(self.topk_fracs):
                tag = f"{frac*100:.1f}".replace(".", "p")
                self.log(f"val_top{tag}_precision_cond", means_p[i], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"val_top{tag}_recall_cond",    means_r[i], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

            p1  = means_p[0] if len(means_p) > 0 else torch.tensor(0.0, device=self.device)
            r1  = means_r[0] if len(means_r) > 0 else torch.tensor(0.0, device=self.device)
            p5  = means_p[1] if len(means_p) > 1 else torch.tensor(0.0, device=self.device)
            r5  = means_r[1] if len(means_r) > 1 else torch.tensor(0.0, device=self.device)
            p15 = means_p[2] if len(means_p) > 2 else torch.tensor(0.0, device=self.device)
            r15 = means_r[2] if len(means_r) > 2 else torch.tensor(0.0, device=self.device)

            f1_1  = self._f1(p1,  r1)
            f1_5  = self._f1(p5,  r5)
            f1_15 = self._f1(p15, r15)

            self.log("val_top1p0_f1_cond",         f1_1,  prog_bar=True,  on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_top5p0_f1_cond",         f1_5,  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_top15p0_f1_cond",        f1_15, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

            composite = 0.5 * f1_1 + 0.3 * f1_5 + 0.2 * f1_15
            self.log("val_topk_f1_composite_cond", composite, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # threshold-free ranking metrics on pooled kept pixels (ROC-AUC, AP).
        # AP is the primary model-selection metric (val_ap): it is where the
        # signal is visible and it is what the wildfire literature selects on
        # (e.g. Huot et al. use AUC-PR). Guarded so a single-class buffer, which
        # would make the metrics undefined, logs a neutral 0 instead of crashing.
        buf_p = getattr(self, "_val_p_buf", None)
        if buf_p:
            p = torch.cat(buf_p).numpy()
            yv = torch.cat(self._val_y_buf).numpy()
            import numpy as _np
            val_ap = 0.0
            val_rocauc = 0.5
            if yv.sum() > 0 and yv.sum() < len(yv):
                try:
                    from sklearn.metrics import average_precision_score, roc_auc_score
                    val_ap = float(average_precision_score(yv, p))
                    val_rocauc = float(roc_auc_score(yv, p))
                except Exception as e:
                    logger.warning("val AP/ROC-AUC failed: %s", e)
            self.log("val_ap",      torch.tensor(val_ap,     device=self.device), prog_bar=True,  on_step=False, on_epoch=True, sync_dist=False)
            self.log("val_roc_auc", torch.tensor(val_rocauc, device=self.device), prog_bar=True,  on_step=False, on_epoch=True, sync_dist=False)
            self._val_p_buf = None
            self._val_y_buf = None

        # val_ap_newfire: same pooled-AP computation, restricted to genuinely
        # NEW ignitions as the positive label (see validation_step). Logged
        # alongside val_ap, not used for checkpoint selection yet -- just makes
        # this visible every epoch instead of only via a separate offline
        # script (operational_stats_dual.py) after training finishes.
        buf_np = getattr(self, "_val_newfire_p_buf", None)
        if buf_np:
            p2 = torch.cat(buf_np).numpy()
            yv2 = torch.cat(self._val_newfire_y_buf).numpy()
            val_ap_newfire = 0.0
            if yv2.sum() > 0 and yv2.sum() < len(yv2):
                try:
                    from sklearn.metrics import average_precision_score
                    val_ap_newfire = float(average_precision_score(yv2, p2))
                except Exception as e:
                    logger.warning("val new-fire AP failed: %s", e)
            self.log("val_ap_newfire", torch.tensor(val_ap_newfire, device=self.device),
                     prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            self._val_newfire_p_buf = None
            self._val_newfire_y_buf = None

        self._log_time_weights(prefix="val")

    def configure_optimizers(self):
        logger.info("Using AdamW optimizer")
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # ------------------------------------------------------------------
    # Inference: combine per-class models into a unified risk score
    # ------------------------------------------------------------------

    @staticmethod
    def combine_lc_predictions(
        prob_maps: Dict[int, torch.Tensor],
        lc_map: torch.Tensor,
        fallback: float = 0.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Combine per-landcover-class probability maps into a single [0, 1] risk score.

        For each pixel, selects the probability from the model trained on that pixel's
        landcover class, then optionally min-max normalises the assembled map.

        Args:
            prob_maps:  {lc_class_idx: (H, W) sigmoid probability map}.
                        Each value is torch.sigmoid() output of the model trained
                        with lc_class_idx == that key.
            lc_map:     (H, W) integer tensor of landcover labels
                        (same encoding as x_cat[:, 0] used during training).
            fallback:   Risk value for pixels whose LC class has no trained model.
                        Defaults to 0.0 (treat unknown = no risk).
            normalize:  If True, min-max scale the assembled map to [0, 1].
                        Useful when class-specific models output probabilities at
                        different absolute scales.

        Returns:
            risk: (H, W) float32 tensor in [0, 1], 1 = highest risk.

        Example::
            prob_maps = {
                10: model_cls10(x, x_cat).sigmoid().squeeze(),
                20: model_cls20(x, x_cat).sigmoid().squeeze(),
                30: model_cls30(x, x_cat).sigmoid().squeeze(),
            }
            risk = ConvLSTMLit.combine_lc_predictions(prob_maps, lc_map)
        """
        H, W   = lc_map.shape
        device = lc_map.device
        risk   = torch.full((H, W), float(fallback), dtype=torch.float32, device=device)

        for cls_idx, prob in prob_maps.items():
            mask = lc_map == int(cls_idx)
            if mask.any():
                risk[mask] = prob.to(device=device, dtype=torch.float32)[mask]

        if normalize:
            r_min = risk.min()
            r_max = risk.max()
            if (r_max - r_min) > 1e-8:
                risk = (risk - r_min) / (r_max - r_min)

        return risk.clamp(0.0, 1.0)
