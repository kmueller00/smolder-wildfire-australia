"""SMOLDER Lightning module: two ConvLSTM encoders fused by cross-attention.

The slow encoder reads the 144-day / 8-day-bin window, the fast encoder the
14-day daily window. At every fast timestep the fast hidden state queries the
slow encoder's final state (multi-head cross-attention, per pixel), and a 5x5
convolution over the 128 fused channels produces fire logits. The head is
applied at every fast timestep (deep supervision, auxiliary weight 0.3);
predictions use the last timestep.
Training losses, the pos_weight schedule and validation metrics are inherited
from ConvLSTMLitV2 / ConvLSTMLit.
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from smolder.models.conv_lstm_lit_v2 import ConvLSTMLitV2
from smolder.models.backbones.conv_lstm import ConvLSTMSegDual

logger = logging.getLogger(__name__)


class ConvLSTMLitDual(ConvLSTMLitV2):
    def __init__(
        self,
        *args,
        input_dim_slow: int = 7,
        input_dim_fast: int = 7,
        fuse: str = "concat",     # "concat" or "cross_attn" (FireSenseNet-style)
        attn_heads: int = 4,
        static_dim: int = 0,      # pointwise path for constant-over-time statics (0=disabled)
        dilation: int = 1,        # same-padding conv dilation, constant param count (see ConvLSTMSegDual)
        **kwargs,
    ):
        # Parent builds a single-branch ConvLSTMSeg from input_dim_grid_nodes; we
        # replace it below. Pass the fast dim so any parent-side shape checks and
        # the saved hparams stay self-consistent.
        kwargs.setdefault("input_dim_grid_nodes", input_dim_fast)
        super().__init__(*args, **kwargs)

        emb = int(self.hparams["emb_dim_kg"]) + int(self.hparams.get("emb_dim_lc", 0) or 0)
        self.model = ConvLSTMSegDual(
            input_dim_slow=int(input_dim_slow) + emb,
            input_dim_fast=int(input_dim_fast) + emb,
            hidden_dim=int(self.hparams["hidden_dim"]),
            kernel_size=tuple(self.hparams["kernel_size"]),
            num_layers=int(self.hparams["hidden_layers"]),
            num_classes=int(self.hparams["num_classes"]),
            fuse=str(fuse),
            attn_heads=int(attn_heads),
            static_dim=int(static_dim),   # deliberately NOT +emb: no kg/lc embeddings here
            dilation=int(dilation),
        )
        self.save_hyperparameters()

    # ------------------------------------------------------------------

    def _prep_branch(self, x: torch.Tensor, x_cat: Optional[torch.Tensor], c_expected: int) -> torch.Tensor:
        """(B,T,H,W,C) -> (B,T,C+emb,H,W) with lc/kg embeddings concatenated.

        Reuses the parent's _prepare_input, which handles layout, embeddings and
        nan_to_num. It reads input_dim_grid_nodes to infer the channel axis, so
        temporarily point that at this branch's width.
        """
        prev = self.hparams["input_dim_grid_nodes"]
        self.hparams["input_dim_grid_nodes"] = int(c_expected)
        try:
            return self._prepare_input(x, x_cat)
        finally:
            self.hparams["input_dim_grid_nodes"] = prev

    def _prep_static(self, x_static: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """(B,H,W,C_static) -> (B,C_static,H,W). No embeddings, no nan_to_num --
        upstream statics are already NaN-free (same invariant x_slow/x_fast rely
        on for this same block before it's broadcast into them)."""
        if x_static is None:
            return None
        return x_static.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

    def forward_seq(self, x_slow: torch.Tensor, x_fast: torch.Tensor,
                    x_cat: Optional[torch.Tensor] = None,
                    x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-timestep logits on the FAST axis: (B, T_fast, H, W)."""
        c_slow = int(self.hparams["input_dim_slow"])
        c_fast = int(self.hparams["input_dim_fast"])
        xs = self._prep_branch(x_slow, x_cat, c_slow)
        xf = self._prep_branch(x_fast, x_cat, c_fast)

        h_slow, h_fast_seq = self.model.encode(xs, xf)   # (B,C,H,W), (B,T,C,H,W)
        B, T, Ch, H, W = h_fast_seq.shape
        # fuse the slow summary with each fast timestep (concat or cross-attn),
        # folding time into batch so fuse_states sees (N,C,H,W)
        h_slow_rep = h_slow.unsqueeze(1).expand(B, T, Ch, H, W).reshape(B * T, Ch, H, W)
        h_fast_flat = h_fast_seq.reshape(B * T, Ch, H, W)
        fused = self.model.fuse_states(h_slow_rep, h_fast_flat)  # (B*T, 2C, H, W)

        static_emb = self.model.encode_static(self._prep_static(x_static))  # (B,hidden,H,W) or None
        if static_emb is not None:
            Cs = static_emb.shape[1]
            static_rep = static_emb.unsqueeze(1).expand(B, T, Cs, H, W).reshape(B * T, Cs, H, W)
            fused = torch.cat([fused, static_rep], dim=1)

        logits = self.model.classification_layer(fused)
        return logits.reshape(B, T, -1, H, W)[:, :, 0]

    def forward(self, x_slow: torch.Tensor, x_fast: torch.Tensor,
                x_cat: Optional[torch.Tensor] = None,
                x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Last-timestep logits (B, 1, H, W)."""
        return self.forward_seq(x_slow, x_fast, x_cat, x_static)[:, -1].unsqueeze(1)

    # ------------------------------------------------------------------

    def _shared_step(self, batch: Dict[str, Any]):
        x_slow = batch["x_slow"]
        x_fast = batch["x_fast"]
        x_cat = batch.get("x_cat")
        x_static = batch.get("x_static")
        y_seq = self._prepare_y_seq(batch.get("y"))     # (B,T_fast,H,W)
        logits_seq = self.forward_seq(x_slow, x_fast, x_cat, x_static)
        return logits_seq, y_seq, x_cat

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        logits_seq, y_seq, x_cat = self._shared_step(batch)
        y_last = y_seq[:, -1]
        mask = self._get_lc_mask(x_cat, y_last.shape[-2], y_last.shape[-1],
                                 land_mask=batch.get("mask"))

        pdist = batch.get("past_dist")              # (B, T, H, W) or None
        loss_main, comps = self._compute_loss(
            logits_seq[:, -1], y_last, mask=mask,
            extra_weight=self._past_fire_weight(y_last, pdist[:, -1] if pdist is not None else None))

        T = logits_seq.shape[1]
        if self.aux_loss_weight > 0 and T > 1:
            la = logits_seq[:, :-1].reshape(-1, *logits_seq.shape[-2:])
            ya = y_seq[:, :-1].reshape(-1, *y_seq.shape[-2:])
            ma = mask.repeat_interleave(T - 1, dim=0) if mask is not None else None
            da = pdist[:, :-1].reshape(-1, *pdist.shape[-2:]) if pdist is not None else None
            loss_aux, _ = self._compute_loss(la, ya, mask=ma, extra_weight=self._past_fire_weight(ya, da))
            loss = loss_main + self.aux_loss_weight * loss_aux
            self.log("train_loss_aux", loss_aux.detach(), on_step=False, on_epoch=True)
        else:
            loss = loss_main

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_main", loss_main.detach(), on_step=False, on_epoch=True)
        for k, v in comps.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True)

        with torch.no_grad():
            probs = torch.sigmoid(logits_seq[:, -1])
            self.log("train_pos_rate", (y_last.sum(dim=(1, 2)) > 0).float().mean(),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_p_mean", probs.mean(), on_step=False, on_epoch=True)
        return loss

    def _val_logits_and_target(self, batch: Dict[str, Any]):
        """Two-input override of the parent's validation hook.

        Everything downstream (top-k conditional F1, val_p_mean, ...) is then
        inherited unchanged, so dual results stay directly comparable to
        focal30v2_* and focal30v2_unified.
        """
        logits = self.forward(batch["x_slow"], batch["x_fast"], batch.get("x_cat"), batch.get("x_static"))
        logits = self._select_logits_map(logits)
        y = self._prepare_y(batch.get("y"))     # last timestep, (B,H,W)
        return logits, y, batch.get("x_cat")
