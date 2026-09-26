"""V2 Lightning module: ConvLSTMLit + deep supervision + cosine LR schedule.

Deep supervision: the ConvLSTM already computes a hidden state for every
timestep; V1 discarded 29 of 30 by supervising only the last. Here the
classification head is applied to every timestep and an auxiliary loss over
t < T-1 (weight aux_loss_weight) is added to the main last-timestep loss.
The target y is (B, T, H, W): y[:, t] = fire in the 3 days after day t, so
every timestep has a valid label.

Validation and all metrics remain last-timestep only (comparable to V1).
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from smolder.models.conv_lstm_lit import ConvLSTMLit

logger = logging.getLogger(__name__)


class ConvLSTMLitV2(ConvLSTMLit):

    def __init__(self, *args, aux_loss_weight: float = 0.3, cosine_t_max: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = float(aux_loss_weight)
        self.cosine_t_max = int(cosine_t_max)
        self.save_hyperparameters()

    # ------------------------------------------------------------------

    def _prepare_input(self, x: torch.Tensor, x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """Channel-first x + KG embedding concat (mirrors parent.forward preamble)."""
        if x is None or x.ndim != 5:
            raise ValueError(f"Expected 5D x. Got {None if x is None else tuple(x.shape)}")
        C_expected = int(self.hparams.get("input_dim_grid_nodes", x.shape[-1]))
        if x.shape[-1] == C_expected:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        elif x.shape[2] == C_expected:
            x = x.contiguous()
        else:
            raise ValueError(f"Cannot infer channel dim from {tuple(x.shape)} (C={C_expected})")

        B, T = int(x.shape[0]), int(x.shape[1])
        Hx, Wx = int(x.shape[3]), int(x.shape[4])

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
                raise ValueError(f"Bad x_cat shape {tuple(xc.shape)}")
            if (int(xc.shape[2]), int(xc.shape[3])) == (Wx, Hx):
                xc = xc.transpose(2, 3).contiguous()
            xc = xc.clamp(0, self.cat_num_embeddings - 1).long()
            if self.lc_emb is not None and xc.shape[1] >= 1:
                lc_e = self.lc_emb(xc[:, 0]).unsqueeze(1).expand(B, T, -1, -1, -1)
            if xc.shape[1] >= 2:
                kg_e = self.kg_emb(xc[:, 1]).unsqueeze(1).expand(B, T, -1, -1, -1)

        if kg_e is None:
            kg_e = torch.zeros(
                (B, T, Hx, Wx, int(self.hparams["emb_dim_kg"])), device=x.device, dtype=x.dtype
            )
        if self.lc_emb is not None and lc_e is None:
            lc_e = torch.zeros(
                (B, T, Hx, Wx, int(self.hparams["emb_dim_lc"])), device=x.device, dtype=x.dtype
            )

        embs = [kg_e] if self.lc_emb is None else [lc_e, kg_e]
        emb = torch.cat(embs, dim=-1)
        x = torch.cat([x, emb.permute(0, 1, 4, 2, 3).contiguous()], dim=2)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def forward_seq(self, x: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-timestep logits (B, T, H, W)."""
        x = self._prepare_input(x, x_cat)
        layer_outputs, _ = self.model.convlstm(x)
        h_seq = layer_outputs[0]                      # (B, T, hidden, H, W)
        B, T, Ch, H, W = h_seq.shape
        logits = self.model.classification_layer(h_seq.reshape(B * T, Ch, H, W))
        return logits.reshape(B, T, -1, H, W)[:, :, 0]  # (B, T, H, W)

    def forward(self, x: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Last-timestep logits (B, 1, H, W) — keeps V1 inference interface."""
        return self.forward_seq(x, x_cat)[:, -1].unsqueeze(1)

    # ------------------------------------------------------------------

    def _prepare_y_seq(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 5:
            y = y[:, 0]
        if y.ndim != 4:
            raise ValueError(f"Expected y (B,T,H,W). Got {tuple(y.shape)}")
        y = torch.nan_to_num(y.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(y, 0.0, 1.0)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = self._prepare_x(batch.get("x"))
        y_seq = self._prepare_y_seq(batch.get("y"))
        x_cat = batch.get("x_cat")

        logits_seq = self.forward_seq(x, x_cat)          # (B,T,H,W)
        y_last = y_seq[:, -1]
        mask = self._get_lc_mask(x_cat, y_last.shape[-2], y_last.shape[-1],
                                 land_mask=batch.get("mask"))

        loss_main, comps = self._compute_loss(logits_seq[:, -1], y_last, mask=mask)

        # auxiliary deep supervision on t < T-1 (time folded into batch)
        T = logits_seq.shape[1]
        if self.aux_loss_weight > 0 and T > 1:
            la = logits_seq[:, :-1].reshape(-1, *logits_seq.shape[-2:])
            ya = y_seq[:, :-1].reshape(-1, *y_seq.shape[-2:])
            ma = mask.repeat_interleave(T - 1, dim=0) if mask is not None else None
            loss_aux, _ = self._compute_loss(la, ya, mask=ma)
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

    # validation_step / metrics inherited from parent (last-timestep, V1-comparable)

    def configure_optimizers(self):
        logger.info("AdamW + CosineAnnealingLR(T_max=%d)", self.cosine_t_max)
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_t_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
