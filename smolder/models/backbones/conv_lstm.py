import lightning as L
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#
# Model from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
#


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dilation=1):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        k = kernel_size[0]
        d = dilation
        p = d * (k - 1) // 2
        self.padding = p, p
        self.bias = bias
        self.dilation = dilation, dilation
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
        dilation=1,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dilation = dilation
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    dilation=self.dilation,
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMSeg(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        num_classes: number of classes

    Input:
        A tensor of size B, T, C, H, W
    Output:
        A tensor of size B, 1, H, W
    Example:
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        num_classes,
    ):
        super(ConvLSTMSeg, self).__init__()

        if (
            not isinstance(kernel_size, tuple)
            or len(kernel_size) != 2
            or not all(isinstance(i, int) for i in kernel_size)
        ):
            raise ValueError("Invalid kernel size, expected a tuple e.g. (5, 5)")

        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dilation=1,
        )
        padding = (kernel_size[0] - 1) // 2
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        _, last_states = self.convlstm(x)
        out = last_states[0][0]  # 0 for layer index, 0 for h index
        out = self.classification_layer(out)
        return out


class ConvLSTMSegDual(nn.Module):
    """Two-branch ConvLSTM: separate temporal encoders for slow and fast predictors.

    Motivation: the lagged-skill analysis (lagged_skill_extended_agg.csv, 6 years
    of patched data, lags 0-180) shows the predictors carry information on very
    different time scales. NDVI/LAI peak at lag 130, SM/PPT at lag 150, while
    VPD peaks at lag 0 and LST at lag 10 (both collapsing to ~0.45 by 90 days).
    A single uniform window is therefore too short for one group and too long for
    the other. Each branch gets its own ConvLSTM over its own window/cadence; the
    two final hidden states are concatenated before the segmentation head.

    Parameters:
        input_dim_slow: channels in the slow input (predictors + statics + emb)
        input_dim_fast: channels in the fast input (predictors + statics + emb)
        hidden_dim: hidden channels per branch (head sees 2 * hidden_dim)
        kernel_size: conv kernel, e.g. (5, 5)
        num_layers: ConvLSTM layers per branch
        num_classes: output channels

    Input:
        x_slow: (B, T_slow, C_slow, H, W)
        x_fast: (B, T_fast, C_fast, H, W)
    Output:
        (B, num_classes, H, W)
    """

    def __init__(
        self,
        input_dim_slow,
        input_dim_fast,
        hidden_dim,
        kernel_size,
        num_layers,
        num_classes,
        fuse="concat",   # "concat" (default) or "cross_attn"
        attn_heads=4,
        static_dim=0,    # extra pointwise path for constant-over-time statics (0=disabled)
        dilation=1,      # grows receptive field ((k-1)*dilation+1 per conv) at CONSTANT
                         # parameter count -- unlike a bigger kernel (kernel=9x9 tripled
                         # params to 3.7M and collapsed to 0.20 val_ap in a
                         # development run), same-padding dilation keeps params identical to
                         # the kernel=5x5 baseline while still widening what each conv sees.
    ):
        super(ConvLSTMSegDual, self).__init__()

        if (
            not isinstance(kernel_size, tuple)
            or len(kernel_size) != 2
            or not all(isinstance(i, int) for i in kernel_size)
        ):
            raise ValueError("Invalid kernel size, expected a tuple e.g. (5, 5)")

        self.fuse = str(fuse)
        self.hidden_dim = int(hidden_dim)
        self.dilation = int(dilation)

        self.convlstm_slow = ConvLSTM(
            input_dim=input_dim_slow,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dilation=self.dilation,
        )
        self.convlstm_fast = ConvLSTM(
            input_dim=input_dim_fast,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dilation=self.dilation,
        )
        padding = self.dilation * (kernel_size[0] - 1) // 2

        # Constant-over-time statics (elevation, lightning, agb, aspect_sin, ...)
        # get broadcast through the recurrent branches above, but permutation
        # importance on the first full-feature checkpoint (2026-07-25/26) showed
        # elevation/lightning/aspect_sin contribute ~0 through that path -- most
        # likely diluted by 14-18 recurrent steps of a value that never changes,
        # or too spatially flat within a patch for a small conv kernel to find
        # local contrast in. This gives them a second, POINTWISE (1x1 kernel, no
        # spatial mixing) route straight to the fusion head, since they're
        # per-pixel absolute levels, not spatial gradients -- additive alongside
        # the existing broadcast path, not a replacement for it.
        self.static_dim = int(static_dim)
        head_in = 2 * hidden_dim
        if self.static_dim > 0:
            self.static_head = nn.Sequential(
                nn.Conv2d(self.static_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            )
            head_in += hidden_dim

        if self.fuse == "cross_attn":
            # Per-pixel cross-attention: the fast (weather) state queries the slow
            # (fuel/moisture) state, so each location's short-term forcing is read
            # in the context of its slow-varying predisposition. This is the
            # fusion in the current next-day-fire SOTA (FireSenseNet's dual-branch
            # cross-attentive design). Attention is over the CHANNEL feature at
            # each pixel (tokens = the two branch vectors), which keeps it O(HW)
            # rather than O((HW)^2) and preserves spatial resolution.
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=int(attn_heads), batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
            # head sees attended-fast + slow context (+ static_emb if enabled)
            self.classification_layer = nn.Conv2d(
                in_channels=head_in, out_channels=num_classes,
                kernel_size=kernel_size, padding=padding, dilation=self.dilation,
            )
        else:
            # head sees both branches' hidden states concatenated on the channel
            # axis (+ static_emb if enabled)
            self.classification_layer = nn.Conv2d(
                in_channels=head_in, out_channels=num_classes,
                kernel_size=kernel_size, padding=padding, dilation=self.dilation,
            )

    def encode_static(self, x_static):
        """x_static: (B, C_static, H, W) -> (B, hidden_dim, H, W), or None if disabled."""
        if self.static_dim <= 0 or x_static is None:
            return None
        return self.static_head(x_static)

    def _cross_attend(self, h_fast, h_slow):
        """Fast queries slow, per pixel. h_*: (B, C, H, W) -> fused (B, 2C, H, W)."""
        B, C, H, W = h_fast.shape
        q = h_fast.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)   # (N,1,C) query
        kv = h_slow.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)  # (N,1,C) key/value
        attended, _ = self.attn(q, kv, kv)                        # (N,1,C)
        attended = self.attn_norm(attended + q).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return torch.cat([attended, h_slow], dim=1)

    def encode(self, x_slow, x_fast):
        """Return (h_slow, h_fast_seq): slow final hidden state (B,C,H,W) and the
        fast branch's per-timestep hidden states (B,T_fast,C,H,W).

        Deep supervision uses the fast axis (it is daily, so y[t] aligns with it),
        with the slow branch's final state broadcast across those timesteps.
        """
        _, slow_last = self.convlstm_slow(x_slow)
        h_slow = slow_last[0][0]                    # (B, hidden, H, W)
        fast_layers, _ = self.convlstm_fast(x_fast)
        h_fast_seq = fast_layers[0]                 # (B, T_fast, hidden, H, W)
        return h_slow, h_fast_seq

    def fuse_states(self, h_slow, h_fast):
        """Fuse a slow and fast hidden state (both (B,C,H,W)) -> (B,2C,H,W)."""
        if self.fuse == "cross_attn":
            return self._cross_attend(h_fast, h_slow)
        return torch.cat([h_slow, h_fast], dim=1)

    def forward(self, x_slow, x_fast, x_static=None):
        h_slow, h_fast_seq = self.encode(x_slow, x_fast)
        h_fast = h_fast_seq[:, -1]                  # (B, hidden, H, W)
        fused = self.fuse_states(h_slow, h_fast)
        static_emb = self.encode_static(x_static)
        if static_emb is not None:
            fused = torch.cat([fused, static_emb], dim=1)
        return self.classification_layer(fused)


class ConvLSTMSegTriple(ConvLSTMSegDual):
    """Three-branch ConvLSTM: fast / slow / very-slow temporal encoders.

    Adds a YEAR-SCALE branch (432 d at 24-day bins) on top of the dual model's
    144 d and 14 d branches. Rationale: the single-lag skill curve DECLINES after
    ~150 days, but ACCUMULATED signal keeps improving -- measured on identical
    fire patches, mean-over-window AUC for soil moisture goes
    0.477 (single lag @150) -> 0.515 (144 d) -> 0.602 (432 d); PPT 0.501 -> 0.577
    -> 0.616; LAI 0.672 -> 0.679 -> 0.711. Cumulative moisture deficit therefore
    carries multi-season drought information that no single lag exposes, which is
    why SM had ~zero permutation importance in the 30-day model.

    Fusion: the fast branch queries the concatenated (slow, very-slow) context
    when fuse="cross_attn", else all three final states are concatenated.
    """

    def __init__(self, input_dim_slow, input_dim_fast, hidden_dim, kernel_size,
                 num_layers, num_classes, input_dim_vslow=None,
                 fuse="concat", attn_heads=4):
        super().__init__(input_dim_slow=input_dim_slow, input_dim_fast=input_dim_fast,
                         hidden_dim=hidden_dim, kernel_size=kernel_size,
                         num_layers=num_layers, num_classes=num_classes,
                         fuse=fuse, attn_heads=attn_heads)
        self.convlstm_vslow = ConvLSTM(
            input_dim=int(input_dim_vslow if input_dim_vslow is not None else input_dim_slow),
            hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers,
            batch_first=True, bias=True, return_all_layers=False, dilation=1,
        )
        padding = (kernel_size[0] - 1) // 2
        # head now sees fast + slow + very-slow
        self.classification_layer = nn.Conv2d(
            in_channels=3 * hidden_dim, out_channels=num_classes,
            kernel_size=kernel_size, padding=padding,
        )

    def encode3(self, x_slow, x_fast, x_vslow):
        h_slow, h_fast_seq = self.encode(x_slow, x_fast)
        _, vlast = self.convlstm_vslow(x_vslow)
        return h_slow, h_fast_seq, vlast[0][0]

    def fuse3(self, h_slow, h_fast, h_vslow):
        """(B,C,H,W) x3 -> (B,3C,H,W)."""
        if self.fuse == "cross_attn":
            # fast attends over the slow context, then very-slow is appended
            fused = self._cross_attend(h_fast, h_slow)       # (B,2C,H,W)
            return torch.cat([fused, h_vslow], dim=1)
        return torch.cat([h_slow, h_fast, h_vslow], dim=1)

    def forward(self, x_slow, x_fast, x_vslow):
        h_slow, h_fast_seq, h_vslow = self.encode3(x_slow, x_fast, x_vslow)
        return self.classification_layer(
            self.fuse3(h_slow, h_fast_seq[:, -1], h_vslow))
