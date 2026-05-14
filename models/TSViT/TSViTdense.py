import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.TSViT.module import Attention, PreNorm, FeedForward
import numpy as np
from utils.config_files_utils import get_params_values
from mamba_ssm import Mamba2
from hilbert import decode, encode

class BiHiSSMamba1D(nn.Module):
    """
    Bidirectional Hilbert 1D Mamba with Snake Horizontal for orthogonal coverage.

    Four scan directions for comprehensive 2D coverage:
    1. Hilbert forward: Space-filling curve preserving 2D locality
    2. Hilbert backward: Reversed Hilbert for bidirectional context
    3. Snake H forward: Row-wise snake (horizontal continuity)
    4. Snake H backward: Reversed Snake H for bidirectional context

    All four outputs are fused to combine these perspectives.
    """

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.dim = dim
        mamba_kw = dict(d_model=dim, d_state=d_state, d_conv=d_conv,
                        expand=expand)

        self.mamba_hilbert_forward = Mamba2(**mamba_kw)
        self.mamba_hilbert_backward = Mamba2(**mamba_kw)
        self.mamba_snake_h_forward = Mamba2(**mamba_kw)
        self.mamba_snake_h_backward = Mamba2(**mamba_kw)

        # Fusion layer to combine all 4 directions
        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # Cache for ordering indices
        self._order_cache = {}

    def get_hilbert_order(self, H, W, device):
        """Get cached Hilbert ordering indices."""
        key = ('hilbert', H, W, str(device))
        if key not in self._order_cache:
            num_bits = int(np.ceil(np.log2(max(H, W))))

            rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            coords = np.stack([rows.flatten(), cols.flatten()], axis=1)

            hilbert_indices = encode(coords, 2, num_bits)
            to_hilbert = np.argsort(hilbert_indices)
            from_hilbert = np.argsort(to_hilbert)

            self._order_cache[key] = (
                torch.from_numpy(to_hilbert).long().to(device),
                torch.from_numpy(from_hilbert).long().to(device)
            )

        return self._order_cache[key]

    def get_snake_h_order(self, H, W, device):
        """
        Get horizontal snake ordering (row-wise).

        Snake H pattern for 4x4:
         0  1  2  3  →
         7  6  5  4  ←
         8  9 10 11  →
        15 14 13 12  ←
        """
        key = ('snake_h', H, W, str(device))
        if key not in self._order_cache:
            indices = []
            for i in range(H):
                if i % 2 == 0:
                    for j in range(W):
                        indices.append(i * W + j)
                else:
                    for j in range(W - 1, -1, -1):
                        indices.append(i * W + j)

            to_snake = np.array(indices)
            from_snake = np.argsort(to_snake)

            self._order_cache[key] = (
                torch.from_numpy(to_snake).long().to(device),
                torch.from_numpy(from_snake).long().to(device)
            )

        return self._order_cache[key]

    def get_snake_v_order(self, H, W, device):
        """
        Get vertical snake ordering (column-wise boustrophedon).

        Snake V pattern for 4x4:
         0  7  8 15
         1  6  9 14
         2  5 10 13
         3  4 11 12
         ↓  ↑  ↓  ↑
        """
        key = ('snake_v', H, W, str(device))
        if key not in self._order_cache:
            indices = []
            for j in range(W):
                if j % 2 == 0:
                    for i in range(H):
                        indices.append(i * W + j)
                else:
                    for i in range(H - 1, -1, -1):
                        indices.append(i * W + j)

            to_snake = np.array(indices)
            from_snake = np.argsort(to_snake)

            self._order_cache[key] = (
                torch.from_numpy(to_snake).long().to(device),
                torch.from_numpy(from_snake).long().to(device)
            )

        return self._order_cache[key]

    def get_raster_row_order(self, H, W, device):
        """Row-major (standard raster) flattening: (0,0),(0,1),…,(H-1,W-1)."""
        key = ('raster_row', H, W, str(device))
        if key not in self._order_cache:
            to_r = np.arange(H * W, dtype=np.int64)
            from_r = np.arange(H * W, dtype=np.int64)
            self._order_cache[key] = (
                torch.from_numpy(to_r).long().to(device),
                torch.from_numpy(from_r).long().to(device),
            )
        return self._order_cache[key]

    def get_raster_col_order(self, H, W, device):
        """Column-major flattening: down each column, left to right in column index."""
        key = ('raster_col', H, W, str(device))
        if key not in self._order_cache:
            indices = [i * W + j for j in range(W) for i in range(H)]
            to_c = np.array(indices, dtype=np.int64)
            from_c = np.argsort(to_c)
            self._order_cache[key] = (
                torch.from_numpy(to_c).long().to(device),
                torch.from_numpy(from_c).long().to(device),
            )
        return self._order_cache[key]

    def get_row_flip_lr_order(self, H, W, device):
        """Every row traversed right-to-left (mirror of row-major; not boustrophedon)."""
        key = ('row_flip_lr', H, W, str(device))
        if key not in self._order_cache:
            indices = [i * W + (W - 1 - j) for i in range(H) for j in range(W)]
            to_f = np.array(indices, dtype=np.int64)
            from_f = np.argsort(to_f)
            self._order_cache[key] = (
                torch.from_numpy(to_f).long().to(device),
                torch.from_numpy(from_f).long().to(device),
            )
        return self._order_cache[key]

    def get_trans_hilbert_order(self, H, W, device):
        """Transposed Hilbert: encode (col, row) instead of (row, col)."""
        key = ('trans_hilbert', H, W, str(device))
        if key not in self._order_cache:
            num_bits = int(np.ceil(np.log2(max(H, W))))
            rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            coords_transposed = np.stack([cols.flatten(), rows.flatten()], axis=1)
            th_indices = encode(coords_transposed, 2, num_bits)
            to_th = np.argsort(th_indices)
            from_th = np.argsort(to_th)
            self._order_cache[key] = (
                torch.from_numpy(to_th).long().to(device),
                torch.from_numpy(from_th).long().to(device),
            )
        return self._order_cache[key]

    def get_zorder_order(self, H, W, device):
        """Morton (Z-order) curve: bit-interleaved (row, col) indices."""
        key = ('zorder', H, W, str(device))
        if key not in self._order_cache:
            bits = int(np.ceil(np.log2(max(H, W))))
            morton = np.zeros(H * W, dtype=np.int64)
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    c = 0
                    for b in range(bits):
                        c |= ((i >> b) & 1) << (2 * b)
                        c |= ((j >> b) & 1) << (2 * b + 1)
                    morton[idx] = c
            to_z = np.argsort(morton)
            from_z = np.argsort(to_z)
            self._order_cache[key] = (
                torch.from_numpy(to_z).long().to(device),
                torch.from_numpy(from_z).long().to(device),
            )
        return self._order_cache[key]

    def get_diagonal_nwse_order(self, H, W, device):
        """
        NW–SE diagonals (constant i+j): visit anti-diagonal lines from NW corner.
        """
        key = ('diag_nwse', H, W, str(device))
        if key not in self._order_cache:
            indices = []
            for s in range(H + W - 1):
                for i in range(H):
                    j = s - i
                    if 0 <= j < W:
                        indices.append(i * W + j)
            to_d = np.array(indices, dtype=np.int64)
            from_d = np.argsort(to_d)
            self._order_cache[key] = (
                torch.from_numpy(to_d).long().to(device),
                torch.from_numpy(from_d).long().to(device),
            )
        return self._order_cache[key]

    def get_antidiagonal_nesw_order(self, H, W, device):
        """
        NE–SW diagonals (constant i−j): complements :meth:`get_diagonal_nwse_order`.
        """
        key = ('antidiag_nesw', H, W, str(device))
        if key not in self._order_cache:
            indices = []
            for d in range(-(W - 1), H):
                for i in range(H):
                    j = i - d
                    if 0 <= j < W:
                        indices.append(i * W + j)
            to_a = np.array(indices, dtype=np.int64)
            from_a = np.argsort(to_a)
            self._order_cache[key] = (
                torch.from_numpy(to_a).long().to(device),
                torch.from_numpy(from_a).long().to(device),
            )
        return self._order_cache[key]

    def forward(self, x, grid_size):
        """
        Forward pass with Hilbert + Snake H bidirectional scanning.

        Process:
        1. Hilbert forward: scan in Hilbert order (2D locality)
        2. Hilbert backward: scan in reversed Hilbert order
        3. Snake H forward: row-wise boustrophedon (horizontal continuity)
        4. Snake H backward: reversed Snake H order
        5. Fuse all four for comprehensive 2D coverage

        Args:
            x: [B, N, D] where N = H * W
            grid_size: H = W (assumes square grid)

        Returns:
            [B, N, D] fused output combining all scan directions
        """
        B, N, D = x.shape
        H = W = grid_size

        # Get orderings
        to_hilbert, from_hilbert = self.get_hilbert_order(H, W, x.device)
        to_snake_h, from_snake_h = self.get_snake_h_order(H, W, x.device)

        # === Hilbert scans (2D locality) ===
        x_hilbert = x[:, to_hilbert, :]

        # Forward scan
        y_hilbert_fwd = self.mamba_hilbert_forward(x_hilbert)

        # Backward scan
        x_hilbert_flipped = x_hilbert.flip(dims=[1])
        y_hilbert_bwd = self.mamba_hilbert_backward(x_hilbert_flipped)
        y_hilbert_bwd = y_hilbert_bwd.flip(dims=[1])

        # === Snake H scans (horizontal continuity) ===
        x_snake_h = x[:, to_snake_h, :]

        # Forward scan
        y_snake_h_fwd = self.mamba_snake_h_forward(x_snake_h)

        # Backward scan
        x_snake_h_flipped = x_snake_h.flip(dims=[1])
        y_snake_h_bwd = self.mamba_snake_h_backward(x_snake_h_flipped)
        y_snake_h_bwd = y_snake_h_bwd.flip(dims=[1])

        # === Reorder all to spatial order and fuse ===
        y_hilbert_fwd_spatial = y_hilbert_fwd[:, from_hilbert, :]
        y_hilbert_bwd_spatial = y_hilbert_bwd[:, from_hilbert, :]
        y_snake_h_fwd_spatial = y_snake_h_fwd[:, from_snake_h, :]
        y_snake_h_bwd_spatial = y_snake_h_bwd[:, from_snake_h, :]

        # Fuse all four directions
        y_fused = self.fusion(torch.cat([
            y_hilbert_fwd_spatial,
            y_hilbert_bwd_spatial,
            y_snake_h_fwd_spatial,
            y_snake_h_bwd_spatial
        ], dim=-1))

        return self.dropout(y_fused)


class BiHiSSMamba2D(BiHiSSMamba1D):
    """
    Main Bi-HiSS Mamba 2D block.
    Args:
            x: [B, N, D] where N = H * W
            grid_size: H = W (assumes square grid)
    Returns:
            [B, H, W, Dm] fused output combining all scan directions
    """

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, dropout=0.):
        mamba_dim = min(max(dim // 2, 32), dim)
        super().__init__(mamba_dim, d_state, d_conv, expand, dropout)
        self.dim = dim
        self.mamba_dim = mamba_dim
        del self.fusion

        self.in_proj = (
            nn.Linear(dim, mamba_dim) if mamba_dim < dim else nn.Identity()
        )
        c4 = mamba_dim * 4
        self.dwconv = nn.Conv2d(
            c4, c4, kernel_size=3, padding=1, groups=c4, bias=True
        )
        self.dw_norm = nn.GroupNorm(min(32, c4), c4)
        self.pw_fuse = nn.Conv2d(c4, dim, kernel_size=1, bias=True)
        self.out_norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid_size):
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {D}")
        H = W = grid_size
        Dm = self.mamba_dim

        x = self.in_proj(x)

        to_hilbert, from_hilbert = self.get_hilbert_order(H, W, x.device)
        to_snake_h, from_snake_h = self.get_snake_h_order(H, W, x.device)

        x_hilbert = x[:, to_hilbert, :]
        y_hf = self.mamba_hilbert_forward(x_hilbert)
        y_hb = self.mamba_hilbert_backward(x_hilbert.flip(1)).flip(1)

        x_snake = x[:, to_snake_h, :]
        y_sf = self.mamba_snake_h_forward(x_snake)
        y_sb = self.mamba_snake_h_backward(x_snake.flip(1)).flip(1)

        y_hf_2d = y_hf[:, from_hilbert, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_hb_2d = y_hb[:, from_hilbert, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_sf_2d = y_sf[:, from_snake_h, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_sb_2d = y_sb[:, from_snake_h, :].view(B, H, W, Dm).permute(0, 3, 1, 2)

        y = torch.cat([y_hf_2d, y_hb_2d, y_sf_2d, y_sb_2d], dim=1)
        y = self.act(self.dw_norm(self.dwconv(y)))
        y = self.act(self.out_norm(self.pw_fuse(y)))
        return self.drop(y)


class BiHiSSMamba2DAblation(BiHiSSMamba1D):
    """
    Ablation between different spatial serializations.
    - hilbert_snake — Hilbert + horizontal snake (default / main arch).
     This is the default BiHiSSMamba2D but if you want better performance use the original BiHiSSMamba2D.
    - raster_uni — row-major only: **one** serialization, **one** Mamba,
      **one** deserialize (``1·mamba_dim`` fuse).
    - raster_bidir — row-major only: **one** serialization, bidirectional
      SSM (fwd + rev on that path), **two** deserializes into ``2·mamba_dim``.
    - quad_uni — **four unidirectional** passes only: row raster, horizontal
      zigzag (snake H), Hilbert, vertical snake; ``4·mamba_dim`` fusion.
    - raster_col — row-major bidir + column-major bidir (VMamba-style).
    - raster_flip — row-major bidir + row-wise L↔R mirror bidir.
    - hilbert_trans — Hilbert + transposed Hilbert.
    - snake_hv — horizontal + vertical snake (boustrophedon on both axes).
    - zorder_zorder — Morton Z-curve for both groups.
    - hilbert_zorder — Hilbert + Z-order mix.
    - snake_zorder — Snake + Z-order mix.
    """
    _SERIALIZATION_MODES = frozenset({
        'hilbert_snake', 'raster_uni', 'raster_bidir', 'raster_col',
        'raster_flip', 'hilbert_trans', 'snake_hv', 'zorder_zorder',
        'hilbert_zorder', 'quad_uni', 'snake_zorder',
    })

    @staticmethod
    def spatial_mamba_dim(dim):
        """Mamba ``d_model`` inside spatial conv-fused blocks (matches Hilbert+snake budget)."""
        return min(max(dim // 2, 32), dim)

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, dropout=0.,
                 spatial_serialization_mode='hilbert_snake'):
        sm = (
            'raster_bidir' if spatial_serialization_mode == 'raster_raster'
            else spatial_serialization_mode
        )
        if sm not in self._SERIALIZATION_MODES:
            raise ValueError(
                f"spatial_serialization_mode must be one of "
                f"{sorted(self._SERIALIZATION_MODES)} (or legacy 'raster_raster'), "
                f"got {spatial_serialization_mode!r}"
            )
        mamba_dim = self.spatial_mamba_dim(dim)

        if sm in ('raster_uni', 'raster_bidir'):
            nn.Module.__init__(self)
            self.dim = dim
            self.mamba_dim = mamba_dim
            self.spatial_serialization_mode = sm
            mamba_kw = dict(
                d_model=mamba_dim, d_state=d_state, d_conv=d_conv,
                expand=expand,
            )
            self.in_proj = (
                nn.Linear(dim, mamba_dim) if mamba_dim < dim else nn.Identity()
            )
            if sm == 'raster_uni':
                fuse_c = mamba_dim
                self.mamba_raster = Mamba2(**mamba_kw)
            else:
                fuse_c = 2 * mamba_dim
                self.mamba_raster_fwd = Mamba2(**mamba_kw)
                self.mamba_raster_bwd = Mamba2(**mamba_kw)
            self.dwconv = nn.Conv2d(
                fuse_c, fuse_c, kernel_size=3, padding=1, groups=fuse_c,
                bias=True,
            )
            self.dw_norm = nn.GroupNorm(min(32, fuse_c), fuse_c)
            self.pw_fuse = nn.Conv2d(fuse_c, dim, kernel_size=1, bias=True)
            self.out_norm = nn.GroupNorm(min(32, dim), dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            oh = type('_OrderCacheRaster', (), {})()
            oh._order_cache = {}
            self._order_helper = oh
            return

        super().__init__(mamba_dim, d_state, d_conv, expand, dropout)
        self.dim = dim
        self.mamba_dim = mamba_dim
        self.spatial_serialization_mode = sm
        del self.fusion

        self.in_proj = (
            nn.Linear(dim, mamba_dim) if mamba_dim < dim else nn.Identity()
        )
        c4 = mamba_dim * 4
        self.dwconv = nn.Conv2d(
            c4, c4, kernel_size=3, padding=1, groups=c4, bias=True
        )
        self.dw_norm = nn.GroupNorm(min(32, c4), c4)
        self.pw_fuse = nn.Conv2d(c4, dim, kernel_size=1, bias=True)
        self.out_norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _serialization_permutations(self, H, W, device):
        m = self.spatial_serialization_mode
        if m == 'hilbert_snake':
            return (
                self.get_hilbert_order(H, W, device),
                self.get_snake_h_order(H, W, device),
            )
        if m == 'raster_col':
            return (
                self.get_raster_row_order(H, W, device),
                self.get_raster_col_order(H, W, device),
            )
        if m == 'raster_flip':
            return (
                self.get_raster_row_order(H, W, device),
                self.get_row_flip_lr_order(H, W, device),
            )
        if m == 'hilbert_trans':
            return (
                self.get_hilbert_order(H, W, device),
                self.get_trans_hilbert_order(H, W, device),
            )
        if m == 'snake_hv':
            return (
                self.get_snake_h_order(H, W, device),
                self.get_snake_v_order(H, W, device),
            )
        if m == 'zorder_zorder':
            o = self.get_zorder_order(H, W, device)
            return o, o
        if m == 'hilbert_zorder':
            return (
                self.get_hilbert_order(H, W, device),
                self.get_zorder_order(H, W, device),
            )
        if m == 'snake_zorder':
            return (
                self.get_snake_h_order(H, W, device),
                self.get_zorder_order(H, W, device),
            )
        raise RuntimeError(f'unhandled spatial_serialization_mode {m!r}')

    def forward(self, x, grid_size):
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {D}")
        H = W = grid_size
        Dm = self.mamba_dim

        x = self.in_proj(x)

        if self.spatial_serialization_mode == 'raster_uni':
            to, from_ = BiHiSSMamba1D.get_raster_row_order(
                self._order_helper, H, W, x.device)
            seq = x[:, to, :]
            y = self.mamba_raster(seq)
            y_2d = y[:, from_, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y = self.act(self.dw_norm(self.dwconv(y_2d)))
            y = self.act(self.out_norm(self.pw_fuse(y)))
            return self.drop(y)

        if self.spatial_serialization_mode == 'raster_bidir':
            to, from_ = BiHiSSMamba1D.get_raster_row_order(
                self._order_helper, H, W, x.device)
            seq = x[:, to, :]
            yf = self.mamba_raster_fwd(seq)
            yb = self.mamba_raster_bwd(seq.flip(1)).flip(1)
            yf_2d = yf[:, from_, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            yb_2d = yb[:, from_, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y = torch.cat([yf_2d, yb_2d], dim=1)
            y = self.act(self.dw_norm(self.dwconv(y)))
            y = self.act(self.out_norm(self.pw_fuse(y)))
            return self.drop(y)

        if self.spatial_serialization_mode == 'quad_uni':
            to_r, from_r = self.get_raster_row_order(H, W, x.device)
            to_z, from_z = self.get_snake_h_order(H, W, x.device)
            to_h, from_h = self.get_hilbert_order(H, W, x.device)
            to_sv, from_sv = self.get_snake_v_order(H, W, x.device)
            y0 = self.mamba_hilbert_forward(x[:, to_r, :])
            y1 = self.mamba_hilbert_backward(x[:, to_z, :])
            y2 = self.mamba_snake_h_forward(x[:, to_h, :])
            y3 = self.mamba_snake_h_backward(x[:, to_sv, :])
            y0_2d = y0[:, from_r, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y1_2d = y1[:, from_z, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y2_2d = y2[:, from_h, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y3_2d = y3[:, from_sv, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
            y = torch.cat([y0_2d, y1_2d, y2_2d, y3_2d], dim=1)
            y = self.act(self.dw_norm(self.dwconv(y)))
            y = self.act(self.out_norm(self.pw_fuse(y)))
            return self.drop(y)

        (to_a, from_a), (to_b, from_b) = self._serialization_permutations(
            H, W, x.device
        )

        x_a = x[:, to_a, :]
        y_hf = self.mamba_hilbert_forward(x_a)
        y_hb = self.mamba_hilbert_backward(x_a.flip(1)).flip(1)

        x_b = x[:, to_b, :]
        y_sf = self.mamba_snake_h_forward(x_b)
        y_sb = self.mamba_snake_h_backward(x_b.flip(1)).flip(1)

        y_hf_2d = y_hf[:, from_a, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_hb_2d = y_hb[:, from_a, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_sf_2d = y_sf[:, from_b, :].view(B, H, W, Dm).permute(0, 3, 1, 2)
        y_sb_2d = y_sb[:, from_b, :].view(B, H, W, Dm).permute(0, 3, 1, 2)

        y = torch.cat([y_hf_2d, y_hb_2d, y_sf_2d, y_sb_2d], dim=1)
        y = self.act(self.dw_norm(self.dwconv(y)))
        y = self.act(self.out_norm(self.pw_fuse(y)))
        return self.drop(y)


class BiHiSSMamba2DAblation1(nn.Module):
    """
    RS-Mamba OSSM-style omnidirectional scans (Zhao et al., arXiv:2404.02668):.
    """

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.dim = dim
        mamba_dim = BiHiSSMamba2DAblation.spatial_mamba_dim(dim)
        self.mamba_dim = mamba_dim
        self.branch_dim = max(mamba_dim // 2, 1)

        mamba_kw = dict(
            d_model=self.branch_dim, d_state=d_state, d_conv=d_conv,
            expand=expand,
        )
        self.mambas = nn.ModuleList([Mamba2(**mamba_kw) for _ in range(8)])

        self.in_proj = (
            nn.Linear(dim, self.branch_dim)
            if self.branch_dim < dim else nn.Identity()
        )
        c8 = self.branch_dim * 8
        self.dwconv = nn.Conv2d(
            c8, c8, kernel_size=3, padding=1, groups=c8, bias=True
        )
        self.dw_norm = nn.GroupNorm(min(32, c8), c8)
        self.pw_fuse = nn.Conv2d(c8, dim, kernel_size=1, bias=True)
        self.out_norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        oh = type('_OrderCacheOSSM', (), {})()
        oh._order_cache = {}
        self._order_helper = oh

    def forward(self, x, grid_size):
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {D}")
        H = W = grid_size
        bd = self.branch_dim
        oh = self._order_helper

        x = self.in_proj(x)
        to_r, from_r = BiHiSSMamba1D.get_raster_row_order(
            oh, H, W, x.device)
        to_c, from_c = BiHiSSMamba1D.get_raster_col_order(
            oh, H, W, x.device)
        to_d, from_d = BiHiSSMamba1D.get_diagonal_nwse_order(
            oh, H, W, x.device)
        to_a, from_a = BiHiSSMamba1D.get_antidiagonal_nesw_order(
            oh, H, W, x.device)

        branches = (
            (to_r, from_r), (to_r, from_r), (to_c, from_c), (to_c, from_c),
            (to_d, from_d), (to_d, from_d), (to_a, from_a), (to_a, from_a),
        )
        maps2d = []
        for k, ((to, from_), rev) in enumerate(zip(branches,
                                                   (False, True, False, True, False, True, False, True))):
            seq = x[:, to, :]
            if rev:
                seq = seq.flip(1)
            y = self.mambas[k](seq)
            if rev:
                y = y.flip(1)
            maps2d.append(
                y[:, from_, :].view(B, H, W, bd).permute(0, 3, 1, 2))

        y = torch.cat(maps2d, dim=1)
        y = self.act(self.dw_norm(self.dwconv(y)))
        y = self.act(self.out_norm(self.pw_fuse(y)))
        return self.drop(y)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MicroConvFFN(nn.Module):
    """
    Ultra-lightweight Position-Enhanced FFN with SwiGLU gating.
    """

    def __init__(self, dim, gate_mult=1, dropout=0., kernel_size=3):
        super().__init__()
        self.dim = dim
        gate_dim = int(dim * gate_mult)
        self.gate_dim = gate_dim

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=True
        )

        self.fc_gate = nn.Linear(dim, gate_dim * 2)
        self.fc_out = nn.Linear(gate_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid_size):
        B, N, D = x.shape
        H = W = grid_size

        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, D, N).transpose(1, 2)

        x_proj = self.fc_gate(x)
        value, gate = x_proj.chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.drop(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x


class MicroConv2DFFN(nn.Module):
    """
    2D-native version of MicroConvFFN — same architecture, no 2D→1D reshaping.

    Input/output: [B, dim, H, W]  (channels-first throughout)
    Uses Conv2d 1×1 in place of Linear for the gated channel mixing.
    """

    def __init__(self, dim, gate_mult=1, dropout=0., kernel_size=3):
        super().__init__()
        gate_dim = max(1, int(dim * gate_mult))
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim, bias=True)
        self.fc_gate = nn.Conv2d(dim, gate_dim * 2, 1)  # was nn.Linear
        self.fc_out = nn.Conv2d(gate_dim, dim, 1)  # was nn.Linear
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # no grid_size needed
        x = self.dwconv(x)  # [B, dim, H, W]
        x_proj = self.fc_gate(x)  # [B, 2*gate_dim, H, W]
        value, gate = x_proj.chunk(2, dim=1)  # dim=1 not dim=-1
        x = value * F.silu(gate)
        x = self.drop(x)
        x = self.fc_out(x)  # [B, dim, H, W]
        x = self.drop(x)
        return x


class MambaSpatialEncoder(nn.Module):
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand=1,
                 dropout=0., share_weights=True, conv2d_fusion=False,
                 spatial_output=False, dwconv_kernel=7,
                 spatial_serialization_mode="hilbert_snake",
                 num_patches_1d=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.conv2d_fusion = conv2d_fusion
        self.spatial_output = spatial_output
        self.num_patches_1d = num_patches_1d
        ssm = spatial_serialization_mode
        if ssm == 'raster_raster':
            ssm = 'raster_bidir'
        self.spatial_serialization_mode = ssm

        self.layers = nn.ModuleList([])
        for _ in range(depth):

            if conv2d_fusion and not spatial_output:
                mamba_block = BiHiSSMamba2D(
                    dim, d_state=d_state, d_conv=d_conv,
                    expand=expand, dropout=dropout
                )

            elif spatial_output:
                mode = ssm
                if mode == "ossm8":
                    mamba_block = BiHiSSMamba2DAblation1(
                        dim, d_state=d_state, d_conv=d_conv,
                        expand=expand, dropout=dropout,
                    )
                elif mode in BiHiSSMamba2DAblation._SERIALIZATION_MODES:
                    mamba_block = BiHiSSMamba2DAblation(
                        dim, d_state=d_state, d_conv=d_conv,
                        expand=expand, dropout=dropout,
                        spatial_serialization_mode=mode,
                    )
                else:
                    raise ValueError(
                        f"Unknown mamba_spatial_serialization {mode!r}; "
                        f"expected 'ossm8' or one of "
                        f"{sorted(BiHiSSMamba2DAblation._SERIALIZATION_MODES)} "
                        f"(legacy 'raster_raster' is normalized to 'raster_bidir')"
                    )

            else:
                mamba_block = BiHiSSMamba1D(
                    dim, d_state=d_state, d_conv=d_conv, expand=expand,
                    dropout=dropout
                )

            if (conv2d_fusion or self.spatial_output):
                self.layers.append(nn.ModuleList([
                    nn.LayerNorm(dim),
                    mamba_block,
                    nn.GroupNorm(min(32, dim), dim),
                    MicroConv2DFFN(dim, dropout=dropout)
                ]))

            else:
                self.layers.append(nn.ModuleList([
                    nn.LayerNorm(dim),
                    mamba_block,
                    nn.LayerNorm(dim),
                    MicroConvFFN(dim, dropout=dropout)
                ]))

        self.norm = nn.LayerNorm(dim)

    def _grid_size(self, n_tokens):
        """Side length H=W of the patch grid; n_tokens must equal H*W."""
        if n_tokens <= 1:
            return 1
        if self.num_patches_1d is not None:
            g = self.num_patches_1d
            if g * g != n_tokens:
                raise ValueError(
                    f"MambaSpatialEncoder: num_patches_1d={g} implies N={g * g}, "
                    f"but sequence length is N={n_tokens}"
                )
            return g
        g = math.isqrt(n_tokens)
        if g * g != n_tokens:
            raise ValueError(
                f"MambaSpatialEncoder: patch token count N={n_tokens} must be a "
                f"perfect square (math.isqrt -> {g}, g*g={g * g}). "
                f"Pass num_patches_1d=image_size//patch_size from the parent model."
            )
        return g

    @staticmethod
    def _norm2d_before_conv_ffn(norm2, x_nchw):
        """norm2 on [B,C,H,W]: GroupNorm/ConvNorm expect NCHW; LayerNorm(dim) needs channel-last."""
        if isinstance(norm2, nn.LayerNorm):
            return norm2(x_nchw.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return norm2(x_nchw)

    def forward(self, x):
        B, N, D = x.shape
        grid_size = self._grid_size(N)

        if self.spatial_output:
            for norm1, mamba, norm2, ffn in self.layers:
                x_2d = mamba(norm1(x), grid_size)  # 1D → 2D (deserialized)
                x_res = x_2d
                x_norm = self._norm2d_before_conv_ffn(norm2, x_2d)
                # MicroConv2DFFN expects NCHW; MicroConvFFN expects [B, N, D] + grid_size.
                if isinstance(ffn, MicroConvFFN):
                    Bf, Cf, Hf, Wf = x_norm.shape
                    x_seq = x_norm.permute(0, 2, 3, 1).reshape(Bf, Hf * Wf, Cf)
                    x_2d = ffn(x_seq, grid_size).reshape(Bf, Hf, Wf, Cf).permute(0, 3, 1, 2) + x_res
                else:
                    x_2d = ffn(x_norm) + x_res
                x = x_2d.permute(0, 2, 3, 1).reshape(B, N, D)  # back to 1D for next layer

            return self.norm(x)

        if self.conv2d_fusion:
            for norm1, mamba, norm2, ffn in self.layers:
                x_2d = mamba(norm1(x), grid_size)  # 1D → 2D (deserialized)
                x_2d = ffn(norm2(x_2d)) + x_2d  # 2D FFN + 2D residual
                x = x_2d.permute(0, 2, 3, 1).reshape(B, N, D)  # back to 1D for next layer

            return self.norm(x)


        else:
            for norm1, mamba, norm2, ffn in self.layers:
                x = mamba(norm1(x), grid_size) + x
                x = ffn(norm2(x), grid_size) + x

            return self.norm(x)

class STViT(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = model_config['num_channels'] * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        print('pos embedding: ', self.pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        print('temporal token: ', self.temporal_token.shape)
        self.temporal_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding#[:, :, :(n + 1)]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x, '(b t) ... -> b t ...', b=b)  # use only space token, location 0
        cls_temporal_tokens = repeat(self.temporal_token, '() () d -> b t k d', b=b, t=1, k=self.num_patches_1d**2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b * self.num_patches_1d**2, self.num_frames+1, self.dim)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x = x.reshape(B, H*W, self.num_classes)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_single_token(nn.Module):
    """
    Temporal-Spatial ViT with single cls token (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.shape_pattern = get_params_values(model_config, 'shape_pattern', 'NTHWC')
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        print('temporal token: ', self.temporal_token.shape)
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        print('space pos embedding: ', self.space_pos_embedding.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))

    def forward(self, x):
        if self.shape_pattern == 'NTHWC':
            x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() () d -> b t d', b=B * self.num_patches_1d ** 2, t=1)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = x.reshape(B, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x)
        x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x = x.reshape(B, H*W, self.num_classes)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_static_position_encodings(nn.Module):
    """
    Temporal-Spatial ViT with static (no lookup) position encodings (used in ablation, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, self.dim))
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += self.temporal_pos_embedding  #.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_global_attention_spatial_encoder(nn.Module):
    """
    Temporal-Spatial ViT where spatial encoder attends to all cls tokens (used in ablation, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.depth + 2, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)        # print(x.shape)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = x.reshape(B, self.num_classes * self.num_patches_1d**2, self.dim)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TViT(nn.Module):
    """
    Temporal-only ViT5 (no spatial transformer, used in ablations, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_seg(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))

        # Mamba spatial encoder config
        self.space_mamba = MambaSpatialEncoder(
            dim=self.dim,
            depth=self.spatial_depth,
            d_state=model_config.get('mamba_d_state', 16),
            d_conv=model_config.get('mamba_d_conv', 4),
            expand=model_config.get('mamba_expand', 1),
            dropout=self.dropout
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x += self.space_pos_embedding  # [:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_mamba(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_seg_v2(nn.Module):
    """
    TSViT_seg with Conv2D-fused spatial Mamba encoder.

    Identical to TSViT_seg (Transformer temporal + Mamba spatial) except the
    spatial BiHiSSMamba2D block reassembles each scan's output back to its
    native 2D spatial layout and fuses via depthwise-separable Conv2D instead
    of a linear layer.
    """

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))

        self.space_mamba = MambaSpatialEncoder(
            dim=self.dim,
            depth=self.spatial_depth,
            d_state=model_config.get('mamba_d_state', 16),
            d_conv=model_config.get('mamba_d_conv', 4),
            expand=model_config.get('mamba_expand', 1),
            dropout=self.dropout,
            conv2d_fusion=True,
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_mamba(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_seg_v3(nn.Module):
    """
    TSViT_seg with Conv2D-fused spatial Mamba encoder.

    Identical to TSViT_seg (Transformer temporal + Mamba spatial) but the
    spatial Mamba block (BiHiSSMamba2DAblation / BiHiSSMamba2DAblation1)
    reassembles every scan back to its native 2D spatial layout and applies a
    depthwise-separable Conv2D fusion followed by a 2D-native MicroConv FFN.
    The serialization mode is selected via ``mamba_spatial_serialization``.
    """

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))

        self.space_mamba = MambaSpatialEncoder(
            dim=self.dim,
            depth=self.spatial_depth,
            d_state=model_config.get('mamba_d_state', 16),
            d_conv=model_config.get('mamba_d_conv', 4),
            expand=model_config.get('mamba_expand', 1),
            dropout=self.dropout,
            spatial_output=True,
            spatial_serialization_mode=model_config.get(
                'mamba_spatial_serialization', 'ossm8'),

            num_patches_1d=self.num_patches_1d,
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_mamba(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

class TSViT_lookup(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    This is a general implementation of lookup position encodings for all dates found in a training set.
    During inference, position encodings are calculated via linear interpolation between dates found in the training data.
    """
    def __init__(self, model_config, train_dates):
        super().__init__()
        train_dates = sorted(train_dates)
        self.train_dates = torch.nn.Parameter(data=torch.tensor(train_dates), requires_grad=False)#.cuda()
        self.eval_dates = torch.nn.Parameter(data=torch.arange(1, 366), requires_grad=False)
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(len(self.train_dates), self.dim), requires_grad=True)
        self.update_inference_temporal_position_embeddings()
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, inference=False):
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)

        if inference:
            self.update_inference_temporal_position_embeddings()
            temporal_pos_embedding = self.get_inference_temporal_position_embeddings(xt).to(x.device)
        else:
            temporal_pos_embedding = self.get_temporal_position_embeddings(xt)

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

    def update_inference_temporal_position_embeddings(self):
        train_dates_idx = torch.arange(self.train_dates.shape[0])
        min_val = torch.min(self.train_dates).item()
        min_idx = torch.argmin(self.train_dates).item()
        if min_val == 0:
            min_val = torch.min(self.train_dates[1:]).item()
            min_idx += 1
        max_val = torch.max(self.train_dates).item()
        max_idx = torch.argmax(self.train_dates).item()
        pos_eval = torch.zeros(self.eval_dates.shape[0], self.dim)
        for i, evdate in enumerate(self.eval_dates):
            if evdate < min_val:
                pos_eval[i] = self.temporal_pos_embedding[min_idx]
            elif evdate > max_val:
                pos_eval[i] = self.temporal_pos_embedding[max_idx]
            else:
                dist = evdate - self.train_dates
                if 0 in dist:
                    pos_eval[i] = self.temporal_pos_embedding[dist == 0]
                    continue
                lower_idx = train_dates_idx[dist >= 0].max().item()
                upper_idx = train_dates_idx[dist <= 0].min().item()
                lower_date = self.train_dates[lower_idx].item()
                upper_date = self.train_dates[upper_idx].item()
                pos_eval[i] = (upper_date - evdate) / (upper_date - lower_date) * self.temporal_pos_embedding[
                    lower_idx] + \
                              (evdate - lower_date) / (upper_date - lower_date) * self.temporal_pos_embedding[upper_idx]
        self.inference_temporal_pos_embedding = nn.Parameter(pos_eval, requires_grad=False)

    def get_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.train_dates)
        return self.temporal_pos_embedding[index].reshape((B, T, self.dim))

    def get_inference_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.eval_dates)
        return self.inference_temporal_pos_embedding[index].reshape((B, T, self.dim))



if __name__ == "__main__":
    res = 24
    model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 20,
                    'max_seq_len': 16, 'dim': 128, 'temporal_depth': 6, 'spatial_depth': 2,
                    'heads': 4, 'pool': 'cls', 'num_channels': 14, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}
    train_config = {'dataset': "psetae_repl_2018_100_3", 'label_map': "labels_20k2k", 'max_seq_len': 16, 'batch_size': 5,
                    'extra_data': [], 'num_workers': 4}

    x = torch.rand((3, 16, res, res, 14))

    # model = TViT(model_config).cuda()
    # model = STViT(model_config)#.cuda()
    # model = TSViT_global_attention_spatial_encoder(model_config)#.cuda()
    model = TSViT_single_token(model_config)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(x)

    # torch.norm(cls, dim=2).shape
    print("Shape of out :", out.shape)  # [B, num_classes]