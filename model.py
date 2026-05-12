import math
import torch
import torch.nn as nn

from features import SEQ_DIM


CROP_CLASSES  = ["rice", "corn", "soybean", "background"]
PHENO_CLASSES = ["Greenup", "MidGreenup", "Peak", "Maturity",
                 "Senescence", "MidSenescence", "Dormancy"]

INPUT_DIM = SEQ_DIM   # 24: 12 bands + 10 indices + 2 per-timestep DOY sin/cos
DOY_ENC_DIM = 4       # sin/cos × 2 harmonics for observation-level DOY context


class MultiHeadAttnPool(nn.Module):
    """
    Multi-head attention pooling: (B, T, H) → (B, H).
    Each head learns independent temporal attention weights.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 2):
        super().__init__()
        self.scorers = nn.Linear(hidden_dim, n_heads)
        self.proj    = nn.Linear(hidden_dim * n_heads, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)  mask: (B, T) bool — True for valid timesteps
        B, T, H = x.shape
        scores  = self.scorers(x)                            # (B, T, n_heads)
        scores  = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)               # (B, T, n_heads)
        ctx     = torch.einsum('bth,btn->bnh', x, weights)  # (B, n_heads, H)
        return self.proj(ctx.reshape(B, -1))                 # (B, H)


class CropPhenologyLSTM(nn.Module):
    """
    BiLSTM with multi-head temporal attention for dual-head crop + phenology prediction.

    Input:
      x       : (B, T, 24)  padded temporal sequence (bands + indices + per-timestep DOY)
      lengths : (B,)          actual sequence lengths
      doy     : (B, 1)        observation DOY normalized [0, 1]

    Output:
      crop_logits  : (B, 4)   rice / corn / soybean / background
      pheno_logits : (B, 7)   phenological stage
    """

    def __init__(
        self,
        input_dim: int  = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_crop: int     = len(CROP_CLASSES),
        n_pheno: int    = len(PHENO_CLASSES),
        dropout: float  = 0.3,
        n_heads: int    = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # 256 for bidirectional

        self.attn    = MultiHeadAttnPool(lstm_out_dim, n_heads=n_heads)
        self.dropout = nn.Dropout(dropout)

        head_in = lstm_out_dim + DOY_ENC_DIM  # 256 + 4 = 260

        self.crop_head = nn.Sequential(
            nn.Linear(head_in, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),      nn.ReLU(),
            nn.Linear(64, n_crop),
        )
        self.pheno_head = nn.Sequential(
            nn.Linear(head_in, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),      nn.ReLU(),
            nn.Linear(64, n_pheno),
        )

    def _encode_doy(self, doy: torch.Tensor) -> torch.Tensor:
        """doy: (B, 1) in [0,1] → (B, 4) two-harmonic sinusoidal encoding."""
        pi2 = 2 * math.pi
        return torch.cat([
            torch.sin(pi2 * doy),
            torch.cos(pi2 * doy),
            torch.sin(2 * pi2 * doy),
            torch.cos(2 * pi2 * doy),
        ], dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, doy: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        B, T, _ = lstm_out.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        context = self.attn(lstm_out, mask)   # (B, lstm_out_dim)
        context = self.dropout(context)

        doy_enc = self._encode_doy(doy)       # (B, 4)
        ctx     = torch.cat([context, doy_enc], dim=1)  # (B, 260)

        return self.crop_head(ctx), self.pheno_head(ctx)
