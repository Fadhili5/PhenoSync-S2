import torch
import torch.nn as nn


CROP_CLASSES  = ["rice", "corn", "soybean", "background"]
PHENO_CLASSES = ["Greenup", "MidGreenup", "Peak", "Maturity",
                 "Senescence", "MidSenescence", "Dormancy"]

N_BANDS   = 12   # Sentinel-2 bands
N_INDICES = 4    # NDVI, LSWI, NDWI, EVI
INPUT_DIM = N_BANDS + N_INDICES  # 16 per timestep


class CropPhenologyLSTM(nn.Module):
    """
    BiLSTM with attention pooling for dual-head crop + phenology prediction.

    Input:
      x       : (B, T, 16)   padded temporal sequence of bands + indices
      lengths : (B,)          actual sequence lengths per sample
      doy     : (B, 1)        normalized day-of-year of observation date [0, 1]

    Output:
      crop_logits  : (B, 4)  rice / corn / soybean / background
      pheno_logits : (B, 7)  phenological stage
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_crop: int = len(CROP_CLASSES),
        n_pheno: int = len(PHENO_CLASSES),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn    = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

        head_in = hidden_dim * 2 + 1  # +1 for DOY scalar

        self.crop_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_crop),
        )
        self.pheno_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_pheno),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, doy: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        B, T, _ = out.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        attn_scores = self.attn(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (attn_weights * out).sum(dim=1)
        context = self.dropout(context)

        ctx = torch.cat([context, doy], dim=1)
        return self.crop_head(ctx), self.pheno_head(ctx)
