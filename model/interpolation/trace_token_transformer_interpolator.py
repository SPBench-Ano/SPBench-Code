"""Trace-as-token Transformer for 2D seismic patch interpolation.

Each trace becomes a token: ``(B, C, H, W) -> (B, H, embed_dim)``.
Sin-cos positional encoding for trace index, Transformer encoder-decoder,
then linear reconstruction head back to ``(B, 1, H, W)``.

Reference: standard Transformer (Vaswani et al. 2017) adapted for seismic
trace-level tokenisation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..registry import register_model


class _SincosPositionEncoding(nn.Module):
    """Sinusoidal positional encoding for trace indices.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension.
    max_len : int
        Maximum sequence length to pre-compute.
    coord_dim : int
        Dimensionality of position coordinates. Currently only ``1``
        (trace index) is implemented. Reserved for future 5D extension.
    """

    def __init__(self, embed_dim: int, max_len: int = 512, coord_dim: int = 1) -> None:
        super().__init__()
        if coord_dim != 1:
            raise NotImplementedError(
                f"coord_dim={coord_dim} is reserved for future extension. "
                f"Currently only coord_dim=1 (trace index) is implemented."
            )
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.coord_dim = coord_dim

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tokens.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, H, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, H, embed_dim)`` with positional encoding added.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                f"Re-instantiate with a larger max_len."
            )
        return x + self.pe[:seq_len, :].unsqueeze(0)


@register_model("trace_token_transformer_interpolator")
class TraceTokenTransformerInterpolator(nn.Module):
    """Trace-as-token Transformer for 2D seismic patch interpolation.

    Each trace in the patch is treated as one token. The model projects
    per-trace samples to an embedding, encodes with a Transformer encoder,
    decodes with a learned-query Transformer decoder, and reconstructs
    the full seismic patch.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for masked-only, 2 for masked + mask channel).
    trace_length : int
        Time samples per trace (W dimension). Must match ``patch_time``.
    embed_dim : int
        Token embedding dimension.
    encoder_depth : int
        Number of Transformer encoder layers.
    decoder_depth : int
        Number of Transformer decoder layers.
    num_heads : int
        Attention heads.
    dropout : float
        Dropout probability.
    pos_encoding : str
        Position encoding type. Currently only ``"sincos"``.
    coord_dim : int
        Placeholder for future multi-dimensional coordinate encoding.
        Currently only ``1`` (trace index) is implemented.
    """

    def __init__(
        self,
        in_channels: int = 2,
        trace_length: int = 256,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        pos_encoding: str = "sincos",
        coord_dim: int = 1,
    ) -> None:
        super().__init__()
        if pos_encoding not in ("sincos",):
            raise ValueError(
                f"Unsupported pos_encoding '{pos_encoding}'. Only 'sincos' is available."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.in_channels = in_channels
        self.trace_length = trace_length
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim

        # Project per-trace (C * trace_length) to embed_dim.
        self.input_proj = nn.Linear(in_channels * trace_length, embed_dim)

        if pos_encoding == "sincos":
            self.pos_encoding = _SincosPositionEncoding(
                embed_dim=embed_dim,
                max_len=512,
                coord_dim=coord_dim,
            )
        else:
            self.pos_encoding = None

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_depth, enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # Learned query for the decoder.
        self.query_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Reconstruction head: embed_dim -> trace_length per token.
        self.output_proj = nn.Linear(embed_dim, trace_length)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.query_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input seismic patch of shape ``(B, C, H, W)`` where ``H`` is trace
            count, ``W`` is time samples, and ``C`` is in_channels.

        Returns
        -------
        torch.Tensor
            Reconstructed seismic patch of shape ``(B, 1, H, W)``.
        """
        B, C, H, W = x.shape
        if W != self.trace_length:
            raise ValueError(
                f"Input time dimension {W} does not match trace_length {self.trace_length}."
            )

        # (B, C, H, W) -> (B, H, C*W) -> project -> (B, H, embed_dim)
        tokens = x.permute(0, 2, 1, 3).reshape(B, H, C * W)
        tokens = self.input_proj(tokens)
        tokens = self.dropout(tokens)

        # Add positional encoding.            (B, H, embed_dim)
        if self.pos_encoding is not None:
            tokens = self.pos_encoding(tokens)

        # Transpose to (H, B, embed_dim) for PyTorch Transformer (batch_first=False).
        src = tokens.transpose(0, 1)  # (H, B, embed_dim)

        # Encoder.
        memory = self.encoder(src)  # (H, B, embed_dim)

        # Decoder with learned query.
        query = self.query_embed.unsqueeze(0).expand(H, B, self.embed_dim)
        decoded = self.decoder(tgt=query, memory=memory)  # (H, B, embed_dim)

        # Reconstruction head.
        decoded = decoded.transpose(0, 1)  # (B, H, embed_dim)
        out = self.output_proj(decoded)  # (B, H, W)
        out = out.unsqueeze(1)  # (B, 1, H, W)

        return out


if __name__ == "__main__":
    # Quick shape smoke test.
    model = TraceTokenTransformerInterpolator(
        in_channels=2, trace_length=64, embed_dim=256,
        encoder_depth=6, decoder_depth=4, num_heads=8, dropout=0.0,
    )
    x = torch.randn(2, 2, 32, 64)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 32, 64), f"Expected (2,1,32,64), got {out.shape}"
    print("PASS: trace_token_transformer_interpolator shape test")

    # Test C=1.
    model_c1 = TraceTokenTransformerInterpolator(
        in_channels=1, trace_length=64, embed_dim=256,
        encoder_depth=6, decoder_depth=4, num_heads=8, dropout=0.0,
    )
    x1 = torch.randn(2, 1, 32, 64)
    with torch.no_grad():
        out1 = model_c1(x1)
    assert out1.shape == (2, 1, 32, 64), f"Expected (2,1,32,64), got {out1.shape}"
    print("PASS: trace_token_transformer_interpolator C=1 shape test")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
