"""
Advanced Transformer Model for Next-Location Prediction
========================================================

Architecture designed for >=70% Acc@1 with <3M parameters on diy_skip_first_part dataset.

Key Features:
1. Multi-head self-attention with relative position encoding
2. Efficient feature fusion (location, user, temporal)
3. Gated residual connections
4. Layer normalization (pre-norm for stability)
5. Sinusoidal positional encodings
6. Advanced temporal encoding (cyclic features)
7. Hierarchical attention pooling
8. Label smoothing and dropout regularization

Design Philosophy:
- Every feature is carefully encoded and integrated
- Spatial: Location embeddings with learned representations
- Temporal: Cyclic time-of-day, day-of-week, duration, time gaps
- User: User embeddings to capture personalization
- Attention: Multi-head attention to learn complex dependencies
- Regularization: Dropout, layer norm, label smoothing to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionBias(nn.Module):
    """Relative position bias for attention mechanism."""
    def __init__(self, num_heads, max_distance=60):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # Learnable relative position embeddings
        self.relative_attention_bias = nn.Embedding(2 * max_distance + 1, num_heads)
    
    def forward(self, seq_len):
        # Create relative position matrix
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # Clip to max distance
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_distance, 
            self.max_distance
        ) + self.max_distance
        # Get bias
        bias = self.relative_attention_bias(relative_positions)  # (seq_len, seq_len, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)


class GatedResidualConnection(nn.Module):
    """Gated residual connection for adaptive feature combination."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer_output):
        gate_values = torch.sigmoid(self.gate(x))
        return x + gate_values * self.dropout(sublayer_output)


class TemporalEncoding(nn.Module):
    """Advanced temporal encoding with cyclic features."""
    def __init__(self, output_dim):
        super().__init__()
        # Project temporal features to embedding space
        # Input: [time_sin, time_cos, dur, wd_sin, wd_cos, gap]  = 6 features
        self.temporal_proj = nn.Sequential(
            nn.Linear(6, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, start_min, dur, weekday, diff):
        """
        Args:
            start_min: Start time in minutes (B, L)
            dur: Duration in minutes (B, L)
            weekday: Weekday index 0-6 (B, L)
            diff: Time gap in days (B, L)
        """
        # Time of day (cyclic)
        hours = start_min / 60.0
        time_rad = (hours / 24.0) * 2 * math.pi
        time_sin = torch.sin(time_rad)
        time_cos = torch.cos(time_rad)
        
        # Duration (log-normalized)
        dur_norm = torch.log1p(dur) / 10.0
        
        # Weekday (cyclic)
        wd_rad = (weekday.float() / 7.0) * 2 * math.pi
        wd_sin = torch.sin(wd_rad)
        wd_cos = torch.cos(wd_rad)
        
        # Time gap (normalized)
        gap_norm = torch.clamp(diff.float() / 7.0, 0, 1)
        
        # Stack features
        temporal_feats = torch.stack([
            time_sin, time_cos, dur_norm, wd_sin, wd_cos, gap_norm
        ], dim=-1)
        
        return self.temporal_proj(temporal_feats)


class MultiHeadAttentionWithRelativePosition(nn.Module):
    """Multi-head attention with relative position bias."""
    def __init__(self, d_model, num_heads, dropout=0.1, max_distance=60):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias = RelativePositionBias(num_heads, max_distance)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, L, L)
        
        # Add relative position bias
        rel_bias = self.relative_position_bias(L)  # (num_heads, L, L)
        attn = attn + rel_bias.unsqueeze(0)
        
        # Apply mask
        if mask is not None:
            # mask shape: (B, L), need to expand for attention
            # Create attention mask: (B, 1, 1, L)
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and gated residuals."""
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        
        # Pre-layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention with relative position
        self.attn = MultiHeadAttentionWithRelativePosition(
            d_model, num_heads, attention_dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # Gated residual connections
        self.gate1 = GatedResidualConnection(d_model, dropout)
        self.gate2 = GatedResidualConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        # Attention with pre-norm and gated residual
        x = self.gate1(x, self.attn(self.norm1(x), mask))
        # Feed-forward with pre-norm and gated residual
        x = self.gate2(x, self.ff(self.norm2(x)))
        return x


class AdvancedTransformerV2(nn.Module):
    """
    Advanced Transformer model for next-location prediction.
    
    Target: >=70% Acc@1 with <3M parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration (optimized for <3M params)
        self.num_locations = config.num_locations
        self.num_users = config.num_users
        self.d_model = getattr(config, 'd_model', 136)
        self.num_heads = getattr(config, 'nhead', 8)
        self.num_layers = getattr(config, 'num_layers', 3)
        self.dim_feedforward = getattr(config, 'dim_feedforward', 272)
        self.dropout = getattr(config, 'dropout', 0.15)
        self.attention_dropout = getattr(config, 'attention_dropout', 0.1)
        
        # Embedding dimensions (optimized for <3M params)
        loc_emb_dim = getattr(config, 'loc_emb_dim', 53)
        user_emb_dim = getattr(config, 'user_emb_dim', 20)
        temporal_dim = getattr(config, 'temporal_dim', 27)
        
        # Core embeddings
        self.loc_emb = nn.Embedding(self.num_locations, loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(self.num_users, user_emb_dim, padding_idx=0)
        
        # Temporal encoding
        self.temporal_enc = TemporalEncoding(temporal_dim)
        
        # Input projection to d_model
        input_dim = loc_emb_dim + user_emb_dim + temporal_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Sinusoidal positional encoding
        max_len = getattr(config, 'max_seq_len', 100)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                             (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.d_model//2])
        self.register_buffer('pe', pe)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.num_heads,
                self.dim_feedforward,
                self.dropout,
                self.attention_dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Hierarchical pooling (attention-based)
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.pool_attn = nn.MultiheadAttention(
            self.d_model, 
            self.num_heads, 
            dropout=self.dropout,
            batch_first=True
        )
        
        # Prediction head (optimized to reduce parameters)
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.num_locations)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        """
        Forward pass.
        
        Args:
            loc_seq: Location sequence (B, L)
            user_seq: User sequence (B, L)
            weekday_seq: Weekday sequence (B, L)
            start_min_seq: Start time in minutes (B, L)
            dur_seq: Duration sequence (B, L)
            diff_seq: Time gap sequence (B, L)
            mask: Attention mask (B, L)
        
        Returns:
            logits: Prediction logits (B, num_locations)
        """
        batch_size, seq_len = loc_seq.shape
        
        # Embeddings
        loc_emb = self.loc_emb(loc_seq)  # (B, L, loc_emb_dim)
        user_emb = self.user_emb(user_seq)  # (B, L, user_emb_dim)
        
        # Temporal encoding
        temporal_emb = self.temporal_enc(
            start_min_seq, dur_seq, weekday_seq, diff_seq
        )  # (B, L, temporal_dim)
        
        # Concatenate all features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        
        # Project to d_model
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Hierarchical attention pooling
        # Expand pool query for batch
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        
        # Create key padding mask for attention pooling
        key_padding_mask = ~mask
        
        # Attention pooling
        pooled, _ = self.pool_attn(
            pool_query, x, x,
            key_padding_mask=key_padding_mask
        )
        pooled = pooled.squeeze(1)  # (B, d_model)
        
        # Prediction
        logits = self.predictor(pooled)  # (B, num_locations)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
