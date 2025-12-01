"""
Optimized History-Aware Transformer for Next-Location Prediction
=================================================================

Key Architecture Principles:
1. Explicit location history tracking and scoring
2. Transformer for sequence modeling
3. Multi-scale temporal features
4. User personalization through embeddings
5. Frequency and recency-aware predictions

Target: >=70% Acc@1 with <3M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocationHistoryEncoder(nn.Module):
    """Encode location visit history with recency and frequency."""
    def __init__(self, num_locations, hidden_dim=128):
        super().__init__()
        self.num_locations = num_locations
        self.hidden_dim = hidden_dim
        
        # Learnable recency decay parameter
        self.recency_weight = nn.Parameter(torch.tensor(0.85))
        self.frequency_weight = nn.Parameter(torch.tensor(1.5))
        
    def forward(self, loc_seq, mask):
        """
        Compute history-based location scores.
        
        Args:
            loc_seq: (B, L) location IDs
            mask: (B, L) validity mask
        
        Returns:
            history_scores: (B, num_locations)
        """
        batch_size, seq_len = loc_seq.shape
        device = loc_seq.device
        
        # Initialize scoring tensors
        recency_scores = torch.zeros(batch_size, self.num_locations, device=device)
        frequency_counts = torch.zeros(batch_size, self.num_locations, device=device)
        
        # Compute recency and frequency for each position
        for t in range(seq_len):
            valid_mask = mask[:, t].float()  # (B,)
            locs = loc_seq[:, t]  # (B,)
            
            # Recency: exponentially decaying weight from the end
            position_from_end = seq_len - t - 1
            recency_factor = torch.pow(self.recency_weight, position_from_end)
            
            # Update recency scores (keep max for each location)
            scatter_recency = torch.zeros(batch_size, self.num_locations, device=device)
            scatter_recency.scatter_(1, locs.unsqueeze(1), (recency_factor * valid_mask).unsqueeze(1))
            recency_scores = torch.maximum(recency_scores, scatter_recency)
            
            # Update frequency counts
            frequency_counts.scatter_add_(1, locs.unsqueeze(1), valid_mask.unsqueeze(1))
        
        # Normalize frequency
        max_freq = frequency_counts.max(dim=1, keepdim=True)[0].clamp(min=1.0)
        frequency_normalized = frequency_counts / max_freq
        
        # Combine recency and frequency
        history_scores = recency_scores + self.frequency_weight * frequency_normalized
        
        return history_scores


class CompactTransformerBlock(nn.Module):
    """Efficient transformer block."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class OptimizedHistoryAwareTransformer(nn.Module):
    """
    Optimized model combining history-awareness with Transformer.
    
    Architecture:
    - Location + User + Temporal embeddings
    - Transformer encoder layers
    - History-based scoring
    - Ensemble prediction
    
    Parameters: <3M
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        self.num_users = config.num_users
        
        # Optimized dimensions for parameter budget (<3M)
        self.loc_emb_dim = getattr(config, 'loc_emb_dim', 55)
        self.user_emb_dim = getattr(config, 'user_emb_dim', 20)
        self.d_model = getattr(config, 'd_model', 144)
        self.nhead = getattr(config, 'nhead', 8)
        self.num_layers = getattr(config, 'num_layers', 3)
        self.dim_feedforward = getattr(config, 'dim_feedforward', 288)
        self.dropout = getattr(config, 'dropout', 0.2)
        
        # Embeddings
        self.loc_emb = nn.Embedding(self.num_locations, self.loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(self.num_users, self.user_emb_dim, padding_idx=0)
        
        # Temporal encoding (6 features -> compact)
        self.temporal_proj = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Project to d_model
        input_dim = self.loc_emb_dim + self.user_emb_dim + 32
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        max_len = getattr(config, 'max_seq_len', 100)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.d_model//2])
        self.register_buffer('pe', pe)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CompactTransformerBlock(
                self.d_model, self.nhead, self.dim_feedforward, self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # History encoder
        self.history_encoder = LocationHistoryEncoder(self.num_locations)
        
        # Prediction head (from transformer output)
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.num_locations)
        )
        
        # Learnable ensemble weight
        self.history_scale = nn.Parameter(torch.tensor(12.0))
        self.model_scale = nn.Parameter(torch.tensor(0.3))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        """
        Forward pass.
        
        Args:
            loc_seq: (B, L) location sequence
            user_seq: (B, L) user sequence
            weekday_seq: (B, L) weekday sequence
            start_min_seq: (B, L) start time in minutes
            dur_seq: (B, L) duration sequence
            diff_seq: (B, L) time gap sequence
            mask: (B, L) attention mask
        
        Returns:
            logits: (B, num_locations) prediction logits
        """
        batch_size, seq_len = loc_seq.shape
        
        # === History-based scoring ===
        history_scores = self.history_encoder(loc_seq, mask)
        
        # === Transformer-based modeling ===
        # Embeddings
        loc_emb = self.loc_emb(loc_seq)  # (B, L, loc_emb_dim)
        user_emb = self.user_emb(user_seq)  # (B, L, user_emb_dim)
        
        # Temporal features (cyclic encoding)
        hours = start_min_seq / 60.0
        time_rad = (hours / 24.0) * 2 * math.pi
        time_sin = torch.sin(time_rad)
        time_cos = torch.cos(time_rad)
        
        dur_norm = torch.log1p(dur_seq) / 10.0
        
        wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
        wd_sin = torch.sin(wd_rad)
        wd_cos = torch.cos(wd_rad)
        
        gap_norm = torch.clamp(diff_seq.float() / 7.0, 0, 1)
        
        temporal_feats = torch.stack([
            time_sin, time_cos, dur_norm, wd_sin, wd_cos, gap_norm
        ], dim=-1)
        temporal_emb = self.temporal_proj(temporal_feats)  # (B, L, 32)
        
        # Combine all features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1  # (B,)
        last_indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, last_indices).squeeze(1)  # (B, d_model)
        
        # Transformer predictions
        transformer_logits = self.predictor(last_hidden)  # (B, num_locations)
        
        # === Ensemble: History + Transformer ===
        # Normalize transformer logits to similar scale
        transformer_probs = F.softmax(transformer_logits, dim=1)
        transformer_scaled = transformer_probs * self.num_locations
        
        # Combine
        final_logits = (self.history_scale * history_scores + 
                       self.model_scale * transformer_scaled)
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
