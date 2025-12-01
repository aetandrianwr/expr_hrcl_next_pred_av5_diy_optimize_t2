"""
Ultra-Optimized History-First Model for Next-Location Prediction  
=================================================================

Core Strategy:
Since 83.5% of next locations are in history and 75.2% are in the last 5 visits,
we build a model that:
1. Aggressively scores history locations (especially recent ones)
2. Uses a lightweight transformer to learn reranking weights
3. Combines multiple complementary scoring strategies

Target: >=70% Acc@1 with <3M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UltraOptimizedHistoryModel(nn.Module):
    """
    Ultra-optimized model focusing on history patterns.
    
    Key innovations:
    - Strong recency bias (last 5 locations get highest scores)
    - Frequency counting with decay
    - Learned location-to-location transitions
    - User-specific preferences
    - Lightweight transformer for context
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        self.num_users = config.num_users
        
        # Compact embeddings
        self.loc_emb_dim = 48
        self.user_emb_dim = 16
        self.d_model = 96
        
        # Embeddings
        self.loc_emb = nn.Embedding(self.num_locations, self.loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(self.num_users, self.user_emb_dim, padding_idx=0)
        
        # Lightweight temporal encoding
        self.temporal_proj = nn.Linear(6, 24)
        
        # Input projection
        input_dim = self.loc_emb_dim + self.user_emb_dim + 24
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        
        # Single lightweight transformer block
        self.attn = nn.MultiheadAttention(self.d_model, 4, dropout=0.15, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Compact prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.d_model * 2, self.num_locations)
        )
        
        # Learnable scoring parameters (optimized for this dataset)
        self.recency_decay = nn.Parameter(torch.tensor(0.75))  # Strong recency
        self.frequency_weight = nn.Parameter(torch.tensor(1.8))
        self.last_location_boost = nn.Parameter(torch.tensor(3.5))  # Boost if repeating last
        self.history_scale = nn.Parameter(torch.tensor(15.0))  # Strong history prior
        self.model_scale = nn.Parameter(torch.tensor(0.25))
        
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
    
    def compute_history_scores(self, loc_seq, mask):
        """
        Compute history scores with strong recency bias.
        
        Key insight: 75% of next locations are in last 5 visits!
        """
        batch_size, seq_len = loc_seq.shape
        device = loc_seq.device
        
        # Initialize scores
        recency_scores = torch.zeros(batch_size, self.num_locations, device=device)
        frequency_scores = torch.zeros(batch_size, self.num_locations, device=device)
        last_location_mask = torch.zeros(batch_size, self.num_locations, device=device)
        
        # Process sequence
        for t in range(seq_len):
            valid_mask = mask[:, t].float()
            locs = loc_seq[:, t]
            
            # Recency: exponential decay from end (aggressive for recent)
            position_from_end = seq_len - t - 1
            if position_from_end < 5:  # Extra boost for last 5
                recency_weight = torch.pow(self.recency_decay, position_from_end / 2.0)
            else:
                recency_weight = torch.pow(self.recency_decay, position_from_end)
            
            # Update recency
            scatter_recency = torch.zeros(batch_size, self.num_locations, device=device)
            scatter_recency.scatter_(1, locs.unsqueeze(1), (recency_weight * valid_mask).unsqueeze(1))
            recency_scores = torch.maximum(recency_scores, scatter_recency)
            
            # Frequency
            frequency_scores.scatter_add_(1, locs.unsqueeze(1), valid_mask.unsqueeze(1))
            
            # Track last location (for repeat boost)
            if t == seq_len - 1:
                last_location_mask.scatter_(1, locs.unsqueeze(1), valid_mask.unsqueeze(1))
        
        # Normalize frequency
        max_freq = frequency_scores.max(dim=1, keepdim=True)[0].clamp(min=1.0)
        frequency_normalized = frequency_scores / max_freq
        
        # Combine scores
        history_scores = (recency_scores + 
                         self.frequency_weight * frequency_normalized +
                         self.last_location_boost * last_location_mask)
        
        return history_scores
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        """Forward pass."""
        batch_size, seq_len = loc_seq.shape
        
        # === Strong history-based scoring ===
        history_scores = self.compute_history_scores(loc_seq, mask)
        
        # === Lightweight transformer modeling ===
        # Embeddings
        loc_emb = self.loc_emb(loc_seq)
        user_emb = self.user_emb(user_seq)
        
        # Temporal features
        hours = start_min_seq / 60.0
        time_rad = (hours / 24.0) * 2 * math.pi
        time_sin, time_cos = torch.sin(time_rad), torch.cos(time_rad)
        
        dur_norm = torch.log1p(dur_seq) / 10.0
        
        wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
        wd_sin, wd_cos = torch.sin(wd_rad), torch.cos(wd_rad)
        
        gap_norm = torch.clamp(diff_seq.float() / 7.0, 0, 1)
        
        temporal_feats = torch.stack([
            time_sin, time_cos, dur_norm, wd_sin, wd_cos, gap_norm
        ], dim=-1)
        temporal_emb = self.temporal_proj(temporal_feats)
        
        # Combine and project
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        x = self.norm(self.input_proj(x))
        
        # Single transformer block
        key_padding_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        last_indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, last_indices).squeeze(1)
        
        # Transformer predictions
        transformer_logits = self.predictor(last_hidden)
        
        # === Final ensemble: Heavy bias toward history ===
        transformer_probs = F.softmax(transformer_logits, dim=1)
        transformer_scaled = transformer_probs * self.num_locations
        
        final_logits = (self.history_scale * history_scores + 
                       self.model_scale * transformer_scaled)
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
