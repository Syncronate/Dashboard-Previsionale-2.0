"""
TCN con Temporal Attention per serie temporali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNWithAttention(nn.Module):
    """
    Temporal Convolutional Network con Self-Attention opzionale.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_blocks: int = 5,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # TCN network
        self.tcn = nn.ModuleDict()
        self.tcn['network'] = nn.ModuleList()
        
        for i in range(num_blocks):
            dilation = 2 ** i
            self.tcn['network'].append(
                TCNBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Temporal attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Calcola receptive field
        self.receptive_field = 1 + sum([2 ** i * (kernel_size - 1) for i in range(num_blocks)])
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, timesteps, num_features)
        
        Returns:
            (batch, timesteps, hidden_dim)
        """
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        for block in self.tcn['network']:
            x = block(x)
        
        # Temporal attention
        if self.use_attention:
            if return_attention_weights:
                attn_out, attn_weights = self.attention(x, x, x)
                x = self.attn_norm(x + attn_out)
                return x, attn_weights
            else:
                attn_out, _ = self.attention(x, x, x)
                x = self.attn_norm(x + attn_out)
        
        return x


class TCNBlock(nn.Module):
    """Blocco TCN con causal dilated convolution"""
    
    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        # Convolutional layers
        self.conv1 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Residual connection
        self.residual = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, timesteps, hidden_dim)
        """
        residual = x
        
        # Conv1 - BatchNorm - ReLU
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch, timesteps, hidden) -> (batch, hidden, timesteps)
        out = self.bn1(out)
        out = out.transpose(1, 2)  # Back to (batch, timesteps, hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Conv2 - BatchNorm
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)
        
        # Residual
        out = out + self.residual(residual)
        out = self.relu(out)
        
        return out


class CausalConv1d(nn.Module):
    """Causal 1D Convolution (no future leakage)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, timesteps, channels)
        Returns:
            (batch, timesteps, channels)
        """
        # Conv1d expects (batch, channels, timesteps)
        x = x.transpose(1, 2)
        
        # Convolution
        out = self.conv(x)
        
        # Remove future (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        # Back to (batch, timesteps, channels)
        out = out.transpose(1, 2)
        
        return out
