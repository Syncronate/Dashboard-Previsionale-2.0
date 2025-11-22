"""
TCN con Temporal Attention - STRUTTURA CORRETTA
"""

import torch
import torch.nn as nn


class TCNWithAttention(nn.Module):
    """
    TCN con optional temporal attention.
    Struttura che match il checkpoint:
    self.temporal_attention.attention.*
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
        
        # TCN network
        self.tcn = nn.ModuleDict()
        self.tcn['network'] = nn.ModuleList()
        
        for i in range(num_blocks):
            dilation = 2 ** i
            in_channels = num_features if i == 0 else hidden_dim
            
            self.tcn['network'].append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # ✅ Temporal attention con struttura corretta
        if use_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Receptive field
        self.receptive_field = 1 + sum([2 ** i * (kernel_size - 1) for i in range(num_blocks)])
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, timesteps, num_features)
        Returns:
            (batch, timesteps, hidden_dim)
        """
        # TCN blocks
        for block in self.tcn['network']:
            x = block(x)
        
        # Temporal attention
        if self.use_attention:
            if return_attention_weights:
                x, attn_weights = self.temporal_attention(x, return_weights=True)
                return x, attn_weights
            else:
                x = self.temporal_attention(x)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal Attention wrapper.
    Struttura che match checkpoint:
    self.attention = MultiheadAttention
    self.norm = LayerNorm
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # ✅ Nomi esatti come nel checkpoint
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: (batch, timesteps, hidden_dim)
        Returns:
            (batch, timesteps, hidden_dim)
        """
        # Self-attention
        if return_weights:
            attn_out, attn_weights = self.attention(x, x, x)
            out = self.norm(x + attn_out)
            return out, attn_weights
        else:
            attn_out, _ = self.attention(x, x, x)
            out = self.norm(x + attn_out)
            return out


class TCNBlock(nn.Module):
    """Blocco TCN con causal convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convolutional layers
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual (Conv1d per match checkpoint)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, timesteps, in_channels)
        """
        residual_input = x
        
        # Conv1
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.bn1(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Conv2
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)
        
        # Residual
        if isinstance(self.residual, nn.Conv1d):
            res = residual_input.transpose(1, 2)
            res = self.residual(res)
            res = res.transpose(1, 2)
        else:
            res = residual_input
        
        out = out + res
        out = self.relu(out)
        
        return out


class CausalConv1d(nn.Module):
    """Causal 1D Convolution"""
    
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
        x = x.transpose(1, 2)
        out = self.conv(x)
        
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = out.transpose(1, 2)
        return out
