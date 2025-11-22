"""
TCN con Temporal Attention - VERSIONE TRAINING (no input_proj)
"""

import torch
import torch.nn as nn


class TCNWithAttention(nn.Module):
    """
    TCN compatibile con checkpoint training.
    NO input_proj - il primo blocco prende num_features direttamente.
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
        
        # ✅ NO input_proj (come nel checkpoint!)
        
        # TCN network
        self.tcn = nn.ModuleDict()
        self.tcn['network'] = nn.ModuleList()
        
        for i in range(num_blocks):
            dilation = 2 ** i
            
            # ✅ Primo blocco prende num_features, altri hidden_dim
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
        
        # Temporal attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Receptive field
        self.receptive_field = 1 + sum([2 ** i * (kernel_size - 1) for i in range(num_blocks)])
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, timesteps, num_features)
        
        Returns:
            (batch, timesteps, hidden_dim)
        """
        # ✅ NO input projection - passa direttamente ai TCN blocks
        
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
    """Blocco TCN con dimensioni configurabili"""
    
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
        
        # Residual connection (se dimensioni cambiano)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, timesteps, in_channels)
        """
        residual = x
        
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
        out = out + self.residual(residual)
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
        """
        x = x.transpose(1, 2)
        
        out = self.conv(x)
        
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = out.transpose(1, 2)
        
        return out
