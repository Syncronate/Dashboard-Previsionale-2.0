"""
TCN - VERSIONE ESATTA DAL CHECKPOINT TRAINING
NO attention, residual come Conv1d
"""

import torch
import torch.nn as nn


class TCNWithAttention(nn.Module):
    """
    TCN compatibile con checkpoint.
    use_attention è ignorato - il checkpoint NON ha attention!
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_blocks: int = 5,
        kernel_size: int = 3,
        num_heads: int = 4,  # Ignorato
        dropout: float = 0.1,
        use_attention: bool = True  # Ignorato - checkpoint NON ha attention
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        # ✅ Forza use_attention=False (checkpoint non ha questi layer!)
        self.use_attention = False
        
        # TCN network
        self.tcn = nn.ModuleDict()
        self.tcn['network'] = nn.ModuleList()
        
        for i in range(num_blocks):
            dilation = 2 ** i
            
            # Primo blocco: num_features -> hidden_dim
            # Altri: hidden_dim -> hidden_dim
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
        
        # ✅ NO attention (checkpoint non l'ha!)
        
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
        
        # ✅ NO attention
        
        if return_attention_weights:
            return x, None
        else:
            return x


class TCNBlock(nn.Module):
    """Blocco TCN ESATTO dal checkpoint"""
    
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
        
        # ✅ Residual come Conv1d (NON Linear!)
        # Checkpoint: [out_channels, in_channels, 1]
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
        # Salva per residual
        residual_input = x
        
        # Conv1
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch, timesteps, channels) -> (batch, channels, timesteps)
        out = self.bn1(out)
        out = out.transpose(1, 2)  # Back
        out = self.relu(out)
        out = self.dropout(out)
        
        # Conv2
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)
        
        # ✅ Residual (anche questo va attraverso Conv1d se dimensioni cambiano)
        # residual_input: (batch, timesteps, in_channels)
        # Dobbiamo passarlo attraverso self.residual che è Conv1d
        
        if isinstance(self.residual, nn.Conv1d):
            # Conv1d expects (batch, channels, timesteps)
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
        # Conv1d expects (batch, channels, timesteps)
        x = x.transpose(1, 2)
        
        out = self.conv(x)
        
        # Remove future
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        # Back to (batch, timesteps, channels)
        out = out.transpose(1, 2)
        
        return out
