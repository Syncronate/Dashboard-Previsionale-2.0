"""
SimpleTCN per Bettolelle - Compatibile con checkpoint esistenti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTCN(nn.Module):
    """
    TCN con default allineati al config Bettolelle.
    Tutti i parametri opzionali per backward compatibility.
    """
    
    def __init__(
        self,
        # ✅ DEFAULT DAL TUO CONFIG
        num_nodes: int = 5,              # graph.nodes (5 nodi)
        num_features: int = 20,          # Calcolato dalle feature effettive
        output_window: int = 12,         # data.output_window (6h = 12 step)
        hidden_dim: int = 256,           # model.hidden_dim
        output_dim: int = 1,             # Predici solo livello
        num_quantiles: int = 3,          # model.num_quantiles [0.1, 0.5, 0.9]
        num_blocks: int = 5,             # model.tcn.num_blocks
        kernel_size: int = 3,            # model.tcn.kernel_size
        dropout: float = 0.2,            # model.dropout
        use_temporal_attention: bool = True,  # model.tcn.use_temporal_attention
        # Parametri deprecati
        input_dim: int = None,
        tcn_blocks: int = None,          # ✅ AGGIUNTO per il tuo checkpoint
        attention_heads: int = None,     # ✅ AGGIUNTO (non usato in SimpleTCN)
        asymmetric_loss: bool = None,   # ✅ AGGIUNTO (gestito dal wrapper)
        **kwargs  # Ignora altri parametri dal checkpoint
    ):
        super().__init__()
        
        # ✅ Gestisci nomi alternativi
        if input_dim is not None:
            num_features = input_dim
        
        if tcn_blocks is not None:
            num_blocks = tcn_blocks
        
        # Salva parametri
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.output_window = output_window
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        self.num_blocks = num_blocks
        self.use_attention = use_temporal_attention
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Temporal attention (opzionale)
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(
            hidden_dim, 
            output_window * output_dim * num_quantiles
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch, input_timesteps, nodes, features]
        
        Returns:
            [batch, output_window, nodes, output_dim, num_quantiles]
        """
        batch_size, input_seq_len, num_nodes, features = x.shape
        
        # Flatten nodes
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * num_nodes, input_seq_len, features)
        
        # Input projection
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Temporal attention
        if self.use_attention:
            x = self.temporal_attention(x)
        
        # Usa ultimo timestep
        last_hidden = x[:, -1, :]
        
        # Output projection
        out = self.output_proj(last_hidden)
        
        # Reshape
        out = out.view(
            batch_size,
            num_nodes,
            self.output_window,
            self.output_dim,
            self.num_quantiles
        )
        
        # Permuta
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        
        return out


class TCNBlock(nn.Module):
    """Blocco TCN con causal convolution"""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = x.transpose(1, 2)
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = out.transpose(1, 2)
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class TemporalAttention(nn.Module):
    """Self-attention temporale"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)
