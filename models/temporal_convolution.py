"""
Temporal Convolutional Networks (TCN) con dilation e residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Convoluzione 1D causale: vede solo il passato, mai il futuro.
    Usa padding asimmetrico per garantire causalità.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        # Padding solo a sinistra (passato)
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.conv(x)
        
        # Rimuovi padding futuro (right-side)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        
        return x


class TemporalBlock(nn.Module):
    """
    Blocco TCN con:
    - 2 convoluzioni causali
    - Batch normalization
    - Dropout
    - Residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        
        # Prima convoluzione
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Seconda convoluzione
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv se dimensioni diverse)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu_final = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, in_channels, time)
        
        # Salva input per residual
        residual = x
        
        # Prima conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Seconda conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        # Somma + attivazione finale
        out = self.relu_final(out + residual)
        
        return out


class TemporalConvNetwork(nn.Module):
    """
    Stack di TemporalBlock con dilation crescente.
    
    Receptive field totale:
        RF = 1 + num_blocks * (kernel_size - 1) * (2^num_blocks - 1)
        
    Esempio con kernel_size=3, num_blocks=5:
        RF = 1 + 5 * 2 * (2^5 - 1) = 1 + 10 * 31 = 311 timestep!
        Ma in pratica: layer 1 vede 3, layer 2 vede 7, layer 3 vede 15...
    """
    def __init__(self, num_features, hidden_dim, num_blocks=5, kernel_size=3, dropout=0.1):
        super().__init__()
        
        layers = []
        num_channels = [num_features] + [hidden_dim] * num_blocks
        
        for i in range(num_blocks):
            dilation = 2 ** i  # Dilation: 1, 2, 4, 8, 16, ...
            
            layers.append(
                TemporalBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Receptive field teorico
        self.receptive_field = 1 + sum([2**i * (kernel_size - 1) for i in range(num_blocks)])
        
    def forward(self, x):
        """
        Args:
            x: (batch, time, features)
        
        Returns:
            out: (batch, time, hidden_dim)
        """
        # Conv1d lavora su (batch, channels, time)
        x = x.transpose(1, 2)  # → (batch, features, time)
        
        x = self.network(x)
        
        # Torna a (batch, time, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention sulla dimensione temporale.
    Permette al modello di pesare quali timestep passati sono più rilevanti.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq, feature)
        )
        
        # Layer normalization per residual connection
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, time, hidden_dim)
            return_attention_weights: Se True, ritorna anche i pesi di attenzione
        
        Returns:
            out: (batch, time, hidden_dim)
            attn_weights (opzionale): (batch, num_heads, time, time)
        """
        # Self-attention (query = key = value = x)
        attn_out, attn_weights = self.attention(x, x, x, need_weights=return_attention_weights)
        
        # Residual connection + normalization
        out = self.norm(x + self.dropout(attn_out))
        
        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class TCNWithAttention(nn.Module):
    """
    Combina TCN (pattern locali) + Temporal Attention (dipendenze globali).
    
    TCN cattura pattern locali ripetuti (es. "pioggia → picco dopo 3h")
    Attention pesa quali timestep passati sono rilevanti per la predizione
    """
    def __init__(self, num_features, hidden_dim, num_blocks=5, kernel_size=3, 
                 num_heads=4, dropout=0.1, use_attention=True):
        super().__init__()
        
        # TCN encoder
        self.tcn = TemporalConvNetwork(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Temporal Attention (opzionale)
        self.use_attention = use_attention
        if use_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        self.receptive_field = self.tcn.receptive_field
        
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, time, features)
        
        Returns:
            out: (batch, time, hidden_dim)
            attn_weights (opzionale): attention weights se richiesti
        """
        # TCN encoding
        h = self.tcn(x)  # → (batch, time, hidden_dim)
        
        # Temporal attention (opzionale)
        if self.use_attention:
            if return_attention_weights:
                h, attn_weights = self.temporal_attention(h, return_attention_weights=True)
                return h, attn_weights
            else:
                h = self.temporal_attention(h)
        
        return h


# ============================================================================
# UTILITY: Calcola receptive field
# ============================================================================

def calculate_receptive_field(num_blocks, kernel_size=3):
    """
    Calcola il receptive field di una TCN.
    
    Args:
        num_blocks: Numero di blocchi TCN
        kernel_size: Dimensione del kernel
    
    Returns:
        receptive_field: Numero di timestep "visti" dall'ultimo layer
    """
    rf = 1
    for i in range(num_blocks):
        dilation = 2 ** i
        rf += (kernel_size - 1) * dilation
    
    return rf


if __name__ == "__main__":
    # Test
    print("="*70)
    print("🧪 TEST TEMPORAL CONVOLUTION NETWORK")
    print("="*70)
    
    # Config test
    batch_size = 4
    seq_len = 48  # 24 ore
    num_features = 8
    hidden_dim = 128
    num_blocks = 5
    
    # Crea modello
    model = TCNWithAttention(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        kernel_size=3,
        num_heads=4,
        use_attention=True
    )
    
    print(f"\n📊 Configurazione:")
    print(f"   Input: ({batch_size}, {seq_len}, {num_features})")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   TCN blocks: {num_blocks}")
    print(f"   Receptive field: {model.receptive_field} timestep")
    
    # Calcola parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔢 Parametri:")
    print(f"   Totali: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, num_features)
    
    with torch.no_grad():
        out, attn_weights = model(x, return_attention_weights=True)
    
    print(f"\n✅ Forward pass:")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test receptive field
    print(f"\n🔍 Receptive Field per Layer:")
    rf = 1
    for i in range(num_blocks):
        dilation = 2 ** i
        rf += (3 - 1) * dilation
        print(f"   Layer {i+1} (dilation={dilation}): RF = {rf} timestep")
    
    print("\n" + "="*70)
    print("✅ Test completato!")
    print("="*70)