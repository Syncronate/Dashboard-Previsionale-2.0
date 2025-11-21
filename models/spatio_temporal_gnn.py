"""
Modello GNN Spazio-Temporale con Attenzione.
Versione con edge_weight batching fix + Monotonic Quantile Prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MonotonicQuantileHead(nn.Module):
    """
    Predice quantili con monotonicità garantita.
    
    Invece di predire [Q10, Q50, Q90] direttamente, predice:
    - Q_low (Q10)
    - Δ_mid (Q50 - Q10, softplus per positività)
    - Δ_high (Q90 - Q50, softplus per positività)
    
    Garantisce Q10 ≤ Q50 ≤ Q90 per costruzione.
    """
    
    def __init__(self, hidden_dim, output_window, output_dim=1, num_quantiles=3):
        super().__init__()
        
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        
        # Layer per il quantile più basso (Q10)
        self.low_head = nn.Linear(hidden_dim, output_window * output_dim)
        
        # Layer per gli incrementi tra quantili successivi
        # (num_quantiles - 1) perché il primo è Q_low
        self.increment_head = nn.Linear(
            hidden_dim, 
            output_window * output_dim * (num_quantiles - 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_nodes, hidden_dim)
        
        Returns:
            quantiles: (batch, output_window, num_nodes, output_dim, num_quantiles)
        """
        batch, num_nodes, hidden_dim = x.shape
        
        # 1. Predici il quantile più basso (Q10)
        q_low = self.low_head(x)  # (batch, num_nodes, output_window * output_dim)
        q_low = q_low.view(batch, num_nodes, self.output_window, self.output_dim)
        
        # 2. Predici gli incrementi (differenze tra quantili consecutivi)
        increments = self.increment_head(x)  # (batch, num_nodes, output_window * output_dim * (Q-1))
        increments = increments.view(
            batch, num_nodes, self.output_window, self.output_dim, self.num_quantiles - 1
        )
        
        # 3. Applica Softplus per garantire positività degli incrementi
        # Softplus(x) = log(1 + exp(x)) → sempre ≥ 0
        increments = F.softplus(increments)
        
        # 4. Costruisci i quantili cumulativamente
        quantiles = [q_low.unsqueeze(-1)]  # Q10: (batch, num_nodes, output_window, output_dim, 1)
        
        current_q = q_low
        for i in range(self.num_quantiles - 1):
            # Q_next = Q_current + Δ_i
            current_q = current_q + increments[..., i]
            quantiles.append(current_q.unsqueeze(-1))
        
        # 5. Concatena lungo la dimensione quantili
        quantiles = torch.cat(quantiles, dim=-1)  
        # → (batch, num_nodes, output_window, output_dim, num_quantiles)
        
        # 6. Riordina per matchare il formato atteso 
        # (batch, output_window, num_nodes, output_dim, num_quantiles)
        quantiles = quantiles.permute(0, 2, 1, 3, 4)
        
        return quantiles


class SpatioTemporalAttentionGNN(nn.Module):
    """
    GNN con attenzione spazio-temporale per forecasting.
    
    Architettura:
    1. GCN Layer → Cattura relazioni spaziali sul grafo
    2. Attention Layer → Focus su nodi/timesteps rilevanti
    3. GRU Layer → Cattura dinamiche temporali
    4. Monotonic Quantile Head → Predizione con incertezza calibrata
    """
    
    def __init__(self, num_nodes, num_features, hidden_dim, rnn_layers, 
                 output_window, output_dim, num_quantiles=1, dropout=0.2,
                 attention_heads=1):
        super().__init__()
        
        # Salva parametri
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout
        self.attention_heads = attention_heads
        
        # === LAYER 1: GCN (Graph Convolutional Network) ===
        self.gcn = GCNConv(num_features, hidden_dim)
        self.gcn_activation = nn.ELU()
        self.gcn_dropout = nn.Dropout(dropout)
        
        # === LAYER 2: Attenzione Spazio-Temporale ===
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Query
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Key
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Value
        self.v_att = nn.Linear(hidden_dim, 1)  # Attention scoring
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # === LAYER 3: GRU (Gated Recurrent Unit) ===
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        self.rnn_norm = nn.LayerNorm(hidden_dim)
        
        # === LAYER 4: Pre-Head (espansione a per-node features) ===
        self.pre_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_nodes * hidden_dim)
        )
        
        # === LAYER 5: Quantile Prediction Head ===
        if num_quantiles > 1:
            # ✅ NUOVO: Head con monotonicità garantita
            self.quantile_head = MonotonicQuantileHead(
                hidden_dim=hidden_dim,
                output_window=output_window,
                output_dim=output_dim,
                num_quantiles=num_quantiles
            )
        else:
            # Singolo output (regressione deterministica)
            self.quantile_head = nn.Linear(hidden_dim, output_window * output_dim)
        
        # Inizializza pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializzazione Xavier/Orthogonal per stabilità training"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                if 'rnn' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass del modello.
        
        Args:
            x: (batch, seq_len, num_nodes, num_features)
                - batch: numero di sequenze nel batch
                - seq_len: lunghezza sequenza temporale (es. 48 = 24 ore)
                - num_nodes: numero nodi nel grafo (es. 5 idrometri)
                - num_features: features per nodo (es. 8)
            
            edge_index: (2, num_edges)
                - Indici archi del grafo [source_nodes, target_nodes]
            
            edge_weight: (num_edges,) optional
                - Pesi degli archi (es. basati su tempo corrivazione)
            
        Returns:
            predictions: (batch, output_window, num_nodes, output_dim, num_quantiles)
                - Predizioni future con quantili monotoni
            
            attention_weights: (batch, seq_len, num_nodes)
                - Pesi di attenzione per visualizzazione
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # ======================================================================
        # STEP 1: GCN - Graph Convolution per ogni timestep
        # ======================================================================
        gcn_outputs = []
        
        for t in range(seq_len):
            # Estrai dati al tempo t
            x_t = x[:, t, :, :]  # (batch, num_nodes, num_features)
            
            # Flatten batch e nodi per GCN
            x_t_flat = x_t.reshape(batch_size * num_nodes, num_features)
            
            # Crea edge_index per tutto il batch
            edge_index_batch = self._create_batch_edge_index(edge_index, batch_size, num_nodes)
            
            # Replica edge_weight per ogni elemento del batch
            if edge_weight is not None:
                edge_weight_batch = edge_weight.repeat(batch_size)
            else:
                edge_weight_batch = None
            
            # Applica GCN
            h_t = self.gcn(x_t_flat, edge_index_batch, edge_weight_batch)
            h_t = self.gcn_activation(h_t)
            h_t = self.gcn_dropout(h_t)
            
            # Reshape a (batch, num_nodes, hidden_dim)
            h_t = h_t.view(batch_size, num_nodes, self.hidden_dim)
            
            gcn_outputs.append(h_t)
        
        # Stack lungo dimensione temporale
        x_gcn_seq = torch.stack(gcn_outputs, dim=1)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # ======================================================================
        # STEP 2: Attenzione Spazio-Temporale
        # ======================================================================
        # Flatten spazio-tempo per attention
        keys = x_gcn_seq.view(batch_size, seq_len * num_nodes, self.hidden_dim)
        
        # Query: usa ultimo timestep, media su tutti i nodi
        query = x_gcn_seq[:, -1, :, :].mean(dim=1)  # (batch, hidden_dim)
        
        # Proiezioni Query-Key-Value
        query_proj = self.W_q(query).unsqueeze(1)  # (batch, 1, hidden_dim)
        keys_proj = self.W_k(keys)  # (batch, seq_len*num_nodes, hidden_dim)
        values_proj = self.W_v(keys)  # (batch, seq_len*num_nodes, hidden_dim)
        
        # Calcola attention scores
        energy = torch.tanh(query_proj + keys_proj)  # (batch, seq_len*num_nodes, hidden_dim)
        scores = self.v_att(energy).squeeze(-1)  # (batch, seq_len*num_nodes)
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len*num_nodes)
        
        # ======================================================================
        # STEP 3: Context Vector (weighted sum)
        # ======================================================================
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len*num_nodes)
            values_proj  # (batch, seq_len*num_nodes, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        context = self.attention_norm(context)
        
        # ======================================================================
        # STEP 4: GRU - Temporal Refinement
        # ======================================================================
        rnn_input = context.unsqueeze(1)  # (batch, 1, hidden_dim)
        rnn_output, _ = self.rnn(rnn_input)
        final_vector = self.rnn_norm(rnn_output.squeeze(1))  # (batch, hidden_dim)
        
        # ======================================================================
        # STEP 5: Predizione Finale con Quantili Monotoni
        # ======================================================================
        
        # 5.1: Espandi a rappresentazione per-nodo
        node_features = self.pre_head(final_vector)  # (batch, num_nodes * hidden_dim)
        node_features = node_features.view(batch_size, num_nodes, self.hidden_dim)
        
        # 5.2: Predici quantili
        if self.num_quantiles > 1:
            # ✅ Usa MonotonicQuantileHead (garantisce Q10 ≤ Q50 ≤ Q90)
            predictions = self.quantile_head(node_features)
            # → (batch, output_window, num_nodes, output_dim, num_quantiles)
        else:
            # Singolo output (deterministic)
            out = self.quantile_head(node_features)  # (batch, num_nodes, output_window * output_dim)
            out = out.view(batch_size, num_nodes, self.output_window, self.output_dim)
            predictions = out.permute(0, 2, 1, 3).unsqueeze(-1)
            # → (batch, output_window, num_nodes, output_dim, 1)
        
        # Reshape attention weights per visualizzazione
        attention_weights_reshaped = attention_weights.view(batch_size, seq_len, num_nodes)
        
        return predictions, attention_weights_reshaped
    
    def _create_batch_edge_index(self, edge_index, batch_size, num_nodes):
        """
        Crea edge_index per batch di grafi.
        
        Ogni sample nel batch ha lo stesso grafo, ma con nodi offsettati.
        
        Args:
            edge_index: (2, num_edges) - grafo singolo
            batch_size: numero di grafi nel batch
            num_nodes: numero nodi per grafo
            
        Returns:
            edge_index_batch: (2, num_edges * batch_size)
        """
        edge_indices = []
        
        for b in range(batch_size):
            offset = b * num_nodes
            edge_indices.append(edge_index + offset)
        
        return torch.cat(edge_indices, dim=1)
    
    def get_config(self):
        """
        Ritorna configurazione modello per serializzazione.
        """
        return {
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'hidden_dim': self.hidden_dim,
            'rnn_layers': self.rnn_layers,
            'output_window': self.output_window,
            'output_dim': self.output_dim,
            'num_quantiles': self.num_quantiles,
            'dropout': self.dropout_rate,
            'attention_heads': self.attention_heads
        }
    
    def count_parameters(self):
        """Conta parametri trainabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, x, edge_index, edge_weight=None):
        """
        Estrae solo attention weights (per visualizzazione).
        
        Args:
            x: (batch, seq_len, num_nodes, num_features)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional
            
        Returns:
            attention_weights: (batch, seq_len, num_nodes)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index, edge_weight)
        return attention_weights