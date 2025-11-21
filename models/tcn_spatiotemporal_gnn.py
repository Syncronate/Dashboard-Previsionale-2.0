"""
Architettura completa TCN + Temporal Attention + Spatial GNN.
VERSIONE AUTOREGRESSIVA con Teacher Forcing e Curriculum Learning.

FIX:
- Edge index salvato come buffer per compatibilità
- Supporto num_features variabili (opzionale)
- Clipping predizioni per stabilità
- Teacher forcing con curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random

from .temporal_convolution import TCNWithAttention


class TCNSpatioTemporalAttentionGNN(nn.Module):
    """
    Modello completo per previsione spazio-temporale:
    
    Supporta 2 modalità:
    - DIRECT: Predice tutti i timestep futuri in una volta (veloce)
    - AUTOREGRESSIVE: Predice un timestep alla volta, lo ri-usa (più realistico)
    
    Con Teacher Forcing durante training per stabilità.
    """
    
    def __init__(self, num_nodes, num_features, hidden_dim, 
                 output_window, output_dim=1, num_quantiles=1,
                 # TCN params
                 tcn_blocks=5, tcn_kernel_size=3,
                 # Attention params
                 attention_heads=4, use_temporal_attention=True,
                 # GNN params
                 gnn_layers=3,
                 # Prediction mode
                 prediction_mode='direct',  # 'direct' o 'autoregressive'
                 teacher_forcing_ratio=0.5,
                 # Regularization
                 dropout=0.1,
                 # ✅ NUOVO: Opzionale edge_index pre-registrato
                 edge_index=None,
                 edge_weight=None,
                 # ✅ NUOVO: Clipping bounds per stabilità
                 pred_min=-5.0,
                 pred_max=10.0):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        self.attention_heads = attention_heads
        
        # ✅ NUOVO: Modalità predizione
        self.prediction_mode = prediction_mode
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # ✅ NUOVO: Bounds per clipping predizioni (livelli realistici fiume)
        self.pred_min = pred_min
        self.pred_max = pred_max
        
        # ✅ NUOVO: Registra edge_index/weight se forniti
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.edge_index = None
            
        if edge_weight is not None:
            self.register_buffer('edge_weight', edge_weight)
        else:
            self.edge_weight = None
        
        # ====================================================================
        # TEMPORAL ENCODER: TCN + Attention
        # ====================================================================
        
        self.temporal_encoder = TCNWithAttention(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=tcn_blocks,
            kernel_size=tcn_kernel_size,
            num_heads=attention_heads,
            dropout=dropout,
            use_attention=use_temporal_attention
        )
        
        print(f"   TCN Receptive Field: {self.temporal_encoder.receptive_field} timestep "
              f"({self.temporal_encoder.receptive_field * 0.5:.1f} ore)")
        print(f"   Prediction Mode: {prediction_mode.upper()}")
        if prediction_mode == 'autoregressive':
            print(f"   Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")
            print(f"   Prediction Clipping: [{pred_min}, {pred_max}]")
        
        # ====================================================================
        # SPATIAL ENCODER: Graph Attention Networks
        # ====================================================================
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=attention_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True
                )
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))
        
        # ====================================================================
        # OUTPUT NETWORK
        # ====================================================================
        
        if prediction_mode == 'direct':
            # Predice TUTTI i timestep insieme
            output_size = output_window * output_dim * num_quantiles
        else:  # autoregressive
            # Predice UN timestep alla volta
            output_size = output_dim * num_quantiles
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size)
        )
        
        # ✅ NUOVO: Per autoregressive, serve proiettare predizione a features
        if prediction_mode == 'autoregressive':
            self.pred_to_features = nn.Linear(output_dim, num_features)
    
    def forward(self, x, edge_index=None, edge_weight=None, 
                target=None, return_attention=False):
        """
        Args:
            x: (batch, timesteps, num_nodes, num_features)
            edge_index: (2, num_edges) - opzionale se registrato nel __init__
            edge_weight: (num_edges,) - opzionale
            target: (batch, output_window, num_nodes, output_dim) - solo per training autoregressivo
            return_attention: Se True, ritorna anche pesi di attenzione
        
        Returns:
            predictions: (batch, output_window, num_nodes, output_dim, num_quantiles)
            info (opzionale): Dict con attention weights
        """
        # ✅ USA edge_index registrato se non fornito
        if edge_index is None:
            edge_index = self.edge_index
        if edge_weight is None:
            edge_weight = self.edge_weight
        
        # ✅ VERIFICA che edge_index sia disponibile
        if edge_index is None:
            raise ValueError(
                "edge_index must be provided either in __init__ or forward()"
            )
        
        if self.prediction_mode == 'direct':
            return self._forward_direct(x, edge_index, edge_weight, return_attention)
        else:
            return self._forward_autoregressive(x, edge_index, edge_weight, target, return_attention)
    
    def _forward_direct(self, x, edge_index, edge_weight, return_attention):
        """Modalità DIRECT: predice tutti i timestep in una volta"""
        batch_size, timesteps, num_nodes, num_features = x.shape
        
        # ====================================================================
        # STEP 1: TEMPORAL ENCODING
        # ====================================================================
        
        node_embeddings = []
        temporal_attn_weights = [] if return_attention else None
        
        for node_idx in range(num_nodes):
            node_ts = x[:, :, node_idx, :]
            
            if return_attention:
                h_node, attn_w = self.temporal_encoder(node_ts, return_attention_weights=True)
                temporal_attn_weights.append(attn_w)
            else:
                h_node = self.temporal_encoder(node_ts)
            
            node_emb = h_node[:, -1, :]
            node_embeddings.append(node_emb)
        
        x_temporal = torch.stack(node_embeddings, dim=1)
        
        # ====================================================================
        # STEP 2: SPATIAL ENCODING
        # ====================================================================
        
        x_spatial = self._apply_gnn(x_temporal, edge_index, edge_weight, batch_size)
        
        # ====================================================================
        # STEP 3: OUTPUT (tutti i timestep insieme)
        # ====================================================================
        
        predictions = []
        
        for node_idx in range(num_nodes):
            node_emb = x_spatial[:, node_idx, :]
            out = self.output_net(node_emb)
            out = out.reshape(batch_size, self.output_window, self.output_dim, self.num_quantiles)
            predictions.append(out)
        
        predictions = torch.stack(predictions, dim=2)
        
        if return_attention:
            info = {'temporal_attention': temporal_attn_weights}
            return predictions, info
        else:
            return predictions
    
    def _forward_autoregressive(self, x, edge_index, edge_weight, target, return_attention):
        """
        Modalità AUTOREGRESSIVE: predice un timestep alla volta.
        
        Con Teacher Forcing durante training:
        - Random choice: usa ground truth o predizione come input
        - Durante inference (target=None): usa sempre predizioni
        
        ✅ NUOVO: Con clipping per stabilità
        """
        batch_size, timesteps, num_nodes, num_features = x.shape
        
        predictions = []
        current_input = x.clone()  # Input iniziale: dati storici
        
        # ====================================================================
        # LOOP AUTOREGRESSIVO
        # ====================================================================
        
        for step in range(self.output_window):
            # ================================================================
            # STEP 1: TEMPORAL ENCODING (su current_input)
            # ================================================================
            
            node_embeddings = []
            
            for node_idx in range(num_nodes):
                node_ts = current_input[:, :, node_idx, :]
                h_node = self.temporal_encoder(node_ts)
                node_emb = h_node[:, -1, :]
                node_embeddings.append(node_emb)
            
            x_temporal = torch.stack(node_embeddings, dim=1)
            
            # ================================================================
            # STEP 2: SPATIAL ENCODING
            # ================================================================
            
            x_spatial = self._apply_gnn(x_temporal, edge_index, edge_weight, batch_size)
            
            # ================================================================
            # STEP 3: PREDICI PROSSIMO TIMESTEP
            # ================================================================
            
            step_predictions = []
            
            for node_idx in range(num_nodes):
                node_emb = x_spatial[:, node_idx, :]
                out = self.output_net(node_emb)  # (batch, output_dim * num_quantiles)
                out = out.reshape(batch_size, self.output_dim, self.num_quantiles)
                step_predictions.append(out)
            
            # (batch, num_nodes, output_dim, num_quantiles)
            step_pred = torch.stack(step_predictions, dim=1)
            
            predictions.append(step_pred)
            
            # ================================================================
            # STEP 4: PREPARA INPUT PER PROSSIMA ITERAZIONE
            # ================================================================
            
            # Estrai predizione mediana per usarla come input
            median_idx = self.num_quantiles // 2
            next_pred = step_pred[..., median_idx]  # (batch, num_nodes, output_dim)
            
            # ✅ NUOVO: CLIPPING per prevenire esplosioni numeriche
            next_pred = torch.clamp(next_pred, min=self.pred_min, max=self.pred_max)
            
            # ✅ TEACHER FORCING (solo durante training)
            if self.training and target is not None:
                # Decide: uso ground truth o predizione?
                use_target = random.random() < self.teacher_forcing_ratio
                
                if use_target and step < target.shape[1]:
                    # Usa ground truth
                    next_input = target[:, step, :, :]  # (batch, num_nodes, output_dim)
                    if next_input.dim() == 4:  # Squeeze se (batch, num_nodes, output_dim, 1)
                        next_input = next_input.squeeze(-1)
                else:
                    # Usa predizione (già clippata)
                    next_input = next_pred
            else:
                # Inference: usa sempre predizione (già clippata)
                next_input = next_pred
            
            # Proietta predizione (output_dim) a features (num_features)
            next_features = []
            for node_idx in range(num_nodes):
                node_pred = next_input[:, node_idx, :]  # (batch, output_dim)
                node_feat = self.pred_to_features(node_pred)  # (batch, num_features)
                next_features.append(node_feat)
            
            next_features = torch.stack(next_features, dim=1)  # (batch, num_nodes, num_features)
            
            # Shift input sequence: rimuovi primo timestep, aggiungi predizione
            current_input = torch.cat([
                current_input[:, 1:, :, :],  # Rimuovi t-oldest
                next_features.unsqueeze(1)    # Aggiungi t+step
            ], dim=1)
        
        # ====================================================================
        # STACK PREDICTIONS
        # ====================================================================
        
        # predictions: list di (batch, num_nodes, output_dim, num_quantiles)
        # Stack: (batch, output_window, num_nodes, output_dim, num_quantiles)
        predictions = torch.stack(predictions, dim=1)
        
        if return_attention:
            info = {}  # Attention non implementata in autoregressive (troppo costoso)
            return predictions, info
        else:
            return predictions
    
    def _apply_gnn(self, x_temporal, edge_index, edge_weight, batch_size):
        """Helper per applicare GNN layers"""
        num_nodes = x_temporal.shape[1]
        
        # Reshape per GNN
        x_spatial = x_temporal.reshape(batch_size * num_nodes, -1)
        
        # Replica edge_index per batch
        if batch_size > 1:
            edge_index_batch = []
            edge_weight_batch = [] if edge_weight is not None else None
            
            for b in range(batch_size):
                offset = b * num_nodes
                edge_index_batch.append(edge_index + offset)
                
                if edge_weight is not None:
                    edge_weight_batch.append(edge_weight)
            
            edge_index_batch = torch.cat(edge_index_batch, dim=1)
            edge_weight_batch = torch.cat(edge_weight_batch) if edge_weight is not None else None
        else:
            edge_index_batch = edge_index
            edge_weight_batch = edge_weight
        
        # GNN layers
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_spatial = gnn(x_spatial, edge_index_batch, edge_weight_batch)
            x_spatial = norm(x_spatial)
            x_spatial = F.relu(x_spatial)
        
        # Reshape back
        x_spatial = x_spatial.reshape(batch_size, num_nodes, -1)
        
        return x_spatial
    
    def set_teacher_forcing_ratio(self, ratio):
        """
        Cambia teacher forcing ratio (utile per curriculum learning).
        
        Args:
            ratio: float in [0, 1]
                0 = sempre usa predizioni (difficile)
                1 = sempre usa ground truth (facile)
        """
        self.teacher_forcing_ratio = max(0.0, min(1.0, ratio))
    
    def get_config(self):
        """Ritorna configurazione del modello (per logging/checkpointing)"""
        return {
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'hidden_dim': self.hidden_dim,
            'output_window': self.output_window,
            'output_dim': self.output_dim,
            'num_quantiles': self.num_quantiles,
            'prediction_mode': self.prediction_mode,
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'pred_min': self.pred_min,
            'pred_max': self.pred_max,
            'receptive_field': self.temporal_encoder.receptive_field
        }