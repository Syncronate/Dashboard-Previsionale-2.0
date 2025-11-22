"""
Simple TCN Model for Spatio-Temporal Forecasting.
Flattens spatial dimension (nodes) into features.
Supports:
- Temporal Attention
- Autoregressive Prediction (with Teacher Forcing)
- Quantile Output
"""

import torch
import torch.nn as nn
import random
from .temporal_convolution import TCNWithAttention

class SimpleTCN(nn.Module):
    """
    Simplified TCN model that treats all nodes as a single feature vector.
    
    Input: (batch, timesteps, num_nodes, num_features)
    Internal: Flattens to (batch, timesteps, num_nodes * num_features)
    Output: (batch, output_window, num_nodes, output_dim, num_quantiles)
    """
    
    def __init__(self, num_nodes, num_features, hidden_dim, 
                 output_window, output_dim=1, num_quantiles=1,
                 # TCN params
                 tcn_blocks=5, tcn_kernel_size=3,
                 # Attention params
                 attention_heads=4, use_temporal_attention=True,
                 # Prediction mode
                 prediction_mode='direct',  # 'direct' or 'autoregressive'
                 teacher_forcing_ratio=0.5,
                 # Regularization
                 dropout=0.1,
                 # Bounds
                 pred_min=-5.0,
                 pred_max=10.0,
                 **kwargs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        
        self.prediction_mode = prediction_mode
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.pred_min = pred_min
        self.pred_max = pred_max
        
        # Flattened input dimension
        self.input_dim = num_nodes * num_features
        
        # ====================================================================
        # TEMPORAL ENCODER: TCN + Attention
        # ====================================================================
        
        self.temporal_encoder = TCNWithAttention(
            num_features=self.input_dim,
            hidden_dim=hidden_dim,
            num_blocks=tcn_blocks,
            kernel_size=tcn_kernel_size,
            num_heads=attention_heads,
            dropout=dropout,
            use_attention=use_temporal_attention
        )
        
        print(f"   SimpleTCN Receptive Field: {self.temporal_encoder.receptive_field} timestep")
        
        # ====================================================================
        # OUTPUT NETWORK
        # ====================================================================
        
        if prediction_mode == 'direct':
            # Predict ALL output steps at once
            output_size = output_window * num_nodes * output_dim * num_quantiles
        else:  # autoregressive
            # Predict ONE step at a time
            output_size = num_nodes * output_dim * num_quantiles
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size)
        )
        
        # For autoregressive: project prediction back to feature space
        if prediction_mode == 'autoregressive':
            self.pred_to_features = nn.Linear(output_dim, num_features)

    def forward(self, x, target=None, return_attention=False, **kwargs):
        """
        Args:
            x: (batch, timesteps, num_nodes, num_features)
            target: (batch, output_window, num_nodes, output_dim) - for teacher forcing
        """
        # Flatten nodes into features: (batch, timesteps, num_nodes * num_features)
        batch_size, timesteps, num_nodes, num_features = x.shape
        x_flat = x.reshape(batch_size, timesteps, num_nodes * num_features)
        
        if self.prediction_mode == 'direct':
            return self._forward_direct(x_flat, batch_size, return_attention)
        else:
            return self._forward_autoregressive(x_flat, target, batch_size, return_attention)

    def _forward_direct(self, x_flat, batch_size, return_attention):
        
        if return_attention:
            h, attn_weights = self.temporal_encoder(x_flat, return_attention_weights=True)
        else:
            h = self.temporal_encoder(x_flat)
            attn_weights = None
            
        # Take last timestep embedding
        emb = h[:, -1, :]  # (batch, hidden_dim)
        
        # Predict
        out = self.output_net(emb)
        
        # Reshape to (batch, output_window, num_nodes, output_dim, num_quantiles)
        predictions = out.reshape(
            batch_size, 
            self.output_window, 
            self.num_nodes, 
            self.output_dim, 
            self.num_quantiles
        )
        
        if return_attention:
            return predictions, {'temporal_attention': attn_weights}
        else:
            return predictions

    def _forward_autoregressive(self, x_flat, target, batch_size, return_attention):
        
        predictions = []
        current_input = x_flat.clone()
        
        for step in range(self.output_window):
            # Encoder
            h = self.temporal_encoder(current_input)
            emb = h[:, -1, :]
            
            # Predict next step
            out_flat = self.output_net(emb)
            
            # Reshape: (batch, num_nodes, output_dim, num_quantiles)
            step_pred = out_flat.reshape(
                batch_size, 
                self.num_nodes, 
                self.output_dim, 
                self.num_quantiles
            )
            
            predictions.append(step_pred)
            
            # Prepare next input
            median_idx = self.num_quantiles // 2
            next_pred = step_pred[..., median_idx] # (batch, num_nodes, output_dim)
            
            # Clipping
            next_pred = torch.clamp(next_pred, min=self.pred_min, max=self.pred_max)
            
            # Teacher Forcing
            if self.training and target is not None:
                use_target = random.random() < self.teacher_forcing_ratio
                if use_target and step < target.shape[1]:
                    next_input_vals = target[:, step, :, :]
                    if next_input_vals.dim() == 4:
                        next_input_vals = next_input_vals.squeeze(-1)
                else:
                    next_input_vals = next_pred
            else:
                next_input_vals = next_pred
            
            # Project to feature space
            next_input_vals_reshaped = next_input_vals.reshape(-1, self.output_dim)
            next_features_reshaped = self.pred_to_features(next_input_vals_reshaped)
            next_features = next_features_reshaped.reshape(batch_size, self.num_nodes * self.num_features)
            
            # Update input sequence
            current_input = torch.cat([
                current_input[:, 1:, :],  # Remove oldest
                next_features.unsqueeze(1) # Add newest
            ], dim=1)
            
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)
        
        if return_attention:
            return predictions, {}
        else:
            return predictions

    def set_teacher_forcing_ratio(self, ratio):
        self.teacher_forcing_ratio = max(0.0, min(1.0, ratio))
