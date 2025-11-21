"""
Lightning wrapper SEMPLIFICATO - Solo SimpleTCN
Rimuove dipendenze da LSTM, GNN complessi, routing, torch_geometric
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .simple_tcn import SimpleTCN
import pandas as pd


class LitSpatioTemporalGNN(pl.LightningModule):
    """
    Lightning Module semplificato con:
    - Solo SimpleTCN (no LSTM, no GNN avanzati)
    - Weighted loss (piene + post-2024 + rising limb)
    - Quantile regression
    - Metriche event-based
    """
    
    def __init__(self, 
                 # Parametri modello
                 num_nodes, num_features, hidden_dim, rnn_layers,
                 output_window, output_dim, num_quantiles=1, dropout=0.2,
                 attention_heads=1,
                 # Encoder temporale (ignorato, sempre SimpleTCN)
                 temporal_encoder='simple_tcn',
                 tcn_blocks=5,
                 tcn_kernel_size=3,
                 use_temporal_attention=True,
                 # Parametri training
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 scheduler_patience=10,
                 scheduler_factor=0.5,
                 # Weighted loss
                 flood_threshold=2.0,
                 flood_weight=10.0,
                 regime_shift_date='2023-12-31',
                 post_2024_weight=3.0,
                 # Rising limb
                 use_rising_limb_weighting=False,
                 rising_limb_weight=1.0,
                 rising_limb_slope_threshold=0.05,
                 # Parametri ignorati (per compatibilità checkpoint)
                 use_hybrid_routing=False,
                 routing_mode='hybrid',
                 config=None,
                 input_noise_std=0.0,
                 edge_index=None,
                 edge_weight=None,
                 **kwargs):  # Cattura altri parametri non usati
        
        super().__init__()
        
        # Salva hyperparameters
        self.save_hyperparameters(ignore=['edge_index', 'edge_weight', 'config'])
        
        # ====================================================================
        # ESTRAI PARAMETRI TCN DAL CONFIG
        # ====================================================================
        
        tcn_config = {
            'tcn_blocks': tcn_blocks,
            'tcn_kernel_size': tcn_kernel_size,
            'use_temporal_attention': use_temporal_attention
        }
        
        if config and 'model' in config and 'tcn' in config['model']:
            tcn_cfg = config['model']['tcn']
            tcn_config.update({
                'tcn_blocks': tcn_cfg.get('num_blocks', tcn_blocks),
                'tcn_kernel_size': tcn_cfg.get('kernel_size', tcn_kernel_size),
                'use_temporal_attention': tcn_cfg.get('use_temporal_attention', use_temporal_attention),
            })
        
        # ✅ QUANTILE LEVELS
        self.quantile_levels = [0.5]  # Default
        
        if config and 'model' in config:
            self.quantile_levels = config['model'].get('quantile_levels', [0.5])
        
        if num_quantiles == 3 and len(self.quantile_levels) != 3:
            self.quantile_levels = [0.1, 0.5, 0.9]
        
        self.register_buffer('quantiles', torch.tensor(self.quantile_levels, dtype=torch.float32))
        
        # ====================================================================
        # CREA MODELLO SimpleTCN
        # ====================================================================
        
        print(f"\n🧠 Modello: SimpleTCN (lightweight, no dependencies)")
        print(f"   - Nodes: {num_nodes}")
        print(f"   - Features: {num_features}")
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - TCN blocks: {tcn_config['tcn_blocks']}")
        print(f"   - Quantiles: {self.quantile_levels}")
        
        self.model = SimpleTCN(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_quantiles=num_quantiles,
            num_blocks=tcn_config['tcn_blocks'],
            kernel_size=tcn_config['tcn_kernel_size'],
            dropout=dropout,
            use_temporal_attention=tcn_config['use_temporal_attention']
        )
        
        # ====================================================================
        # SETUP
        # ====================================================================
        
        self.config = config
        self.num_nodes = num_nodes
        self.output_window = output_window
        self.output_dim = output_dim
        
        # Converti regime shift date
        self.regime_shift_ts = pd.to_datetime(self.hparams.regime_shift_date).timestamp()

        # Accumula predizioni validation
        self.val_predictions = []
        self.val_targets = []
        self.val_timestamps = []
    
    def quantile_loss(self, preds, target, quantiles):
        """
        Quantile Loss (Pinball Loss).
        
        Args:
            preds: [batch, seq, nodes, output_dim, num_quantiles]
            target: [batch, seq, nodes, 1] o [batch, seq, nodes]
            quantiles: [num_quantiles]
        """
        # Normalizza target
        if target.dim() == 4 and target.size(-1) == 1:
            target = target.squeeze(-1)
        elif target.dim() == 4:
            target = target[..., 0]
        
        losses = []
        for i, q in enumerate(quantiles):
            pred_q = preds[..., i]
            
            if pred_q.dim() == 4 and pred_q.size(-1) == 1:
                pred_q = pred_q.squeeze(-1)
            elif pred_q.dim() == 4:
                pred_q = pred_q[..., 0]
            
            errors = target - pred_q
            loss_q = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_q)
        
        total_loss = torch.stack(losses, dim=-1).sum(dim=-1)
        return total_loss.mean()
    
    def forward(self, x, target=None):
        """
        Forward pass.
        
        Args:
            x: [batch, timesteps, num_nodes, num_features]
        Returns:
            [batch, output_window, num_nodes, output_dim, num_quantiles]
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # SimpleTCN forward
        out = self.model(x)  # [batch, seq_len, nodes, output_dim, num_quantiles]
        
        # Prendi ultimi output_window timestep
        out = out[:, -self.output_window:, :, :, :]
        
        return out
        
    def training_step(self, batch, batch_idx):
        """Training step con weighted loss."""
        x, y, ts = batch
        
        # Forward
        predictions = self(x)
        
        # ====================================================================
        # LOSS
        # ====================================================================
        
        if self.hparams.num_quantiles > 1:
            # Quantile Loss
            loss = self._compute_weighted_quantile_loss(predictions, y, ts)
            pred_median = predictions[..., self.hparams.num_quantiles // 2].squeeze(-1)
        else:
            # MSE
            pred_median = predictions.squeeze(-1)
            if pred_median.dim() == 4:
                pred_median = pred_median.squeeze(-1)
            loss = self._compute_weighted_mse_loss(pred_median, y, ts)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        self.log('train_mae', F.l1_loss(pred_median, y_true), on_step=True, on_epoch=True)
        
        return loss
    
    def _compute_weighted_quantile_loss(self, predictions, y, ts):
        """Quantile Loss con pesi."""
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        
        # Loss base
        base_loss = self.quantile_loss(predictions, y.unsqueeze(-1) if y.dim() == 3 else y, self.quantiles)
        
        # Predizione mediana per weighted errors
        pred_median = predictions[..., self.hparams.num_quantiles // 2]
        if pred_median.dim() == 4:
            pred_median = pred_median.squeeze(-1)
        
        # Pesi
        sample_weights = torch.ones_like(y_true)
        
        # 1. Flood
        flood_mask = (y_true > self.hparams.flood_threshold)
        if flood_mask.any():
            sample_weights[flood_mask] *= self.hparams.flood_weight
        
        # 2. Regime shift
        regime_mask = (ts.view(-1, 1, 1) > self.regime_shift_ts).expand_as(y_true)
        if regime_mask.any():
            sample_weights[regime_mask] *= self.hparams.post_2024_weight
        
        # 3. Rising limb
        if self.hparams.use_rising_limb_weighting and y.size(1) > 1:
            if y.dim() == 4:
                y_diff = y[:, 1:, :, :] - y[:, :-1, :, :]
                y_diff_squeezed = y_diff.squeeze(-1)
            else:
                y_diff_squeezed = y[:, 1:, :] - y[:, :-1, :]
            
            rising_mask = torch.zeros_like(y_true, dtype=torch.bool)
            rising_mask[:, 1:, :] = (y_diff_squeezed > self.hparams.rising_limb_slope_threshold)
            
            if rising_mask.any():
                sample_weights[rising_mask] *= self.hparams.rising_limb_weight
        
        # Weighted MSE
        weighted_errors = sample_weights * (pred_median - y_true) ** 2
        weighted_loss = weighted_errors.mean()
        
        total_loss = base_loss + weighted_loss
        
        self.log('train_quantile_loss', base_loss, on_step=False, on_epoch=True)
        self.log('train_weighted_mse', weighted_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def _compute_weighted_mse_loss(self, pred_median, y, ts):
        """MSE Loss con pesi."""
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        
        loss = F.mse_loss(pred_median, y_true)
        
        # 1. Flood
        flood_mask = (y_true > self.hparams.flood_threshold)
        if flood_mask.any():
            flood_loss = F.mse_loss(pred_median[flood_mask], y_true[flood_mask])
            loss = loss + self.hparams.flood_weight * flood_loss
            self.log('train_flood_loss', flood_loss, on_step=False, on_epoch=True)
        
        # 2. Regime
        regime_mask = (ts.view(-1, 1, 1) > self.regime_shift_ts).expand_as(y_true)
        if regime_mask.any():
            regime_loss = F.mse_loss(pred_median[regime_mask], y_true[regime_mask])
            loss = loss + self.hparams.post_2024_weight * regime_loss
            self.log('train_regime_loss', regime_loss, on_step=False, on_epoch=True)
        
        # 3. Rising limb
        if self.hparams.use_rising_limb_weighting and y.size(1) > 1:
            if y.dim() == 4:
                y_diff = y[:, 1:, :, :] - y[:, :-1, :, :]
                y_diff_squeezed = y_diff.squeeze(-1)
            else:
                y_diff_squeezed = y[:, 1:, :] - y[:, :-1, :]
            
            rising_mask = torch.zeros_like(y_true, dtype=torch.bool)
            rising_mask[:, 1:, :] = (y_diff_squeezed > self.hparams.rising_limb_slope_threshold)
            
            if rising_mask.any():
                rising_pred = pred_median[rising_mask]
                rising_true = y_true[rising_mask]
                if rising_pred.numel() > 0:
                    rising_loss = F.mse_loss(rising_pred, rising_true)
                    loss = loss + self.hparams.rising_limb_weight * rising_loss
                    self.log('train_rising_loss', rising_loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y, ts = batch
        
        predictions = self(x)
        
        # Estrai mediana
        if self.hparams.num_quantiles > 1:
            pred_median = predictions[..., self.hparams.num_quantiles // 2].squeeze(-1)
        else:
            pred_median = predictions.squeeze(-1)
            if pred_median.dim() == 4:
                pred_median = pred_median.squeeze(-1)
        
        y_true = y.squeeze(-1) if y.dim() == 4 else y

        # Loss
        loss = F.mse_loss(pred_median, y_true)
        
        if self.hparams.num_quantiles > 1:
            val_q_loss = self.quantile_loss(
                predictions, 
                y.unsqueeze(-1) if y.dim() == 3 else y, 
                self.quantiles
            )
            self.log('val_quantile_loss', val_q_loss, on_epoch=True)
        
        # MAE
        mae = F.l1_loss(pred_median, y_true)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        
        # Accumula
        self.val_predictions.append(pred_median.detach().cpu())
        self.val_targets.append(y_true.detach().cpu())
        self.val_timestamps.append(ts.detach().cpu())
        
        return loss
        
    def on_validation_epoch_end(self):
        """Metriche event-based."""
        
        if len(self.val_predictions) == 0:
            return
        
        all_preds = torch.cat(self.val_predictions, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)
        
        # Bettolelle (nodo 3)
        preds_bett = all_preds[:, :, 3]
        targets_bett = all_targets[:, :, 3]
        
        preds_flat = preds_bett.flatten()
        targets_flat = targets_bett.flatten()
        
        # Peak error
        peak_true = targets_flat.max()
        peak_pred = preds_flat.max()
        peak_error = abs(peak_pred - peak_true).item()
        
        self.log('val_event_peak_error', peak_error, prog_bar=True)
        self.log('val_event_peak_true', peak_true.item())
        self.log('val_event_peak_pred', peak_pred.item())
        
        # Time to peak
        time_true = targets_flat.argmax().item()
        time_pred = preds_flat.argmax().item()
        time_error_hours = abs(time_pred - time_true) * 0.5
        
        self.log('val_event_time_to_peak_error_hours', time_error_hours)
        
        # POD/FAR/CSI
        threshold = self.hparams.flood_threshold
        hits = ((targets_flat > threshold) & (preds_flat > threshold)).sum().item()
        misses = ((targets_flat > threshold) & (preds_flat <= threshold)).sum().item()
        false_alarms = ((targets_flat <= threshold) & (preds_flat > threshold)).sum().item()
        
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
        
        self.log('val_event_POD', pod)
        self.log('val_event_FAR', far)
        self.log('val_event_CSI', csi)
        
        # Reset
        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_timestamps.clear()
    
    def configure_optimizers(self):
        """Optimizer e scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=1e-7,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_event_peak_error",
                "frequency": 1
            }
        }
