"""
Lightning wrapper con weighted loss e metriche event-based.

AGGIORNATO: 
- ASYMMETRIC LOSS: Penalità 3x per sottostime (meglio falso allarme che mancato allarme)
- Curriculum Learning fixato per TCN autoregressive
- Quantile Loss con asimmetria direzionale
- Logging dettagliato per monitorare l'effetto della penalità
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .spatio_temporal_gnn import SpatioTemporalAttentionGNN
from .tcn_spatiotemporal_gnn import TCNSpatioTemporalAttentionGNN
from .simple_tcn import SimpleTCN
from .routing_physics import BettolleRoutingModule, HybridFloodModel
import pandas as pd
import warnings


class LitSpatioTemporalGNN(pl.LightningModule):
    """
    Lightning Module con:
    - Scelta tra LSTM o TCN (con supporto autoregressive)
    - Triplo weighted loss (piene + post-2024 + fase di salita)
    - ASYMMETRIC LOSS per penalizzare sottostime
    - Quantile regression opzionale
    - Metriche event-based
    - Supporto routing ibrido (opzionale)
    """
    
    def __init__(self, 
                 # Parametri modello
                 num_nodes, num_features, hidden_dim, rnn_layers,
                 output_window, output_dim, num_quantiles=1, dropout=0.2,
                 attention_heads=1,
                 # Encoder temporale
                 temporal_encoder='lstm',
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
                 # NUOVO: Asymmetric loss parameters
                 use_asymmetric_loss=True,
                 underestimation_penalty=3.0,
                 asymmetric_threshold=1.0,
                 # Routing ibrido
                 use_hybrid_routing=False,
                 routing_mode='hybrid',
                 config=None,
                 # Noise
                 input_noise_std=0.0,
                 # Grafo
                 edge_index=None,
                 edge_weight=None):
        
        super().__init__()
        
        # Salva hyperparameters
        self.save_hyperparameters(ignore=['edge_index', 'edge_weight', 'config'])
        
        # ====================================================================
        # ASYMMETRIC LOSS PARAMETERS
        # ====================================================================
        
        if use_asymmetric_loss:
            print(f"\n[ASYMMETRIC LOSS] Configurazione:")
            print(f"   ⚠️  Sottostima penalizzata: {underestimation_penalty}x")
            print(f"   📊  Soglia attivazione: {asymmetric_threshold}m")
            print(f"   🎯  Obiettivo: Preferire sovrastima conservativa")
        
        # ====================================================================
        # ESTRAI PARAMETRI TCN DAL CONFIG
        # ====================================================================
        
        tcn_config = {}
        self.tcn_prediction_mode = 'direct'  # Default
        self.tcn_teacher_forcing_ratio = 0.5  # Default
        
        if config and 'model' in config and 'tcn' in config['model']:
            tcn_cfg = config['model']['tcn']
            tcn_config = {
                'tcn_blocks': tcn_cfg.get('num_blocks', tcn_blocks),
                'tcn_kernel_size': tcn_cfg.get('kernel_size', tcn_kernel_size),
                'use_temporal_attention': tcn_cfg.get('use_temporal_attention', use_temporal_attention),
                'prediction_mode': tcn_cfg.get('prediction_mode', 'direct'),
                'teacher_forcing_ratio': tcn_cfg.get('teacher_forcing_ratio', 0.5)
            }
            # Salva per accesso diretto
            self.tcn_prediction_mode = tcn_config['prediction_mode']
            self.tcn_teacher_forcing_ratio = tcn_config['teacher_forcing_ratio']
            
            print(f"\n[CONFIG] TCN Config estratto dal YAML:")
            print(f"   - Prediction mode: {tcn_config['prediction_mode']}")
            print(f"   - Teacher forcing ratio: {tcn_config['teacher_forcing_ratio']}")
        
        # ESTRAI QUANTILE LEVELS
        self.quantile_levels = [0.5]  # Default: solo mediana
        
        if config and 'model' in config:
            self.quantile_levels = config['model'].get('quantile_levels', [0.5])
        
        # Se num_quantiles > 1 ma levels non forniti, usa default simmetrici
        if num_quantiles == 3 and len(self.quantile_levels) != 3:
            self.quantile_levels = [0.1, 0.5, 0.9]
            print(f"   [WARN] Quantile levels non specificati, uso default: {self.quantile_levels}")
        
        print(f"   [INFO] Quantile levels: {self.quantile_levels}")
        
        # REGISTRA SEMPRE (anche se num_quantiles=1) per compatibilità checkpoint
        self.register_buffer('quantiles', torch.tensor(self.quantile_levels, dtype=torch.float32))
        
        # ====================================================================
        # CREA MODELLO BASE (LSTM o TCN)
        # ====================================================================
        
        print(f"\n[INFO] Inizializzazione modello con encoder: {temporal_encoder.upper()}")
        
        if temporal_encoder == 'lstm':
            # Modello originale con LSTM
            self.gnn_model = SpatioTemporalAttentionGNN(
                num_nodes=num_nodes,
                num_features=num_features,
                hidden_dim=hidden_dim,
                rnn_layers=rnn_layers,
                output_window=output_window,
                output_dim=output_dim,
                num_quantiles=num_quantiles,
                dropout=dropout,
                attention_heads=attention_heads
            )
            self.is_autoregressive = False
        
        elif temporal_encoder == 'tcn':
            # Modello TCN con parametri autoregressive + edge_index
            self.gnn_model = TCNSpatioTemporalAttentionGNN(
                num_nodes=num_nodes,
                num_features=num_features,
                hidden_dim=hidden_dim,
                output_window=output_window,
                output_dim=output_dim,
                num_quantiles=num_quantiles,
                attention_heads=attention_heads,
                gnn_layers=rnn_layers,
                dropout=dropout,
                edge_index=edge_index,
                edge_weight=edge_weight,
                **tcn_config
            )
            self.is_autoregressive = (self.tcn_prediction_mode == 'autoregressive')
            
        elif temporal_encoder == 'simple_tcn':
            # Nuovo modello SimpleTCN (senza GNN, spatial flattening)
            self.gnn_model = SimpleTCN(
                num_nodes=num_nodes,
                num_features=num_features,
                hidden_dim=hidden_dim,
                output_window=output_window,
                output_dim=output_dim,
                num_quantiles=num_quantiles,
                attention_heads=attention_heads,
                tcn_blocks=tcn_config.get('tcn_blocks', tcn_blocks),
                tcn_kernel_size=tcn_config.get('tcn_kernel_size', tcn_kernel_size),
                use_temporal_attention=tcn_config.get('use_temporal_attention', use_temporal_attention),
                prediction_mode=tcn_config.get('prediction_mode', 'direct'),
                teacher_forcing_ratio=tcn_config.get('teacher_forcing_ratio', 0.5),
                dropout=dropout
            )
            self.is_autoregressive = (self.tcn_prediction_mode == 'autoregressive')
            
            print(f"\n[DEBUG] SimpleTCN Configurazione:")
            print(f"   - Temporal encoder: {temporal_encoder}")
            print(f"   - Prediction mode: {self.tcn_prediction_mode}")
            print(f"   - Is autoregressive: {self.is_autoregressive}")
            print(f"   - Initial TF ratio: {self.tcn_teacher_forcing_ratio}")
        
        else:
            raise ValueError(f"❌ temporal_encoder deve essere 'lstm', 'tcn' o 'simple_tcn', ricevuto: {temporal_encoder}")
        
        # Salva config
        self.config = config
        
        # ====================================================================
        # PARAMETRI CURRICULUM LEARNING (se autoregressive)
        # ====================================================================
        
        if self.is_autoregressive:
            # Parametri curriculum (personalizzabili dal config se disponibili)
            if config and 'training' in config and 'curriculum' in config['training']:
                curriculum_cfg = config['training']['curriculum']
                self.curriculum_initial_tf = curriculum_cfg.get('initial_tf', 0.9)
                self.curriculum_final_tf = curriculum_cfg.get('final_tf', 0.2)
                self.curriculum_decay_epochs = curriculum_cfg.get('decay_epochs', 20)
            else:
                # Default values
                self.curriculum_initial_tf = 0.9
                self.curriculum_final_tf = 0.2
                self.curriculum_decay_epochs = 20
            
            print(f"\n[CURRICULUM] Configurazione Curriculum Learning:")
            print(f"   - Initial TF: {self.curriculum_initial_tf:.0%}")
            print(f"   - Final TF: {self.curriculum_final_tf:.0%}")
            print(f"   - Decay epochs: {self.curriculum_decay_epochs}")
        
        # ====================================================================
        # AGGIUNGI ROUTING (se abilitato)
        # ====================================================================
        
        self.use_routing = use_hybrid_routing
        
        if use_hybrid_routing:
            if config is None:
                warnings.warn(
                    "[WARN] Config non fornito per routing ibrido. "
                    "Verrà usato un config di default minimale."
                )
                self.config = self._create_default_config(output_window, routing_mode)
            
            self.routing_module = BettolleRoutingModule(self.config)
            self.model = HybridFloodModel(self.gnn_model, self.routing_module, output_window)
        else:
            self.model = self.gnn_model
            self.routing_module = None
        
        # ====================================================================
        # SETUP
        # ====================================================================
        
        # Grafo
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        if edge_weight is not None:
            self.register_buffer('edge_weight', edge_weight)
            
        # Converti data regime shift
        self.regime_shift_ts = pd.to_datetime(self.hparams.regime_shift_date).timestamp()

        # Accumula predizioni validation
        self.val_predictions = []
        self.val_targets = []
        self.val_timestamps = []
        
        # Traccia statistiche asimmetria
        self.underestimation_count = 0
        self.overestimation_count = 0
    
    def _create_default_config(self, output_window, routing_mode):
        """Crea config minimale per routing."""
        return {
            'routing': {
                'enabled': True,
                'mode': routing_mode,
                'warmup_gnn_epochs': 5,
                'metadata_path': None,
                'K_init': [0.5, 0.5, 0.5],
                'X_init': [0.2, 0.2, 0.2],
                'attenuation_init': [0.95, 0.95, 0.95],
                'learnable_params': True
            },
            'data': {
                'output_window': output_window
            }
        }
    
    def update_config(self, new_config):
        """Aggiorna config dopo caricamento da checkpoint."""
        self.config = new_config
        
        if self.use_routing and self.routing_module is not None:
            print("🔄 Aggiornamento routing module...")
            old_state = self.routing_module.state_dict()
            self.routing_module = BettolleRoutingModule(new_config)
            
            try:
                self.routing_module.load_state_dict(old_state, strict=False)
                print("   [OK] Parametri routing preservati")
            except Exception as e:
                print(f"   [WARN] Alcuni parametri reinizializzati: {e}")
            
            self.model = HybridFloodModel(
                self.gnn_model, 
                self.routing_module, 
                self.hparams.output_window
            )
    
    def _get_base_model(self):
        """Helper per ottenere il modello base (bypassa routing wrapper)."""
        if self.use_routing and hasattr(self, 'model') and hasattr(self.model, 'gnn_model'):
            return self.model.gnn_model
        elif hasattr(self, 'gnn_model'):
            return self.gnn_model
        else:
            return self.model
            
    def _apply_input_noise(self, x):
        """Applica rumore gaussiano all'input per robustezza."""
        if self.hparams.input_noise_std > 0 and self.training:
            noise = torch.randn_like(x) * self.hparams.input_noise_std
            return x + noise
        return x

    def quantile_loss(self, preds, target, quantiles):
        """
        Calcola la Quantile Loss (Pinball Loss) con penalità asimmetrica.
        
        Args:
            preds: [batch, seq, nodes, output_dim, num_quantiles]
            target: [batch, seq, nodes, 1] o [batch, seq, nodes]
            quantiles: [num_quantiles]
        
        Returns:
            loss: scalar
        """
        assert not target.requires_grad, "Target non deve richiedere gradienti"
        
        # Normalizza target: [batch, seq, nodes]
        if target.dim() == 4 and target.size(-1) == 1:
            target = target.squeeze(-1)
        elif target.dim() == 4:
            target = target[..., 0]
        
        losses = []
        for i, q in enumerate(quantiles):
            # Estrai predizioni per il quantile corrente
            pred_q = preds[..., i]
            
            # Normalizza pred_q: [batch, seq, nodes]
            if pred_q.dim() == 4 and pred_q.size(-1) == 1:
                pred_q = pred_q.squeeze(-1)
            elif pred_q.dim() == 4:
                pred_q = pred_q[..., 0]
            
            # Errore: y - ŷ
            errors = target - pred_q
            
            # Quantile loss standard
            loss_q = torch.max((q - 1) * errors, q * errors)
            
            # NUOVO: Penalità asimmetrica per sottostime
            if self.hparams.use_asymmetric_loss:
                underestimation_mask = (pred_q < target) & (target > self.hparams.asymmetric_threshold)
                loss_q[underestimation_mask] *= self.hparams.underestimation_penalty
            
            losses.append(loss_q)
        
        # Somma su tutti i quantili e media su batch/seq/nodes
        total_loss = torch.stack(losses, dim=-1).sum(dim=-1)
        return total_loss.mean()
    
    def forward(self, x, target=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, timesteps, num_nodes, num_features)
            target: Target per teacher forcing (opzionale)
        
        Returns:
            predictions: (batch, output_window, num_nodes, output_dim, num_quantiles)
        """
        if hasattr(self, 'edge_index') and self.edge_index is not None:
            return self.model(
                x, 
                edge_index=self.edge_index, 
                edge_weight=self.edge_weight if hasattr(self, 'edge_weight') else None,
                target=target
            )
        else:
            return self.model(x, target=target)
        
    def training_step(self, batch, batch_idx):
        """
        Training step con weighted loss unificata e penalità asimmetrica.
        """
        x, y, ts = batch
        
        # Applica rumore all'input se abilitato
        x = self._apply_input_noise(x)
        
        # Forward pass (passa target per teacher forcing se TCN autoregressive)
        output = self(x, target=y)
        
        if isinstance(output, tuple):
            predictions, info = output
        else:
            predictions = output
            info = None
        
        # ====================================================================
        # LOSS UNIFICATA con ASIMMETRIA
        # ====================================================================
        
        if self.hparams.num_quantiles > 1:
            # Quantile Loss pesata con asimmetria
            loss = self._compute_weighted_quantile_loss(predictions, y, ts)
            pred_median = predictions[..., self.hparams.num_quantiles // 2].squeeze(-1)
        else:
            # MSE pesata con asimmetria
            pred_median = predictions.squeeze(-1)
            if pred_median.dim() == 4:
                pred_median = pred_median.squeeze(-1)
            loss = self._compute_weighted_mse_loss(pred_median, y, ts)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # MAE sempre calcolata sulla predizione mediana
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        self.log('train_mae', F.l1_loss(pred_median, y_true), on_step=True, on_epoch=True)
        
        return loss
    
    def _compute_weighted_quantile_loss(self, predictions, y, ts):
        """
        Quantile Loss con pesi per piene/regime/rising limb + PENALITÀ ASIMMETRICA.
        
        Args:
            predictions: [batch, seq, nodes, num_quantiles]
            y: [batch, seq, nodes, 1] o [batch, seq, nodes]
            ts: [batch] - timestamps
        
        Returns:
            loss: scalar
        """
        # Prepara target
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        
        # Loss base (quantile loss con asimmetria già applicata)
        base_loss = self.quantile_loss(predictions, y.unsqueeze(-1) if y.dim() == 3 else y, self.quantiles)
        
        # Estrai predizione mediana per calcolare weighted errors
        pred_median = predictions[..., self.hparams.num_quantiles // 2]
        if pred_median.dim() == 4:
            pred_median = pred_median.squeeze(-1)
        
        # Pesi per ogni sample
        sample_weights = torch.ones_like(y_true)
        
        # 1. Flood weighting
        flood_mask = (y_true > self.hparams.flood_threshold)
        if flood_mask.any():
            sample_weights[flood_mask] *= self.hparams.flood_weight
        
        # 2. Post-regime shift weighting
        regime_mask = (ts.view(-1, 1, 1) > self.regime_shift_ts).expand_as(y_true)
        if regime_mask.any():
            sample_weights[regime_mask] *= self.hparams.post_2024_weight
        
        # 3. Rising limb weighting
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
        
        # ============================================================
        # NUOVO: ASYMMETRIC PENALTY (Penalità extra per sottostime)
        # ============================================================
        if self.hparams.use_asymmetric_loss:
            # Identifica sottostime sopra la soglia di rilevanza
            underestimation_mask = (pred_median < y_true) & (y_true > self.hparams.asymmetric_threshold)
            
            if underestimation_mask.any():
                # Applica moltiplicatore extra ai pesi dove sottostimiamo
                sample_weights[underestimation_mask] *= self.hparams.underestimation_penalty
                
                # Logging statistiche (ogni 100 step)
                if self.global_step % 100 == 0:
                    under_ratio = underestimation_mask.sum().item() / underestimation_mask.numel()
                    self.log('train_underestimation_ratio', under_ratio, on_step=True)
        
        # Applica pesi alla loss (errore quadratico pesato con asimmetria)
        weighted_errors = sample_weights * (pred_median - y_true) ** 2
        weighted_loss = weighted_errors.mean()
        
        # Combina base loss (quantile) + weighted loss (MSE pesato)
        total_loss = base_loss + weighted_loss
        
        # Log separati
        self.log('train_quantile_loss', base_loss, on_step=False, on_epoch=True)
        self.log('train_weighted_mse', weighted_loss, on_step=False, on_epoch=True)
        
        # Log bias direzionale (sottostima vs sovrastima)
        if self.global_step % 100 == 0:
            bias = (pred_median - y_true).mean().item()
            self.log('train_prediction_bias', bias, on_step=True)
        
        return total_loss
    
    def _compute_weighted_mse_loss(self, pred_median, y, ts):
        """
        MSE Loss con pesi per piene/regime/rising limb + PENALITÀ ASIMMETRICA.
        
        Args:
            pred_median: [batch, seq, nodes]
            y: [batch, seq, nodes, 1] o [batch, seq, nodes]
            ts: [batch] - timestamps
        
        Returns:
            loss: scalar
        """
        y_true = y.squeeze(-1) if y.dim() == 4 else y
        
        # Calcola errore quadratico elemento per elemento
        squared_errors = (pred_median - y_true) ** 2
        
        # Inizializza mappa dei pesi
        weights_map = torch.ones_like(y_true)
        
        # 1. Flood weighting
        flood_mask = (y_true > self.hparams.flood_threshold)
        if flood_mask.any():
            weights_map[flood_mask] *= self.hparams.flood_weight
            
            # Log flood loss separatamente
            flood_errors = squared_errors[flood_mask]
            if flood_errors.numel() > 0:
                self.log('train_flood_loss', flood_errors.mean(), on_step=False, on_epoch=True)
        
        # 2. Post-regime shift weighting
        regime_mask = (ts.view(-1, 1, 1) > self.regime_shift_ts).expand_as(y_true)
        if regime_mask.any():
            weights_map[regime_mask] *= self.hparams.post_2024_weight
            
            # Log regime loss
            regime_errors = squared_errors[regime_mask]
            if regime_errors.numel() > 0:
                self.log('train_regime_loss', regime_errors.mean(), on_step=False, on_epoch=True)
        
        # 3. Rising limb weighting
        if self.hparams.use_rising_limb_weighting and y.size(1) > 1:
            if y.dim() == 4:
                y_diff = y[:, 1:, :, :] - y[:, :-1, :, :]
                y_diff_squeezed = y_diff.squeeze(-1)
            else:
                y_diff_squeezed = y[:, 1:, :] - y[:, :-1, :]
            
            rising_mask = torch.zeros_like(y_true, dtype=torch.bool)
            rising_mask[:, 1:, :] = (y_diff_squeezed > self.hparams.rising_limb_slope_threshold)
            
            if rising_mask.any():
                weights_map[rising_mask] *= self.hparams.rising_limb_weight
                
                # Log rising loss
                rising_errors = squared_errors[rising_mask]
                if rising_errors.numel() > 0:
                    self.log('train_rising_loss', rising_errors.mean(), on_step=False, on_epoch=True)
        
        # ============================================================
        # NUOVO: ASYMMETRIC PENALTY - IL TRUCCO PER ALZARE IL PICCO
        # ============================================================
        if self.hparams.use_asymmetric_loss:
            # Identifica dove stiamo SOTTOSTIMANDO sopra una soglia rilevante
            underestimation_mask = (pred_median < y_true) & (y_true > self.hparams.asymmetric_threshold)
            
            if underestimation_mask.any():
                # "Fear Factor" - Moltiplicatore di paura per sottostime
                weights_map[underestimation_mask] *= self.hparams.underestimation_penalty
                
                # Traccia statistiche
                self.underestimation_count += underestimation_mask.sum().item()
                self.overestimation_count += ((pred_median > y_true) & (y_true > self.hparams.asymmetric_threshold)).sum().item()
                
                # Log ratio ogni 100 step
                if self.global_step % 100 == 0:
                    total_relevant = self.underestimation_count + self.overestimation_count
                    if total_relevant > 0:
                        under_ratio = self.underestimation_count / total_relevant
                        self.log('train_underestimation_ratio', under_ratio, on_step=True)
                        
                        # Reset contatori
                        self.underestimation_count = 0
                        self.overestimation_count = 0
                
                # Log penalità applicata
                under_errors = squared_errors[underestimation_mask]
                if under_errors.numel() > 0:
                    self.log('train_underestimation_loss', under_errors.mean(), on_step=False, on_epoch=True)
        
        # Calcola loss finale pesata
        loss = (squared_errors * weights_map).mean()
        
        # Log bias complessivo (negativo = sottostima, positivo = sovrastima)
        if self.global_step % 100 == 0:
            prediction_bias = (pred_median - y_true).mean().item()
            self.log('train_prediction_bias', prediction_bias, on_step=True)
            
            # Log bias solo per livelli alti (>1m)
            high_level_mask = y_true > 1.0
            if high_level_mask.any():
                high_level_bias = (pred_median[high_level_mask] - y_true[high_level_mask]).mean().item()
                self.log('train_high_level_bias', high_level_bias, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step con metriche unificate e tracking bias."""
        x, y, ts = batch
        
        # Forward pass (no teacher forcing in validation)
        output = self(x, target=None)
        
        if isinstance(output, tuple):
            predictions, info = output
        else:
            predictions = output
            info = None
        
        # Estrai predizione mediana
        if self.hparams.num_quantiles > 1:
            pred_median = predictions[..., self.hparams.num_quantiles // 2].squeeze(-1)
        else:
            pred_median = predictions.squeeze(-1)
            if pred_median.dim() == 4:
                pred_median = pred_median.squeeze(-1)
        
        y_true = y.squeeze(-1) if y.dim() == 4 else y

        # Loss principale (MSE)
        loss = F.mse_loss(pred_median, y_true)
        
        # Log Quantile Loss in Validation (se applicabile)
        if self.hparams.num_quantiles > 1:
            val_q_loss = self.quantile_loss(
                predictions, 
                y.unsqueeze(-1) if y.dim() == 3 else y, 
                self.quantiles
            )
            self.log('val_quantile_loss', val_q_loss, on_epoch=True)
        
        # MAE
        mae = F.l1_loss(pred_median, y_true)
        
        # NUOVO: Calcola bias direzionale in validation
        bias = (pred_median - y_true).mean().item()
        high_level_mask = y_true > 1.0
        if high_level_mask.any():
            high_level_bias = (pred_median[high_level_mask] - y_true[high_level_mask]).mean().item()
            self.log('val_high_level_bias', high_level_bias, on_epoch=True)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_prediction_bias', bias, on_epoch=True, prog_bar=True)
        
        # Accumula per metriche event-based
        self.val_predictions.append(pred_median.detach().cpu())
        self.val_targets.append(y_true.detach().cpu())
        self.val_timestamps.append(ts.detach().cpu())
        
        return loss
        
    def on_train_epoch_start(self):
        """Chiamato all'inizio di ogni epoca. Gestisce curriculum learning e reporting."""
        
        # ====================================================================
        # CURRICULUM LEARNING (TCN/SimpleTCN Autoregressive)
        # ====================================================================
        
        if hasattr(self, 'is_autoregressive') and self.is_autoregressive:
            base_model = self._get_base_model()
            
            if hasattr(base_model, 'set_teacher_forcing_ratio'):
                # Calcola il teacher forcing ratio corrente
                progress = min(self.current_epoch / self.curriculum_decay_epochs, 1.0)
                current_tf = self.curriculum_initial_tf - (self.curriculum_initial_tf - self.curriculum_final_tf) * progress
                
                # Aggiorna il modello
                base_model.set_teacher_forcing_ratio(current_tf)
                
                # Log su TensorBoard/WandB
                self.log('curriculum/teacher_forcing_ratio', current_tf, prog_bar=True)
                
                # Print dettagliato
                if self.current_epoch == 0:
                    print("\n" + "="*80)
                    print("🎓 CURRICULUM LEARNING ATTIVATO")
                    print("="*80)
                    print(f"   Modello: {self.hparams.temporal_encoder.upper()} (autoregressive)")
                    print(f"   Strategia: Linear decay in {self.curriculum_decay_epochs} epoche")
                    print(f"   Teacher Forcing: {self.curriculum_initial_tf:.0%} → {self.curriculum_final_tf:.0%}")
                    print(f"   Epoch {self.current_epoch}: TF = {current_tf:.2%}")
                    print("="*80)
                elif self.current_epoch < self.curriculum_decay_epochs:
                    print(f"\n📚 [Epoch {self.current_epoch}] Teacher Forcing = {current_tf:.2%}")
                elif self.current_epoch == self.curriculum_decay_epochs:
                    print(f"\n✅ [Epoch {self.current_epoch}] Curriculum completato! TF fisso a {current_tf:.2%}")
        
        # ====================================================================
        # ASYMMETRIC LOSS REPORTING
        # ====================================================================
        
        if self.hparams.use_asymmetric_loss and self.current_epoch % 5 == 0:
            print(f"\n⚠️  [Epoch {self.current_epoch}] Asymmetric Loss Active:")
            print(f"   - Sottostima penalizzata {self.hparams.underestimation_penalty}x")
            print(f"   - Obiettivo: Spingere previsioni verso l'alto nei picchi")
        
        # ====================================================================
        # ROUTING INFO
        # ====================================================================
        
        if not self.use_routing or self.config is None:
            print(f"\n[Epoch {self.current_epoch}] Training mode: STANDARD (GNN only)")
            return

        routing_enabled = self.config.get('routing', {}).get('enabled', True)
        routing_mode = self.config.get('routing', {}).get('mode', 'hybrid')
        warmup_epochs = self.config.get('routing', {}).get('warmup_gnn_epochs', 5)
        
        if not routing_enabled or routing_mode == 'gnn_only':
            return
        
        if self.current_epoch == warmup_epochs:
            print("\n" + "="*80)
            print("🔄 ATTIVAZIONE ROUTING IBRIDO!")
            print("="*80)
            print(f"   Epoca corrente: {self.current_epoch}")
            print(f"   Warmup completato: {warmup_epochs} epoche")
            print(f"   Routing mode: {routing_mode.upper()} ATTIVO")
            print("="*80 + "\n")

    def on_validation_epoch_end(self):
        """Calcola metriche event-based e statistiche di bias."""
        
        if len(self.val_predictions) == 0:
            return
        
        # Concatena batch
        all_preds = torch.cat(self.val_predictions, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)
        
        # Focus Bettolelle (nodo 3)
        preds_bett = all_preds[:, :, 3]
        targets_bett = all_targets[:, :, 3]
        
        # Flatten
        preds_flat = preds_bett.flatten()
        targets_flat = targets_bett.flatten()
        
        # === METRICHE EVENT-BASED ===
        
        # 1. Peak Level Error (con segno per capire direzione)
        peak_true = targets_flat.max()
        peak_pred = preds_flat.max()
        peak_error = abs(peak_pred - peak_true).item()
        peak_error_signed = (peak_pred - peak_true).item()  # NUOVO: errore con segno
        
        self.log('val_event_peak_error', peak_error, prog_bar=True)
        self.log('val_event_peak_error_signed', peak_error_signed)  # Negativo = sottostima
        self.log('val_event_peak_true', peak_true.item())
        self.log('val_event_peak_pred', peak_pred.item())
        
        # 2. Time to Peak Error
        time_true = targets_flat.argmax().item()
        time_pred = preds_flat.argmax().item()
        time_error = abs(time_pred - time_true)
        time_error_hours = time_error * 0.5
        
        self.log('val_event_time_to_peak_error_steps', time_error)
        self.log('val_event_time_to_peak_error_hours', time_error_hours)
        
        # 3. Rising Limb MAE
        if time_true > 0:
            rising_true = targets_flat[:time_true]
            rising_pred = preds_flat[:time_true]
            rising_mae = F.l1_loss(rising_pred, rising_true).item()
            self.log('val_event_rising_mae', rising_mae)
            
            # NUOVO: Rising limb bias (sottostima in salita?)
            rising_bias = (rising_pred - rising_true).mean().item()
            self.log('val_event_rising_bias', rising_bias)
        
        # 4. POD/FAR/CSI
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
        
        # 5. Event MAE/RMSE
        event_mae = F.l1_loss(preds_flat, targets_flat).item()
        event_rmse = torch.sqrt(F.mse_loss(preds_flat, targets_flat)).item()
        
        self.log('val_event_mae', event_mae)
        self.log('val_event_rmse', event_rmse)
        
        # 6. NUOVO: Statistiche di bias per livelli alti
        high_mask = targets_flat > self.hparams.flood_threshold
        if high_mask.any():
            high_bias = (preds_flat[high_mask] - targets_flat[high_mask]).mean().item()
            self.log('val_event_flood_bias', high_bias, prog_bar=True)
        
        # Reset
        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_timestamps.clear()
    
    def configure_optimizers(self):
        """Configura optimizer e scheduler."""
        
        print("\n[CONFIG] OPTIMIZER CONFIGURATION:")
        print(f"   - Type: AdamW")
        print(f"   - Learning rate: {self.hparams.learning_rate}")
        print(f"   - Weight decay: {self.hparams.weight_decay}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=1e-7
        )
        
        print(f"   [OK] Optimizer initialized\n")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_event_peak_error",
                "interval": "epoch",
                "frequency": 1
            }
        }
