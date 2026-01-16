import os
import json
from datetime import datetime, timedelta
import pytz
import gspread
from google.oauth2.service_account import Credentials
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import traceback
import io
import csv
import random
import math

# ============================================
# CONFIGURAZIONE RIPRODUCIBILITÀ
# ============================================
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# --- HORIZON FEATURE CATEGORIES ---
HORIZON_FEATURE_CATEGORIES = {
    "pioggia_montana": ["Arcevia", "1295"],
    "pioggia_intermedia": ["Corinaldo", "Barbara", "2858", "2964"],
    "pioggia_locale": ["Bettolelle", "2637"],
    "idrometri_monte": ["Serra dei Conti", "1008"],
    "idrometri_intermedi": ["Pianello", "Nevola", "3072", "1283", "Passo Ripe"],
    "suolo": ["soil_moisture"],
    "stagionalita": ["Seasonality"]
}
# --- END HORIZON FEATURE CATEGORIES ---

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc(x)) * self.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x
        hidden = self.fc1(x)
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        output = self.glu(hidden)
        return self.layer_norm(output + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, 
                 context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, context_dim, dropout)
            for _ in range(num_inputs)
        ])
        self.selection_grn = GatedResidualNetwork(
            hidden_dim * num_inputs, hidden_dim, num_inputs, context_dim, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: list, context: torch.Tensor = None) -> tuple:
        processed = []
        for i, (var_input, grn) in enumerate(zip(inputs, self.variable_grns)):
            if context is not None:
                ctx_expanded = context.unsqueeze(1).expand(-1, var_input.size(1), -1)
                processed.append(grn(var_input, ctx_expanded))
            else:
                processed.append(grn(var_input))
        stacked = torch.cat(processed, dim=-1)
        if context is not None:
            ctx_expanded = context.unsqueeze(1).expand(-1, stacked.size(1), -1)
            selection_weights = self.selection_grn(stacked, ctx_expanded)
        else:
            selection_weights = self.selection_grn(stacked)
        weights = self.softmax(selection_weights)
        processed_stacked = torch.stack(processed, dim=-2)
        output = (processed_stacked * weights.unsqueeze(-1)).sum(dim=-2)
        return output, weights

class HorizonFeatureGating(nn.Module):
    def __init__(self, num_features: int, feature_names: list, output_window: int, 
                 horizon_weights: dict = None, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.feature_names = feature_names
        self.output_window = output_window
        self.hidden_dim = hidden_dim
        self.feature_category_map = self._build_category_map(feature_names)
        
        if horizon_weights is None:
            self.use_gating = False
        else:
            self.use_gating = True
            # Pre-compute weight tensors for each step group
            self.register_buffer('weight_group_0', self._build_weight_tensor(horizon_weights.get('group_0', {})))
            self.register_buffer('weight_group_1', self._build_weight_tensor(horizon_weights.get('group_1', {})))
            self.register_buffer('weight_group_2', self._build_weight_tensor(horizon_weights.get('group_2', {})))
    
    def _build_category_map(self, feature_names: list) -> dict:
        """Map each feature index to its category."""
        category_map = {}
        for idx, fname in enumerate(feature_names):
            fname_lower = fname.lower()
            found_category = "other"  # Default category
            for category, keywords in HORIZON_FEATURE_CATEGORIES.items():
                for keyword in keywords:
                    if keyword.lower() in fname_lower:
                        found_category = category
                        break
                if found_category != "other":
                    break
            category_map[idx] = found_category
        return category_map
    
    def _build_weight_tensor(self, category_weights: dict) -> torch.Tensor:
        """Build a weight tensor for all features based on category weights."""
        weights = torch.ones(self.num_features)
        for idx, category in self.feature_category_map.items():
            if category in category_weights:
                weights[idx] = category_weights[category]
        return weights
    
    def get_weights_for_step(self, step_idx: int) -> torch.Tensor:
        """Get the weight tensor for a specific prediction step."""
        if not self.use_gating:
            return torch.ones(self.num_features, device=self.weight_group_0.device if hasattr(self, 'weight_group_0') else 'cpu')
        
        # Determine which group this step belongs to
        steps_per_group = max(1, self.output_window // 3)
        group_idx = min(step_idx // steps_per_group, 2)
        
        if group_idx == 0:
            return self.weight_group_0
        elif group_idx == 1:
            return self.weight_group_1
        else:
            return self.weight_group_2
    
    def forward(self, variable_weights: torch.Tensor, step_idx: int = None) -> torch.Tensor:
        """
        Apply horizon-aware gating to variable weights.
        
        Args:
            variable_weights: (batch, seq_len, num_features) - weights from VSN
            step_idx: If provided, applies weights for specific step. 
                      If None, applies average weights.
        
        Returns:
            Gated variable weights (batch, seq_len, num_features)
        """
        if not self.use_gating:
            return variable_weights
        
        if step_idx is not None:
            gate_weights = self.get_weights_for_step(step_idx)
        else:
            # Average across all groups
            gate_weights = (self.weight_group_0 + self.weight_group_1 + self.weight_group_2) / 3.0
        
        # Apply gating: multiply then renormalize to sum to 1
        # gate_weights: (num_features,) -> expand to match variable_weights
        gated = variable_weights * gate_weights.unsqueeze(0).unsqueeze(0)
        
        # Renormalize so weights sum to 1 along feature dimension
        gated_sum = gated.sum(dim=-1, keepdim=True) + 1e-8
        gated_normalized = gated / gated_sum
        
        return gated_normalized


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention per TFT.
    A differenza della MHA standard, questa versione permette di estrarre
    pesi di attenzione interpretabili su base temporale.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Proiezioni Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Proiezione output
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple:
        batch_size = query.size(0)
        
        # (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # (batch, n_heads, seq_len, seq_len) x (batch, n_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)
        
        # (batch, seq_len, n_heads * d_k) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 num_past_inputs: int,          # Numero di feature passate (livelli, pioggia, etc.)
                 num_future_inputs: int,        # Numero di feature future note (stagionalità)
                 num_static_inputs: int = 0,    # Numero di feature statiche (opzionale)
                 hidden_dim: int = 64,          # Dimensione hidden state
                 num_heads: int = 4,            # Numero di attention heads
                 num_encoder_layers: int = 1,   # Numero di layer LSTM encoder
                 dropout: float = 0.1,
                 output_dim: int = 1,           # Numero di target (es. 1 = solo Bettolelle)
                 output_window: int = 6,        # Passi futuri da predire
                 num_quantiles: int = 1,        # Quantili per uncertainty (1 = point forecast)
                 feature_names: list = None,    # NEW: List of past feature names for horizon gating
                 horizon_weights: dict = None): # NEW: Horizon-aware feature weights config
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_window = output_window
        self.num_quantiles = num_quantiles
        self.num_past_inputs = num_past_inputs
        self.num_future_inputs = num_future_inputs
        self.num_static_inputs = num_static_inputs
        
        # --- 1. Embedding delle variabili ---
        # Ogni feature viene embedded in hidden_dim
        self.past_var_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_past_inputs)
        ])
        self.future_var_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_future_inputs)
        ])
        
        if num_static_inputs > 0:
            self.static_var_embeddings = nn.ModuleList([
                nn.Linear(1, hidden_dim) for _ in range(num_static_inputs)
            ])
            # Static context encoder
            self.static_context_grn = GatedResidualNetwork(
                hidden_dim * num_static_inputs, hidden_dim, hidden_dim, dropout=dropout
            )
            context_dim = hidden_dim
        else:
            self.static_var_embeddings = None
            self.static_context_grn = None
            context_dim = None
        
        # --- 2. Variable Selection Networks ---
        self.past_vsn = VariableSelectionNetwork(
            hidden_dim, num_past_inputs, hidden_dim, context_dim, dropout
        )
        if num_future_inputs > 0:
            self.future_vsn = VariableSelectionNetwork(
                hidden_dim, num_future_inputs, hidden_dim, context_dim, dropout
            )
        else:
            self.future_vsn = None
        
        # --- 2b. Horizon Feature Gating (NEW) ---
        self.feature_names = feature_names if feature_names else []
        self.horizon_weights = horizon_weights
        if feature_names and horizon_weights:
            self.horizon_gating = HorizonFeatureGating(
                num_features=num_past_inputs,
                feature_names=feature_names,
                output_window=output_window,
                horizon_weights=horizon_weights,
                hidden_dim=hidden_dim
            )
        else:
            self.horizon_gating = None
        
        # --- 3. Encoder LSTM (per sequenza passata) ---
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_encoder_layers,
            batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0
        )
        
        # --- 4. Decoder LSTM (per sequenza futura) ---
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_encoder_layers,
            batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0
        )
        
        # --- 5. Gated Layer per combinare encoder/decoder ---
        self.post_lstm_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        # --- 6. Temporal Self-Attention ---
        self.temporal_attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_dim)
        
        # --- 7. Position-wise Feed-Forward ---
        self.positionwise_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        # --- 8. Output layer (multi-horizon direct) ---
        self.output_layer = nn.Linear(hidden_dim, output_dim * num_quantiles)

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, 
                x_static: torch.Tensor = None) -> tuple:
        """
        Args:
            x_past: (batch, past_seq_len, num_past_inputs) - Serie storica
            x_future: (batch, future_seq_len, num_future_inputs) - Covariate future note
            x_static: (batch, num_static_inputs) - Feature statiche (opzionale)
        
        Returns:
            predictions: (batch, output_window, output_dim, num_quantiles) o (batch, output_window, output_dim)
            interpretability: Dict con pesi attenzione e importanza variabili
        """
        batch_size = x_past.size(0)
        past_seq_len = x_past.size(1)
        future_seq_len = x_future.size(1)
        
        # --- 1. Embed variabili ---
        # Past
        past_embedded = []
        for i in range(self.num_past_inputs):
            var_input = x_past[:, :, i:i+1]  # (batch, seq, 1)
            past_embedded.append(self.past_var_embeddings[i](var_input))
        
        # Future
        future_embedded = []
        for i in range(self.num_future_inputs):
            var_input = x_future[:, :, i:i+1]
            future_embedded.append(self.future_var_embeddings[i](var_input))
        
        # Static context
        static_context = None
        if x_static is not None and self.num_static_inputs > 0:
            static_embedded = []
            for i in range(self.num_static_inputs):
                var_input = x_static[:, i:i+1]
                static_embedded.append(self.static_var_embeddings[i](var_input))
            static_concat = torch.cat(static_embedded, dim=-1)
            static_context = self.static_context_grn(static_concat)
        
        # --- 2. Variable Selection ---
        past_selected, past_var_weights = self.past_vsn(past_embedded, static_context)
        
        # --- 2b. Apply Horizon Feature Gating (NEW) ---
        # Store original weights for interpretability, apply gated weights to processing
        past_var_weights_gated = past_var_weights
        if self.horizon_gating is not None:
            # Apply gating to variable weights (modifies attention distribution)
            past_var_weights_gated = self.horizon_gating(past_var_weights)
            # RE-CALCULATE past_selected with gated weights
            past_selected = (torch.stack(past_embedded, dim=-2) * past_var_weights_gated.unsqueeze(-1)).sum(dim=-2)
        
        # Gestione caso senza feature future
        if self.num_future_inputs > 0 and self.future_vsn is not None and len(future_embedded) > 0:
            future_selected, future_var_weights = self.future_vsn(future_embedded, static_context)
        else:
            # Se non ci sono feature future, usa l'ultimo hidden state dell'encoder ripetuto
            # Creiamo un placeholder di zeri per il decoder
            batch_size = x_past.size(0)
            future_selected = torch.zeros(batch_size, self.output_window, self.hidden_dim, device=x_past.device)
            future_var_weights = None
        
        # --- 3. Encoder LSTM ---
        encoder_output, (h_n, c_n) = self.encoder_lstm(past_selected)
        
        # --- 4. Decoder LSTM ---
        decoder_output, _ = self.decoder_lstm(future_selected, (h_n, c_n))
        
        # --- 5. Concatena encoder e decoder outputs ---
        # (batch, past_seq + future_seq, hidden)
        combined = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Gated skip connection
        combined = self.post_lstm_gate(combined)
        
        # --- 6. Temporal Self-Attention ---
        # Maschera causale: il futuro può vedere solo il passato e se stesso
        total_len = past_seq_len + future_seq_len
        
        attention_output, temporal_attention_weights = self.temporal_attention(
            combined, combined, combined
        )
        
        # Add & Norm
        attention_output = self.attention_layer_norm(combined + self.post_attention_grn(attention_output))
        
        # --- 7. Position-wise Feed-Forward ---
        output_temporal = self.positionwise_grn(attention_output)
        
        # --- 8. Estrai solo gli output futuri (ultimi future_seq_len steps) ---
        future_output = output_temporal[:, past_seq_len:, :]  # (batch, future_seq, hidden)
        
        # Usa solo i primi output_window steps come predizioni
        predictions_input = future_output[:, :self.output_window, :]
        
        # --- 9. Output projection ---
        predictions = self.output_layer(predictions_input)  # (batch, output_window, output_dim * num_quantiles)
        
        # Reshape per quantili
        if self.num_quantiles > 1:
            predictions = predictions.view(batch_size, self.output_window, self.output_dim, self.num_quantiles)
        
        # Interpretability dict
        interpretability = {
            'past_variable_weights': past_var_weights,      # (batch, past_seq, num_past)
            'future_variable_weights': future_var_weights,  # (batch, future_seq, num_future)
            'temporal_attention': temporal_attention_weights  # (batch, heads, total_seq, total_seq)
        }
        
        return predictions, interpretability


# ============================================
# COSTANTI
# ============================================
MODELS_DIR = "models"
MODEL_BASE_NAME = os.environ.get("TFT_MODEL_NAME", "modello_temporal_20260115_1726")  # Default: 3h Horizon-Gated

GSHEET_ID = os.environ.get("GSHEET_ID", "1pQI6cKrrT-gcVAfl-9ZhUx5b3J-edZRRj6nzDcCBRcA")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "DATI METEO CON FEATURE"
GSHEET_FORECAST_DATA_SHEET_NAME = "Previsioni Cumulate Feature ICON"
GSHEET_PREDICTIONS_SHEET_NAME = os.environ.get("GSHEET_PREDICTIONS_SHEET", "Previsioni Idro-Bettolelle 3h")
GSHEET_FEATURE_IMPORTANCE_SHEET_NAME = os.environ.get("GSHEET_IMPORTANCE_SHEET", "TFT Feature Importance 3h")

GSHEET_DATE_COL_INPUT = 'Data e Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
GSHEET_FORECAST_DATE_COL = 'Data e Ora'
GSHEET_FORECAST_DATE_FORMAT = '%d/%m/%Y %H:%M'
italy_tz = pytz.timezone('Europe/Rome')


def log_environment_info():
    print("=== INFORMAZIONI AMBIENTE ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except:
        pass
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 30)


def load_model_and_scalers(model_base_name, models_dir):
    print(f"\n=== CARICAMENTO MODELLO TFT ===")
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_past_features_path = os.path.join(models_dir, f"{model_base_name}_past_features.joblib")
    scaler_forecast_features_path = os.path.join(models_dir, f"{model_base_name}_forecast_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")
    
    for p in [config_path, model_path, scaler_past_features_path, scaler_targets_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File non trovato -> {p}")
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TFT specific config
    num_past = len(config["all_past_feature_columns"])
    num_future = len(config["forecast_input_columns"])
    num_targets = len(config["target_columns"])
    
    hidden_dim = config["hidden_dim"]
    num_heads = config.get("num_heads", 4)
    num_layers = config["num_encoder_layers"]
    dropout = config["dropout"]
    out_win = config["output_window_steps"]
    
    quantiles = config.get("quantiles", [0.5])
    num_quantiles = len(quantiles)
    
    print(f"Configurazione modello TFT:")
    print(f"  - Past features: {num_past}")
    print(f"  - Future features: {num_future}")
    print(f"  - Output: {num_targets} target(s)")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - Encoder layers: {num_layers}")
    print(f"  - Quantili: {num_quantiles}")
    print(f"  - Output window: {out_win} steps")
    
    model = TemporalFusionTransformer(
        num_past_inputs=num_past,
        num_future_inputs=num_future,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        dropout=dropout,
        output_dim=num_targets,
        output_window=out_win,
        num_quantiles=num_quantiles,
        feature_names=config.get("all_past_feature_columns", []),  # NEW: backward compatible
        horizon_weights=config.get("horizon_weights", None)        # NEW: backward compatible
    ).to(device)
    
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler_past_features = joblib.load(scaler_past_features_path)
        try:
            scaler_forecast_features = joblib.load(scaler_forecast_features_path)
        except:
            print("Warning: Scaler forecast non caricato (potrebbe non essere necessario).")
            scaler_forecast_features = None
        scaler_targets = joblib.load(scaler_targets_path)
    
    print("Modello TFT e scaler caricati")
        
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device


def fetch_and_prepare_data(gc, sheet_id, config):
    print(f"\n=== CARICAMENTO E PREPARAZIONE DATI ===")
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    target_columns = config.get("target_columns", [])
    forecast_feature_columns = config["forecast_input_columns"]
    
    # Filter columns that are also targets (not to be searched in forecast sheet)
    actual_forecast_cols_to_fetch = [c for c in forecast_feature_columns if c not in target_columns]
    
    sh = gc.open_by_key(sheet_id)
    
    def values_to_csv_string(data):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(data)
        return output.getvalue()

    print(f"Caricamento dati storici da '{GSHEET_HISTORICAL_DATA_SHEET_NAME}'...")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    historical_values = historical_ws.get_all_values()
    df_historical_raw = pd.read_csv(io.StringIO(values_to_csv_string(historical_values)), decimal=',')
    
    df_forecast_raw = None
    if len(actual_forecast_cols_to_fetch) > 0:
        print(f"Caricamento dati previsionali da '{GSHEET_FORECAST_DATA_SHEET_NAME}'...")
        try:
            forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
            forecast_values = forecast_ws.get_all_values()
            df_forecast_raw = pd.read_csv(io.StringIO(values_to_csv_string(forecast_values)), decimal=',')
        except gspread.exceptions.WorksheetNotFound:
            print(f"Warning: Foglio '{GSHEET_FORECAST_DATA_SHEET_NAME}' non trovato. Si prosegue senza previsioni esterne.")

    # Rename columns helper
    def build_rename_map(columns):
        rename_map = {}
        for col in columns:
            if 'Giornaliera ' in col:
                if '_cumulata_30min' in col:
                    new_name = col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
                    rename_map[col] = new_name
                else:
                    new_name = col.replace('Giornaliera ', '')
                    if not col.strip().endswith(')'):
                        rename_map[col] = new_name
        return rename_map

    hist_rename_map = build_rename_map(df_historical_raw.columns)
    if hist_rename_map:
        df_historical_raw.rename(columns=hist_rename_map, inplace=True)
    
    if df_forecast_raw is not None:
        fcst_rename_map = build_rename_map(df_forecast_raw.columns)
        if fcst_rename_map:
            df_forecast_raw.rename(columns=fcst_rename_map, inplace=True)

    df_historical_raw = df_historical_raw.loc[:, ~df_historical_raw.columns.duplicated(keep='last')]
    if df_forecast_raw is not None:
        df_forecast_raw = df_forecast_raw.loc[:, ~df_forecast_raw.columns.duplicated(keep='last')]

    # Parse historical dates
    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(
        df_historical_raw[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce'
    )
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    print(f"Ultimo timestamp storico: {latest_valid_timestamp}")
    
    # Prepare historical data
    for col in past_feature_columns:
        if col not in df_historical.columns:
            raise ValueError(f"Colonna storica '{col}' non trovata.")
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values
    print(f"Shape dati storici: {input_data_historical.shape}")
    
    # Prepare forecast data
    input_data_forecast = None
    
    if len(actual_forecast_cols_to_fetch) > 0 and df_forecast_raw is not None:
        df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(
            df_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce'
        )
        future_forecasts = df_forecast_raw[df_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
        
        if len(future_forecasts) < output_window_steps:
            raise ValueError(f"Previsioni insufficienti: {len(future_forecasts)} < {output_window_steps}")
        
        future_data = future_forecasts.head(output_window_steps).copy()
        
        for col in actual_forecast_cols_to_fetch:
            if col not in future_data.columns:
                raise ValueError(f"Colonna previsionale '{col}' non trovata.")
            future_data[col] = pd.to_numeric(future_data[col], errors='coerce')

        # Handle mixed case (external features + target columns as placeholders)
        if len(actual_forecast_cols_to_fetch) < len(forecast_feature_columns):
             df_complete_forecast = pd.DataFrame(index=future_data.index, columns=forecast_feature_columns)
             for col in actual_forecast_cols_to_fetch:
                 df_complete_forecast[col] = future_data[col]
             df_complete_forecast = df_complete_forecast.fillna(0)
             input_data_forecast = df_complete_forecast.values
        else:
             input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
             
        print(f"Shape dati forecast: {input_data_forecast.shape}")
        
    elif len(actual_forecast_cols_to_fetch) == 0:
        # Create dummy forecast with only seasonal features (Dummy, Sin, Cos)
        print("Nessuna colonna forecast esterna richiesta. Creazione input dummy per variabili temporali.")
        input_data_forecast = np.zeros((output_window_steps, len(forecast_feature_columns)))
    else:
        print("Dati forecast non disponibili.")
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp

    
def make_prediction(model, scalers, config, data_inputs, device):
    print(f"\n=== GENERAZIONE PREVISIONI TFT ===")
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    
    print(f"DEBUG: Historical Data Shape: {historical_data_np.shape}")
    
    # Normalize past data
    historical_normalized = scaler_past_features.transform(historical_data_np)
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    
    print(f"DEBUG: Past Tensor Shape: {historical_tensor.shape}")
    print(f"DEBUG: Past Tensor Mean: {historical_tensor.mean().item():.6f}")
    print(f"DEBUG: Past Tensor Std: {historical_tensor.std().item():.6f}")
    
    # Normalize forecast data
    if forecast_data_np is not None and scaler_forecast_features is not None:
        forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
        forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    else:
        # Create dummy forecast tensor
        output_steps = config['output_window_steps']
        num_forecast_features = len(config["forecast_input_columns"])
        forecast_tensor = torch.zeros(1, output_steps, num_forecast_features).to(device)

    print(f"DEBUG: Forecast Tensor Shape: {forecast_tensor.shape}")

    # TFT Inference (simpler than Seq2Seq - no autoregressive loop)
    model.eval()
    set_seed(SEED)
    
    with torch.no_grad():
        predictions_normalized, attention_info = model(historical_tensor, forecast_tensor)
    
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    num_targets = len(config["target_columns"])
    
    # De-normalize predictions
    quantiles = config.get("quantiles", [0.5])
    num_quantiles = len(quantiles)
    
    if num_quantiles > 1:
        # Shape: (output_steps, num_targets * num_quantiles)
        # Reshape to (output_steps, num_quantiles, num_targets) for inverse transform
        seq_len = predictions_np.shape[0]
        preds_reshaped = predictions_np.reshape(seq_len, num_targets, num_quantiles)
        
        # Inverse transform each quantile
        preds_unscaled = np.zeros_like(preds_reshaped)
        for q in range(num_quantiles):
            preds_unscaled[:, :, q] = scaler_targets.inverse_transform(preds_reshaped[:, :, q])
        
        predictions_scaled_back = preds_unscaled  # Shape: (output_steps, num_targets, num_quantiles)
    else:
        predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)

    # Log predictions
    print(f"\nPrevisioni TFT generate (shape: {predictions_scaled_back.shape}):")
    if predictions_scaled_back.ndim == 3:
        median_idx = predictions_scaled_back.shape[2] // 2
        for i in range(predictions_scaled_back.shape[0]):
            val = predictions_scaled_back[i, 0, median_idx]
            print(f"  Step {i+1}: {val:.3f} m")
    else:
        for i in range(predictions_scaled_back.shape[0]):
            print(f"  Step {i+1}: {predictions_scaled_back[i][0]:.3f} m")
    
    return predictions_scaled_back, attention_info


def append_feature_importance_to_gsheet(gc, sheet_id_str, importance_sheet_name, attention_info, config):
    """Export feature importance (variable selection weights) to Google Sheets for operational transparency."""
    print(f"\n=== SALVATAGGIO FEATURE IMPORTANCE ===")
    
    if attention_info is None or "past_variable_weights" not in attention_info:
        print("Warning: No variable weights available for export.")
        return
    
    past_var_weights = attention_info["past_variable_weights"]  # Shape: [1, input_window, num_features]
    
    # Aggregate weights over the input window (mean)
    if isinstance(past_var_weights, torch.Tensor):
        weights_np = past_var_weights.detach().cpu().numpy()
    else:
        weights_np = past_var_weights
    
    # Mean over batch and time dimensions
    avg_weights = weights_np.squeeze(0).mean(axis=0)  # Shape: [num_features]
    
    feature_names = config.get("all_past_feature_columns", [])
    
    # Map features to categories
    def get_category(feature_name):
        for category, keywords in HORIZON_FEATURE_CATEGORIES.items():
            for kw in keywords:
                if kw.lower() in feature_name.lower():
                    return category
        return "altro"
    
    # Build importance data
    importance_data = []
    for i, (name, weight) in enumerate(zip(feature_names, avg_weights)):
        importance_data.append({
            "Feature": name,
            "Peso": float(weight),
        })
    
    # Sort by weight descending
    importance_data.sort(key=lambda x: x["Peso"], reverse=True)
    
    # Add rank
    for rank, item in enumerate(importance_data, 1):
        item["Rank"] = rank
    
    # Write to GSheet
    sh = gc.open_by_key(sheet_id_str)
    
    try:
        worksheet = sh.worksheet(importance_sheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=importance_sheet_name, rows=50, cols=10)
    
    # Header
    header = ["Rank", "Feature", "Peso (%)"]
    worksheet.append_row(header, value_input_option='USER_ENTERED')
    
    # Data rows
    rows_to_append = []
    for item in importance_data:
        row = [
            item["Rank"],
            item["Feature"],
            f"{item['Peso']*100:.2f}".replace('.', ',')
        ]
        rows_to_append.append(row)
    
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Salvate {len(rows_to_append)} righe di feature importance")



def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    print(f"\n=== SALVATAGGIO PREVISIONI ===")
    sh = gc.open_by_key(sheet_id_str)
    
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows=100, cols=20)
    
    quantiles = config.get("quantiles")
    target_cols = config["target_columns"]
    header = ["Timestamp Previsione"]

    if quantiles and isinstance(quantiles, list) and len(quantiles) > 1:
        for col in target_cols:
            for q in quantiles:
                header.append(f"Previsto: {col} (Q{int(q * 100)})")
    else:
        for col in target_cols:
            header.append(f"Previsto: {col}")
    
    worksheet.append_row(header, value_input_option='USER_ENTERED')
    
    rows_to_append = []
    prediction_start_time = config["_prediction_start_time"]
    
    for i in range(predictions_np.shape[0]):
        timestamp = prediction_start_time + timedelta(minutes=30 * (i + 1))
        row = [timestamp.strftime('%d/%m/%Y %H:%M')]
        
        step_data = predictions_np[i]
        if step_data.ndim > 1:
            step_data = step_data.flatten()
            
        row.extend([f"{val:.3f}".replace('.', ',') for val in step_data])
        rows_to_append.append(row)
    
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Salvate {len(rows_to_append)} righe di previsione")


def main():
    print(f"\n{'='*60}")
    print(f"AVVIO SCRIPT TFT - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modello: {MODEL_BASE_NAME}")
    print(f"{'='*60}")
    
    try:
        log_environment_info()
        
        credentials_file_path = "credentials.json"
        if not os.path.exists(credentials_file_path):
            raise FileNotFoundError("File credenziali non trovato.")

        credentials = Credentials.from_service_account_file(
            credentials_file_path,
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        print("Autenticazione Google Sheets riuscita")
        
        model, scaler_past, scaler_forecast, scaler_target, config, device = \
            load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts
        
        scalers = (scaler_past, scaler_forecast, scaler_target)
        data_inputs = (hist_data, fcst_data)
        predictions, attention_info = make_prediction(model, scalers, config, data_inputs, device)
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        append_feature_importance_to_gsheet(gc, GSHEET_ID, GSHEET_FEATURE_IMPORTANCE_SHEET_NAME, attention_info, config)
        
        print(f"\n{'='*60}")
        print(f"COMPLETATO - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\nERRORE: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
