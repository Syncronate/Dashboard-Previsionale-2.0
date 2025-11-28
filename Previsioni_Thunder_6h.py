import os
import json
from datetime import datetime, timedelta
import pytz
import gspread
from google.oauth2.service_account import Credentials
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import traceback
import io
import csv
import random

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

# ============================================
# DEFINIZIONE MODELLO (Seq2SeqWithAttention)
# ============================================

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        if bidirectional:
            self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        x = self.input_proj(x)
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.layer_norm(outputs)

        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
            hidden = self.hidden_proj(torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1))
            cell = self.cell_proj(torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=-1))

        return outputs, hidden, cell


class ImprovedAttention(nn.Module):
    def __init__(self, hidden_size, attention_type='additive'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type

        if attention_type == 'additive':
            self.W_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        elif attention_type == 'dot':
            self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.shape

        if self.attention_type == 'additive':
            decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1).expand(-1, seq_len, -1)
            encoder_proj = self.W_encoder(encoder_outputs)
            energy = torch.tanh(decoder_proj + encoder_proj)
            scores = self.v(energy).squeeze(-1)
        elif self.attention_type == 'dot':
            decoder_proj = self.W(decoder_hidden)
            scores = torch.bmm(encoder_outputs, decoder_proj.unsqueeze(2)).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attn_weights = self.softmax(scores)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size,
                 num_layers=2, dropout=0.2, num_quantiles=1, attention_type='additive'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.num_quantiles = num_quantiles

        self.attention = ImprovedAttention(hidden_size, attention_type)
        self.lstm = nn.LSTM(
            forecast_input_size + hidden_size * 2, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.gate = nn.Sequential(
            nn.Linear(forecast_input_size + hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * num_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_forecast_step, hidden, cell, encoder_outputs):
        batch_size = x_forecast_step.size(0)
        decoder_hidden = hidden[-1]
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs)
        lstm_input = torch.cat([x_forecast_step, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        projected_context = self.context_proj(context)
        gate_input = torch.cat([x_forecast_step.squeeze(1), projected_context, output], dim=1)
        gate_value = self.gate(gate_input)
        gated_output = gate_value * output + (1 - gate_value) * projected_context
        gated_output = self.dropout(gated_output)
        prediction = self.fc(gated_output)
        return prediction, hidden, cell, attn_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0, target_sequence=None):
        batch_size = x_past.shape[0]
        target_output_size = self.decoder.output_size
        num_quantiles = self.decoder.num_quantiles
        decoder_output_dim = target_output_size * num_quantiles

        outputs = torch.zeros(batch_size, self.output_window, decoder_output_dim).to(x_past.device)
        attention_weights_history = torch.zeros(batch_size, self.output_window, x_past.shape[1]).to(x_past.device)

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        # Primo input del decoder
        decoder_input_step = x_future_forecast[:, 0:1, :]

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )

            outputs[:, t, :] = decoder_output_step
            attention_weights_history[:, t, :] = attn_weights.squeeze()

            if t < self.output_window - 1:
                # Autoregressive logic handled in make_prediction for inference script
                # Here we just expect x_future_forecast to contain the correct next step input
                # But since we are in a script where we might need to feed back predictions...
                # Actually, for the script, we will call the model step-by-step or pass the full tensor
                # IF the full tensor is already prepared.
                # However, since the script needs to handle dynamic autoregression (updating the input),
                # it's better if the script calls the model in a way that allows this, OR
                # we implement the autoregressive logic HERE inside the forward method if x_future_forecast is not full.
                
                # For consistency with the script logic which prepares the tensor:
                # We will assume x_future_forecast is fully populated OR we are in a mode where we don't update it here.
                # BUT, wait! In the Transformer script, we pass the full tensor.
                # If we want true autoregression where prediction t feeds into t+1, we need to do it here OR in the script loop.
                # The Transformer script does it inside the model forward (for autoregressive mode).
                # So let's add the autoregressive feedback logic here too!
                
                if teacher_forcing_ratio == 0 and target_sequence is None:
                     # Autoregressive Inference Mode
                     
                     # 1. Get prediction
                    if num_quantiles > 1:
                        pred_reshaped = decoder_output_step.view(batch_size, target_output_size, num_quantiles)
                        pred_targets = pred_reshaped[:, :, 1] # Median
                    else:
                        pred_targets = decoder_output_step.view(batch_size, target_output_size)
                    
                    # 2. Prepare next input
                    # We need to take the base features from x_future_forecast (e.g. weather) and update the target columns
                    next_base_input = x_future_forecast[:, t+1:t+2, :].clone()
                    
                    # Assuming target columns are the LAST N columns of the decoder input
                    # This is a strong assumption but standard in this project
                    num_target_features = target_output_size
                    
                    # Check if we actually have target features in the input (we might not if use_future_forecasts=False)
                    # If use_future_forecasts=False, input dim is just num_targets
                    if next_base_input.shape[-1] == num_target_features:
                        next_base_input = pred_targets.unsqueeze(1)
                    else:
                        # We have other features + targets. Update only targets.
                        next_base_input[:, :, -num_target_features:] = pred_targets
                    
                    decoder_input_step = next_base_input
                else:
                    decoder_input_step = x_future_forecast[:, t+1:t+2, :]

        if num_quantiles > 1:
            outputs = outputs.view(batch_size, self.output_window, target_output_size, num_quantiles)

        return outputs, attention_weights_history


# ============================================
# COSTANTI
# ============================================
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_1054_final" # Default, will be overwritten by config if needed

GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "DATI METEO CON FEATURE"
GSHEET_FORECAST_DATA_SHEET_NAME = "Previsioni Cumulate Feature ICON"
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Idro-Bettolelle 6h"

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
    print(f"\n=== CARICAMENTO MODELLO E SCALER ===")
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
    
    enc_input_size = len(config["all_past_feature_columns"])
    
    # Determine decoder input size
    use_future_forecasts = config.get("use_future_forecasts", True)
    dec_output_size = len(config["target_columns"])
    
    if use_future_forecasts:
        dec_input_size = len(config["forecast_input_columns"])
    else:
        dec_input_size = dec_output_size # Only past targets
    
    quantiles = config.get("quantiles")
    num_quantiles = len(quantiles) if quantiles and isinstance(quantiles, list) else 1
    
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    drop = config["dropout"]
    out_win = config["output_window_steps"]
    
    print(f"Configurazione modello:")
    print(f"  - Encoder input: {enc_input_size} features")
    print(f"  - Decoder input: {dec_input_size} features")
    print(f"  - Output: {dec_output_size} target(s)")
    print(f"  - Quantili: {num_quantiles}")
    print(f"  - Output window: {out_win} steps")
    print(f"  - Use Future Forecasts: {use_future_forecasts}")
    
    encoder = EncoderLSTM(enc_input_size, hidden, layers, drop)
    decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop, num_quantiles=num_quantiles)
    model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
    
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
    
    print("✓ Modello e scaler caricati")
    
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device


def fetch_and_prepare_data(gc, sheet_id, config):
    print(f"\n=== CARICAMENTO E PREPARAZIONE DATI ===")
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    target_columns = config.get("target_columns", [])
    
    use_future_forecasts = config.get("use_future_forecasts", True)
    forecast_feature_columns = []
    actual_forecast_cols_to_fetch = []

    if use_future_forecasts:
        forecast_feature_columns = config["forecast_input_columns"]
        # Filtra le colonne che sono anche target (non vanno cercate nel foglio previsioni)
        actual_forecast_cols_to_fetch = [c for c in forecast_feature_columns if c not in target_columns]
        
        if len(actual_forecast_cols_to_fetch) == 0:
            print("Info: Le colonne di input previsionale coincidono con i target (Modello Autoregressivo Puro).")
            print("      Non verranno caricati dati dal foglio previsioni.")
    
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
    # Carica il foglio previsioni SOLO se ci sono feature esterne da cercare
    if use_future_forecasts and len(actual_forecast_cols_to_fetch) > 0:
        print(f"Caricamento dati previsionali da '{GSHEET_FORECAST_DATA_SHEET_NAME}'...")
        try:
            forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
            forecast_values = forecast_ws.get_all_values()
            df_forecast_raw = pd.read_csv(io.StringIO(values_to_csv_string(forecast_values)), decimal=',')
        except gspread.exceptions.WorksheetNotFound:
            print(f"Warning: Foglio '{GSHEET_FORECAST_DATA_SHEET_NAME}' non trovato. Si prosegue senza previsioni esterne.")

    # Rinomina colonne
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

    # Parsing date storiche
    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(
        df_historical_raw[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce'
    )
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    print(f"Ultimo timestamp storico: {latest_valid_timestamp}")
    
    # Prepara dati storici
    for col in past_feature_columns:
        if col not in df_historical.columns:
            raise ValueError(f"Colonna storica '{col}' non trovata.")
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values
    print(f"Shape dati storici: {input_data_historical.shape}")
    
    # Prepara dati forecast
    input_data_forecast = None
    
    if use_future_forecasts and len(actual_forecast_cols_to_fetch) > 0 and df_forecast_raw is not None:
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

        # Gestione caso misto (Rain + Level) dove Level è target
        if len(actual_forecast_cols_to_fetch) < len(forecast_feature_columns):
             df_complete_forecast = pd.DataFrame(index=future_data.index, columns=forecast_feature_columns)
             for col in actual_forecast_cols_to_fetch:
                 df_complete_forecast[col] = future_data[col]
             df_complete_forecast = df_complete_forecast.fillna(0)
             input_data_forecast = df_complete_forecast.values
        else:
             input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
             
        print(f"Shape dati forecast: {input_data_forecast.shape}")
        
    elif use_future_forecasts and len(actual_forecast_cols_to_fetch) == 0:
        print("Nessuna colonna forecast esterna richiesta. Si procede in modalità autoregressiva pura.")
        input_data_forecast = None
    else:
        print("Nessuna colonna forecast richiesta o dati forecast non caricati.")
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp


def make_prediction(model, scalers, config, data_inputs, device):
    print(f"\n=== GENERAZIONE PREVISIONI ===")
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    
    # Normalizzazione
    historical_normalized = scaler_past_features.transform(historical_data_np)
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    
    forecast_tensor = None
    use_future_forecasts = config.get("use_future_forecasts", True)
    
    if use_future_forecasts and forecast_data_np is not None:
        # CRITICO: Se siamo in modalità autoregressiva (anche parziale), le colonne dei target in forecast_data_np
        # sono state riempite con 0. Dobbiamo inserire l'ultimo valore storico noto nella prima riga
        # per "innescare" correttamente il loop autoregressivo.
        
        past_cols = config.get("all_past_feature_columns", [])
        target_cols = config.get("target_columns", [])
        forecast_cols = config.get("forecast_input_columns", [])
        
        # Identifica gli indici dei target nelle feature passate e future
        # Assumiamo che in forecast_data_np i target siano nelle posizioni corrispondenti a forecast_cols
        
        for t_col in target_cols:
            if t_col in past_cols and t_col in forecast_cols:
                past_idx = past_cols.index(t_col)
                fcst_idx = forecast_cols.index(t_col)
                
                # historical_data_np è raw (non scalato)
                last_val = historical_data_np[-1, past_idx]
                
                # Inseriamo il valore solo nella prima riga (step t=0 per predire t=1)
                # Le righe successive verranno sovrascritte dal modello durante l'inferenza
                forecast_data_np[0, fcst_idx] = last_val
                
        forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
        forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    else:
        # No future forecasts: costruiamo il dummy input con l'ultimo target noto
        past_cols = config.get("all_past_feature_columns", [])
        target_cols = config.get("target_columns", [])
        num_targets = len(target_cols)
        output_steps = config['output_window_steps']
        
        last_target_values = []
        for t_col in target_cols:
            if t_col in past_cols:
                idx = past_cols.index(t_col)
                last_target_values.append(historical_normalized[-1, idx])
            else:
                last_target_values.append(0.0)
        
        last_targets_tensor = torch.tensor(last_target_values, dtype=torch.float32).view(1, 1, -1).to(device)
        
        # Create dummy future tensor (batch, seq_len, num_targets)
        # We initialize with zeros, but the first step will be the last known target
        dummy_future = torch.zeros(1, output_steps, num_targets).to(device)
        dummy_future[:, 0:1, :] = last_targets_tensor
        forecast_tensor = dummy_future

    # Inferenza
    model.eval()
    set_seed(SEED)
    
    with torch.no_grad():
        # The model's forward method now handles the autoregressive loop if teacher_forcing_ratio=0
        predictions_normalized, _ = model(historical_tensor, forecast_tensor, teacher_forcing_ratio=0.0)
    
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    num_targets = len(config["target_columns"])
    
    # De-normalizzazione
    if predictions_np.ndim == 3:
        seq_len, n_targets, n_quantiles = predictions_np.shape
        preds_permuted = predictions_np.transpose(0, 2, 1)
        preds_flat = preds_permuted.reshape(-1, num_targets)
        preds_unscaled_flat = scaler_targets.inverse_transform(preds_flat)
        preds_unscaled = preds_unscaled_flat.reshape(seq_len, n_quantiles, num_targets)
        predictions_scaled_back = preds_unscaled.transpose(0, 2, 1)
    elif predictions_np.ndim == 2:
        predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)
    else:
        predictions_np_flat = predictions_np.reshape(-1, num_targets)
        predictions_scaled_back_flat = scaler_targets.inverse_transform(predictions_np_flat)
        predictions_scaled_back = predictions_scaled_back_flat.reshape(predictions_np.shape)

    # Log previsioni
    print(f"\nPrevisioni generate (shape: {predictions_scaled_back.shape}):")
    if predictions_scaled_back.ndim == 3:
        median_idx = predictions_scaled_back.shape[2] // 2
        for i in range(predictions_scaled_back.shape[0]):
            val = predictions_scaled_back[i, 0, median_idx]
            print(f"  Step {i+1}: {val:.2f}")
    else:
        for i in range(predictions_scaled_back.shape[0]):
            print(f"  Step {i+1}: {predictions_scaled_back[i]}")
    
    return predictions_scaled_back


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
        print(f"✓ Salvate {len(rows_to_append)} righe di previsione")


def main():
    print(f"\n{'='*60}")
    print(f"AVVIO SCRIPT SEQ2SEQ - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}")
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
        print("✓ Autenticazione Google Sheets riuscita")
        
        model, scaler_past, scaler_forecast, scaler_target, config, device = \
            load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts
        
        scalers = (scaler_past, scaler_forecast, scaler_target)
        data_inputs = (hist_data, fcst_data)
        predictions = make_prediction(model, scalers, config, data_inputs, device)
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETATO - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
