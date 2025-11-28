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
# CONFIGURAZIONE RIPRODUCIBILITÀ COMPLETA
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
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    
    print(f"✓ Seed impostato a {seed} per riproducibilità completa")

set_seed(SEED)

# ============================================
# DEFINIZIONE MODELLO
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
        self.debug_mode = True  # Attiva debug

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0, target_sequence=None):
        batch_size = x_past.shape[0]
        target_output_size = self.decoder.output_size
        num_quantiles = self.decoder.num_quantiles
        decoder_output_dim = target_output_size * num_quantiles

        outputs = torch.zeros(batch_size, self.output_window, decoder_output_dim).to(x_past.device)
        attention_weights_history = torch.zeros(batch_size, self.output_window, x_past.shape[1]).to(x_past.device)

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        decoder_input_step = x_future_forecast[:, 0:1, :]

        if self.debug_mode:
            print(f"\n{'='*60}")
            print("DEBUG AUTOREGRESSIVE LOOP")
            print(f"{'='*60}")
            print(f"Output window: {self.output_window}")
            print(f"Target output size: {target_output_size}")
            print(f"Num quantiles: {num_quantiles}")
            print(f"Decoder input dim: {x_future_forecast.shape[2]}")

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )

            outputs[:, t, :] = decoder_output_step
            attention_weights_history[:, t, :] = attn_weights.squeeze()

            if t < self.output_window - 1:
                if teacher_forcing_ratio > 0:
                    use_teacher_forcing = random.random() < teacher_forcing_ratio
                else:
                    use_teacher_forcing = False

                if use_teacher_forcing and target_sequence is not None:
                    decoder_input_step = x_future_forecast[:, t+1:t+2, :]
                else:
                    # AUTOREGRESSIVE
                    if num_quantiles > 1:
                        pred_reshaped = decoder_output_step.view(batch_size, target_output_size, num_quantiles)
                        pred_targets = pred_reshaped[:, :, num_quantiles // 2]
                    else:
                        pred_targets = decoder_output_step.view(batch_size, target_output_size)

                    next_forecast_features = x_future_forecast[:, t+1, :].clone()
                    
                    # DEBUG: Mostra cosa stiamo sostituendo
                    if self.debug_mode and t < 3:
                        print(f"\n--- Step {t} -> {t+1} ---")
                        print(f"  Predizione (normalizzata): {pred_targets[0].detach().cpu().numpy()}")
                        print(f"  Ultime {target_output_size} feature PRIMA: {next_forecast_features[0, -target_output_size:].detach().cpu().numpy()}")
                    
                    num_target_features = target_output_size
                    next_forecast_features[:, -num_target_features:] = pred_targets

                    if self.debug_mode and t < 3:
                        print(f"  Ultime {target_output_size} feature DOPO:  {next_forecast_features[0, -target_output_size:].detach().cpu().numpy()}")

                    decoder_input_step = next_forecast_features.unsqueeze(1)

        if num_quantiles > 1:
            outputs = outputs.view(batch_size, self.output_window, target_output_size, num_quantiles)

        return outputs, attention_weights_history


# ============================================
# COSTANTI
# ============================================
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20251126_1100_best"

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
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 30)


def load_model_and_scalers(model_base_name, models_dir):
    print(f"\n=== CARICAMENTO MODELLO E SCALER ===")
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_past_features_path = os.path.join(models_dir, f"{model_base_name}_past_features.joblib")
    scaler_forecast_features_path = os.path.join(models_dir, f"{model_base_name}_forecast_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")
    
    for p in [config_path, model_path, scaler_past_features_path, scaler_forecast_features_path, scaler_targets_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File non trovato -> {p}")
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    enc_input_size = len(config["all_past_feature_columns"])
    dec_input_size = len(config["forecast_input_columns"])
    
    quantiles = config.get("quantiles")
    num_quantiles = len(quantiles) if quantiles and isinstance(quantiles, list) else 1
    
    dec_output_size = len(config["target_columns"])
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    drop = config["dropout"]
    out_win = config["output_window_steps"]
    
    encoder = EncoderLSTM(enc_input_size, hidden, layers, drop)
    decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop, num_quantiles=num_quantiles)
    model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
    
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    
    # ============================================
    # DEBUG CRITICO: Verifica allineamento colonne
    # ============================================
    print(f"\n{'='*60}")
    print("DEBUG CRITICO: VERIFICA ALLINEAMENTO COLONNE")
    print(f"{'='*60}")
    
    target_cols = config["target_columns"]
    forecast_cols = config["forecast_input_columns"]
    
    print(f"\nTarget columns ({len(target_cols)}):")
    for i, col in enumerate(target_cols):
        print(f"  {i}: {col}")
    
    print(f"\nForecast input columns ({len(forecast_cols)}):")
    for i, col in enumerate(forecast_cols):
        print(f"  {i}: {col}")
    
    print(f"\nULTIME {len(target_cols)} colonne delle forecast features:")
    for i, col in enumerate(forecast_cols[-len(target_cols):]):
        print(f"  {i}: {col}")
    
    # Verifica se le ultime colonne corrispondono ai target
    print(f"\n⚠️  VERIFICA CORRISPONDENZA:")
    match = True
    for i, (target, forecast) in enumerate(zip(target_cols, forecast_cols[-len(target_cols):])):
        matches = target == forecast or target in forecast or forecast in target
        status = "✓" if matches else "❌"
        print(f"  {status} Target '{target}' vs Forecast '{forecast}'")
        if not matches:
            match = False
    
    if not match:
        print("\n❌ ATTENZIONE: Le colonne target NON corrispondono alle ultime colonne forecast!")
        print("   Questo causa previsioni errate nella logica autoregressiva!")
    
    print(f"{'='*60}\n")
    
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device


def fetch_and_prepare_data(gc, sheet_id, config):
    print(f"\n=== CARICAMENTO E PREPARAZIONE DATI ===")
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    forecast_feature_columns = config["forecast_input_columns"]
    
    sh = gc.open_by_key(sheet_id)
    
    def values_to_csv_string(data):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(data)
        return output.getvalue()

    print(f"Caricamento dati storici...")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    historical_values = historical_ws.get_all_values()
    historical_csv_string = values_to_csv_string(historical_values)
    df_historical_raw = pd.read_csv(io.StringIO(historical_csv_string), decimal=',')
    
    print(f"Caricamento dati previsionali...")
    forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
    forecast_values = forecast_ws.get_all_values()
    forecast_csv_string = values_to_csv_string(forecast_values)
    df_forecast_raw = pd.read_csv(io.StringIO(forecast_csv_string), decimal=',')

    def build_rename_map(columns):
        rename_map = {}
        for col in columns:
            if 'Giornaliera ' in col:
                if '_cumulata_30min' in col:
                    new_name = col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
                    rename_map[col] = new_name
                else:
                    new_name = col.replace('Giornaliera ', '')
                    if col.strip().endswith(')'):
                        continue
                    rename_map[col] = new_name
        return rename_map

    hist_rename_map = build_rename_map(df_historical_raw.columns)
    fcst_rename_map = build_rename_map(df_forecast_raw.columns)

    if hist_rename_map:
        df_historical_raw.rename(columns=hist_rename_map, inplace=True)
    if fcst_rename_map:
        df_forecast_raw.rename(columns=fcst_rename_map, inplace=True)

    df_historical_raw = df_historical_raw.loc[:, ~df_historical_raw.columns.duplicated(keep='last')]
    df_forecast_raw = df_forecast_raw.loc[:, ~df_forecast_raw.columns.duplicated(keep='last')]

    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(
        df_historical_raw[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce'
    )
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    
    for col in past_feature_columns:
        if col not in df_historical.columns:
            raise ValueError(f"Colonna storica '{col}' non trovata.")
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values
    
    df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(
        df_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce'
    )
    future_forecasts = df_forecast_raw[df_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
    if len(future_forecasts) < output_window_steps:
        raise ValueError(f"Previsioni insufficienti: {len(future_forecasts)} < {output_window_steps}")
    future_data = future_forecasts.head(output_window_steps).copy()
    
    for col in forecast_feature_columns:
        if col not in future_data.columns:
            raise ValueError(f"Colonna previsionale '{col}' non trovata.")
        future_data[col] = pd.to_numeric(future_data[col], errors='coerce')

    input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
    
    # ============================================
    # DEBUG: Mostra i valori effettivi delle ultime colonne
    # ============================================
    target_cols = config["target_columns"]
    print(f"\n{'='*60}")
    print("DEBUG: VALORI FORECAST FEATURES (ultime colonne = target)")
    print(f"{'='*60}")
    print(f"Shape forecast data: {input_data_forecast.shape}")
    print(f"\nValori delle ULTIME {len(target_cols)} colonne (dovrebbero essere i livelli idrometrici):")
    for i in range(min(5, output_window_steps)):
        last_vals = input_data_forecast[i, -len(target_cols):]
        print(f"  Step {i}: {last_vals}")
    print(f"{'='*60}\n")
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp


def make_prediction(model, scalers, config, data_inputs, device):
    print(f"\n=== GENERAZIONE PREVISIONI ===")
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    
    # Normalizzazione
    historical_normalized = scaler_past_features.transform(historical_data_np)
    forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
    
    # DEBUG: Valori normalizzati delle ultime colonne
    target_cols = config["target_columns"]
    print(f"\nDEBUG: Valori NORMALIZZATI delle ultime {len(target_cols)} colonne forecast:")
    for i in range(min(3, forecast_normalized.shape[0])):
        last_vals = forecast_normalized[i, -len(target_cols):]
        print(f"  Step {i}: {last_vals}")
    
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    
    model.eval()
    set_seed(SEED)
    
    with torch.no_grad():
        predictions_normalized, attention_weights = model(historical_tensor, forecast_tensor)
    
    predictions_cpu = predictions_normalized.cpu().numpy()
    predictions_np = predictions_cpu.squeeze(0)
    
    num_targets = len(config["target_columns"])
    
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
        original_shape = predictions_np.shape
        predictions_np_flat = predictions_np.reshape(-1, num_targets)
        predictions_scaled_back_flat = scaler_targets.inverse_transform(predictions_np_flat)
        predictions_scaled_back = predictions_scaled_back_flat.reshape(original_shape)

    # DEBUG: Mostra previsioni finali
    print(f"\n{'='*60}")
    print("DEBUG: PREVISIONI FINALI (de-normalizzate)")
    print(f"{'='*60}")
    print(f"Shape: {predictions_scaled_back.shape}")
    
    if predictions_scaled_back.ndim == 3:
        # Con quantili
        print("Previsioni per ogni step (mediana):")
        for i in range(predictions_scaled_back.shape[0]):
            median_idx = predictions_scaled_back.shape[2] // 2
            vals = predictions_scaled_back[i, :, median_idx]
            print(f"  Step {i+1}: {vals}")
    else:
        print("Previsioni per ogni step:")
        for i in range(predictions_scaled_back.shape[0]):
            vals = predictions_scaled_back[i]
            print(f"  Step {i+1}: {vals}")
    
    print(f"{'='*60}\n")
    
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
        print(f"✓ Salvate {len(rows_to_append)} righe.")


def main():
    print(f"\n{'='*60}")
    print(f"AVVIO SCRIPT DEBUG - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        log_environment_info()
        
        credentials_file_path = "credentials.json"
        if not os.path.exists(credentials_file_path):
            raise FileNotFoundError(f"File credenziali non trovato.")

        credentials = Credentials.from_service_account_file(
            credentials_file_path,
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        
        model, scaler_past, scaler_forecast, scaler_target, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts
        
        scalers = (scaler_past, scaler_forecast, scaler_target)
        data_inputs = (hist_data, fcst_data)
        predictions = make_prediction(model, scalers, config, data_inputs, device)
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        
        print(f"\n✓ COMPLETATO")

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
