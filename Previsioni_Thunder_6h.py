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
import random # Importato per compatibilità con il forward di Seq2Seq

# Imposta seed per riproducibilità
torch.manual_seed(42)
np.random.seed(42)

# --- INIZIO BLOCCO MODELLO AGGIORNATO ---
# Queste classi sono state copiate dalla tua app di training Streamlit per corrispondere
# esattamente all'architettura del modello salvato.

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
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

# NOTA: Questa è la classe 'ImprovedAttention' dalla tua app, rinominata in 'Attention' per compatibilità.
class Attention(nn.Module):
    def __init__(self, hidden_size, attention_type='additive'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        # Questa implementazione corrisponde a quella che ha generato le chiavi nel file .pth
        self.W_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False) # *2 per encoder bidirezionale
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.shape
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1).expand(-1, seq_len, -1)
        encoder_proj = self.W_encoder(encoder_outputs)
        energy = torch.tanh(decoder_proj + encoder_proj)
        scores = self.v(energy).squeeze(-1)
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
        self.attention = Attention(hidden_size, attention_type)

        # L'input LSTM ora include forecast features + context vector (da encoder bidirezionale)
        self.lstm = nn.LSTM(
            forecast_input_size + hidden_size * 2,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Questi layer ora esistono e corrispondono al file .pth
        self.gate = nn.Sequential(
            nn.Linear(forecast_input_size + hidden_size * 2 + hidden_size, hidden_size), # Corretto per includere output
            nn.Sigmoid()
        )
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * num_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_forecast_step, hidden, cell, encoder_outputs):
        decoder_hidden = hidden[-1]
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs)
        
        lstm_input = torch.cat([x_forecast_step, context.unsqueeze(1)], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        
        projected_context = self.context_proj(context)
        gate_input = torch.cat([
            x_forecast_step.squeeze(1),
            projected_context,
            output
        ], dim=1)
        
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
        
        decoder_input_step = x_future_forecast[:, 0:1, :]
        
        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )
            outputs[:, t, :] = decoder_output_step
            attention_weights_history[:, t, :] = attn_weights.squeeze(0) # Squeeze per rimuovere la dimensione batch che è sempre 1

            if t < self.output_window - 1:
                # Per l'inferenza, usiamo sempre l'input futuro fornito (no teacher forcing o autoregressione complessa)
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]

        # Non è necessario fare il reshape qui per l'inferenza, la funzione chiamante lo gestisce.
        return outputs, attention_weights_history

# --- FINE BLOCCO MODELLO AGGIORNATO ---


# --- Costanti ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20251111_1356"

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
    """Log delle informazioni sull'ambiente di esecuzione"""
    print("=== INFORMAZIONI AMBIENTE ===")
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"gspread version: {gspread.__version__}")
    print(f"joblib version: {joblib.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Device utilizzato: {'cuda' if torch.cuda.is_available() else 'cpu'}")
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
            raise FileNotFoundError(f"ERRORE CRITICO: File non trovato -> {p}")
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_type = config.get("model_type")
    if model_type != "Seq2SeqAttention":
        raise ValueError(f"Tipo di modello non supportato: '{model_type}'.")
    
    enc_input_size = len(config["all_past_feature_columns"])
    dec_input_size = len(config["forecast_input_columns"])
    
    quantiles = config.get("quantiles")
    num_quantiles = 1
    if quantiles and isinstance(quantiles, list):
        num_quantiles = len(quantiles)
        print(f"Rilevato modello a quantili. Quantili: {quantiles}")
    else:
        print("Rilevato modello con output singolo (non a quantili).")
    
    dec_output_size = len(config["target_columns"])
    # NOTA: La dimensione di output del decoder è solo il numero di target. I quantili sono gestiti internamente.
    final_output_dim = dec_output_size * num_quantiles
    print(f"Dimensione output del decoder impostata a: {final_output_dim}")
    
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    drop = config["dropout"]
    out_win = config["output_window_steps"]
    
    # Istanziazione delle classi del modello NUOVE E CORRETTE
    encoder = EncoderLSTM(enc_input_size, hidden, layers, drop)
    # Passiamo num_quantiles al decoder
    decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop, num_quantiles=num_quantiles)
    model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
    
    model_state = torch.load(model_path, map_location=device)
    
    # Caricamento dello state_dict. Questo ora dovrebbe funzionare.
    model.load_state_dict(model_state)
    model.eval()
    
    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    
    print("Modello e scaler caricati con successo.")
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device

# --- Il resto dello script (da qui in poi) rimane invariato ---

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

    print(f"Caricamento dati storici da '{GSHEET_HISTORICAL_DATA_SHEET_NAME}'...")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    historical_values = historical_ws.get_all_values()
    historical_csv_string = values_to_csv_string(historical_values)
    df_historical_raw = pd.read_csv(io.StringIO(historical_csv_string), decimal=',')
    
    print(f"Caricamento dati previsionali da '{GSHEET_FORECAST_DATA_SHEET_NAME}'...")
    forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
    forecast_values = forecast_ws.get_all_values()
    forecast_csv_string = values_to_csv_string(forecast_values)
    df_forecast_raw = pd.read_csv(io.StringIO(forecast_csv_string), decimal=',')

    print("Avvio rinomina aggressiva per standardizzare le colonne...")
    hist_rename_map = {
        col: col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
        for col in df_historical_raw.columns
        if 'Giornaliera ' in col or '_cumulata_30min' in col
    }
    fcst_rename_map = {
        col: col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
        for col in df_forecast_raw.columns
        if 'Giornaliera ' in col or '_cumulata_30min' in col
    }

    if hist_rename_map:
        df_historical_raw.rename(columns=hist_rename_map, inplace=True)
        print(f"Rinominate {len(hist_rename_map)} colonne nel set di dati storici.")
    if fcst_rename_map:
        df_forecast_raw.rename(columns=fcst_rename_map, inplace=True)
        print(f"Rinominate {len(fcst_rename_map)} colonne nel set di dati previsionali.")

    df_historical_raw = df_historical_raw.loc[:, ~df_historical_raw.columns.duplicated(keep='last')]
    df_forecast_raw = df_forecast_raw.loc[:, ~df_forecast_raw.columns.duplicated(keep='last')]
    print("✓ Eventuali colonne duplicate rimosse, mantenendo l'ultima occorrenza (keep='last').")

    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(
        df_historical_raw[GSHEET_DATE_COL_INPUT], 
        format=GSHEET_DATE_FORMAT_INPUT, 
        errors='coerce'
    )
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    print(f"Ultimo timestamp valido: {latest_valid_timestamp}")
    
    for col in past_feature_columns:
        if col not in df_historical.columns:
            raise ValueError(f"Colonna storica '{col}' non trovata. Controllare la mappatura e il foglio di input.")
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values
    
    min_val, max_val = input_data_historical.min(), input_data_historical.max()
    print(f"Statistiche dati storici (finali): Min={min_val:.2f}, Max={max_val:.2f}, Mean={input_data_historical.mean():.2f}")
    min_ragionevole, max_ragionevole = -1000, 200000
    if min_val < min_ragionevole or max_val > max_ragionevole:
        raise ValueError(f"Valori anomali rilevati. Min: {min_val}, Max: {max_val}. Controllare i dati sorgente.")
    else:
        print("✓ Controllo di validità dei dati storici superato.")
    
    df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(
        df_forecast_raw[GSHEET_FORECAST_DATE_COL], 
        format=GSHEET_FORECAST_DATE_FORMAT, 
        errors='coerce'
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
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp

def make_prediction(model, scalers, config, data_inputs, device):
    print(f"\n=== GENERAZIONE PREVISIONI ===")
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    
    historical_normalized = scaler_past_features.transform(historical_data_np)
    forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
    
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions_normalized, attention_weights = model(historical_tensor, forecast_tensor)
    
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    
    original_shape = predictions_np.shape
    num_targets = len(config["target_columns"])
    
    if predictions_np.ndim > 1 and predictions_np.shape[1] > num_targets:
        print(f"Rilevato output multi-colonna ({original_shape[1]} colonne) per {num_targets} target(s). Appiattimento per lo scaling.")
        predictions_np_flat = predictions_np.reshape(-1, num_targets)
        predictions_scaled_back_flat = scaler_targets.inverse_transform(predictions_np_flat)
        predictions_scaled_back = predictions_scaled_back_flat.reshape(original_shape)
    else:
        predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)

    print(f"Previsioni finali stats: min={predictions_scaled_back.min():.4f}, max={predictions_scaled_back.max():.4f}")
    
    return predictions_scaled_back

def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    print(f"\n=== SALVATAGGIO PREVISIONI ===")
    sh = gc.open_by_key(sheet_id_str)
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        worksheet.clear()
        print(f"Foglio '{predictions_sheet_name}' pulito.")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows=100, cols=20)
        print(f"Foglio '{predictions_sheet_name}' creato.")
    
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
        row.extend([f"{val:.3f}".replace('.', ',') for val in predictions_np[i, :]])
        rows_to_append.append(row)
    
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Salvate {len(rows_to_append)} righe di previsione.")

def main():
    print(f"AVVIO SCRIPT - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Modello: {MODEL_BASE_NAME}")
    
    try:
        log_environment_info()
        
        credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not credentials_json:
            raise ValueError("La variabile d'ambiente GOOGLE_CREDENTIALS_JSON non è impostata.")
        
        credentials_info = json.loads(credentials_json)
        credentials = Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        print("✓ Autenticazione Google Sheets riuscita.")
        
        model, scaler_past, scaler_forecast, scaler_target, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts
        
        scalers = (scaler_past, scaler_forecast, scaler_target)
        data_inputs = (hist_data, fcst_data)
        predictions = make_prediction(model, scalers, config, data_inputs, device)
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        
        print(f"\n✓ SCRIPT COMPLETATO - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
