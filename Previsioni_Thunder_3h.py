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
import io  # <-- IMPORTANTE: Assicurati che questo import sia presente

# Imposta seed per riproducibilità
torch.manual_seed(42)
np.random.seed(42)

# --- Le classi del modello rimangono identiche ---
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([hidden_repeated, encoder_outputs], dim=2)))
        energy = energy.permute(0, 2, 1)
        v_exp = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        scores = torch.bmm(v_exp, energy).squeeze(1)
        return torch.softmax(scores, dim=1)

class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(forecast_input_size + hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_forecast_step, hidden, cell, encoder_outputs):
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat([x_forecast_step, context_vector], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0):
        batch_size = x_past.shape[0]
        target_output_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(x_past.device)
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :]
        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )
            outputs[:, t, :] = decoder_output_step
            if t < self.output_window - 1:
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]
        return outputs, attn_weights

# --- Costanti ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20250801_1755"

GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "DATI METEO CON FEATURE"
GSHEET_FORECAST_DATA_SHEET_NAME = "Previsioni Cumulate Feature ICON"
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Idro-Bettolelle 3h"

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
    
    required_files = [config_path, model_path, scaler_past_features_path, scaler_forecast_features_path, scaler_targets_path]
    for p in required_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"ERRORE CRITICO: File non trovato -> {p}")
        print(f"✓ File trovato: {p}")
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    print(f"Configurazione caricata: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device selezionato: {device}")
    
    model_type = config.get("model_type")
    if model_type != "Seq2SeqAttention":
        raise ValueError(f"Tipo di modello non supportato: '{model_type}'.")
    
    enc_input_size = len(config["all_past_feature_columns"])
    dec_input_size = len(config["forecast_input_columns"])
    dec_output_size = len(config["target_columns"])
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    drop = config["dropout"]
    out_win = config["output_window_steps"]
    
    print(f"Parametri modello:")
    print(f"  - Encoder input size: {enc_input_size}")
    print(f"  - Decoder input size: {dec_input_size}")
    print(f"  - Decoder output size: {dec_output_size}")
    print(f"  - Hidden size: {hidden}")
    print(f"  - Num layers: {layers}")
    print(f"  - Dropout: {drop}")
    print(f"  - Output window: {out_win}")
    
    encoder = EncoderLSTM(enc_input_size, hidden, layers, drop)
    decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop)
    model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
    
    model_state = torch.load(model_path, map_location=device)
    print(f"Checksum parametri modello: {hash(str(list(model_state.keys())))}")
    
    model.load_state_dict(model_state)
    model.eval()
    
    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    
    # Log corretto per MinMaxScaler
    print(f"Scaler past features - N. features: {getattr(scaler_past_features, 'n_features_in_', 'N/A')}")
    print(f"Scaler forecast features - N. features: {getattr(scaler_forecast_features, 'n_features_in_', 'N/A')}")
    print(f"Scaler targets - N. features: {getattr(scaler_targets, 'n_features_in_', 'N/A')}")
    
    print("Modello e scaler caricati con successo.")
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device

# --- INIZIO FUNZIONE CORRETTA E ROBUSTA ---
def fetch_and_prepare_data(gc, sheet_id, config):
    print(f"\n=== CARICAMENTO E PREPARAZIONE DATI (METODO CSV ROBUSTO) ===")
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    forecast_feature_columns = config["forecast_input_columns"]
    
    print(f"Parametri:")
    print(f"  - Input window steps: {input_window_steps}")
    print(f"  - Output window steps: {output_window_steps}")
    
    sh = gc.open_by_key(sheet_id)
    
    # --- CARICAMENTO DATI STORICI COME CSV ---
    print(f"Caricamento dati storici da '{GSHEET_HISTORICAL_DATA_SHEET_NAME}' come CSV...")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    export_historical = historical_ws.export(format='csv')
    # Usiamo il motore di parsing di Pandas, molto più robusto
    df_historical_raw = pd.read_csv(io.StringIO(export_historical.decode('utf-8')))
    print(f"Righe dati storici grezzi: {len(df_historical_raw)}")

    # --- CARICAMENTO DATI PREVISIONALI COME CSV ---
    print(f"Caricamento dati previsionali da '{GSHEET_FORECAST_DATA_SHEET_NAME}' come CSV...")
    forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
    export_forecast = forecast_ws.export(format='csv')
    df_forecast_raw = pd.read_csv(io.StringIO(export_forecast.decode('utf-8')))
    print(f"Righe dati previsionali grezzi: {len(df_forecast_raw)}")

    # Mappatura colonne (per sicurezza)
    column_mapping = {
        "Cumulata Sensore 1295 (Arcevia)_cumulata_30min": "Cumulata Sensore 1295 (Arcevia)",
        "Cumulata Sensore 2637 (Bettolelle)_cumulata_30min": "Cumulata Sensore 2637 (Bettolelle)",
        "Cumulata Sensore 2858 (Barbara)_cumulata_30min": "Cumulata Sensore 2858 (Barbara)",  
        "Cumulata Sensore 2964 (Corinaldo)_cumulata_30min": "Cumulata Sensore 2964 (Corinaldo)"
    }
    df_historical_raw.rename(columns=column_mapping, inplace=True, errors='ignore')
    df_forecast_raw.rename(columns=column_mapping, inplace=True, errors='ignore')

    # --- PROCESSAMENTO DATI STORICI ---
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
            raise ValueError(f"Colonna storica '{col}' non trovata dopo il caricamento CSV.")
        # La lettura da CSV gestisce meglio i tipi, ma forziamo a numerico per sicurezza
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values
    
    # CONTROLLO DI VALIDITA' FINALE
    min_val, max_val = input_data_historical.min(), input_data_historical.max()
    print(f"Statistiche dati storici (dopo caricamento CSV): Min={min_val:.2f}, Max={max_val:.2f}, Mean={input_data_historical.mean():.2f}")
    min_ragionevole, max_ragionevole = -1000, 100000
    if min_val < min_ragionevole or max_val > max_ragionevole:
        raise ValueError(f"Valori anomali rilevati dopo caricamento CSV. Min: {min_val}, Max: {max_val}. Controllare il foglio Google o il processo di pulizia.")
    else:
        print("✓ Controllo di validità dei dati storici superato.")
    
    # --- PROCESSAMENTO DATI PREVISIONALI ---
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
            raise ValueError(f"Colonna previsionale '{col}' non trovata dopo caricamento CSV.")
        future_data[col] = pd.to_numeric(future_data[col], errors='coerce')

    input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
    print(f"Dati previsionali finali shape: {input_data_forecast.shape}")

    return input_data_historical, input_data_forecast, latest_valid_timestamp
# --- FINE FUNZIONE CORRETTA E ROBUSTA ---

def make_prediction(model, scalers, config, data_inputs, device):
    print(f"\n=== GENERAZIONE PREVISIONI ===")
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    
    print(f"Dati input - Storici shape: {historical_data_np.shape}")
    print(f"Dati input - Previsionali shape: {forecast_data_np.shape}")
    
    historical_normalized = scaler_past_features.transform(historical_data_np)
    forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
    
    print(f"Dati normalizzati - Storici stats: min={historical_normalized.min():.4f}, max={historical_normalized.max():.4f}")
    print(f"Dati normalizzati - Previsionali stats: min={forecast_normalized.min():.4f}, max={forecast_normalized.max():.4f}")
    
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    
    print(f"Tensori shape - Storici: {historical_tensor.shape}, Previsionali: {forecast_tensor.shape}")
    
    with torch.no_grad():
        predictions_normalized, attention_weights = model(historical_tensor, forecast_tensor)
    
    print(f"Previsioni normalizzate shape: {predictions_normalized.shape}")
    print(f"Previsioni normalizzate stats: min={predictions_normalized.min():.4f}, max={predictions_normalized.max():.4f}")
    
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)
    
    print(f"Previsioni finali shape: {predictions_scaled_back.shape}")
    print(f"Previsioni finali stats: min={predictions_scaled_back.min():.4f}, max={predictions_scaled_back.max():.4f}")
    print(f"Prime 3 previsioni: {predictions_scaled_back[:3].tolist()}")
    
    return predictions_scaled_back

def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    print(f"\n=== SALVATAGGIO PREVISIONI ===")
    sh = gc.open_by_key(sheet_id_str)
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        worksheet.clear()
        print(f"Foglio '{predictions_sheet_name}' pulito.")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows=config["output_window_steps"] + 10, cols=len(config["target_columns"]) + 10)
        print(f"Foglio '{predictions_sheet_name}' creato.")
    
    header = ["Timestamp Previsione"] + [f"Previsto: {col}" for col in config["target_columns"]]
    worksheet.append_row(header, value_input_option='USER_ENTERED')
    
    rows_to_append = []
    prediction_start_time = config["_prediction_start_time"]
    for i in range(predictions_np.shape[0]):
        timestamp = prediction_start_time + timedelta(minutes=30 * (i + 1))
        row = [timestamp.strftime('%d/%m/%Y %H:%M')]
        row.extend([f"{val:.3f}".replace('.', ',') for val in predictions_np[i, :]])
        rows_to_append.append(row)
        
        if i < 3:
            print(f"Riga {i+1}: {row}")
    
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Salvate {len(rows_to_append)} righe di previsione.")

def main():
    print(f"AVVIO SCRIPT - {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Modello: {MODEL_BASE_NAME}")
    
    try:
        log_environment_info()
        
        # Le credenziali devono essere in un file 'credentials.json' o gestite come secrets
        credentials = Credentials.from_service_account_file(
            "credentials.json", 
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
