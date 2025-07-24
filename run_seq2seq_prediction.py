import os
import json
from datetime import datetime, timedelta
import pytz
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Costanti ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20250723_1608_posttrained_20250723_1615_posttrained_20250723_1628"
GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "Previsioni Idrometri"
GSHEET_RAIN_FORECAST_SHEET_NAME = "Previsioni Cumulate"
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Modello seq2seq 6 ore"
GSHEET_DATE_COL_INPUT = 'Data_Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
GSHEET_FORECAST_DATE_COL = 'Timestamp'
GSHEET_FORECAST_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

italy_tz = pytz.timezone('Europe/Rome')

# --- Definizione Modello Seq2Seq ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, rain_forecast_size, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(output_size + rain_forecast_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, rain_forecast, hidden, cell):
        decoder_input = torch.cat((x, rain_forecast), dim=2)
        output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class HydroSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, target_len=12):
        super(HydroSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len
        self.device = device
    def forward(self, src, rain_forecasts):
        batch_size = src.shape[0]
        target_len = self.target_len
        decoder_output_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, target_len, decoder_output_size).to(self.device)
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, decoder_output_size).to(self.device)
        for t in range(target_len):
            rain_forecast_step = rain_forecasts[:, t:t+1, :]
            output, hidden, cell = self.decoder(decoder_input, rain_forecast_step, hidden, cell)
            outputs[:, t:t+1, :] = output
            decoder_input = output
        return outputs

# --- Funzioni Utilità ---
def load_model_and_scalers(model_base_name, models_dir):
    """Carica il modello, la configurazione e gli scaler."""
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_past_features_path = os.path.join(models_dir, f"{model_base_name}_past_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")
    scaler_forecast_features_path = os.path.join(models_dir, f"{model_base_name}_forecast_features.joblib")
    
    required_files = [config_path, model_path, scaler_past_features_path, scaler_targets_path, scaler_forecast_features_path]
    if not all(os.path.exists(p) for p in required_files):
        for p in required_files:
            if not os.path.exists(p): print(f"ERRORE CRITICO: File non trovato -> {p}")
        raise FileNotFoundError(f"Uno o più file per il modello '{model_base_name}' non trovati in '{models_dir}'.")

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_columns = config["all_past_feature_columns"]
    target_columns = config["target_columns"]
    rain_forecast_columns = config["forecast_input_columns"]
    output_window = config["output_window_steps"]

    encoder = Encoder(input_size=len(feature_columns), hidden_size=config["hidden_size"], num_layers=config["num_layers"], dropout=config["dropout"])
    decoder = Decoder(output_size=len(target_columns), hidden_size=config["hidden_size"], rain_forecast_size=len(rain_forecast_columns), num_layers=config["num_layers"], dropout=config["dropout"])
    model = HydroSeq2Seq(encoder, decoder, device, output_window).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)

    print(f"Modello '{model_base_name}' e scaler caricati con successo su {device}.")
    return model, scaler_past_features, scaler_targets, scaler_forecast_features, config, device


def fetch_and_prepare_data(gc, sheet_id, config, column_mapping):
    """Recupera e prepara i dati storici e di previsione delle piogge."""
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    model_feature_columns = config["all_past_feature_columns"]
    rain_forecast_columns = config["forecast_input_columns"]

    sh = gc.open_by_key(sheet_id)
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    historical_values = historical_ws.get_all_values()
    if not historical_values or len(historical_values) < 2:
        raise ValueError(f"Foglio storico '{GSHEET_HISTORICAL_DATA_SHEET_NAME}' vuoto o con solo intestazione.")
    
    historical_headers = historical_values[0]
    num_rows_to_fetch = input_window_steps + 20
    start_index = max(1, len(historical_values) - num_rows_to_fetch)
    historical_data_rows = historical_values[start_index:]
    df_historical_raw = pd.DataFrame(historical_data_rows, columns=historical_headers)

    rain_forecast_ws = sh.worksheet(GSHEET_RAIN_FORECAST_SHEET_NAME)
    rain_forecast_values = rain_forecast_ws.get_all_values()
    if not rain_forecast_values or len(rain_forecast_values) < 2:
        raise ValueError(f"Foglio previsioni pioggia '{GSHEET_RAIN_FORECAST_SHEET_NAME}' vuoto o con solo intestazione.")
    
    df_rain_forecast_raw = pd.DataFrame(rain_forecast_values[1:], columns=rain_forecast_values[0])
    df_historical = df_historical_raw.rename(columns=column_mapping)
    date_col_model_name = column_mapping.get(GSHEET_DATE_COL_INPUT, GSHEET_DATE_COL_INPUT)
    
    df_historical[date_col_model_name] = pd.to_datetime(df_historical[date_col_model_name], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
    df_historical = df_historical.dropna(subset=[date_col_model_name])
    df_historical = df_historical.sort_values(by=date_col_model_name)
    latest_valid_timestamp = df_historical[date_col_model_name].iloc[-1]

    for col in model_feature_columns:
        if col in df_historical.columns: df_historical[col] = pd.to_numeric(df_historical[col].astype(str).str.replace(',', '.'), errors='coerce')
        else: df_historical[col] = np.nan

    df_features_filled = df_historical[model_feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
    if len(df_features_filled) < input_window_steps:
        raise ValueError(f"Dati storici insufficienti ({len(df_features_filled)} righe) per l'input (richiesti {input_window_steps}).")
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values

    df_rain_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(df_rain_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce')
    future_rain_forecasts = df_rain_forecast_raw[df_rain_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
    if len(future_rain_forecasts) < output_window_steps:
        raise ValueError(f"Previsioni di pioggia insufficienti ({len(future_rain_forecasts)} righe) per l'output (richiesti {output_window_steps}).")
    future_rain_data = future_rain_forecasts.head(output_window_steps)
    for col in rain_forecast_columns:
         if col in future_rain_data.columns: future_rain_data[col] = pd.to_numeric(future_rain_data[col].astype(str).str.replace(',', '.'), errors='coerce')
         else: future_rain_data[col] = np.nan
    input_data_rain_forecast = future_rain_data[rain_forecast_columns].fillna(0).values
    print(f"Dati storici e di previsione pioggia processati. Ultimo timestamp usato: {latest_valid_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    return input_data_historical, input_data_rain_forecast, latest_valid_timestamp

def predict_with_model(model, historical_data_np, rain_forecast_np, scaler_past_features, scaler_targets, scaler_forecast_features, device):
    """Esegue la previsione con il modello Seq2Seq."""
    historical_normalized = scaler_past_features.transform(historical_data_np)
    rain_forecast_normalized = scaler_forecast_features.transform(rain_forecast_np)
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    rain_forecast_tensor = torch.FloatTensor(rain_forecast_normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(historical_tensor, rain_forecast_tensor)
    output_np = output_tensor.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(output_np)
    return predictions_scaled_back

def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    """Aggiunge le previsioni a un foglio Google."""
    sh = gc.open_by_key(sheet_id_str)
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        worksheet.clear()
        print(f"Foglio '{predictions_sheet_name}' trovato e pulito.")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows=config["output_window_steps"] + 10, cols=len(config["target_columns"]) + 10)
        print(f"Foglio '{predictions_sheet_name}' creato.")
    header = ["Timestamp Previsione"] + [f"Previsto: {col}" for col in config["target_columns"]]
    worksheet.append_row(header, value_input_option='USER_ENTERED')
    rows_to_append = []
    prediction_start_time = config["_prediction_start_time"]
    for i, step in enumerate(range(predictions_np.shape[0])):
        timestamp = prediction_start_time + timedelta(minutes=30 * (i + 1))
        row = [timestamp.strftime('%d/%m/%Y %H:%M')]
        row.extend([f"{val:.3f}".replace('.', ',') for val in predictions_np[step, :]])
        rows_to_append.append(row)
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Aggiunte {len(rows_to_append)} righe di previsione.")

def main():
    """Funzione principale."""
    print(f"Avvio script di previsione Seq2Seq alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if not GSHEET_ID:
        print("Errore: GSHEET_ID non è impostato.")
        return
    try:
        credentials = Credentials.from_service_account_file("credentials.json", scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        print("Autenticazione a Google Sheets riuscita.")
        model, scaler_past_features, scaler_targets, scaler_forecast_features, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        column_mapping = {
            'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
            'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
            'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
            'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
            'Umidita\' Sensore 3452 (Montemurello)': 'Umidita\' Sensore 3452 (Montemurello)',
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
            GSHEET_DATE_COL_INPUT: GSHEET_DATE_COL_INPUT
        }

        historical_data, rain_forecast_data, last_input_timestamp = fetch_and_prepare_data(gc, GSHEET_ID, config, column_mapping)
        
        config["_prediction_start_time"] = last_input_timestamp

        predictions = predict_with_model(model, historical_data, rain_forecast_data, scaler_past_features, scaler_targets, scaler_forecast_features, device)
        print(f"Previsioni generate con shape: {predictions.shape}")
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        print(f"Script completato con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
