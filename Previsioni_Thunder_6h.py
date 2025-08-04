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

# --- NOTA: Le classi del modello (EncoderLSTM, Attention, etc.) sono identiche e vengono omesse per brevità ---
# --- Sono necessarie e devono rimanere nello script. ---
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

# --- Costanti Aggiornate per il NUOVO modello ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20250803_1711"

GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "DATI METEO CON FEATURE"
GSHEET_FORECAST_DATA_SHEET_NAME = "Previsioni Cumulate Feature ICON"
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Idro-Bettolelle 6h"

GSHEET_DATE_COL_INPUT = 'Data e Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
GSHEET_FORECAST_DATE_COL = 'Data e Ora'
GSHEET_FORECAST_DATE_FORMAT = '%d/%m/%Y %H:%M'
italy_tz = pytz.timezone('Europe/Rome')


def load_model_and_scalers(model_base_name, models_dir):
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_past_features_path = os.path.join(models_dir, f"{model_base_name}_past_features.joblib")
    scaler_forecast_features_path = os.path.join(models_dir, f"{model_base_name}_forecast_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")
    required_files = [config_path, model_path, scaler_past_features_path, scaler_forecast_features_path, scaler_targets_path]
    for p in required_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"ERRORE CRITICO: File non trovato -> {p}")
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get("model_type")
    if model_type != "Seq2SeqAttention":
        raise ValueError(f"Tipo di modello non supportato da questo script: '{model_type}'. Richiesto 'Seq2SeqAttention'.")
    enc_input_size = len(config["all_past_feature_columns"])
    dec_input_size = len(config["forecast_input_columns"])
    dec_output_size = len(config["target_columns"])
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    drop = config["dropout"]
    out_win = config["output_window_steps"]
    encoder = EncoderLSTM(enc_input_size, hidden, layers, drop)
    decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop)
    model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
    print(f"Architettura modello '{model_type}' istanziata correttamente.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Modello '{model_base_name}' caricato con successo su {device}.")
    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    print("Scaler caricati con successo.")
    return model, scaler_past_features, scaler_forecast_features, scaler_targets, config, device


def fetch_and_prepare_data(gc, sheet_id, config):
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    forecast_feature_columns = config["forecast_input_columns"]
    
    sh = gc.open_by_key(sheet_id)
    
    print(f"Caricamento dati storici da: '{GSHEET_HISTORICAL_DATA_SHEET_NAME}'")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    df_historical_raw = pd.DataFrame(historical_ws.get_all_records())
    
    print(f"Caricamento dati previsionali da: '{GSHEET_FORECAST_DATA_SHEET_NAME}'")
    forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
    df_forecast_raw = pd.DataFrame(forecast_ws.get_all_records())

    ### AGGIUNTO: Mappatura dei nomi delle colonne dal Google Sheet al nome atteso dal modello ###
    # Dizionario: {"Nome Colonna nel Google Sheet": "Nome Colonna atteso dal modello"}
    column_mapping = {
        "Cumulata Sensore 1295 (Arcevia)_cumulata_30min": "Cumulata Sensore 1295 (Arcevia)",
        "Cumulata Sensore 2637 (Bettolelle)_cumulata_30min": "Cumulata Sensore 2637 (Bettolelle)",
        "Cumulata Sensore 2858 (Barbara)_cumulata_30min": "Cumulata Sensore 2858 (Barbara)",
        "Cumulata Sensore 2964 (Corinaldo)_cumulata_30min": "Cumulata Sensore 2964 (Corinaldo)"
    }
    
    # Applica la rinomina a entrambi i DataFrame. Se una colonna non esiste, viene ignorata.
    df_historical_raw.rename(columns=column_mapping, inplace=True)
    df_forecast_raw.rename(columns=column_mapping, inplace=True)
    print("Mappatura nomi colonne applicata ai dati caricati.")
    ### FINE PARTE AGGIUNTA ###

    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(df_historical_raw[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    
    for col in past_feature_columns:
        if col not in df_historical.columns:
            # Ora questo errore ti dirà esattamente quale colonna attesa dal modello non è stata trovata
            # nemmeno dopo la mappatura, rendendo il debug più facile.
            raise ValueError(f"Colonna storica '{col}' non trovata nel foglio dopo la mappatura.")
        df_historical.loc[:, col] = pd.to_numeric(df_historical[col].astype(str).str.replace(',', '.'), errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values

    df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(df_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce')
    
    future_forecasts = df_forecast_raw[df_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
    if len(future_forecasts) < output_window_steps:
        raise ValueError(f"Previsioni future insufficienti ({len(future_forecasts)} righe), richieste {output_window_steps}.")
    
    future_data = future_forecasts.head(output_window_steps).copy()

    for col in forecast_feature_columns:
        if col not in future_data.columns:
            raise ValueError(f"Colonna previsionale '{col}' non trovata nel foglio dopo la mappatura.")
        future_data.loc[:, col] = pd.to_numeric(future_data[col].astype(str).str.replace(',', '.'), errors='coerce')

    input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
    
    print(f"Dati storici preparati con shape: {input_data_historical.shape}")
    print(f"Dati previsionali preparati con shape: {input_data_forecast.shape}")
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp

def make_prediction(model, scalers, config, data_inputs, device):
    scaler_past_features, scaler_forecast_features, scaler_targets = scalers
    historical_data_np, forecast_data_np = data_inputs
    historical_normalized = scaler_past_features.transform(historical_data_np)
    forecast_normalized = scaler_forecast_features.transform(forecast_data_np)
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions_normalized, _ = model(historical_tensor, forecast_tensor)
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)
    return predictions_scaled_back

def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
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
    for i in range(predictions_np.shape[0]):
        timestamp = prediction_start_time + timedelta(minutes=30 * (i + 1))
        row = [timestamp.strftime('%d/%m/%Y %H:%M')]
        row.extend([f"{val:.3f}".replace('.', ',') for val in predictions_np[i, :]])
        rows_to_append.append(row)
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Aggiunte {len(rows_to_append)} righe di previsione.")

def main():
    print(f"Avvio script di previsione (Modello: {MODEL_BASE_NAME}) alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    try:
        credentials = Credentials.from_service_account_file("credentials.json", scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        print("Autenticazione a Google Sheets riuscita.")
        
        model, scaler_past, scaler_forecast, scaler_target, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts

        scalers = (scaler_past, scaler_forecast, scaler_target)
        data_inputs = (hist_data, fcst_data)
        
        predictions = make_prediction(model, scalers, config, data_inputs, device)
        
        print(f"Previsioni generate con shape: {predictions.shape}")
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        print(f"Script completato con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except Exception as e:
        print(f"ERRORE CRITICO DURANTE L'ESECUZIONE: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()```
