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

# --- NOTA: Definizioni delle Classi di Modello ---
# In questa sezione dovrai inserire le definizioni Python del tuo nuovo modello.
# Il file di configurazione menziona "Seq2SeqAttention", quindi dovresti avere
# classi come Encoder, Attention, Decoder e la classe principale del modello.
#
# ESEMPIO (DA SOSTITUIRE CON IL TUO CODICE REALE):
# class Encoder(nn.Module):
#     # ... tuo codice ...
#
# class Attention(nn.Module):
#     # ... tuo codice ...
#
# class Decoder(nn.Module):
#     # ... tuo codice ...
#
# class Seq2SeqAttention(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#
#     def forward(self, src, future_inputs, teacher_forcing_ratio=0):
#         # La firma di questo metodo è un'ipotesi.
#         # Adattala alla tua implementazione reale.
#         # ... logica della forward pass ...
#         return outputs


# --- Costanti Aggiornate ---
MODELS_DIR = "models"
# NOTA: Nome del nuovo modello aggiornato.
MODEL_BASE_NAME = "modello_seq2seq_20250731_2004"
GSHEET_ID = os.environ.get("GSHEET_ID")
# NOTA: Nomi dei fogli di input aggiornati.
GSHEET_HISTORICAL_DATA_SHEET_NAME = "DATI METEO CON FEATURE"
GSHEET_FORECAST_DATA_SHEET_NAME = "Previsioni Cumulate Feature ICON"
# NOTA: Nome del foglio di output cambiato per chiarezza.
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Thunder-ICON 24h"

# Costanti per le date (presumibilmente invariate)
GSHEET_DATE_COL_INPUT = 'Data_Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
GSHEET_FORECAST_DATE_COL = 'Timestamp'
GSHEET_FORECAST_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
italy_tz = pytz.timezone('Europe/Rome')


# --- Funzione di Caricamento Modello Aggiornata ---
def load_model_and_scalers(model_base_name, models_dir):
    """
    Carica il modello, i file di configurazione e gli scaler associati.
    """
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_past_features_path = os.path.join(models_dir, f"{model_base_name}_past_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")
    # NOTA: Assumiamo che ci sia un solo scaler per le feature future, come nello script originale.
    # Se hai scaler separati, dovrai aggiornare questa logica.
    scaler_forecast_features_path = os.path.join(models_dir, f"{model_base_name}_forecast_features.joblib")

    required_files = [config_path, model_path, scaler_past_features_path, scaler_targets_path, scaler_forecast_features_path]
    for p in required_files:
        if not os.path.exists(p):
            print(f"ERRORE CRITICO: File non trovato -> {p}")
            raise FileNotFoundError(f"Uno o più file per il modello '{model_base_name}' non trovati.")

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # NOTA: L'istanziazione del modello dipende dalle tue classi.
    # Questo è un ESEMPIO e dovrai adattarlo.
    # Assumiamo che la tua classe principale si chiami 'Seq2SeqAttention'.
    # encoder = Encoder(input_dim=len(config['all_past_feature_columns']), ...)
    # attention = Attention(...)
    # decoder = Decoder(output_dim=len(config['target_columns']), ...)
    # model = Seq2SeqAttention(encoder, decoder, device).to(device)

    # Carica solo lo stato del modello. Prima devi avere istanziato l'architettura corretta.
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    
    # PER ORA, RESTITUIAMO 'None' PER IL MODELLO. DECOMMENTA LE RIGHE SOPRA E RIMUOVI 'model = None'
    # QUANDO AVRAI INSERITO LE CLASSI CORRETTE.
    model = None # <-- SOSTITUISCI QUESTO!
    if model is None:
        print("ATTENZIONE: Il modello non è stato istanziato. Inserire le classi corrette.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Modello '{model_base_name}' caricato con successo su {device}.")


    scaler_past_features = joblib.load(scaler_past_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    scaler_forecast_features = joblib.load(scaler_forecast_features_path)
    print("Scaler caricati con successo.")
    
    return model, scaler_past_features, scaler_targets, scaler_forecast_features, config, device


# --- Funzione di Preparazione Dati Aggiornata ---
def fetch_and_prepare_data(gc, sheet_id, config):
    """
    Recupera e prepara i dati storici e futuri dai nuovi fogli Google.
    """
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    past_feature_columns = config["all_past_feature_columns"]
    forecast_feature_columns = config["forecast_input_columns"]

    sh = gc.open_by_key(sheet_id)
    
    # Carica dati storici
    print(f"Caricamento dati storici dal foglio: '{GSHEET_HISTORICAL_DATA_SHEET_NAME}'")
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    df_historical_raw = pd.DataFrame(historical_ws.get_all_records())
    
    # NOTA: La conversione della data e la gestione dei numerici è cruciale
    df_historical_raw[GSHEET_DATE_COL_INPUT] = pd.to_datetime(df_historical_raw[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
    df_historical = df_historical_raw.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]
    
    for col in past_feature_columns:
        if col not in df_historical.columns:
            raise ValueError(f"Colonna storica '{col}' non trovata nel foglio '{GSHEET_HISTORICAL_DATA_SHEET_NAME}'.")
        df_historical[col] = pd.to_numeric(df_historical[col].astype(str).str.replace(',', '.'), errors='coerce')

    df_features_filled = df_historical[past_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values

    # Carica dati previsionali
    print(f"Caricamento dati previsionali dal foglio: '{GSHEET_FORECAST_DATA_SHEET_NAME}'")
    forecast_ws = sh.worksheet(GSHEET_FORECAST_DATA_SHEET_NAME)
    df_forecast_raw = pd.DataFrame(forecast_ws.get_all_records())
    df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(df_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce')
    
    future_forecasts = df_forecast_raw[df_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
    if len(future_forecasts) < output_window_steps:
        raise ValueError(f"Previsioni future insufficienti ({len(future_forecasts)} righe), richieste {output_window_steps}.")
    
    future_data = future_forecasts.head(output_window_steps)
    for col in forecast_feature_columns:
        if col not in future_data.columns:
            raise ValueError(f"Colonna previsionale '{col}' non trovata nel foglio '{GSHEET_FORECAST_DATA_SHEET_NAME}'.")
        future_data[col] = pd.to_numeric(future_data[col].astype(str).str.replace(',', '.'), errors='coerce')

    input_data_forecast = future_data[forecast_feature_columns].ffill().bfill().fillna(0).values
    
    print(f"Dati storici preparati con shape: {input_data_historical.shape}")
    print(f"Dati previsionali preparati con shape: {input_data_forecast.shape}")
    
    return input_data_historical, input_data_forecast, latest_valid_timestamp

# --- Funzione di Previsione Riscritta ---
def make_prediction(model, scalers, config, data_inputs, device):
    """
    Esegue la previsione utilizzando il nuovo modello.
    """
    scaler_past_features, scaler_targets, scaler_forecast_features = scalers
    historical_data_np, forecast_data_np = data_inputs

    # 1. Normalizza gli input
    historical_normalized = scaler_past_features.transform(historical_data_np)
    forecast_normalized = scaler_forecast_features.transform(forecast_data_np)

    # 2. Converti in tensori PyTorch
    historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
    forecast_tensor = torch.FloatTensor(forecast_normalized).unsqueeze(0).to(device)

    # 3. Esegui la previsione
    # NOTA: La chiamata al modello è un'IPOTESI. La firma del metodo .forward()
    # della tua classe di modello (es. Seq2SeqAttention) potrebbe essere diversa.
    # Adatta questa riga al tuo codice.
    with torch.no_grad():
        predictions_normalized = model(historical_tensor, forecast_tensor)

    # 4. De-normalizza l'output per ottenere i valori reali
    predictions_np = predictions_normalized.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)
    
    return predictions_scaled_back


# --- Funzione `append_predictions_to_gsheet` (invariata) ---
def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    # Questa funzione rimane invariata rispetto alla tua versione originale
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
    print(f"Avvio script di previsione (Modello: {MODEL_BASE_NAME}) alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    try:
        credentials = Credentials.from_service_account_file("credentials.json", scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        print("Autenticazione a Google Sheets riuscita.")
        
        model, scaler_past, scaler_target, scaler_forecast, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        # NOTA: Questo blocco è cruciale. Se il modello non è stato caricato, lo script si ferma.
        if model is None:
            raise RuntimeError("Istanziazione del modello non implementata in 'load_model_and_scalers'. Controllare il codice.")

        # NOTA: il mapping delle colonne è stato rimosso.
        hist_data, fcst_data, last_ts = fetch_and_prepare_data(gc, GSHEET_ID, config)
        config["_prediction_start_time"] = last_ts

        scalers = (scaler_past, scaler_target, scaler_forecast)
        data_inputs = (hist_data, fcst_data)
        
        predictions = make_prediction(model, scalers, config, data_inputs, device)
        
        print(f"Previsioni generate con shape: {predictions.shape}")
        
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        print(f"Script completato con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except Exception as e:
        print(f"ERRORE CRITICO DURANTE L'ESECUZIONE: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()