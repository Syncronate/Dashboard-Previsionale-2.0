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
from sklearn.preprocessing import MinMaxScaler
import traceback

# --- Costanti ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_seq2seq_20250723_1608_posttrained_20250723_1615_posttrained_20250723_1628"
GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_HISTORICAL_DATA_SHEET_NAME = "Dati Meteo Stazioni"
GSHEET_DATE_COL_INPUT = 'Data_Ora'
GSHEET_RAIN_FORECAST_SHEET_NAME = "Previsioni Cumulate"
GSHEET_PREDICTIONS_SHEET_NAME = "Previsioni Modello seq2seq 6 ore"
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
GSHEET_FORECAST_DATE_COL = 'Timestamp'
GSHEET_FORECAST_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
HUMIDITY_COL_MODEL_NAME = "Umidita' Sensore 3452 (Montemurello)"
FORECAST_HUMIDITY_SHEET_COL = "Media Umidità Suolo (4 Stazioni) (%)"
italy_tz = pytz.timezone('Europe/Rome')

# --- NUOVO: Costanti per la gestione dello stato (memoria del modello) ---
# Questa cartella sarà gestita dalla cache di GitHub Actions
STATE_DIR = "model_state"
HIDDEN_STATE_PATH = os.path.join(STATE_DIR, "hidden_state.pt")
CELL_STATE_PATH = os.path.join(STATE_DIR, "cell_state.pt")


# --- Definizione Modello Seq2Seq (invariata) ---
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
    # NOTA: La forward del modello completo non verrà usata direttamente in fase di assimilazione,
    # ma è utile per il training e la validazione. La logica di previsione userà encoder e decoder separatamente.
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

# --- NUOVO: Funzioni per salvare e caricare la memoria del modello ---
def save_model_state(hidden, cell):
    """Salva i tensori di stato (memoria) nella cartella gestita dalla cache."""
    os.makedirs(STATE_DIR, exist_ok=True)
    torch.save(hidden, HIDDEN_STATE_PATH)
    torch.save(cell, CELL_STATE_PATH)
    print(f"Stato del modello (hidden, cell) salvato in '{STATE_DIR}' per la prossima esecuzione.")

def load_model_state(device):
    """Carica i tensori di stato. Se non esistono (cache vuota), ritorna None."""
    if os.path.exists(HIDDEN_STATE_PATH) and os.path.exists(CELL_STATE_PATH):
        try:
            hidden = torch.load(HIDDEN_STATE_PATH, map_location=device)
            cell = torch.load(CELL_STATE_PATH, map_location=device)
            print("Cache trovata. Stato del modello precedente (memoria) caricato.")
            return hidden, cell
        except Exception as e:
            print(f"ATTENZIONE: Errore nel caricare i file di stato dalla cache. Si procederà a freddo. Errore: {e}")
            return None, None
    print("Cache non trovata o prima esecuzione. Si procederà con una inizializzazione a freddo (cold start).")
    return None, None

# --- Funzione `load_model_and_scalers` (invariata) ---
def load_model_and_scalers(model_base_name, models_dir):
    # ... il codice di questa funzione rimane identico ...
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
    output_window = config["output_window_steps"]
    all_forecast_inputs = config["forecast_input_columns"]
    rain_forecast_columns = [col for col in all_forecast_inputs if "Umidita" not in col]
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


# --- MODIFICATO: La funzione di fetch ora estrae anche l'ultima osservazione reale del target ---
def fetch_and_prepare_data(gc, sheet_id, config, column_mapping):
    # ... La logica iniziale per leggere i fogli rimane la stessa ...
    input_window_steps = config["input_window_steps"]
    output_window_steps = config["output_window_steps"]
    model_feature_columns = config["all_past_feature_columns"]
    all_forecast_inputs = config["forecast_input_columns"]
    
    sh = gc.open_by_key(sheet_id)
    historical_ws = sh.worksheet(GSHEET_HISTORICAL_DATA_SHEET_NAME)
    historical_values = historical_ws.get_all_values()
    if not historical_values or len(historical_values) < 2:
        raise ValueError(f"Foglio storico '{GSHEET_HISTORICAL_DATA_SHEET_NAME}' vuoto o con solo intestazione.")
    
    df_historical_raw = pd.DataFrame(historical_values[1:], columns=historical_values[0])
    df_historical = df_historical_raw.rename(columns=column_mapping)
    
    df_historical[GSHEET_DATE_COL_INPUT] = pd.to_datetime(df_historical[GSHEET_DATE_COL_INPUT], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
    df_historical = df_historical.dropna(subset=[GSHEET_DATE_COL_INPUT]).sort_values(by=GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = df_historical[GSHEET_DATE_COL_INPUT].iloc[-1]

    # Processamento colonne feature (invariato)
    for col in model_feature_columns:
        if col in df_historical.columns: df_historical[col] = pd.to_numeric(df_historical[col].astype(str).str.replace(',', '.'), errors='coerce')
        else: df_historical[col] = np.nan
    df_features_filled = df_historical[model_feature_columns].ffill().bfill().fillna(0)
    input_data_historical = df_features_filled.iloc[-input_window_steps:].values

    # --- AGGIUNTA CHIAVE: Estrarre l'ultima osservazione REALE della variabile target per l'assimilazione ---
    target_col_name = config["target_columns"][0] # Assumiamo di prevedere UN solo idrometro
    if target_col_name not in df_features_filled.columns:
        raise ValueError(f"La colonna target '{target_col_name}' non è stata trovata nei dati storici. Controllare 'target_columns' nel config e il `column_mapping`.")
    
    last_real_value = df_features_filled[target_col_name].iloc[-1]
    # Lo scaler si aspetta un array 2D, quindi usiamo [[...]]
    latest_target_observation_np = np.array([[last_real_value]])
    print(f"Ultima osservazione reale per '{target_col_name}' (usata per assimilazione): {last_real_value}")

    # ... La logica per le previsioni di pioggia/umidità rimane la stessa ...
    forecast_ws = sh.worksheet(GSHEET_RAIN_FORECAST_SHEET_NAME)
    forecast_values = forecast_ws.get_all_values()
    df_forecast_raw = pd.DataFrame(forecast_values[1:], columns=forecast_values[0])
    if FORECAST_HUMIDITY_SHEET_COL in df_forecast_raw.columns:
        df_forecast_raw = df_forecast_raw.rename(columns={FORECAST_HUMIDITY_SHEET_COL: HUMIDITY_COL_MODEL_NAME})
    else:
        raise ValueError(f"La colonna '{FORECAST_HUMIDITY_SHEET_COL}' non è trovata in '{GSHEET_RAIN_FORECAST_SHEET_NAME}'.")
    df_forecast_raw[GSHEET_FORECAST_DATE_COL] = pd.to_datetime(df_forecast_raw[GSHEET_FORECAST_DATE_COL], format=GSHEET_FORECAST_DATE_FORMAT, errors='coerce')
    future_forecasts = df_forecast_raw[df_forecast_raw[GSHEET_FORECAST_DATE_COL] > latest_valid_timestamp].copy()
    if len(future_forecasts) < output_window_steps:
        raise ValueError(f"Previsioni future insufficienti ({len(future_forecasts)} righe), richiesti {output_window_steps}.")
    future_data = future_forecasts.head(output_window_steps)
    for col in all_forecast_inputs:
         if col in future_data.columns: 
             future_data[col] = pd.to_numeric(future_data[col].astype(str).str.replace(',', '.'), errors='coerce')
         else:
             raise ValueError(f"Colonna di forecast '{col}' non trovata.")
    input_data_forecast = future_data[all_forecast_inputs].ffill().bfill().fillna(0).values
    
    print(f"Dati storici e di previsione processati. Ultimo timestamp storico: {latest_valid_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Restituisce anche la nuova informazione
    return input_data_historical, input_data_forecast, latest_valid_timestamp, latest_target_observation_np

# --- NUOVO: Funzione che orchestra l'intero ciclo di previsione con assimilazione ---
def run_prediction_cycle(model, scalers, config, data_inputs, device):
    """
    Orchestra il ciclo completo: carica stato, esegue previsione con assimilazione, salva nuovo stato.
    """
    # Unpack degli scaler e dei dati per chiarezza
    scaler_past_features, scaler_targets, scaler_forecast_features = scalers
    historical_data_np, forecast_data_np, latest_real_target_obs = data_inputs

    # 1. Tenta di caricare la memoria (stato) dall'esecuzione precedente
    initial_hidden, initial_cell = load_model_state(device)

    # 2. Se non c'è memoria (cold start), la crea usando l'encoder sui dati storici
    if initial_hidden is None:
        print("Esecuzione ENCODER per creare lo stato iniziale.")
        historical_normalized = scaler_past_features.transform(historical_data_np)
        historical_tensor = torch.FloatTensor(historical_normalized).unsqueeze(0).to(device)
        with torch.no_grad():
            initial_hidden, initial_cell = model.encoder(historical_tensor)

    # 3. Prepara le previsioni di pioggia per il decoder (logica invariata)
    all_forecast_columns = config["forecast_input_columns"]
    rain_only_columns = [col for col in all_forecast_columns if "Umidita" not in col]
    forecast_df = pd.DataFrame(forecast_data_np, columns=all_forecast_columns)
    forecast_scaled_full = scaler_forecast_features.transform(forecast_df.values)
    final_scaled_forecast_df = pd.DataFrame(forecast_scaled_full, columns=all_forecast_columns)
    rain_forecast_tensor = torch.FloatTensor(final_scaled_forecast_df[rain_only_columns].values).unsqueeze(0).to(device)
    
    # 4. Esegui la previsione con il ciclo del DECODER
    predictions_np = np.zeros((model.target_len, model.decoder.output_size))
    
    with torch.no_grad():
        # --- ASSIMILAZIONE DATI ---
        # Usa l'osservazione reale (scalata) come PRIMO input per il decoder per "ancorare" la previsione
        decoder_input_scaled = scaler_targets.transform(latest_real_target_obs)
        decoder_input = torch.FloatTensor(decoder_input_scaled).unsqueeze(0).to(device)
        
        # Inizializza il decoder con lo stato (o appena creato o caricato dalla cache)
        hidden, cell = initial_hidden, initial_cell
        
        for t in range(model.target_len):
            rain_forecast_step = rain_forecast_tensor[:, t:t+1, :]
            
            # Esegui un passo del decoder
            output, hidden, cell = model.decoder(decoder_input, rain_forecast_step, hidden, cell)
            
            # Salva lo stato *dopo* il primo passo. Questa è la nuova memoria aggiornata con la realtà.
            if t == 0:
                final_hidden_to_save = hidden.detach().clone()
                final_cell_to_save = cell.detach().clone()

            # Salva la previsione fatta
            predictions_np[t] = output.cpu().numpy().squeeze(0)
            
            # L'input per il passo successivo è l'output di questo passo (teacher forcing disabilitato per l'inferenza)
            decoder_input = output

    # 5. Salva la nuova memoria per la prossima esecuzione
    save_model_state(final_hidden_to_save, final_cell_to_save)

    # 6. De-normalizza le previsioni per avere i valori reali
    predictions_scaled_back = scaler_targets.inverse_transform(predictions_np)
    return predictions_scaled_back

# --- Funzione `append_predictions_to_gsheet` (invariata) ---
def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, config):
    # ... il codice di questa funzione rimane identico ...
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


# --- MODIFICATO: Flusso principale semplificato per usare le nuove funzioni ---
def main():
    print(f"Avvio script di previsione con Assimilazione Dati alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if not GSHEET_ID:
        print("Errore: la variabile d'ambiente GSHEET_ID non è impostata.")
        return
    try:
        # 1. Setup iniziale (autenticazione, caricamento modello e scaler)
        credentials = Credentials.from_service_account_file("credentials.json", scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        print("Autenticazione a Google Sheets riuscita.")
        
        model, scaler_past, scaler_target, scaler_forecast, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)
        
        column_mapping = {
            'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', 'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
            'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)', 'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
            'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_MODEL_NAME,
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
        }
        
        # 2. Fetch di tutti i dati necessari, inclusa l'ultima osservazione reale
        hist_data, fcst_data, last_ts, real_obs = fetch_and_prepare_data(gc, GSHEET_ID, config, column_mapping)
        config["_prediction_start_time"] = last_ts

        # 3. Esegui il ciclo di previsione completo
        scalers = (scaler_past, scaler_target, scaler_forecast)
        data_inputs = (hist_data, fcst_data, real_obs)
        predictions = run_prediction_cycle(model, scalers, config, data_inputs, device)
        
        print(f"Previsioni generate con shape: {predictions.shape}")
        
        # 4. Scrivi i risultati su Google Sheets
        append_predictions_to_gsheet(gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions, config)
        print(f"Script completato con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except Exception as e:
        print(f"ERRORE CRITICO DURANTE L'ESECUZIONE: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
