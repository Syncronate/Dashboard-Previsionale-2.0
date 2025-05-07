import os
import json
# import base64 # Non più necessario per le credenziali
from datetime import datetime, timedelta
import pytz

import gspread
from google.oauth2.service_account import Credentials # Manteniamo questo per il tipo
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Costanti (adattale se necessario) ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_lstm_20250507_0717"
GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_DATA_SHEET_NAME = os.environ.get("GSHEET_DATA_SHEET_NAME", "Sheet1")
GSHEET_PREDICTIONS_SHEET_NAME = os.environ.get("GSHEET_PREDICTIONS_SHEET_NAME", "Previsioni")
GSHEET_DATE_COL_INPUT = 'Data_Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
HUMIDITY_COL_NAME_INPUT = "Umidita' Sensore 3452 (Montemurello)"

italy_tz = pytz.timezone('Europe/Rome')

# --- Definizione Modello LSTM (invariata) ---
class HydroLSTM(nn.Module):
    # ... (codice del modello come prima) ...
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(out.size(0), self.output_window, self.output_size)
        return out

# --- Funzioni Utilità (load_model_and_scalers, fetch_input_data_from_gsheet, predict_with_model, append_predictions_to_gsheet - invariate rispetto alla mia risposta precedente, ma verifica il mapping colonne in fetch_input_data) ---
# ... (copia queste funzioni dalla mia risposta precedente, assicurandoti che siano corrette) ...
def load_model_and_scalers(model_base_name, models_dir):
    """Carica il modello, la configurazione e gli scaler."""
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_features_path = os.path.join(models_dir, f"{model_base_name}_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")

    if not all(os.path.exists(p) for p in [config_path, model_path, scaler_features_path, scaler_targets_path]):
        raise FileNotFoundError(f"Uno o più file per il modello '{model_base_name}' non trovati in '{models_dir}'.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if "feature_columns" not in config:
        # !! IMPORTANTE: Adatta questa lista alle feature ESATTE del tuo modello !!
        config["feature_columns"] = [ 
            'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
            'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
            HUMIDITY_COL_NAME_INPUT, 
            'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
        ]
        print(f"Warning: 'feature_columns' non trovate nella config. Usate quelle definite nello script: {config['feature_columns']}")

    input_size = len(config["feature_columns"])
    output_size = len(config["target_columns"])
    
    model = HydroLSTM(
        input_size,
        config["hidden_size"],
        output_size,
        config["output_window"],
        config["num_layers"],
        config["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler_features = joblib.load(scaler_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    
    print(f"Modello '{model_base_name}' e scaler caricati su {device}.")
    return model, scaler_features, scaler_targets, config, device

def fetch_input_data_from_gsheet(gc, sheet_id, data_sheet_name, config, column_mapping):
    """Recupera e prepara i dati di input dal Google Sheet."""
    input_window_steps = config["input_window"]
    model_feature_columns = config["feature_columns"]

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.worksheet(data_sheet_name)
    
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        raise ValueError(f"Foglio Google '{data_sheet_name}' vuoto o con solo intestazione.")

    headers_gsheet = all_values[0]
    num_rows_to_fetch = input_window_steps + 20 
    start_index = max(1, len(all_values) - num_rows_to_fetch)
    data_rows = all_values[start_index:]
    df_gsheet_raw = pd.DataFrame(data_rows, columns=headers_gsheet)

    # Nomi colonne GSheet che ci servono dal mapping (devono esistere nel GSheet)
    required_gsheet_cols_from_mapping = [col for col in column_mapping.keys() if col != GSHEET_DATE_COL_INPUT] # Escludi data per ora
    
    # Aggiungi colonna data se non già considerata come una feature
    cols_to_select_in_gsheet = []
    if GSHEET_DATE_COL_INPUT in headers_gsheet:
        cols_to_select_in_gsheet.append(GSHEET_DATE_COL_INPUT)
    else:
        raise ValueError(f"Colonna data input '{GSHEET_DATE_COL_INPUT}' non trovata nel foglio GSheet '{data_sheet_name}'.")

    for gsheet_col_name in required_gsheet_cols_from_mapping:
        if gsheet_col_name in headers_gsheet:
            cols_to_select_in_gsheet.append(gsheet_col_name)
        else:
            print(f"Attenzione: Colonna GSheet '{gsheet_col_name}' specificata nel mapping ma non trovata nel foglio '{data_sheet_name}'.")
    
    df_subset = df_gsheet_raw[list(set(cols_to_select_in_gsheet))].copy() # set per evitare duplicati
    df_mapped = df_subset.rename(columns=column_mapping)
    date_col_model_name = column_mapping.get(GSHEET_DATE_COL_INPUT, GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = None

    for col in df_mapped.columns:
        if col == date_col_model_name:
            try:
                df_mapped[col] = pd.to_datetime(df_mapped[col], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
                if not df_mapped[col].empty and df_mapped[col].notna().any():
                    if df_mapped[col].dt.tz is None:
                        df_mapped[col] = df_mapped[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                    else:
                        df_mapped[col] = df_mapped[col].dt.tz_convert(italy_tz)
            except Exception as e_date:
                print(f"Errore conversione data per colonna '{col}': {e_date}. Sarà NaT.")
                df_mapped[col] = pd.NaT
        elif col in model_feature_columns:
            try:
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]):
                    col_str = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_num:
                print(f"Errore conversione numerica per colonna '{col}': {e_num}. Sarà NaN.")
                df_mapped[col] = np.nan
    
    if date_col_model_name in df_mapped.columns and pd.api.types.is_datetime64_any_dtype(df_mapped[date_col_model_name]):
        df_mapped = df_mapped.sort_values(by=date_col_model_name)
        valid_dates = df_mapped[date_col_model_name].dropna()
        if not valid_dates.empty:
            latest_valid_timestamp = valid_dates.iloc[-1]
    else:
        raise ValueError(f"Colonna data '{date_col_model_name}' non trovata o non valida dopo mappatura/conversione.")

    df_features_selected = pd.DataFrame() # Inizia vuoto
    for m_col in model_feature_columns:
        if m_col in df_mapped.columns:
            df_features_selected[m_col] = df_mapped[m_col]
        else:
            print(f"Attenzione: Colonna feature modello '{m_col}' non presente in df_mapped. Sarà riempita con NaN.")
            df_features_selected[m_col] = np.nan 

    df_features_filled = df_features_selected[model_feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0) # Assicura ordine e fill
    
    if len(df_features_filled) < input_window_steps:
        raise ValueError(f"Dati insufficienti nel GSheet ({len(df_features_filled)} righe valide) per l'input del modello (richiesti {input_window_steps}).")
    
    input_data = df_features_filled.iloc[-input_window_steps:]
    
    for col in input_data.columns:
        if not pd.api.types.is_numeric_dtype(input_data[col]):
            raise TypeError(f"La colonna '{col}' nei dati di input finali non è numerica (tipo: {input_data[col].dtype}).")

    print(f"Dati di input recuperati e processati. Shape: {input_data.shape}")
    print(f"Ultimo timestamp valido usato per i dati di input: {latest_valid_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if latest_valid_timestamp else 'N/A'}")
    return input_data.values, latest_valid_timestamp

def predict_with_model(model, input_data_np, scaler_features, scaler_targets, device):
    """Esegue la previsione."""
    input_normalized = scaler_features.transform(input_data_np)
    input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_np = output_tensor.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(output_np)
    return predictions_scaled_back

def append_predictions_to_gsheet(gc, sheet_id, predictions_sheet_name, predictions_np, target_columns, prediction_start_time, config):
    """Aggiunge le previsioni a un nuovo foglio Google."""
    sh = gc.open_by_key(sheet_id)
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        print(f"Foglio '{predictions_sheet_name}' trovato.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Foglio '{predictions_sheet_name}' non trovato. Creazione in corso...")
        # Determina l'header in base a target_columns
        header_parts_targets = [f"Previsto: {target_col.split('[')[0].strip()}" for target_col in target_columns]
        header_row = ["Timestamp Esecuzione", "Timestamp Inizio Serie", "Passo (Relativo)"] + header_parts_targets
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows="1", cols=len(header_row) + 5) 
        worksheet.append_row(header_row, value_input_option='USER_ENTERED') # USER_ENTERED per formattazione corretta
        print(f"Foglio '{predictions_sheet_name}' creato con intestazione.")

    output_window_steps = config["output_window"]
    timestamp_esecuzione_dt = datetime.now(italy_tz)
    timestamp_esecuzione_str = timestamp_esecuzione_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Valore di fallback per prediction_start_time se è None
    if prediction_start_time is None:
        print("Attenzione: prediction_start_time è None. Userò il timestamp di esecuzione meno la durata dell'input window come stima.")
        input_duration_minutes = config["input_window"] * 30 # Assumendo step da 30 min
        prediction_start_time = timestamp_esecuzione_dt - timedelta(minutes=input_duration_minutes)
    
    prediction_start_time_str = prediction_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    rows_to_append = []
    for step_idx in range(output_window_steps):
        current_prediction_time_dt = prediction_start_time + timedelta(minutes=30 * (step_idx + 1))
        # current_prediction_time_str = current_prediction_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z') # Non più necessario per cella singola
        
        row_data_for_step = [
            timestamp_esecuzione_str,
            prediction_start_time_str, # Timestamp di inizio della serie di input che ha generato questa previsione
            f"T+{ (step_idx + 1) * 0.5 :.1f}h (per {current_prediction_time_dt.strftime('%H:%M %d/%m')})" 
        ]
        for target_idx in range(predictions_np.shape[1]): # Itera sui target
            row_data_for_step.append(f"{predictions_np[step_idx, target_idx]:.3f}")
        rows_to_append.append(row_data_for_step)
    
    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Aggiunte {len(rows_to_append)} righe di previsione al foglio '{predictions_sheet_name}'.")


def main():
    print(f"Avvio simulazione script alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if not GSHEET_ID:
        print("Errore: GSHEET_ID non impostato come variabile d'ambiente.")
        return

    try:
        # --- MODIFICA AUTENTICAZIONE ---
        # Lo script GitHub Actions creerà un file 'credentials.json'
        # nella directory di lavoro del workflow.
        credentials_path = "credentials.json" 
        if not os.path.exists(credentials_path):
            # Se si esegue localmente e si vuole usare il Base64 per test, si può aggiungere un fallback
            gcp_sa_key_b64 = os.environ.get("GCP_SA_KEY_BASE64_FALLBACK") # Nome diverso per fallback locale
            if gcp_sa_key_b64:
                print("Uso GCP_SA_KEY_BASE64_FALLBACK per le credenziali (test locale).")
                credentials_json_str = base64.b64decode(gcp_sa_key_b64).decode('utf-8')
                credentials_dict = json.loads(credentials_json_str)
                credentials = Credentials.from_service_account_info(credentials_dict,
                    scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
            else:
                 raise FileNotFoundError(f"File '{credentials_path}' non trovato e GCP_SA_KEY_BASE64_FALLBACK non impostato.")
        else: # Metodo preferito per GitHub Actions
            credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            )
        gc = gspread.authorize(credentials)
        print("Autenticazione Google Sheets riuscita.")
        # --- FINE MODIFICA AUTENTICAZIONE ---

        model, scaler_features, scaler_targets, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)

        # !! IMPORTANTE: Rivedi e adatta questo mapping !!
        column_mapping_gsheet_to_model = {
            'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
            'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
            'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
            'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
            HUMIDITY_COL_NAME_INPUT: HUMIDITY_COL_NAME_INPUT,
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
            GSHEET_DATE_COL_INPUT: GSHEET_DATE_COL_INPUT 
        }
        
        missing_features_in_mapping = []
        for fc in config["feature_columns"]:
            is_mapped = False
            for model_name_in_map_val in column_mapping_gsheet_to_model.values():
                if fc == model_name_in_map_val:
                    is_mapped = True
                    break
            if not is_mapped:
                missing_features_in_mapping.append(fc)
        
        if missing_features_in_mapping:
            print(f"ATTENZIONE CRITICA: Le seguenti feature del modello NON sono mappate da alcuna colonna GSheet nel 'column_mapping_gsheet_to_model': {missing_features_in_mapping}")
            print("Queste colonne saranno riempite con NaN e poi con 0, il che potrebbe portare a previsioni errate.")
            # raise ValueError(f"Feature modello non mappate: {missing_features_in_mapping}") # Scommenta per far fallire


        input_data_np, last_input_timestamp = fetch_input_data_from_gsheet(
            gc, GSHEET_ID, GSHEET_DATA_SHEET_NAME, config, column_mapping_gsheet_to_model
        )

        predictions_np = predict_with_model(model, input_data_np, scaler_features, scaler_targets, device)
        print(f"Previsioni generate. Shape: {predictions_np.shape}")
        
        prediction_start_for_series = last_input_timestamp 
        
        append_predictions_to_gsheet(
            gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions_np, 
            config["target_columns"], prediction_start_for_series, config
        )

        print(f"Simulazione completata con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except FileNotFoundError as e:
        print(f"Errore File: {e}")
    except ValueError as e:
        print(f"Errore Valore: {e}")
    except TypeError as e:
        print(f"Errore Tipo: {e}")
    except gspread.exceptions.APIError as e:
        print(f"Errore API Google Sheets: {e}")
    except Exception as e:
        print(f"Errore imprevisto durante la simulazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
