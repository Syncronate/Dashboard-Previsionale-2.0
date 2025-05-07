import os
import json
import base64
from datetime import datetime, timedelta
import pytz # Per fusi orari

import gspread
from google.oauth2.service_account import Credentials
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Costanti (adattale se necessario) ---
MODELS_DIR = "models"  # Assicurati che il percorso sia corretto rispetto alla root del repo
MODEL_BASE_NAME = "modello_lstm_20250507_0717"
GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_DATA_SHEET_NAME = os.environ.get("GSHEET_DATA_SHEET_NAME", "Sheet1") # Foglio da cui leggere i dati
GSHEET_PREDICTIONS_SHEET_NAME = os.environ.get("GSHEET_PREDICTIONS_SHEET_NAME", "Previsioni") # Foglio su cui scrivere
GSHEET_DATE_COL_INPUT = 'Data_Ora' # Colonna data nel foglio di input
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M' # Formato data nel foglio di input
HUMIDITY_COL_NAME_INPUT = "Umidita' Sensore 3452 (Montemurello)" # Assicurati corrisponda al GSheet

italy_tz = pytz.timezone('Europe/Rome')

# --- Definizione Modello LSTM (copia dalla tua app Streamlit) ---
class HydroLSTM(nn.Module):
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

# --- Funzioni Utilità (semplificate/adattate dalla tua app) ---

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
    
    # Assicurati che feature_columns sia nella config o definiscilo se necessario
    if "feature_columns" not in config:
        # Definisci qui le feature columns ESATTE usate per addestrare QUESTO modello
        # Questo è un punto CRITICO e deve corrispondere all'addestramento
        config["feature_columns"] = [ # Esempio, ADATTALO!
            'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
            'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
            HUMIDITY_COL_NAME_INPUT, # Usa la costante definita sopra se presente nel GSheet
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
    model_feature_columns = config["feature_columns"] # Feature che il modello si aspetta

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.worksheet(data_sheet_name)
    
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        raise ValueError(f"Foglio Google '{data_sheet_name}' vuoto o con solo intestazione.")

    headers_gsheet = all_values[0]
    
    # Recupera N righe in più per sicurezza e per il ffill/bfill
    num_rows_to_fetch = input_window_steps + 20 # Un po' di buffer
    start_index = max(1, len(all_values) - num_rows_to_fetch)
    data_rows = all_values[start_index:]
    
    df_gsheet_raw = pd.DataFrame(data_rows, columns=headers_gsheet)

    # Verifica che le colonne GSheet necessarie per il mapping siano presenti
    gsheet_cols_in_mapping = [col for col in column_mapping.keys() if col in headers_gsheet]
    if len(gsheet_cols_in_mapping) < len(column_mapping.keys()):
        missing_map_cols = list(set(column_mapping.keys()) - set(gsheet_cols_in_mapping))
        print(f"Attenzione: Colonne GSheet specificate nel mapping ma non trovate nel foglio: {missing_map_cols}")


    df_subset = df_gsheet_raw[gsheet_cols_in_mapping + ([GSHEET_DATE_COL_INPUT] if GSHEET_DATE_COL_INPUT not in gsheet_cols_in_mapping else [])].copy()
    df_mapped = df_subset.rename(columns=column_mapping)

    # Conversione e pulizia dati
    latest_valid_timestamp = None
    date_col_model_name = column_mapping.get(GSHEET_DATE_COL_INPUT, GSHEET_DATE_COL_INPUT)

    for col in df_mapped.columns:
        if col == date_col_model_name:
            try:
                df_mapped[col] = pd.to_datetime(df_mapped[col], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
                if df_mapped[col].dt.tz is None:
                    df_mapped[col] = df_mapped[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                else:
                    df_mapped[col] = df_mapped[col].dt.tz_convert(italy_tz)
            except Exception as e_date:
                print(f"Errore conversione data per colonna '{col}': {e_date}. Sarà NaT.")
                df_mapped[col] = pd.NaT
        elif col in model_feature_columns: # Pulisci solo le colonne feature del modello
            try:
                # Sostituisci virgola con punto per decimali e converti in numerico
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

    # Seleziona solo le feature richieste dal modello
    # e assicurati che siano nell'ordine corretto
    df_features_selected = pd.DataFrame(columns=model_feature_columns)
    for m_col in model_feature_columns:
        if m_col in df_mapped.columns:
            df_features_selected[m_col] = df_mapped[m_col]
        else:
            print(f"Attenzione: Colonna feature modello '{m_col}' non trovata dopo mappatura. Sarà riempita con NaN e poi con 0.")
            df_features_selected[m_col] = np.nan # Sarà gestita da fillna sotto

    # Gestione NaN (importante!)
    df_features_filled = df_features_selected.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Prendi gli ultimi 'input_window_steps' record
    if len(df_features_filled) < input_window_steps:
        raise ValueError(f"Dati insufficienti nel GSheet ({len(df_features_filled)} righe valide) per l'input del modello (richiesti {input_window_steps}).")
    
    input_data = df_features_filled.iloc[-input_window_steps:]
    
    # Verifica finale che tutte le colonne siano numeriche
    for col in input_data.columns:
        if not pd.api.types.is_numeric_dtype(input_data[col]):
            raise TypeError(f"La colonna '{col}' nei dati di input finali non è numerica (tipo: {input_data[col].dtype}). Controlla la mappatura e la pulizia.")

    print(f"Dati di input recuperati e processati. Shape: {input_data.shape}")
    print(f"Ultimo timestamp valido usato per i dati di input: {latest_valid_timestamp}")
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
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows="1", cols="20") # Inizia con poche colonne, si espanderà
        # Prepara l'intestazione
        header_row = ["Timestamp Esecuzione Previsione", "Ora Previsione (Inizio Serie)", "Passo Previsione (HH:MM)"]
        for target_col in target_columns:
            header_row.append(f"Previsto: {target_col}")
        worksheet.append_row(header_row)
        print(f"Foglio '{predictions_sheet_name}' creato con intestazione.")

    output_window_steps = config["output_window"]
    timestamp_esecuzione = datetime.now(italy_tz)

    # Ogni riga nel foglio "Previsioni" rappresenterà una serie completa di previsioni
    # fatte in un certo momento.
    # La prima colonna sarà il timestamp di quando la previsione è stata eseguita.
    # La seconda colonna sarà il timestamp di inizio della serie di previsioni.
    # Le colonne successive saranno le previsioni per ogni step.

    # Nuova logica: una riga per ogni esecuzione, con tutte le previsioni su quella riga
    row_to_append = [
        timestamp_esecuzione.strftime('%Y-%m-%d %H:%M:%S %Z'),
        prediction_start_time.strftime('%Y-%m-%d %H:%M:%S %Z') if prediction_start_time else "N/A"
    ]
    
    # Aggiungi una colonna per ogni passo temporale per ogni target
    # Esempio: Prev_Target1_T+0.5h, Prev_Target1_T+1.0h, ..., Prev_Target2_T+0.5h, ...
    # Oppure, se preferisci, una riga per ogni passo previsto (ma questo è più complesso da allineare)
    # Scegliamo il formato più semplice: una riga per esecuzione, con le previsioni "appiattite"
    
    # Se vogliamo che ogni riga del GSheet "Previsioni" contenga UNA previsione per UN target per UN passo futuro:
    for step_idx in range(output_window_steps):
        current_prediction_time = prediction_start_time + timedelta(minutes=30 * (step_idx + 1))
        row_data_for_step = [
            timestamp_esecuzione.strftime('%Y-%m-%d %H:%M:%S %Z'), # Quando è stata fatta la previsione
            current_prediction_time.strftime('%Y-%m-%d %H:%M:%S %Z'), # Per quale ora futura è questa previsione
            f"T+{ (step_idx + 1) * 0.5 :.1f}h" # Relativo a prediction_start_time
        ]
        for target_idx, target_col_name in enumerate(target_columns):
            row_data_for_step.append(f"{predictions_np[step_idx, target_idx]:.3f}") # Valore previsto per questo target a questo step
        
        # Se l'intestazione non include i nomi dei target specifici per ogni colonna di previsione, aggiungiamoli la prima volta
        if worksheet.row_count == 1 or worksheet.cell(1,4).value is None: # Controlla se l'header è solo quello base
            current_header = worksheet.row_values(1)
            new_header_parts = []
            if len(current_header) < 3 + len(target_columns): # Se mancano le colonne per i target
                for target_col in target_columns:
                    new_header_parts.append(f"Previsto: {target_col}")
                worksheet.update(f'D1', [new_header_parts]) # Aggiorna da D1 in poi


        worksheet.append_row(row_data_for_step)
        print(f"Aggiunta previsione per {current_prediction_time.strftime('%Y-%m-%d %H:%M')} al foglio '{predictions_sheet_name}'.")


def main():
    print(f"Avvio simulazione alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if not GSHEET_ID:
        print("Errore: GSHEET_ID non impostato come variabile d'ambiente.")
        return

    try:
        gcp_sa_key_b64 = os.environ.get("GCP_SA_KEY_BASE64")
        if not gcp_sa_key_b64:
            raise ValueError("GCP_SA_KEY_BASE64 non trovato nei secret.")
        
        credentials_json_str = base64.b64decode(gcp_sa_key_b64).decode('utf-8')
        credentials_dict = json.loads(credentials_json_str)
        credentials = Credentials.from_service_account_info(credentials_dict, 
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        print("Autenticazione Google Sheets riuscita.")

        model, scaler_features, scaler_targets, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)

        # Mappatura tra nomi colonne GSheet (input) e nomi feature modello
        # ADATTALA MOLTO ATTENTAMENTE AI NOMI DEL TUO GSHEET DI INPUT E ALLE FEATURE DEL MODELLO
        column_mapping_gsheet_to_model = {
            'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
            'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
            'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
            'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
            HUMIDITY_COL_NAME_INPUT: HUMIDITY_COL_NAME_INPUT, # Se il nome è lo stesso
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
            GSHEET_DATE_COL_INPUT: GSHEET_DATE_COL_INPUT # Se il nome è lo stesso o per riferimento
        }
        # Assicurati che tutte le feature_columns della config siano mappate o gestite
        for fc in config["feature_columns"]:
            if fc not in column_mapping_gsheet_to_model.values():
                # Se una feature del modello non è mappata da una colonna GSheet,
                # potrebbe essere un problema a meno che non venga imputata o sia un errore.
                # Per ora, assumiamo che sia un errore di mapping se non è la colonna data.
                if fc != column_mapping_gsheet_to_model.get(GSHEET_DATE_COL_INPUT): # Non lamentarti per la colonna data se mappata a se stessa
                    print(f"ATTENZIONE CRITICA: La feature del modello '{fc}' non ha una colonna GSheet corrispondente nel mapping.")
                    # Potresti voler sollevare un errore qui se tutte le feature DEVONO venire da GSheet
                    # raise ValueError(f"Feature modello '{fc}' non mappata.")


        input_data_np, last_input_timestamp = fetch_input_data_from_gsheet(
            gc, GSHEET_ID, GSHEET_DATA_SHEET_NAME, config, column_mapping_gsheet_to_model
        )

        predictions_np = predict_with_model(model, input_data_np, scaler_features, scaler_targets, device)
        print(f"Previsioni generate. Shape: {predictions_np.shape}")

        # Il prediction_start_time è l'ultimo timestamp dei dati di input usati
        # Le previsioni iniziano dallo step *successivo*
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
