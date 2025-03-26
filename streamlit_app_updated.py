# -*- coding: utf-8 -*-
import streamlit as st
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Non più usato
# import seaborn as sns # Non più usato
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
# import math # Non più usato
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import time
import gspread # Importa gspread
from google.oauth2.service_account import Credentials # Importa Credentials
import re # Importa re per espressioni regolari
import json # Importa json per leggere/scrivere file di configurazione
import glob # Per cercare i file dei modelli
import traceback # Per stampare errori dettagliati
import mimetypes # Per indovinare i tipi MIME per i download
from streamlit_js_eval import streamlit_js_eval # Per forzare refresh periodico
import pytz # Per gestione timezone

# Configurazione della pagina
st.set_page_config(page_title="Modello Predittivo Idrologico", page_icon="🌊", layout="wide")

# --- Costanti ---
MODELS_DIR = "models" # Cartella dove risiedono i modelli pre-addestrati
DEFAULT_DATA_PATH = "dati_idro.csv" # Assumi sia nella stessa cartella dello script
GSHEET_ID = "1pQI6cKrrT-gcVAfl-9ZhUx5b3J-edZRRj6nzDcCBRcA" # ID del Google Sheet per la Dashboard
GSHEET_DATE_COL = 'Data_Ora'
GSHEET_DATE_FORMAT = '%d/%m/%Y %H:%M'
# Colonne di interesse dal Google Sheet per la Dashboard
# *** Rivedi e conferma questi nomi ESATTI dal tuo foglio Google ***
GSHEET_RELEVANT_COLS = [
    GSHEET_DATE_COL,
    'Arcevia - Pioggia Ora (mm)',
    'Barbara - Pioggia Ora (mm)',
    'Corinaldo - Pioggia Ora (mm)',
    'Misa - Pioggia Ora (mm)', # Presumo Bettolelle (Sensore 2637)?
    'Serra dei Conti - Livello Misa (mt)',
    'Pianello di Ostra - Livello Misa (m)',
    'Nevola - Livello Nevola (mt)', # Presumo Corinaldo/Nevola (Sensore 1283)?
    'Misa - Livello Misa (mt)', # Presumo Bettolelle (Sensore 1112)?
    'Ponte Garibaldi - Livello Misa 2 (mt)'
]
DASHBOARD_REFRESH_INTERVAL_SECONDS = 300 # Aggiorna dashboard ogni 5 minuti (300 sec)
DASHBOARD_HISTORY_ROWS = 48 # NUOVA: Numero di righe storiche da recuperare (es. 48 per 2 giorni)
DEFAULT_THRESHOLDS = { # Soglie predefinite (l'utente può modificarle)
    'Arcevia - Pioggia Ora (mm)': 10.0,
    'Barbara - Pioggia Ora (mm)': 10.0,
    'Corinaldo - Pioggia Ora (mm)': 10.0,
    'Misa - Pioggia Ora (mm)': 10.0, # Bettolelle Pioggia?
    'Serra dei Conti - Livello Misa (mt)': 2.5,
    'Pianello di Ostra - Livello Misa (m)': 3.0,
    'Nevola - Livello Nevola (mt)': 2.0, # Corinaldo Nevola?
    'Misa - Livello Misa (mt)': 2.8, # Bettolelle Livello?
    'Ponte Garibaldi - Livello Misa 2 (mt)': 4.0
}
# Define Italy timezone
italy_tz = pytz.timezone('Europe/Rome')

# --- NUOVO: Coordinate Stazioni con LOCATION ID e TYPE ---
# Le coordinate NON sono più usate per la mappa, ma le manteniamo
# per estrarre label/tipo sensore ecc.
STATION_COORDS = {
    # Sensor Column Name: {lat, lon, name (sensor specific), type, location_id}
    'Arcevia - Pioggia Ora (mm)': {'lat': 43.5228, 'lon': 12.9388, 'name': 'Arcevia (Pioggia)', 'type': 'Pioggia', 'location_id': 'Arcevia'},
    'Barbara - Pioggia Ora (mm)': {'lat': 43.5808, 'lon': 13.0277, 'name': 'Barbara (Pioggia)', 'type': 'Pioggia', 'location_id': 'Barbara'},
    'Corinaldo - Pioggia Ora (mm)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Pioggia)', 'type': 'Pioggia', 'location_id': 'Corinaldo'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo'}, # Stessa Loc
    'Misa - Pioggia Ora (mm)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Pioggia)', 'type': 'Pioggia', 'location_id': 'Bettolelle'}, # Assunzione Bettolelle
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle'}, # Stessa Loc, Assunzione Bettolelle
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti'},
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'}, # Coordinate indicative
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'} # Coordinate indicative Ponte Garibaldi Senigallia
}

# --- Definizioni Funzioni Core ML (Dataset, LSTM) ---
# (INVARIATE)
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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

# --- Funzioni Utilità Modello/Dati ---
# (INVARIATE - per ora, prepare_training_data rimane com'è)
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, val_split=20):
    # st.write(f"Prepare Data: Input Window={input_window}, Output Window={output_window}, Val Split={val_split}%")
    # st.write(f"Prepare Data: Feature Cols ({len(feature_columns)}): {', '.join(feature_columns[:3])}...")
    # st.write(f"Prepare Data: Target Cols ({len(target_columns)}): {', '.join(target_columns)}")
    try:
        for col in feature_columns + target_columns:
            if col not in df.columns:
                 raise ValueError(f"Colonna '{col}' richiesta per training non trovata nel DataFrame.")
        if df[feature_columns + target_columns].isnull().sum().sum() > 0:
             st.warning("NaN residui rilevati prima della creazione sequenze! Controlla caricamento dati.")
    except ValueError as e:
        st.error(f"Errore colonne in prepare_training_data: {e}")
        return None, None, None, None, None, None
    X, y = [], []
    total_len = len(df)
    required_len = input_window + output_window
    if total_len < required_len:
         st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe per input+output).")
         return None, None, None, None, None, None
    # st.write(f"Creazione sequenze da {total_len - required_len + 1} punti possibili...")
    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_window][feature_columns].values)
        y.append(df.iloc[i + input_window : i + required_len][target_columns].values)
    if not X or not y:
        st.error("Errore: Nessuna sequenza X/y creata. Controlla finestre e lunghezza dati.")
        return None, None, None, None, None, None
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    # st.write(f"Sequenze create: X shape={X.shape}, y shape={y.shape}")
    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima della normalizzazione.")
        return None, None, None, None, None, None
    num_sequences, seq_len_in, num_features = X.shape
    num_sequences_y, seq_len_out, num_targets = y.shape
    X_flat = X.reshape(-1, num_features); y_flat = y.reshape(-1, num_targets)
    # st.write(f"Shape per scaling: X_flat={X_flat.shape}, y_flat={y_flat.shape}")
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except Exception as e_scale:
         st.error(f"Errore durante scaling: {e_scale}")
         st.error(f"NaN in X_flat: {np.isnan(X_flat).sum()}, NaN in y_flat: {np.isnan(y_flat).sum()}")
         return None, None, None, None, None, None
    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets)
    # st.write(f"Shape post scaling: X_scaled={X_scaled.shape}, y_scaled={y_scaled.shape}")
    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    if split_idx == 0 or split_idx == len(X_scaled):
         st.warning(f"Split indice ({split_idx}) non valido per divisione train/val...")
         if len(X_scaled) < 2:
              st.error("Dataset troppo piccolo per creare set di validazione.")
              return None, None, None, None, None, None
         split_idx = max(1, len(X_scaled) - 1) if split_idx == len(X_scaled) else split_idx
         split_idx = min(len(X_scaled) - 1, split_idx) if split_idx == 0 else split_idx
    X_train = X_scaled[:split_idx]; y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]; y_val = y_scaled[split_idx:]
    # st.write(f"Split Dati: Train={len(X_train)}, Validation={len(X_val)}")
    if X_train.size == 0 or y_train.size == 0:
         st.error("Set di Training vuoto dopo lo split.")
         return None, None, None, None, None, None
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set di Validazione vuoto dopo lo split.")
         X_val = np.empty((0, seq_len_in, num_features), dtype=np.float32)
         y_val = np.empty((0, seq_len_out, num_targets), dtype=np.float32)
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

@st.cache_data
def load_model_config(_config_path):
    try:
        with open(_config_path, 'r') as f: config = json.load(f)
        required = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        if not all(k in config for k in required): st.error(f"Config '{_config_path}' incompleto."); return None
        return config
    except Exception as e: st.error(f"Errore caricamento config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    if not config: st.error("Config non valida."); return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        f_cols_model = config.get("feature_columns", [])
        if not f_cols_model: f_cols_model = st.session_state.get("feature_columns", [])
        if not f_cols_model: st.error("Impossibile determinare input_size."); return None, None
        input_size_model = len(f_cols_model)
        model = HydroLSTM(input_size_model, config["hidden_size"], len(config["target_columns"]),
                          config["output_window"], config["num_layers"], config["dropout"]).to(device)
        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'):
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")
        model.eval()
        # st.success(f"Modello '{config.get('name', os.path.basename(str(_model_path)))}' caricato su {device}.") # Meno verbose
        return model, device
    except Exception as e:
        st.error(f"Errore caricamento modello '{config.get('name', 'N/A')}': {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(_scaler_features_path, _scaler_targets_path):
    try:
        def _load_joblib(path):
             if isinstance(path, str):
                  if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
                  return joblib.load(path)
             elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
             else: raise TypeError("Percorso scaler non valido.")
        sf = _load_joblib(_scaler_features_path)
        st = _load_joblib(_scaler_targets_path)
        # st.success(f"Scaler caricati.") # Meno verbose
        return sf, st
    except Exception as e: st.error(f"Errore caricamento scaler: {e}"); return None, None

def find_available_models(models_dir=MODELS_DIR):
    available = {}
    if not os.path.isdir(models_dir): return available
    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth_path in pth_files:
        base = os.path.splitext(os.path.basename(pth_path))[0]
        cfg_p = os.path.join(models_dir, f"{base}.json")
        scf_p = os.path.join(models_dir, f"{base}_features.joblib")
        sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
        if os.path.exists(cfg_p) and os.path.exists(scf_p) and os.path.exists(sct_p):
            try:
                 with open(cfg_p, 'r') as f: name = json.load(f).get("display_name", base)
            except: name = base
            available[name] = {"config_name": base, "pth_path": pth_path, "config_path": cfg_p,
                               "scaler_features_path": scf_p, "scaler_targets_path": sct_p}
        else: st.warning(f"Modello '{base}' ignorato: file mancanti.")
    return available

def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict: Modello, scaler o config mancanti."); return None
    input_w = config["input_window"]; output_w = config["output_window"]
    target_cols = config["target_columns"]; f_cols_cfg = config.get("feature_columns", [])
    if input_data.shape[0] != input_w: st.error(f"Predict: Input righe {input_data.shape[0]} != Finestra {input_w}."); return None
    if f_cols_cfg and input_data.shape[1] != len(f_cols_cfg): st.error(f"Predict: Input colonne {input_data.shape[1]} != Features {len(f_cols_cfg)}."); return None
    if not f_cols_cfg and hasattr(scaler_features, 'n_features_in_') and scaler_features.n_features_in_ != input_data.shape[1]: st.error(f"Predict: Input colonne {input_data.shape[1]} != Scaler Features {scaler_features.n_features_in_}."); return None
    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(inp_tens)
        out_np = output.cpu().numpy().reshape(output_w, len(target_cols))
        if not hasattr(scaler_targets, 'n_features_in_'): st.error("Predict: Scaler targets non fittato."); return None
        if scaler_targets.n_features_in_ != len(target_cols): st.error(f"Predict: Output targets {len(target_cols)} != Scaler Targets {scaler_targets.n_features_in_}."); return None
        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except Exception as e: st.error(f"Errore durante predict: {e}"); st.error(traceback.format_exc()); return None

def plot_predictions(predictions, config, start_time=None):
    """ Modificata: Usata solo per SIMULAZIONE, non più per grafici individuali dashboard """
    if config is None or predictions is None: return []
    output_w = config["output_window"]; target_cols = config["target_columns"]
    figs = []
    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        if start_time:
            hours = [start_time + timedelta(hours=h+1) for h in range(output_w)]
            x_axis, x_title = hours, "Data e Ora Previste"
        else: hours = np.arange(1, output_w + 1); x_axis, x_title = hours, "Ore Future"

        # Estrai nome stazione per titolo grafico
        station_name_graph = get_station_label(sensor, short=False) # Usa la funzione helper

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsto'))
        fig.update_layout(
            title=f'Previsione Simulazione - {station_name_graph}', # Titolo aggiornato
            xaxis_title=x_title,
            yaxis_title=f'{sensor.split("(")[-1].split(")")[0].strip()}', # Estrae unità
            height=400,
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        figs.append(fig)
    return figs

# --- MODIFICATA FUNZIONE: Fetch Dati Dashboard da Google Sheet ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format, num_rows_to_fetch=DASHBOARD_HISTORY_ROWS):
    """
    Importa le ultime 'num_rows_to_fetch' righe di dati dal Google Sheet.
    Restituisce un DataFrame pulito o None in caso di errore grave.
    """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data (Cache Key: {_cache_key_time}, Rows: {num_rows_to_fetch})") # Debug
    actual_fetch_time = datetime.now(italy_tz)
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets:
            return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        credentials = Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"],
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        all_values = worksheet.get_all_values() # Fetch dei dati grezzi

        if not all_values or len(all_values) < 2:
            return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = all_values[0]
        # Prendi le ultime num_rows_to_fetch righe di DATI (escludendo l'header)
        # Se ci sono meno righe disponibili, prendi quelle che ci sono
        start_index = max(1, len(all_values) - num_rows_to_fetch)
        data_rows = all_values[start_index:]

        if not data_rows:
             return None, "Errore: Nessuna riga di dati trovata (dopo header).", actual_fetch_time

        # Verifica che tutte le colonne richieste siano presenti negli header
        headers_set = set(headers)
        missing_cols = [col for col in relevant_columns if col not in headers_set]
        if missing_cols:
            return None, f"Errore: Colonne GSheet mancanti: {', '.join(missing_cols)}", actual_fetch_time

        # Crea DataFrame
        df = pd.DataFrame(data_rows, columns=headers)

        # Seleziona solo le colonne rilevanti
        df = df[relevant_columns]

        # Converti e pulisci i dati
        error_parsing = []
        for col in relevant_columns:
            if col == date_col:
                try:
                    # Converte in datetime, gli errori diventano NaT
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    # Localizza assumendo fuso orario italiano (se naive)
                    if df[col].dt.tz is None:
                         df[col] = df[col].dt.tz_localize(italy_tz)
                    else: # Se già aware, converte
                         df[col] = df[col].dt.tz_convert(italy_tz)

                    # Controlla se ci sono NaT dopo la conversione
                    if df[col].isnull().any():
                        original_dates_with_errors = df.loc[df[col].isnull(), col] # Trova le righe con errori
                        # (Potresti voler loggare original_dates_with_errors per debug)
                        error_parsing.append(f"Formato data non valido per '{col}' in alcune righe (atteso: {date_format})")

                except Exception as e_date:
                    error_parsing.append(f"Errore conversione data '{col}': {e_date}")
                    df[col] = pd.NaT # Forza NaT su tutta la colonna in caso di errore grave
            else: # Colonne numeriche
                try:
                    # Sostituisci virgola, spazi, etc. e converti in numerico
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    # Gestisci 'N/A' e stringhe vuote comuni
                    df[col] = df[col].replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Controlla se ci sono NaN dopo la conversione
                    if df[col].isnull().any():
                        # Potresti aggiungere un warning qui se necessario
                        pass # Per ora non blocchiamo per singoli valori non numerici

                except Exception as e_num:
                    error_parsing.append(f"Errore conversione numerica '{col}': {e_num}")
                    df[col] = np.nan # Forza NaN in caso di errore grave

        # Gestisci NaT nella colonna data (potrebbe essere meglio rimuovere le righe?)
        if df[date_col].isnull().any():
             st.warning(f"Attenzione: Rilevate date non valide nel GSheet (colonna '{date_col}'). Queste righe potrebbero essere escluse o causare problemi nei grafici.")
             # Opzione 1: Rimuovi righe con date non valide
             # df = df.dropna(subset=[date_col])
             # Opzione 2: Lasciale come NaT (potrebbe interrompere plotly in alcuni casi)

        # Ordina per data/ora per sicurezza (gestisce NaT mettendoli all'inizio o alla fine a seconda di na_position)
        df = df.sort_values(by=date_col, na_position='first').reset_index(drop=True)

        # Gestisci NaN numerici (opzionale: ffill/bfill?)
        nan_numeric_count = df.drop(columns=[date_col]).isnull().sum().sum()
        if nan_numeric_count > 0:
            st.info(f"Info: Rilevati {nan_numeric_count} valori numerici mancanti/non validi nelle ultime {len(df)} righe. Saranno visualizzati come 'N/D' o interruzioni nei grafici.")
            # Potresti applicare ffill/bfill qui se preferito:
            # numeric_cols = df.select_dtypes(include=np.number).columns
            # df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            # if df[numeric_cols].isnull().sum().sum() > 0:
            #     st.warning("NaN residui dopo fill nei dati GSheet.")

        if error_parsing:
            # Restituisce il DataFrame ma segnala gli errori
            error_message = "Attenzione: Errori durante la conversione dei dati GSheet. " + " | ".join(error_parsing)
            return df, error_message, actual_fetch_time

        # Restituisce il DataFrame
        return df, None, actual_fetch_time # Nessun errore grave

    except gspread.exceptions.APIError as api_e:
        # ... (gestione errori API invariata) ...
        try:
            error_details = api_e.response.json()
            error_message = error_details.get('error', {}).get('message', str(api_e))
            status_code = error_details.get('error', {}).get('code', 'N/A')
            if status_code == 403: error_message += " Verifica condivisione foglio."
            elif status_code == 429: error_message += f" Limite API superato. Riprova. TTL: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
        except: error_message = str(api_e)
        return None, f"Errore API Google Sheets: {error_message}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound:
        return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}'). Verifica ID e permessi.", actual_fetch_time
    except Exception as e:
        return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time


# --- Funzione Allenamento Modificata ---
# (INVARIATA)
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) # verbose=False per meno output
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state_dict = None
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        if v_loss and any(v is not None for v in v_loss): # Plotta solo se ci sono dati validi
             fig.add_trace(go.Scatter(y=v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0))
        placeholder.plotly_chart(fig, use_container_width=True)
    st.write(f"Inizio training per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train(); train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch); loss = criterion(outputs, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval(); val_loss = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch); loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else: val_losses.append(None) # Usa None se non c'è validation
        progress_percentage = (epoch + 1) / epochs
        progress_bar.progress(progress_percentage)
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss != 0 else "N/A"
        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss_str} - LR: {current_lr:.6f}')
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)
        time.sleep(0.01) # Minor sleep
    if best_model_state_dict:
        best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
        model.load_state_dict(best_model_state_dict_on_device)
        st.success(f"Caricato modello migliore (Val Loss: {best_val_loss:.6f})")
    elif len(val_loader) == 0: st.warning("Nessun set di validazione, usato modello ultima epoca.")
    else: st.warning("Nessun miglioramento in validazione, usato modello ultima epoca.")
    return model, train_losses, val_losses

# --- Funzioni Helper Download ---
# (INVARIATE)
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False, sep=';', decimal=',')
    b64 = base64.b64encode(csv.encode('utf-8')).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Scarica CSV</a>'

def get_binary_file_download_link(file_object, filename, text):
    file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    buf_html = io.StringIO(); fig.write_html(buf_html)
    b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
    href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'
    href_png = ""
    try:
        import kaleido # Verifica se kaleido è installato
        buf_png = io.BytesIO(); fig.write_image(buf_png, format="png") # Richiede kaleido
        buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode()
        href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
    except ImportError: pass # Silenzia se manca kaleido
    except Exception as e_png: pass # Silenzia altri errori PNG
    return f"{href_html} {href_png}"


def get_download_link_for_file(filepath, link_text=None):
    if not os.path.exists(filepath): return f"<i>File non trovato: {os.path.basename(filepath)}</i>"
    link_text = link_text or f"Scarica {os.path.basename(filepath)}"
    try:
        with open(filepath, "rb") as f: file_content = f.read()
        b64 = base64.b64encode(file_content).decode("utf-8")
        filename = os.path.basename(filepath)
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None:
            if filename.endswith(('.pth', '.joblib')): mime_type = 'application/octet-stream'
            elif filename.endswith('.json'): mime_type = 'application/json'
            else: mime_type = 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e:
        st.error(f"Errore generazione link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet ---
# (INVARIATA)
def extract_sheet_id(url):
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- NUOVA FUNZIONE: Estrazione Etichetta Stazione ---
# (INVARIATA)
def get_station_label(col_name, short=False):
    """Estrae un'etichetta leggibile dal nome colonna GSheet."""
    if col_name in STATION_COORDS:
        location_id = STATION_COORDS[col_name].get('location_id')
        if location_id:
            if short:
                sensor_type = STATION_COORDS[col_name].get('type', '')
                sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == location_id]
                if len(sensors_at_loc) > 1:
                    type_abbr = 'P' if sensor_type == 'Pioggia' else ('L' if sensor_type == 'Livello' else '')
                    return f"{location_id} ({type_abbr})"[:25]
                else:
                    return location_id[:25]
            else:
                return location_id
    parts = col_name.split(' - ')
    if len(parts) > 1:
        location = parts[0].strip()
        measurement = parts[1].split(' (')[0].strip()
        if short: return f"{location} - {measurement}"[:25]
        else: return location
    else: return col_name.split(' (')[0].strip()[:25]


# --- Inizializzazione Session State ---
# (AGGIUNTA chiave per DataFrame dashboard)
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scaler_features' not in st.session_state: st.session_state.active_scaler_features = None
if 'active_scaler_targets' not in st.session_state: st.session_state.active_scaler_targets = None
if 'df' not in st.session_state: st.session_state.df = None
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [ # Per MODELLO
         'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
         'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
         'Umidita\' Sensore 3452 (Montemurello)',
         'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
         'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
         'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
         'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
         'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
     ]
if 'date_col_name_csv' not in st.session_state: st.session_state.date_col_name_csv = 'Data e Ora'
if 'dashboard_thresholds' not in st.session_state: st.session_state.dashboard_thresholds = DEFAULT_THRESHOLDS.copy()
if 'last_dashboard_data' not in st.session_state: st.session_state.last_dashboard_data = None # Ora sarà un DataFrame
if 'last_dashboard_error' not in st.session_state: st.session_state.last_dashboard_error = None
if 'last_dashboard_fetch_time' not in st.session_state: st.session_state.last_dashboard_fetch_time = None
if 'active_alerts' not in st.session_state: st.session_state.active_alerts = [] # Lista di tuple (colonna, valore, soglia)

# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('🌊 Dashboard e Modello Predittivo Idrologico')

# --- Sidebar ---
st.sidebar.header('Impostazioni')

# --- Caricamento Dati Storici (per Analisi/Training) ---
st.sidebar.subheader('Dati Storici (per Analisi/Training)')
uploaded_data_file = st.sidebar.file_uploader('Carica CSV Dati Storici (Opzionale)', type=['csv'], key="data_uploader")

# --- Logica Caricamento DF (INVARIATA) ---
df = None; df_load_error = None; data_source_info = ""
data_path_to_load = None; is_uploaded = False
if uploaded_data_file is not None:
    data_path_to_load = uploaded_data_file; is_uploaded = True
    data_source_info = f"File caricato: **{uploaded_data_file.name}**"
elif os.path.exists(DEFAULT_DATA_PATH):
    data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False
    data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
else: df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato. Carica un CSV."

if data_path_to_load:
    try:
        read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
        df_loaded = False
        for enc in encodings_to_try:
            try:
                if is_uploaded: data_path_to_load.seek(0)
                df = pd.read_csv(data_path_to_load, encoding=enc, **read_args)
                df_loaded = True; break
            except UnicodeDecodeError: continue
            except Exception as read_e: raise read_e
        if not df_loaded: raise ValueError(f"Impossibile leggere CSV con encoding {', '.join(encodings_to_try)}.")

        date_col_csv = st.session_state.date_col_name_csv
        if date_col_csv not in df.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")
        try: df[date_col_csv] = pd.to_datetime(df[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
            try:
                 df[date_col_csv] = pd.to_datetime(df[date_col_csv], errors='coerce')
                 st.sidebar.warning("Formato data CSV non standard, tentata inferenza.")
            except Exception as e_date_csv:
                 raise ValueError(f"Errore conversione data CSV '{date_col_csv}': {e_date_csv}")

        df = df.dropna(subset=[date_col_csv])
        df = df.sort_values(by=date_col_csv).reset_index(drop=True)

        current_f_cols = st.session_state.feature_columns
        missing_features = [col for col in current_f_cols if col not in df.columns]
        # Non bloccare se mancano feature, l'utente potrebbe usarne meno per il training
        if missing_features:
             st.sidebar.warning(f"Attenzione: Le seguenti feature globali non sono nel CSV: {', '.join(missing_features)}. Saranno ignorate se non deselezionate.")
             # Rimuovi le feature mancanti da current_f_cols per la pulizia successiva
             current_f_cols = [col for col in current_f_cols if col in df.columns]


        # Pulizia colonne numeriche più robusta (sulle colonne presenti)
        for col in current_f_cols:
             if col in df.columns: # Sicurezza aggiuntiva
                 if df[col].dtype == 'object':
                      df[col] = df[col].astype(str).str.strip()
                      df[col] = df[col].replace(['N/A', '', '-', 'None', 'null'], np.nan, regex=True)
                      df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        cols_to_check_nan = [c for c in current_f_cols if c in df.columns] # Solo colonne esistenti
        n_nan = df[cols_to_check_nan].isnull().sum().sum()
        if n_nan > 0:
              st.sidebar.caption(f"Trovati {n_nan} NaN/non numerici nel CSV. Eseguito ffill/bfill.")
              df[cols_to_check_nan] = df[cols_to_check_nan].fillna(method='ffill').fillna(method='bfill')
              if df[cols_to_check_nan].isnull().sum().sum() > 0:
                  st.sidebar.error("NaN residui dopo fill. Controlla inizio/fine file CSV.")
                  nan_cols = df[cols_to_check_nan].isnull().sum()
                  st.sidebar.json(nan_cols[nan_cols > 0].to_dict())
                  # df[cols_to_check_nan] = df[cols_to_check_nan].fillna(0) # Opzione: riempire con 0

        st.session_state.df = df
        st.sidebar.success(f"Dati CSV caricati ({len(df)} righe).")
    except Exception as e:
        df = None; st.session_state.df = None
        df_load_error = f'Errore dati CSV ({data_source_info}): {type(e).__name__} - {e}'
        st.sidebar.error(f"Errore CSV: {df_load_error}")

df = st.session_state.get('df', None)
if df is None and df_load_error: st.sidebar.warning(f"Dati CSV non disponibili. {df_load_error}")


# --- Selezione Modello (per Simulazione/Predict basato su CSV) ---
st.sidebar.divider()
st.sidebar.subheader("Modello Predittivo (per Simulazione)")

available_models_dict = find_available_models(MODELS_DIR)
model_display_names = list(available_models_dict.keys())
MODEL_CHOICE_UPLOAD = "Carica File Manualmente..."
MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"
selection_options = [MODEL_CHOICE_NONE] + model_display_names + [MODEL_CHOICE_UPLOAD]
current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
try: current_index = selection_options.index(current_selection_name)
except ValueError: current_index = 0

selected_model_display_name = st.sidebar.selectbox(
    "Modello:", selection_options, index=current_index,
    help="Scegli un modello pre-addestrato o carica i file."
)

# --- Logica Caricamento Modello (INVARIATA) ---
config_to_load = None; model_to_load = None; device_to_load = None
scaler_f_to_load = None; scaler_t_to_load = None; load_error_sidebar = False
st.session_state.active_model_name = None; st.session_state.active_config = None
st.session_state.active_model = None; st.session_state.active_device = None
st.session_state.active_scaler_features = None; st.session_state.active_scaler_targets = None

if selected_model_display_name == MODEL_CHOICE_NONE:
    st.session_state.active_model_name = MODEL_CHOICE_NONE
elif selected_model_display_name == MODEL_CHOICE_UPLOAD:
    st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
    with st.sidebar.expander("Carica File Modello Manualmente", expanded=False):
        m_f = st.file_uploader('.pth', type=['pth'], key="up_pth")
        sf_f = st.file_uploader('.joblib (Features)', type=['joblib'], key="up_scf")
        st_f = st.file_uploader('.joblib (Target)', type=['joblib'], key="up_sct")
        st.caption("Configura parametri modello:")
        c1, c2 = st.columns(2)
        iw = c1.number_input("In Win", 1, 168, 24, 6, key="up_in")
        ow = c1.number_input("Out Win", 1, 72, 12, 1, key="up_out")
        hs = c2.number_input("Hidden", 16, 1024, 128, 16, key="up_hid")
        nl = c2.number_input("Layers", 1, 8, 2, 1, key="up_lay")
        dr = c2.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="up_drop")
        # Le feature globali sono la base per la selezione dei target qui
        targets_global = [col for col in st.session_state.feature_columns if 'Livello' in col]
        targets_up = st.multiselect("Target", targets_global, default=targets_global, key="up_targets")
        if m_f and sf_f and st_f and targets_up:
            # Le feature per il modello caricato SARANNO quelle globali correnti
            current_model_features = st.session_state.feature_columns
            temp_cfg = {"input_window": iw, "output_window": ow, "hidden_size": hs,
                        "num_layers": nl, "dropout": dr, "target_columns": targets_up,
                        "feature_columns": current_model_features, # USA quelle globali
                        "name": "uploaded"}
            model_to_load, device_to_load = load_specific_model(m_f, temp_cfg)
            scaler_f_to_load, scaler_t_to_load = load_specific_scalers(sf_f, st_f)
            if model_to_load and scaler_f_to_load and scaler_t_to_load: config_to_load = temp_cfg
            else: load_error_sidebar = True
        else: st.caption("Carica tutti i file e scegli i target.")
else: # Modello pre-addestrato
    model_info = available_models_dict[selected_model_display_name]
    st.session_state.active_model_name = selected_model_display_name
    config_to_load = load_model_config(model_info["config_path"])
    if config_to_load:
        config_to_load["pth_path"] = model_info["pth_path"]
        config_to_load["scaler_features_path"] = model_info["scaler_features_path"]
        config_to_load["scaler_targets_path"] = model_info["scaler_targets_path"]
        config_to_load["name"] = model_info["config_name"]
        if "feature_columns" not in config_to_load or not config_to_load["feature_columns"]:
             st.warning(f"Config '{selected_model_display_name}' non specifica le feature_columns. Uso quelle globali.")
             config_to_load["feature_columns"] = st.session_state.feature_columns

        model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
        scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])
        if not (model_to_load and scaler_f_to_load and scaler_t_to_load): load_error_sidebar = True; config_to_load = None
    else: load_error_sidebar = True

# Salva nello stato sessione SOLO se tutto è caricato correttamente
if config_to_load and model_to_load and device_to_load and scaler_f_to_load and scaler_t_to_load:
    st.session_state.active_config = config_to_load
    st.session_state.active_model = model_to_load
    st.session_state.active_device = device_to_load
    st.session_state.active_scaler_features = scaler_f_to_load
    st.session_state.active_scaler_targets = scaler_t_to_load
    st.session_state.active_model_name = selected_model_display_name

# Mostra feedback basato sullo stato sessione aggiornato
if st.session_state.active_model and st.session_state.active_config:
    cfg = st.session_state.active_config
    active_name = st.session_state.active_model_name
    st.sidebar.success(f"Modello ATTIVO: '{active_name}' (In:{cfg['input_window']}h, Out:{cfg['output_window']}h)")
elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]: st.sidebar.error("Caricamento modello fallito.")
elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not st.session_state.active_model: st.sidebar.info("Completa caricamento manuale modello.")


# --- Configurazione Soglie Dashboard (nella Sidebar) ---
st.sidebar.divider()
st.sidebar.subheader("Configurazione Soglie Dashboard")
with st.sidebar.expander("Modifica Soglie di Allerta", expanded=False):
    temp_thresholds = st.session_state.dashboard_thresholds.copy()
    monitorable_cols = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
    for col in monitorable_cols:
        label_short = get_station_label(col, short=True)
        is_level = 'Livello' in col or '(m)' in col or '(mt)' in col
        step = 0.1 if is_level else 1.0
        fmt = "%.1f" if is_level else "%.0f"
        min_v = 0.0
        current_threshold = st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, 0.0))
        new_threshold = st.number_input(
            label=f"Soglia {label_short}", value=current_threshold, min_value=min_v, step=step, format=fmt,
            key=f"thresh_{col}", help=f"Soglia di allerta per: {col}"
        )
        if new_threshold != current_threshold: temp_thresholds[col] = new_threshold

    if st.button("Salva Soglie", key="save_thresholds"):
        st.session_state.dashboard_thresholds = temp_thresholds.copy()
        st.success("Soglie aggiornate!")
        st.rerun()


# --- Menu Navigazione ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
model_ready = st.session_state.active_model is not None and st.session_state.active_config is not None
data_ready_csv = df is not None

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
requires_model = ['Simulazione']
requires_csv = ['Analisi Dati Storici', 'Allenamento Modello']

radio_captions = []
disabled_options = []
for opt in radio_options:
    caption = ""; disabled = False
    if opt == 'Dashboard': caption = "Monitoraggio GSheet"
    elif opt in requires_model and not model_ready: caption = "Richiede Modello attivo"; disabled = True
    elif opt in requires_csv and not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
    elif opt == 'Simulazione' and model_ready and not data_ready_csv: caption = "Esegui previsioni (CSV non disp.)"; disabled = False
    elif opt == 'Simulazione' and model_ready and data_ready_csv: caption = "Esegui previsioni custom"
    elif opt == 'Analisi Dati Storici' and data_ready_csv: caption = "Esplora dati CSV caricati"
    elif opt == 'Allenamento Modello' and data_ready_csv: caption = "Allena un nuovo modello"
    else: # Caso di default
         if opt == 'Dashboard': caption = "Monitoraggio GSheet"
         elif opt == 'Simulazione': caption = "Esegui previsioni custom"
         elif opt == 'Analisi Dati Storici': caption = "Esplora dati CSV caricati"
         elif opt == 'Allenamento Modello': caption = "Allena un nuovo modello"

    radio_captions.append(caption); disabled_options.append(disabled)

# Logica selezione pagina (INVARIATA ma con nuove condizioni `disabled_options`)
current_page_index = 0
try:
     if 'current_page' in st.session_state:
          current_page_index = radio_options.index(st.session_state.current_page)
          if disabled_options[current_page_index]:
               st.sidebar.warning(f"Pagina '{st.session_state.current_page}' non disponibile. Reindirizzato a Dashboard.")
               current_page_index = 0
     else: current_page_index = next((i for i, disabled in enumerate(disabled_options) if not disabled), 0)
except ValueError: current_page_index = 0

page = st.sidebar.radio(
    'Scegli una funzionalità', options=radio_options, captions=radio_captions, index=current_page_index,
    key='page_selector'
)
if not disabled_options[radio_options.index(page)]:
     st.session_state.current_page = page
else:
     st.warning(f"La pagina '{page}' non è accessibile. Vai alla Dashboard.")
     st.session_state.current_page = radio_options[0]
     time.sleep(0.5); st.rerun()


# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================

active_config = st.session_state.active_config
active_model = st.session_state.active_model
active_device = st.session_state.active_device
active_scaler_features = st.session_state.active_scaler_features
active_scaler_targets = st.session_state.active_scaler_targets
df_current_csv = st.session_state.get('df', None) # Dati CSV
feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns) if active_config else st.session_state.feature_columns
date_col_name_csv = st.session_state.date_col_name_csv

# --- PAGINA DASHBOARD (RIVISTA SENZA MAPPA, CON TABELLA E GRAFICI) ---
if page == 'Dashboard':
    st.header(f'📊 Dashboard Monitoraggio Idrologico')

    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("🚨 **Errore Configurazione:** Credenziali Google ('GOOGLE_CREDENTIALS') non trovate.")
        st.info("Aggiungi le credenziali del service account come secret per abilitare la dashboard.")
        st.stop()

    # --- Logica Fetch Dati ---
    now_ts = time.time()
    cache_time_key = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)

    # Chiama la funzione cachata (ora restituisce DataFrame)
    df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
        cache_time_key,
        GSHEET_ID,
        GSHEET_RELEVANT_COLS,
        GSHEET_DATE_COL,
        GSHEET_DATE_FORMAT,
        num_rows_to_fetch=DASHBOARD_HISTORY_ROWS # Usa la nuova costante
    )

    # Salva in session state
    st.session_state.last_dashboard_data = df_dashboard # Salva il DataFrame
    st.session_state.last_dashboard_error = error_msg
    if df_dashboard is not None or error_msg is None:
        st.session_state.last_dashboard_fetch_time = actual_fetch_time

    # --- Visualizzazione Dati e Grafici ---
    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        if st.session_state.last_dashboard_fetch_time:
            last_fetch_dt = st.session_state.last_dashboard_fetch_time
            fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt
            fetch_secs_ago = int(fetch_time_ago.total_seconds())
            st.caption(f"Dati GSheet recuperati ({DASHBOARD_HISTORY_ROWS} righe) alle: {last_fetch_dt.strftime('%d/%m/%Y %H:%M:%S')} ({fetch_secs_ago}s fa). Refresh ogni {DASHBOARD_REFRESH_INTERVAL_SECONDS}s.")
        else:
            st.caption("In attesa del primo recupero dati da Google Sheet...")

    with col_refresh_btn:
        if st.button("🔄 Forza Aggiorna", key="dash_refresh"):
            fetch_gsheet_dashboard_data.clear()
            st.success("Cache GSheet pulita. Ricaricamento..."); time.sleep(0.5); st.rerun()

    if error_msg:
        if "API" in error_msg or "Foglio Google non trovato" in error_msg or "Credenziali" in error_msg: st.error(f"🚨 {error_msg}")
        else: st.warning(f"⚠️ {error_msg}") # Errori di parsing etc.

    # Mostra dati se disponibili (ora df_dashboard è un DataFrame)
    if df_dashboard is not None and not df_dashboard.empty:
        # Prendi l'ultima riga per valori attuali e timestamp
        latest_row_data = df_dashboard.iloc[-1]
        last_update_time = latest_row_data.get(GSHEET_DATE_COL)
        time_now_italy = datetime.now(italy_tz)

        if pd.notna(last_update_time):
             # Assicurati sia aware (dovrebbe esserlo)
             if last_update_time.tzinfo is None: last_update_time = italy_tz.localize(last_update_time)
             else: last_update_time = last_update_time.tz_convert(italy_tz)

             time_delta = time_now_italy - last_update_time
             minutes_ago = int(time_delta.total_seconds() // 60)
             time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
             if minutes_ago < 0: time_ago_str = "nel futuro?"
             elif minutes_ago < 2: time_ago_str = "pochi istanti fa"
             elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
             else: time_ago_str = f"{minutes_ago // 60}h {minutes_ago % 60}min fa"
             st.success(f"**Ultimo rilevamento nei dati:** {time_str} ({time_ago_str})")
             if minutes_ago > 30: st.warning(f"⚠️ Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa.")
        else: st.warning("⚠️ Timestamp ultimo rilevamento non disponibile nei dati GSheet.")

        st.divider()

        # --- NUOVO: Tabella con Valori Attuali e Soglie ---
        st.subheader("Tabella Valori Attuali")
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
        table_rows = []
        current_alerts = [] # Ricalcola alert qui

        for col_name in cols_to_monitor:
            current_value = latest_row_data.get(col_name)
            threshold = st.session_state.dashboard_thresholds.get(col_name)
            alert_active = False
            value_numeric = np.nan # Valore numerico per confronto
            value_display = "N/D" # Stringa per display
            unit = ""

            if pd.notna(current_value):
                 value_numeric = current_value # Già numerico da fetch
                 is_level = 'Livello' in col_name or '(m)' in col_name or '(mt)' in col_name
                 unit = '(mm)' if 'Pioggia' in col_name else ('(m)' if is_level else '')
                 value_display = f"{current_value:.1f} {unit}" if unit == '(mm)' else f"{current_value:.2f} {unit}"

                 if threshold is not None and current_value >= threshold:
                      alert_active = True
                      current_alerts.append((col_name, current_value, threshold))

            # Stato per la tabella
            status = "🔴 SUPERAMENTO" if alert_active else ("✅ OK" if pd.notna(current_value) else "⚪ N/D")
            # Soglia per display
            threshold_display = f"{threshold:.1f}" if threshold is not None else "-"

            table_rows.append({
                "Sensore": get_station_label(col_name, short=True), # Nome breve
                "Nome Completo": col_name, # Nome completo per tooltip o riferimento
                "Valore Numerico": value_numeric, # Per styling
                "Valore Attuale": value_display, # Per display
                "Soglia": threshold_display,
                "Soglia Numerica": threshold, # Per styling
                "Stato": status
            })

        df_display = pd.DataFrame(table_rows)

        # Funzione di stile per la tabella
        def highlight_threshold(row):
            color = 'red'
            background = 'rgba(255, 0, 0, 0.15)' # Rosso chiaro per background
            text_color = 'black' # Testo nero di default
            style = [''] * len(row) # Stile vuoto di default
            threshold_val = row['Soglia Numerica']
            current_val = row['Valore Numerico']

            if pd.notna(threshold_val) and pd.notna(current_val) and current_val >= threshold_val:
                # Applica stile alla riga intera o solo ad alcune celle?
                # Applichiamo a tutta la riga per evidenziare
                style = [f'background-color: {background}; color: {text_color}; font-weight: bold;'] * len(row)
                # Potresti voler cambiare colore solo alla cella Stato o Valore
                # style[row.index.get_loc('Valore Attuale')] = f'background-color: {color}; color: white; font-weight: bold;'
                # style[row.index.get_loc('Stato')] = f'background-color: {color}; color: white; font-weight: bold;'
            return style

        # Applica lo stile e mostra la tabella
        # Seleziona colonne da mostrare (nascondi quelle numeriche usate solo per lo stile)
        cols_to_show_in_table = ["Sensore", "Valore Attuale", "Soglia", "Stato"]
        st.dataframe(
            df_display.style.apply(highlight_threshold, axis=1, subset=["Valore Numerico", "Soglia Numerica"]),
            column_order=cols_to_show_in_table,
            hide_index=True,
            use_container_width=True,
            # Configura colonne per tooltip etc.
            column_config={
                "Sensore": st.column_config.TextColumn(help="Nome breve del sensore"),
                "Valore Attuale": st.column_config.TextColumn(help="Ultimo valore misurato con unità"),
                "Soglia": st.column_config.TextColumn(help="Soglia di allerta configurata"),
                "Stato": st.column_config.TextColumn(help="Stato rispetto alla soglia"),
                # Nascondi colonne non volute (alternative a column_order se non funziona bene)
                # "Nome Completo": None,
                # "Valore Numerico": None,
                # "Soglia Numerica": None,
            }
        )

        # Aggiorna alert globali
        st.session_state.active_alerts = current_alerts

        st.divider()

        # --- NUOVO: Grafico Comparativo Configurabile ---
        st.subheader("Grafico Comparativo Storico")
        # Opzioni: usa nomi brevi per selezione
        sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
        # Default: Seleziona i primi 2-3 sensori di livello, se presenti
        default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3]
        if not default_selection_labels and len(sensor_options_compare) > 0: # Fallback se non ci sono livelli
             default_selection_labels = list(sensor_options_compare.keys())[:2]

        selected_labels_compare = st.multiselect(
            "Seleziona sensori da confrontare:",
            options=list(sensor_options_compare.keys()),
            default=default_selection_labels,
            key="compare_select"
        )

        # Mappa le label selezionate ai nomi colonna originali
        selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare]

        if selected_cols_compare:
            fig_compare = go.Figure()
            for col in selected_cols_compare:
                label = get_station_label(col, short=True) # Usa label breve per legenda
                fig_compare.add_trace(go.Scatter(
                    x=df_dashboard[GSHEET_DATE_COL],
                    y=df_dashboard[col],
                    mode='lines', name=label,
                    hovertemplate=f'<b>{label}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>' # Usa nome completo in hover?
                ))
            fig_compare.update_layout(
                title=f"Andamento Storico Comparato (ultime {DASHBOARD_HISTORY_ROWS} ore)",
                xaxis_title='Data e Ora',
                yaxis_title='Valore Misurato',
                height=500,
                hovermode="x unified",
                legend_title_text='Sensori'
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            # Aggiungi link download per grafico comparativo
            compare_filename = f"compare_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.markdown(get_plotly_download_link(fig_compare, compare_filename), unsafe_allow_html=True)

        else:
            st.info("Seleziona almeno un sensore per visualizzare il grafico comparativo.")

        st.divider()

        # --- NUOVO: Grafici Individuali ---
        st.subheader("Grafici Individuali Storici")
        num_cols_individual = 3 # Quanti grafici per riga
        graph_cols = st.columns(num_cols_individual)
        col_idx = 0

        for col_name in cols_to_monitor:
            with graph_cols[col_idx % num_cols_individual]:
                threshold_individual = st.session_state.dashboard_thresholds.get(col_name)
                label_individual = get_station_label(col_name, short=True)
                unit_individual = '(mm)' if 'Pioggia' in col_name else ('(m)' if ('Livello' in col_name or '(m)' in col_name or '(mt)' in col_name) else '')
                yaxis_title_individual = f"Valore {unit_individual}"

                fig_individual = go.Figure()
                fig_individual.add_trace(go.Scatter(
                    x=df_dashboard[GSHEET_DATE_COL],
                    y=df_dashboard[col_name],
                    mode='lines', name=label_individual,
                    line=dict(color='royalblue'),
                    hovertemplate=f'<b>{label_individual}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'
                ))
                # Aggiungi linea soglia se definita
                if threshold_individual is not None:
                    fig_individual.add_hline(
                        y=threshold_individual, line_dash="dash", line_color="red",
                        annotation_text=f"Soglia ({threshold_individual:.1f})",
                        annotation_position="bottom right"
                    )
                fig_individual.update_layout(
                    title=f"{label_individual}",
                    xaxis_title=None, # Riduci spazio, la data è chiara
                    yaxis_title=yaxis_title_individual,
                    height=300, # Grafici più piccoli
                    hovermode="x unified",
                    showlegend=False,
                    margin=dict(t=30, b=20, l=40, r=10) # Margini ridotti
                )
                fig_individual.update_yaxes(rangemode='tozero') # Parti sempre da zero sull'asse Y
                st.plotly_chart(fig_individual, use_container_width=True)
                # Aggiungi link download per grafico individuale
                ind_filename = f"sensor_{label_individual.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_individual, ind_filename, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)

            col_idx += 1


        # Mostra box riepilogativo degli alert ATTIVI sotto i grafici
        st.divider()
        if st.session_state.active_alerts:
            st.warning("**🚨 ALLERTE ATTIVE (Valori Attuali) 🚨**")
            alert_md = ""
            sorted_alerts = sorted(st.session_state.active_alerts, key=lambda x: get_station_label(x[0], short=False))
            for col, val, thr in sorted_alerts:
                label_alert = get_station_label(col, short=False)
                sensor_type_alert = STATION_COORDS.get(col, {}).get('type', '')
                type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                val_fmt = f"{val:.1f}" if 'Pioggia' in col else f"{val:.2f}"
                thr_fmt = f"{thr:.1f}"
                unit = '(mm)' if 'Pioggia' in col else '(m)'
                alert_md += f"- **{label_alert}{type_str}**: Valore attuale **{val_fmt}{unit}** >= Soglia **{thr_fmt}**\n" # Rimosso nome colonna ridondante
            st.markdown(alert_md)
        else:
            st.success("✅ Nessuna soglia superata nell'ultimo rilevamento.")

        # Toast (opzionale, può essere fastidioso)
        # if current_alerts:
        #     alert_summary = f"{len(current_alerts)} sensori in allerta!"
        #     st.toast(alert_summary, icon="🚨")


    elif df_dashboard is not None and df_dashboard.empty: # Se fetch OK ma dataframe vuoto (es. dopo filtro data errato?)
        st.warning("Il recupero dati da Google Sheet ha restituito un set di dati vuoto.")
        if not error_msg: st.info("Controlla che ci siano dati recenti nel foglio Google.")

    else: # Se df_dashboard è None (fetch fallito gravemente)
        st.error("Impossibile visualizzare i dati della dashboard al momento.")
        if not error_msg: st.info("Controlla connessione o configurazione GSheet.")

    # Meccanismo di refresh automatico
    component_key = "dashboard_auto_refresh"
    streamlit_js_eval(js_expressions=f"setInterval(function(){{streamlitHook.rerunScript(null)}}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});", key=component_key)


# --- PAGINA SIMULAZIONE ---
# (MODIFICATA LEGGERMENTE per usare plot_predictions aggiornata)
elif page == 'Simulazione':
    st.header('🧪 Simulazione Idrologica')
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo per usare la Simulazione.")
    else:
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns_model = active_config["target_columns"]
        # feature_columns_current_model definito globalmente

        st.info(f"Simulazione con: **{st.session_state.active_model_name}** (Input: {input_window}h, Output: {output_window}h)")
        target_labels = [get_station_label(t, short=True) for t in target_columns_model]
        st.caption(f"Target previsti: {', '.join(target_labels)}")
        with st.expander("Feature richieste dal modello attivo"):
             st.caption(", ".join(feature_columns_current_model))

        sim_data_input = None
        sim_method_options = ['Manuale Costante', 'Importa da Google Sheet', 'Orario Dettagliato (Avanzato)']
        if data_ready_csv: sim_method_options.append('Usa Ultime Ore da CSV Caricato')
        sim_method = st.radio("Metodo preparazione dati simulazione", sim_method_options, key="sim_method_radio")

        # --- Simulazione: Manuale Costante ---
        if sim_method == 'Manuale Costante':
            st.subheader(f'Inserisci valori costanti per {input_window} ore')
            temp_sim_values = {}
            cols_manual = st.columns(3)
            feature_groups = {'Pioggia': [], 'Umidità': [], 'Livello': [], 'Altro': []}
            for feature in feature_columns_current_model:
                label_feat = get_station_label(feature, short=True)
                if 'Cumulata' in feature or 'Pioggia' in feature: feature_groups['Pioggia'].append((feature, label_feat))
                elif 'Umidita' in feature: feature_groups['Umidità'].append((feature, label_feat))
                elif 'Livello' in feature: feature_groups['Livello'].append((feature, label_feat))
                else: feature_groups['Altro'].append((feature, label_feat))
            col_idx = 0
            if feature_groups['Pioggia']:
                 with cols_manual[col_idx % 3]:
                      st.write("**Pioggia (mm/ora)**")
                      for feature, label_feat in feature_groups['Pioggia']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.0
                           temp_sim_values[feature] = st.number_input(label_feat, 0.0, value=round(default_val,1), step=0.5, format="%.1f", key=f"man_{feature}", help=feature)
                 col_idx += 1
            if feature_groups['Livello']:
                 with cols_manual[col_idx % 3]:
                      st.write("**Livelli (m)**")
                      for feature, label_feat in feature_groups['Livello']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.5
                           temp_sim_values[feature] = st.number_input(label_feat, -5.0, 20.0, value=round(default_val,2), step=0.05, format="%.2f", key=f"man_{feature}", help=feature)
                 col_idx += 1
            if feature_groups['Umidità'] or feature_groups['Altro']:
                with cols_manual[col_idx % 3]:
                     if feature_groups['Umidità']:
                          st.write("**Umidità (%)**")
                          for feature, label_feat in feature_groups['Umidità']:
                               default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 70.0
                               temp_sim_values[feature] = st.number_input(label_feat, 0.0, 100.0, value=round(default_val,1), step=1.0, format="%.1f", key=f"man_{feature}", help=feature)
                     if feature_groups['Altro']:
                          st.write("**Altre Feature**")
                          for feature, label_feat in feature_groups['Altro']:
                              default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.0
                              temp_sim_values[feature] = st.number_input(label_feat, value=round(default_val,2), step=0.1, format="%.2f", key=f"man_{feature}", help=feature)
            try:
                ordered_values = [temp_sim_values[feature] for feature in feature_columns_current_model]
                sim_data_input = np.tile(ordered_values, (input_window, 1)).astype(float)
            except KeyError as ke: st.error(f"Errore: Feature '{ke}' mancante nell'input manuale."); sim_data_input = None
            except Exception as e: st.error(f"Errore creazione dati costanti: {e}"); sim_data_input = None

        # --- Simulazione: Google Sheet ---
        elif sim_method == 'Importa da Google Sheet':
             st.subheader(f'Importa ultime {input_window} ore da Google Sheet')
             st.warning("⚠️ Funzionalità sperimentale. Verifica mappatura colonne!")
             sheet_url_sim = st.text_input("URL Foglio Google (storico)", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url")
             column_mapping_gsheet_to_model_sim = { # MAPPATURA DA VERIFICARE/ADATTARE
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
                # 'NomeColonnaUmiditaGSheet': 'Umidita\' Sensore 3452 (Montemurello)' # Esempio
             }
             with st.expander("Mostra/Modifica Mappatura GSheet -> Modello (Avanzato)"):
                 try:
                     edited_mapping_str = st.text_area("Mappatura JSON", value=json.dumps(column_mapping_gsheet_to_model_sim, indent=2), height=300, key="gsheet_map_edit")
                     edited_mapping = json.loads(edited_mapping_str)
                     if isinstance(edited_mapping, dict): column_mapping_gsheet_to_model_sim = edited_mapping
                     else: st.warning("Formato JSON mappatura non valido.")
                 except json.JSONDecodeError: st.warning("Errore JSON mappatura.")
                 except Exception as e_map: st.error(f"Errore mappatura: {e_map}")

             model_features_set = set(feature_columns_current_model)
             mapped_model_features = set(column_mapping_gsheet_to_model_sim.values())
             missing_model_features_in_map = list(model_features_set - mapped_model_features)
             imputed_values_sim = {}; needs_imputation_input = False
             if missing_model_features_in_map:
                  st.warning(f"Feature modello non mappate da GSheet (richiesto valore costante):")
                  needs_imputation_input = True
                  for missing_f in missing_model_features_in_map:
                       label_missing = get_station_label(missing_f, short=True)
                       default_val = 0.0; fmt = "%.2f"; step = 0.1
                       if data_ready_csv and missing_f in df_current_csv and pd.notna(df_current_csv[missing_f].median()): default_val = df_current_csv[missing_f].median()
                       if 'Umidita' in missing_f: fmt = "%.1f"; step = 1.0
                       elif 'Cumulata' in missing_f: fmt = "%.1f"; step = 0.5; default_val = max(0.0, default_val)
                       elif 'Livello' in missing_f: fmt = "%.2f"; step = 0.05
                       imputed_values_sim[missing_f] = st.number_input(f"Valore per '{label_missing}'", value=round(default_val, 2), step=step, format=fmt, key=f"sim_gsheet_impute_{missing_f}", help=missing_f)

             # Funzione fetch_historical_gsheet_data (INVARIATA - Definita sopra o importata)
             @st.cache_data(ttl=120, show_spinner="Importazione storica da Google Sheet...")
             def fetch_historical_gsheet_data(sheet_id, n_rows, date_col, date_format, col_mapping, required_model_cols, impute_dict):
                try:
                    if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti."
                    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
                    gc = gspread.authorize(credentials)
                    sh = gc.open_by_key(sheet_id)
                    worksheet = sh.sheet1
                    all_data = worksheet.get_all_values()
                    if not all_data or len(all_data) < (n_rows + 1): return None, f"Errore: Dati GSheet insufficienti ({len(all_data)-1} righe, richieste {n_rows})."

                    headers = all_data[0]
                    start_index = max(1, len(all_data) - n_rows)
                    data_rows = all_data[start_index:] # Prendi le ultime n_rows di dati

                    df_gsheet = pd.DataFrame(data_rows, columns=headers)
                    relevant_gsheet_cols = list(col_mapping.keys())
                    missing_gsheet_cols = [c for c in relevant_gsheet_cols if c not in df_gsheet.columns]
                    if missing_gsheet_cols: return None, f"Errore: Colonne GSheet mancanti: {', '.join(missing_gsheet_cols)}"

                    df_mapped = df_gsheet[relevant_gsheet_cols].rename(columns=col_mapping)

                    for model_col, impute_val in impute_dict.items():
                         if model_col not in df_mapped.columns: df_mapped[model_col] = impute_val

                    final_missing = [c for c in required_model_cols if c not in df_mapped.columns]
                    if final_missing: return None, f"Errore: Colonne modello mancanti dopo map/impute: {', '.join(final_missing)}"

                    gsheet_date_col_in_mapping = None
                    for gsheet_c, model_c in col_mapping.items():
                         if gsheet_c == date_col: gsheet_date_col_in_mapping = model_c; break

                    for col in required_model_cols:
                        if col != gsheet_date_col_in_mapping:
                             try:
                                 df_mapped[col] = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                                 df_mapped[col] = df_mapped[col].replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                                 df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
                             except Exception as e_clean:
                                 st.warning(f"Problema pulizia GSheet '{col}': {e_clean}")
                                 df_mapped[col] = np.nan

                    if gsheet_date_col_in_mapping and gsheet_date_col_in_mapping in df_mapped.columns:
                         try:
                             df_mapped[gsheet_date_col_in_mapping] = pd.to_datetime(df_mapped[gsheet_date_col_in_mapping], format=date_format, errors='coerce')
                             if df_mapped[gsheet_date_col_in_mapping].isnull().any(): st.warning(f"Date non valide in '{gsheet_date_col_in_mapping}'.")
                             df_mapped = df_mapped.sort_values(by=gsheet_date_col_in_mapping, na_position='first')
                         except Exception as e_date: return None, f"Errore conversione data GSheet '{gsheet_date_col_in_mapping}': {e_date}"

                    try: df_final = df_mapped[required_model_cols]
                    except KeyError as e_key: return None, f"Errore selezione colonne finali: '{e_key}' non trovata."

                    nan_count_before = df_final.drop(columns=[gsheet_date_col_in_mapping] if gsheet_date_col_in_mapping else [], errors='ignore').isnull().sum().sum()
                    if nan_count_before > 0:
                         st.warning(f"NaN GSheet ({nan_count_before}). Applico ffill/bfill.")
                         numeric_cols_gs = df_final.select_dtypes(include=np.number).columns
                         df_final[numeric_cols_gs] = df_final[numeric_cols_gs].fillna(method='ffill').fillna(method='bfill')
                         if df_final[numeric_cols_gs].isnull().sum().sum() > 0: return None, "Errore: NaN residui dopo fill GSheet."

                    if len(df_final) != n_rows: return None, f"Errore: Righe finali ({len(df_final)}) != richieste ({n_rows})."

                    # Restituisci solo dati numerici nell'ordine corretto
                    df_final_numeric = df_final[required_model_cols] # Seleziona e riordina
                    if gsheet_date_col_in_mapping and gsheet_date_col_in_mapping in df_final_numeric.columns:
                        df_final_numeric = df_final_numeric.drop(columns=[gsheet_date_col_in_mapping])
                        # Verifica che le colonne rimanenti siano le feature del modello
                        numeric_cols_expected = [c for c in required_model_cols if c != gsheet_date_col_in_mapping]
                        if set(df_final_numeric.columns) != set(numeric_cols_expected):
                             return None, f"Errore: Colonne numeriche finali ({list(df_final_numeric.columns)}) diverse da attese ({numeric_cols_expected})."
                        # Riordina per sicurezza rispetto a feature_columns_current_model (che non include data)
                        df_final_numeric = df_final_numeric[[c for c in feature_columns_current_model if c in df_final_numeric.columns]]

                    return df_final_numeric, None # Successo

                except Exception as e:
                    st.error(traceback.format_exc()) # Debug
                    return None, f"Errore imprevisto import GSheet: {type(e).__name__} - {e}"

             if st.button("Importa e Prepara da Google Sheet", key="sim_gsheet_import", disabled=needs_imputation_input and not imputed_values_sim):
                 sheet_id_sim = extract_sheet_id(sheet_url_sim)
                 if not sheet_id_sim: st.error("URL GSheet non valido.")
                 else:
                     imported_df_numeric, import_err = fetch_historical_gsheet_data(
                         sheet_id_sim, input_window, GSHEET_DATE_COL, GSHEET_DATE_FORMAT,
                         column_mapping_gsheet_to_model_sim, feature_columns_current_model, imputed_values_sim
                     )
                     if import_err: st.error(f"Import GSheet fallito: {import_err}"); st.session_state.imported_sim_data_gs = None; sim_data_input = None
                     elif imported_df_numeric is not None:
                          st.success(f"Importate e mappate {len(imported_df_numeric)} righe da GSheet.")
                          if imported_df_numeric.shape == (input_window, len(feature_columns_current_model)):
                              st.session_state.imported_sim_data_gs = imported_df_numeric
                              sim_data_input = imported_df_numeric.values
                              with st.expander("Mostra Dati Numerici Importati (pronti per modello)"): st.dataframe(imported_df_numeric.round(3))
                          else: st.error(f"Errore Shape dati GSheet ({imported_df_numeric.shape}) vs atteso ({input_window}, {len(feature_columns_current_model)})."); st.session_state.imported_sim_data_gs = None; sim_data_input = None
                     else: st.error("Import GSheet non riuscito."); st.session_state.imported_sim_data_gs = None; sim_data_input = None

             elif sim_method == 'Importa da Google Sheet' and 'imported_sim_data_gs' in st.session_state and st.session_state.imported_sim_data_gs is not None:
                 imported_df_state = st.session_state.imported_sim_data_gs
                 if isinstance(imported_df_state, pd.DataFrame) and imported_df_state.shape == (input_window, len(feature_columns_current_model)):
                     sim_data_input = imported_df_state.values; st.info("Uso dati GSheet importati precedentemente.")
                     with st.expander("Mostra Dati Importati (cache)"): st.dataframe(imported_df_state.round(3))
                 else: st.warning("Dati GSheet importati non validi/aggiornati. Rieseguire import."); st.session_state.imported_sim_data_gs = None; sim_data_input = None


        # --- Simulazione: Orario Dettagliato ---
        elif sim_method == 'Orario Dettagliato (Avanzato)':
            st.subheader(f'Inserisci dati orari per le {input_window} ore precedenti')
            session_key_hourly = f"sim_hourly_data_{input_window}_{'_'.join(sorted(feature_columns_current_model))}"
            needs_reinit = (session_key_hourly not in st.session_state or not isinstance(st.session_state[session_key_hourly], pd.DataFrame) or st.session_state[session_key_hourly].shape[0] != input_window or list(st.session_state[session_key_hourly].columns) != feature_columns_current_model)
            if needs_reinit:
                 st.caption("Inizializzazione tabella dati orari...")
                 init_vals = {}
                 for col in feature_columns_current_model:
                      med_val = 0.0
                      if data_ready_csv and col in df_current_csv and pd.notna(df_current_csv[col].median()): med_val = df_current_csv[col].median()
                      elif col in DEFAULT_THRESHOLDS: med_val = DEFAULT_THRESHOLDS.get(col, 0.0) * 0.2
                      if 'Cumulata' in col: med_val = max(0.0, med_val)
                      init_vals[col] = float(med_val)
                 init_df = pd.DataFrame(np.repeat([list(init_vals.values())], input_window, axis=0), columns=feature_columns_current_model)
                 st.session_state[session_key_hourly] = init_df[feature_columns_current_model].fillna(0.0) # Ordina e fillna
            df_for_editor = st.session_state[session_key_hourly].copy()
            df_for_editor = df_for_editor[feature_columns_current_model] # Assicura ordine
            if df_for_editor.isnull().sum().sum() > 0: df_for_editor = df_for_editor.fillna(0.0)
            try: df_for_editor = df_for_editor.astype(float)
            except Exception as e_cast: st.error(f"Errore conversione tabella float: {e_cast}. Reset."); del st.session_state[session_key_hourly]; st.rerun()
            column_config_editor = {}
            for col in feature_columns_current_model:
                 label_edit = get_station_label(col, short=True); fmt = "%.3f"; step = 0.01; min_v=None; max_v=None
                 if 'Cumulata' in col or 'Pioggia' in col: fmt = "%.1f"; step = 0.5; min_v=0.0
                 elif 'Umidita' in col: fmt = "%.1f"; step = 1.0; min_v=0.0; max_v=100.0
                 elif 'Livello' in col: fmt = "%.3f"; step = 0.01; min_v=-5.0; max_v=20.0
                 column_config_editor[col] = st.column_config.NumberColumn(label=label_edit, help=col, format=fmt, step=step, min_value=min_v, max_value=max_v, required=True)
            edited_df = st.data_editor(df_for_editor, height=(input_window + 1) * 35 + 3, use_container_width=True, column_config=column_config_editor, key=f"editor_{session_key_hourly}", num_rows="fixed")
            validation_passed = False
            if edited_df.shape[0] != input_window: st.error(f"Tabella deve avere {input_window} righe."); sim_data_input = None
            elif list(edited_df.columns) != feature_columns_current_model: st.error("Ordine/nomi colonne tabella cambiati."); sim_data_input = None
            elif edited_df.isnull().sum().sum() > 0: st.warning("Valori mancanti in tabella. Compilare."); sim_data_input = None
            else:
                 try:
                      sim_data_input_edit = edited_df[feature_columns_current_model].astype(float).values # Ordina prima di values
                      if sim_data_input_edit.shape == (input_window, len(feature_columns_current_model)): sim_data_input = sim_data_input_edit; validation_passed = True
                      else: st.error("Errore shape dati tabella."); sim_data_input = None
                 except Exception as e_edit: st.error(f"Errore conversione dati tabella: {e_edit}"); sim_data_input = None
            if validation_passed and not st.session_state[session_key_hourly].equals(edited_df): st.session_state[session_key_hourly] = edited_df


        # --- Simulazione: Ultime Ore da CSV ---
        elif sim_method == 'Usa Ultime Ore da CSV Caricato':
             st.subheader(f"Usa le ultime {input_window} ore dai dati CSV caricati")
             if not data_ready_csv: st.error("Dati CSV non caricati.")
             elif len(df_current_csv) < input_window: st.error(f"Dati CSV ({len(df_current_csv)} righe) insufficienti per {input_window} ore.")
             else:
                  try:
                       latest_csv_data_df = df_current_csv.iloc[-input_window:][feature_columns_current_model] # Seleziona e ordina
                       if latest_csv_data_df.isnull().sum().sum() > 0:
                            st.error(f"Trovati NaN nelle ultime {input_window} ore del CSV. Impossibile usare per simulazione."); sim_data_input = None
                            #st.dataframe(latest_csv_data_df[latest_csv_data_df.isnull().any(axis=1)])
                       else:
                            latest_csv_data = latest_csv_data_df.values
                            if latest_csv_data.shape == (input_window, len(feature_columns_current_model)):
                                sim_data_input = latest_csv_data
                                last_ts_csv = df_current_csv.iloc[-1][date_col_name_csv]
                                st.caption(f"Basato su dati CSV fino a: {last_ts_csv.strftime('%d/%m/%Y %H:%M')}")
                                with st.expander("Mostra dati CSV usati"): st.dataframe(df_current_csv.iloc[-input_window:][[date_col_name_csv] + feature_columns_current_model].round(3))
                            else: st.error("Errore shape dati CSV estratti."); sim_data_input = None
                  except KeyError as ke: st.error(f"Errore: Colonna '{ke}' modello non trovata nel CSV."); sim_data_input = None
                  except Exception as e_csv_sim: st.error(f"Errore estrazione dati CSV: {e_csv_sim}"); sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE ---
        st.divider()
        if sim_data_input is not None: st.success(f"Dati input ({sim_method}) pronti ({sim_data_input.shape}).")
        run_simulation = st.button('Esegui simulazione', type="primary", disabled=(sim_data_input is None), key="sim_run")
        if run_simulation and sim_data_input is not None:
             valid_input = False
             if not isinstance(sim_data_input, np.ndarray): st.error("Input non è NumPy array.")
             elif sim_data_input.shape[0] != input_window: st.error(f"Errore righe input. Atteso:{input_window}, Got:{sim_data_input.shape[0]}")
             elif sim_data_input.shape[1] != len(feature_columns_current_model): st.error(f"Errore colonne input. Atteso:{len(feature_columns_current_model)}, Got:{sim_data_input.shape[1]}")
             elif np.isnan(sim_data_input).any(): st.error(f"Errore: NaN nell'input simulazione ({np.isnan(sim_data_input).sum()} valori).")
             else: valid_input = True
             if valid_input:
                  with st.spinner('Simulazione in corso...'):
                       predictions_sim = predict(active_model, sim_data_input, active_scaler_features, active_scaler_targets, active_config, active_device)
                       if predictions_sim is not None:
                           st.subheader(f'Risultato Simulazione: Previsione per {output_window} ore')
                           start_pred_time = datetime.now(italy_tz); last_input_time_found = None
                           if sim_method == 'Usa Ultime Ore da CSV Caricato' and data_ready_csv:
                                try:
                                     last_csv_time = df_current_csv.iloc[-1][date_col_name_csv]
                                     if isinstance(last_csv_time, pd.Timestamp):
                                         if last_csv_time.tzinfo is None: last_input_time_found = italy_tz.localize(last_csv_time)
                                         else: last_input_time_found = last_csv_time.tz_convert(italy_tz)
                                except: pass # Ignora errore tempo CSV
                           elif sim_method == 'Importa da Google Sheet': st.caption("Nota: Timestamp inizio previsione basato su ora corrente per import GSheet.")
                           if last_input_time_found: start_pred_time = last_input_time_found
                           st.caption(f"Previsione a partire da: {start_pred_time.strftime('%d/%m/%Y %H:%M %Z')}")

                           pred_times_sim = [start_pred_time + timedelta(hours=i+1) for i in range(output_window)]
                           results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns_model)
                           results_df_sim.insert(0, 'Ora previsione', [t.strftime('%d/%m %H:%M') for t in pred_times_sim])
                           # Usa etichetta breve + unità per colonne risultati
                           results_df_sim.columns = ['Ora previsione'] + [get_station_label(col, short=True) + f" ({col.split('(')[-1].split(')')[0]})" for col in target_columns_model]
                           st.dataframe(results_df_sim.round(3))
                           st.markdown(get_table_download_link(results_df_sim, f"simulazione_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                           st.subheader('Grafici Previsioni Simulate')
                           # Usa la funzione plot_predictions aggiornata
                           figs_sim = plot_predictions(predictions_sim, active_config, start_pred_time)
                           sim_cols = st.columns(min(len(figs_sim), 2)) # Max 2 grafici simulazione per riga
                           for i, fig_sim in enumerate(figs_sim):
                               with sim_cols[i % len(sim_cols)]:
                                    s_name_file = target_columns_model[i].replace('[','').replace(']','').replace('(','').replace(')','').replace('/','_').replace(' ','_').strip()
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                       else: st.error("Predizione simulazione fallita.")
        elif run_simulation and sim_data_input is None: st.error("Dati input simulazione non pronti/validi.")


# --- PAGINA ANALISI DATI STORICI ---
# (INVARIATA - già usa df_current_csv e feature_columns globali)
elif page == 'Analisi Dati Storici':
    st.header('🔎 Analisi Dati Storici (CSV)')
    if not data_ready_csv:
        st.warning("⚠️ Carica i Dati Storici (CSV) per usare l'Analisi.")
    else:
        st.info(f"Dataset CSV: {len(df_current_csv)} righe, dal {df_current_csv[date_col_name_csv].min().strftime('%d/%m/%Y %H:%M')} al {df_current_csv[date_col_name_csv].max().strftime('%d/%m/%Y %H:%M')}")
        min_date = df_current_csv[date_col_name_csv].min().date()
        max_date = df_current_csv[date_col_name_csv].max().date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input('Data inizio', min_date, min_value=min_date, max_value=max_date, key="analisi_start")
        end_date = col2.date_input('Data fine', max_date, min_value=min_date, max_value=max_date, key="analisi_end")
        if start_date > end_date: st.error("Data inizio successiva a data fine.")
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            if pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]) and df_current_csv[date_col_name_csv].dt.tz is not None:
                 tz_csv = df_current_csv[date_col_name_csv].dt.tz
                 start_dt = tz_csv.localize(start_dt); end_dt = tz_csv.localize(end_dt)
            elif not pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]): st.error("Colonna data CSV non è datetime."); st.stop()

            mask = (df_current_csv[date_col_name_csv] >= start_dt) & (df_current_csv[date_col_name_csv] <= end_dt)
            filtered_df = df_current_csv.loc[mask]
            if len(filtered_df) == 0: st.warning("Nessun dato nel periodo selezionato.")
            else:
                 st.success(f"Trovati {len(filtered_df)} record ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}).")
                 tab1, tab2, tab3 = st.tabs(["Andamento Temporale", "Statistiche/Distribuzione", "Correlazione"])

                 potential_features_analysis = filtered_df.select_dtypes(include=np.number).columns.tolist()
                 potential_features_analysis = [f for f in potential_features_analysis if f not in ['index', 'level_0']]
                 feature_labels_analysis = {get_station_label(f, short=True): f for f in potential_features_analysis}
                 if not feature_labels_analysis: st.warning("Nessuna feature numerica nei dati filtrati."); st.stop()

                 with tab1:
                      st.subheader("Andamento Temporale Features CSV")
                      default_labels_ts = [lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f][:2]
                      if not default_labels_ts: default_labels_ts = list(feature_labels_analysis.keys())[:2]
                      selected_labels_ts = st.multiselect("Seleziona feature", options=list(feature_labels_analysis.keys()), default=default_labels_ts, key="analisi_ts")
                      features_plot = [feature_labels_analysis[lbl] for lbl in selected_labels_ts]
                      if features_plot:
                           fig_ts = go.Figure()
                           for feature in features_plot:
                                legend_name = get_station_label(feature, short=True)
                                fig_ts.add_trace(go.Scatter(x=filtered_df[date_col_name_csv], y=filtered_df[feature], mode='lines', name=legend_name, hovertemplate=f'<b>{legend_name}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'))
                           fig_ts.update_layout(title='Andamento Temporale Selezionato', xaxis_title='Data e Ora', yaxis_title='Valore', height=500, hovermode="x unified")
                           st.plotly_chart(fig_ts, use_container_width=True)
                           st.markdown(get_plotly_download_link(fig_ts, f"andamento_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)
                      else: st.info("Seleziona almeno una feature.")

                 with tab2:
                      st.subheader("Statistiche e Distribuzione")
                      default_stat_label = next((lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f), list(feature_labels_analysis.keys())[0])
                      selected_label_stat = st.selectbox("Seleziona feature", options=list(feature_labels_analysis.keys()), index=list(feature_labels_analysis.keys()).index(default_stat_label), key="analisi_stat")
                      feature_stat = feature_labels_analysis[selected_label_stat]
                      if feature_stat:
                           st.write(f"**Statistiche per: {selected_label_stat}** (`{feature_stat}`)")
                           st.dataframe(filtered_df[[feature_stat]].describe().round(3))
                           st.write(f"**Distribuzione per: {selected_label_stat}**")
                           fig_hist = go.Figure(data=[go.Histogram(x=filtered_df[feature_stat], name=selected_label_stat)])
                           fig_hist.update_layout(title=f'Distribuzione di {selected_label_stat}', xaxis_title='Valore', yaxis_title='Frequenza', height=400)
                           st.plotly_chart(fig_hist, use_container_width=True)
                           st.markdown(get_plotly_download_link(fig_hist, f"distrib_{selected_label_stat.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)

                 with tab3:
                      st.subheader("Matrice di Correlazione")
                      default_corr_labels = list(feature_labels_analysis.keys())
                      selected_labels_corr = st.multiselect("Seleziona feature per correlazione", options=list(feature_labels_analysis.keys()), default=default_corr_labels, key="analisi_corr")
                      features_corr = [feature_labels_analysis[lbl] for lbl in selected_labels_corr]
                      if len(features_corr) > 1:
                           corr_matrix = filtered_df[features_corr].corr()
                           heatmap_labels = [get_station_label(f, short=True) for f in features_corr]
                           fig_hm = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=heatmap_labels, y=heatmap_labels, colorscale='RdBu', zmin=-1, zmax=1, colorbar=dict(title='Corr'), text=corr_matrix.round(2).values, texttemplate="%{text}", hoverongaps=False))
                           fig_hm.update_layout(title='Matrice di Correlazione', height=max(400, len(heatmap_labels)*30), xaxis_tickangle=-45, yaxis_autorange='reversed')
                           st.plotly_chart(fig_hm, use_container_width=True)
                           st.markdown(get_plotly_download_link(fig_hm, f"correlazione_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)
                           if len(selected_labels_corr) <= 10:
                                st.subheader("Scatter Plot Correlazione (2 Feature)")
                                cs1, cs2 = st.columns(2)
                                label_x = cs1.selectbox("Feature X", selected_labels_corr, index=0, key="scat_x")
                                default_y_index = 1 if len(selected_labels_corr) > 1 else 0
                                label_y = cs2.selectbox("Feature Y", selected_labels_corr, index=default_y_index, key="scat_y")
                                fx = feature_labels_analysis.get(label_x); fy = feature_labels_analysis.get(label_y)
                                if fx and fy:
                                    fig_sc = go.Figure(data=[go.Scatter(x=filtered_df[fx], y=filtered_df[fy], mode='markers', marker=dict(size=5, opacity=0.6), name=f'{label_x} vs {label_y}')])
                                    fig_sc.update_layout(title=f'Correlazione: {label_x} vs {label_y}', xaxis_title=label_x, yaxis_title=label_y, height=500)
                                    st.plotly_chart(fig_sc, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sc, f"scatter_{label_x.replace(' ','_')}_vs_{label_y.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)
                           else: st.info("Troppe feature selezionate per scatter plot interattivo.")
                      else: st.info("Seleziona almeno due feature per correlazione.")
                 st.divider()
                 st.subheader('Download Dati Filtrati CSV')
                 st.markdown(get_table_download_link(filtered_df, f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)


# --- PAGINA ALLENAMENTO MODELLO ---
# (INVARIATA - già usa df_current_csv e feature_columns globali)
elif page == 'Allenamento Modello':
    st.header('🎓 Allenamento Nuovo Modello LSTM')
    if not data_ready_csv:
        st.warning("⚠️ Dati storici CSV non caricati. Carica un file CSV valido.")
    else:
        st.success(f"Dati CSV disponibili: {len(df_current_csv)} righe.")
        st.subheader('Configurazione Addestramento')
        default_save_name = f"modello_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
        save_name_input = st.text_input("Nome base per salvare modello (a-z, A-Z, 0-9, _, -)", default_save_name, key="train_save_name")
        save_name = re.sub(r'[^a-zA-Z0-9_-]', '_', save_name_input)
        if save_name != save_name_input: st.caption(f"Nome file corretto in: `{save_name}`")

        st.write("**1. Seleziona Feature Input:**")
        all_available_features = st.session_state.feature_columns # Feature globali definite
        # Ma considera solo quelle presenti nel CSV caricato!
        features_in_csv = [f for f in all_available_features if f in df_current_csv.columns]
        if len(features_in_csv) < len(all_available_features):
            st.caption(f"Nota: Alcune feature globali non sono nel CSV e non sono selezionabili: {', '.join(set(all_available_features) - set(features_in_csv))}")

        selected_features_train = []
        with st.expander("Seleziona Feature Input Modello", expanded=False):
            cols_feat = st.columns(4)
            for i, feat in enumerate(features_in_csv): # Mostra solo quelle nel CSV
                 label_feat_train = get_station_label(feat, short=True)
                 with cols_feat[i % len(cols_feat)]:
                      if st.checkbox(label_feat_train, value=True, key=f"train_feat_{feat}", help=feat):
                          selected_features_train.append(feat)
        if not selected_features_train: st.warning("Seleziona almeno una feature di input.")

        st.write("**2. Seleziona Target Output (Livelli):**")
        selected_targets_train = []
        hydro_features_options_train = [f for f in selected_features_train if 'Livello' in f] # Solo tra quelle selezionate E di livello
        if not hydro_features_options_train: st.warning("Nessuna colonna 'Livello' tra le feature input selezionate.")
        else:
            default_targets_train = []
            if model_ready and active_config.get("target_columns"):
                 valid_active_targets = [t for t in active_config["target_columns"] if t in hydro_features_options_train]
                 if valid_active_targets: default_targets_train = valid_active_targets
            if not default_targets_train: default_targets_train = hydro_features_options_train[:1] # Fallback al primo
            cols_t = st.columns(min(len(hydro_features_options_train), 5))
            for i, feat in enumerate(hydro_features_options_train):
                with cols_t[i % len(cols_t)]:
                     lbl = get_station_label(feat, short=True)
                     if st.checkbox(lbl, value=(feat in default_targets_train), key=f"train_target_{feat}", help=feat):
                         selected_targets_train.append(feat)
        if not selected_targets_train: st.warning("Seleziona almeno un target.")

        st.write("**3. Imposta Parametri:**")
        with st.expander("Parametri Modello e Training", expanded=True):
             c1t, c2t, c3t = st.columns(3)
             default_iw = active_config["input_window"] if model_ready else 24; default_ow = active_config["output_window"] if model_ready else 12
             default_hs = active_config["hidden_size"] if model_ready else 128; default_nl = active_config["num_layers"] if model_ready else 2
             default_dr = active_config["dropout"] if model_ready else 0.2; default_bs = active_config.get("batch_size", 32) if model_ready else 32
             default_vs = active_config.get("val_split_percent", 20) if model_ready else 20; default_lr = active_config.get("learning_rate", 0.001) if model_ready else 0.001
             default_ep = active_config.get("epochs_run", 50) if model_ready else 50
             iw_t = c1t.number_input("Input Win (h)", 6, 168, default_iw, 6, key="t_in")
             ow_t = c1t.number_input("Output Win (h)", 1, 72, default_ow, 1, key="t_out")
             vs_t = c1t.slider("% Validazione", 0, 50, default_vs, 1, key="t_val")
             hs_t = c2t.number_input("Hidden Size", 16, 1024, default_hs, 16, key="t_hid")
             nl_t = c2t.number_input("Num Layers", 1, 8, default_nl, 1, key="t_lay")
             dr_t = c2t.slider("Dropout", 0.0, 0.7, default_dr, 0.05, key="t_drop")
             lr_t = c3t.number_input("Learning Rate", 1e-5, 1e-2, default_lr, format="%.5f", step=1e-4, key="t_lr")
             bs_t = c3t.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], default_bs, key="t_batch")
             ep_t = c3t.number_input("Epoche", 5, 500, default_ep, 5, key="t_epochs")

        st.write("**4. Avvia Addestramento:**")
        valid_name = bool(save_name); valid_features = bool(selected_features_train); valid_targets = bool(selected_targets_train)
        ready_to_train = valid_name and valid_features and valid_targets
        if not valid_features: st.warning("Seleziona feature input.")
        if not valid_targets: st.warning("Seleziona target.")
        if not valid_name: st.warning("Inserisci nome valido.")

        train_button = st.button("Addestra Nuovo Modello", type="primary", disabled=not ready_to_train, key="train_run")
        if train_button and ready_to_train:
             st.info(f"Avvio addestramento per '{save_name}'...")
             with st.spinner('Preparazione dati...'):
                  training_features_selected = selected_features_train
                  st.caption(f"Input: {len(training_features_selected)} feature")
                  st.caption(f"Output: {', '.join(selected_targets_train)}")
                  cols_to_check_df = training_features_selected + selected_targets_train
                  missing_in_df = [c for c in cols_to_check_df if c not in df_current_csv.columns]
                  if missing_in_df: st.error(f"Errore: Colonne mancanti nel DataFrame: {', '.join(missing_in_df)}"); st.stop()

                  X_tr, y_tr, X_v, y_v, sc_f_tr, sc_t_tr = prepare_training_data(df_current_csv.copy(), training_features_selected, selected_targets_train, iw_t, ow_t, vs_t)
                  if X_tr is None: st.error("Preparazione dati fallita."); st.stop()
                  st.success(f"Dati pronti: {len(X_tr)} train, {len(X_v)} val.")

             st.subheader("Addestramento...")
             input_size_train = len(training_features_selected); output_size_train = len(selected_targets_train)
             trained_model = None
             try:
                 trained_model, train_losses, val_losses = train_model(X_tr, y_tr, X_v, y_v, input_size_train, output_size_train, ow_t, hs_t, nl_t, ep_t, bs_t, lr_t, dr_t)
             except Exception as e_train: st.error(f"Errore training: {e_train}"); st.error(traceback.format_exc())

             if trained_model:
                 st.success("Addestramento completato!")
                 st.subheader("Salvataggio Risultati")
                 os.makedirs(MODELS_DIR, exist_ok=True)
                 base_path = os.path.join(MODELS_DIR, save_name)
                 m_path = f"{base_path}.pth"; c_path = f"{base_path}.json"; sf_path = f"{base_path}_features.joblib"; st_path = f"{base_path}_targets.joblib"
                 final_val_loss = None
                 if val_losses and vs_t > 0: valid_val_losses = [v for v in val_losses if v is not None]; final_val_loss = min(valid_val_losses) if valid_val_losses else None
                 config_save = {
                     "input_window": iw_t, "output_window": ow_t, "hidden_size": hs_t, "num_layers": nl_t, "dropout": dr_t,
                     "target_columns": selected_targets_train, "feature_columns": training_features_selected, # Salva feature usate
                     "training_date": datetime.now(italy_tz).isoformat(), "final_val_loss": final_val_loss,
                     "epochs_run": ep_t, "batch_size": bs_t, "val_split_percent": vs_t, "learning_rate": lr_t,
                     "display_name": save_name, "source_data_info": data_source_info
                 }
                 try:
                     torch.save(trained_model.state_dict(), m_path)
                     with open(c_path, 'w') as f: json.dump(config_save, f, indent=4)
                     joblib.dump(sc_f_tr, sf_path); joblib.dump(sc_t_tr, st_path)
                     st.success(f"Modello '{save_name}' salvato in '{MODELS_DIR}/'")
                     st.subheader("Download File Modello")
                     col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                     with col_dl1: st.markdown(get_download_link_for_file(m_path, "Modello (.pth)"), unsafe_allow_html=True)
                     with col_dl2: st.markdown(get_download_link_for_file(c_path, "Config (.json)"), unsafe_allow_html=True)
                     with col_dl3: st.markdown(get_download_link_for_file(sf_path, "Scaler Feat (.joblib)"), unsafe_allow_html=True)
                     with col_dl4: st.markdown(get_download_link_for_file(st_path, "Scaler Targ (.joblib)"), unsafe_allow_html=True)
                     if st.button("Ricarica App per aggiornare lista modelli"): st.session_state.clear(); st.rerun() # Pulisci tutto lo stato
                 except Exception as e_save: st.error(f"Errore salvataggio file: {e_save}"); st.error(traceback.format_exc())
             elif not train_button: pass
             else: st.error("Addestramento fallito/interrotto. Impossibile salvare.")


# --- Footer ---
st.sidebar.divider()
st.sidebar.info('App Idrologica Dashboard & Predict © 2025 tutti i diritti riservati a Alberto Bussaglia')
