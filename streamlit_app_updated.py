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
# Le chiavi DEVONO corrispondere ESATTAMENTE ai nomi delle colonne in GSHEET_RELEVANT_COLS
# Sostituisci lat/lon con le coordinate REALI dei tuoi sensori.
# location_id raggruppa sensori nello stesso punto fisico.
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
# (INVARIATE)
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
        station_name_graph = sensor.split(' - ')[0] if ' - ' in sensor else sensor.split(' [')[0]

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsto'))
        fig.update_layout(
            title=f'Previsione - {station_name_graph}',
            xaxis_title=x_title,
            yaxis_title=f'{sensor.split("(")[-1].split(")")[0].strip()}', # Estrae unità
            height=400,
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        figs.append(fig)
    return figs

# --- NUOVA FUNZIONE: Fetch Dati Dashboard da Google Sheet (con gestione cache modificata) ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format):
    """
    Importa l'ultima riga di dati dal Google Sheet specificato per la dashboard.
    Pulisce e converte i dati numerici, gestendo la virgola come separatore decimale.
    L'argomento _cache_key_time è usato per influenzare la cache di Streamlit.
    """
    # _cache_key_time non viene usato direttamente, ma la sua presenza e il suo
    # cambiamento (basato sull'ora arrotondata) aiutano a invalidare la cache
    # in modo controllato insieme al ttl.
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data (Cache Key Time: {_cache_key_time})") # Debug
    actual_fetch_time = datetime.now(italy_tz) # Tempo dell'esecuzione effettiva
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets:
            st.error("Credenziali Google non trovate nei secrets di Streamlit.")
            return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        credentials = Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"],
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        data = worksheet.get_all_values() # Fetch dei dati grezzi

        if not data or len(data) < 2:
            return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = data[0]
        last_row_values = data[-1] # Prende l'ultima riga

        # Verifica che tutte le colonne richieste siano presenti
        headers_set = set(headers)
        missing_cols = [col for col in relevant_columns if col not in headers_set]
        if missing_cols:
            return None, f"Errore: Colonne GSheet mancanti: {', '.join(missing_cols)}", actual_fetch_time

        # Crea un dizionario con i dati dell'ultima riga, usando gli header come chiavi
        last_data_dict = dict(zip(headers, last_row_values))

        # Seleziona solo le colonne rilevanti
        filtered_data = {col: last_data_dict.get(col) for col in relevant_columns}

        # Converti e pulisci i dati
        cleaned_data = {}
        error_parsing = []
        for col, value in filtered_data.items():
            if value is None or value in ['N/A', '', '-', ' ']:
                cleaned_data[col] = np.nan
                continue # Salta alla prossima colonna

            if col == date_col:
                try:
                    # Parse datetime
                    dt_naive = datetime.strptime(str(value), date_format)
                    # Assume the sheet time is already in Italy timezone, make it aware
                    cleaned_data[col] = italy_tz.localize(dt_naive)
                except ValueError:
                    error_parsing.append(f"Formato data non valido per '{col}': '{value}' (atteso: {date_format})")
                    cleaned_data[col] = pd.NaT # Usa NaT per date non valide
            else: # Colonne numeriche
                try:
                    # Sostituisci la virgola con il punto per la conversione decimale
                    numeric_value = float(str(value).replace(',', '.'))
                    cleaned_data[col] = numeric_value
                except ValueError:
                    error_parsing.append(f"Valore non numerico per '{col}': '{value}'")
                    cleaned_data[col] = np.nan

        if error_parsing:
            # Restituisce i dati parzialmente puliti ma segnala gli errori
            error_message = "Attenzione: Errori durante la conversione dei dati. " + " | ".join(error_parsing)
            return pd.Series(cleaned_data), error_message, actual_fetch_time

        # Restituisce una Pandas Series per facilità d'uso
        return pd.Series(cleaned_data), None, actual_fetch_time # Nessun errore

    except gspread.exceptions.APIError as api_e:
        try:
            # Tenta di ottenere un messaggio di errore più dettagliato dall'API
            error_details = api_e.response.json()
            error_message = error_details.get('error', {}).get('message', str(api_e))
            status_code = error_details.get('error', {}).get('code', 'N/A')
            if status_code == 403: # Forbidden / Permission Denied
                 error_message += " Verifica che l'email del service account sia stata condivisa con il foglio Google con permessi di lettura."
            elif status_code == 429: # Rate Limit Exceeded
                 error_message += f" Limite richieste API superato. Riprova tra qualche minuto. TTL cache: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."

        except: # Fallback se la risposta non è JSON o manca la struttura attesa
            error_message = str(api_e)
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
        # Verifica se kaleido è installato
        import kaleido
        buf_png = io.BytesIO(); fig.write_image(buf_png, format="png") # Richiede kaleido
        buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode()
        href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
    except ImportError:
        # st.warning("Libreria 'kaleido' non trovata. Download PNG disabilitato. Installa con: pip install kaleido")
        pass
    except Exception as e_png:
        # st.warning(f"Errore esportazione PNG: {e_png}")
        pass # Silenzia altri errori PNG
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
def get_station_label(col_name, short=False):
    """Estrae un'etichetta leggibile dal nome colonna GSheet."""
    # Prima prova a cercare una corrispondenza in STATION_COORDS per usare il location_id
    if col_name in STATION_COORDS:
        location_id = STATION_COORDS[col_name].get('location_id')
        if location_id:
            if short:
                # Per short, potremmo voler aggiungere il tipo se ci sono più sensori
                sensor_type = STATION_COORDS[col_name].get('type', '')
                # Verifica se ci sono altri sensori nella stessa location
                sensors_at_loc = [
                    sc['type'] for sc_name, sc in STATION_COORDS.items()
                    if sc.get('location_id') == location_id
                ]
                if len(sensors_at_loc) > 1:
                    # Aggiungi tipo per disambiguare se breve
                    type_abbr = 'P' if sensor_type == 'Pioggia' else ('L' if sensor_type == 'Livello' else '')
                    return f"{location_id} ({type_abbr})"[:25]
                else:
                    return location_id[:25] # Solo nome location se unico sensore
            else:
                return location_id # Per metriche/mappa, usa solo il nome della location

    # Fallback se non in STATION_COORDS o manca location_id
    parts = col_name.split(' - ')
    if len(parts) > 1:
        location = parts[0].strip()
        measurement = parts[1].split(' (')[0].strip()
        if short:
            return f"{location} - {measurement}"[:25] # Limita lunghezza
        else:
            return location # Usa solo la location come label principale
    else:
        # Fallback se non c'è ' - '
        return col_name.split(' (')[0].strip()[:25]


# --- Inizializzazione Session State ---
# (AGGIUNTE chiavi per la dashboard)
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scaler_features' not in st.session_state: st.session_state.active_scaler_features = None
if 'active_scaler_targets' not in st.session_state: st.session_state.active_scaler_targets = None
if 'df' not in st.session_state: st.session_state.df = None
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [ # Queste sono per il MODELLO, non per la dashboard GSheet
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
# NUOVE CHIAVI per la Dashboard
if 'dashboard_thresholds' not in st.session_state: st.session_state.dashboard_thresholds = DEFAULT_THRESHOLDS.copy()
if 'last_dashboard_data' not in st.session_state: st.session_state.last_dashboard_data = None
if 'last_dashboard_error' not in st.session_state: st.session_state.last_dashboard_error = None
if 'last_dashboard_fetch_time' not in st.session_state: st.session_state.last_dashboard_fetch_time = None # NUOVO
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
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1'] # Aggiunta iso-8859-1
        df_loaded = False
        for enc in encodings_to_try:
            try:
                if is_uploaded: data_path_to_load.seek(0)
                df = pd.read_csv(data_path_to_load, encoding=enc, **read_args)
                df_loaded = True; break
            except UnicodeDecodeError: continue
            except Exception as read_e: raise read_e # Altri errori di lettura
        if not df_loaded: raise ValueError(f"Impossibile leggere CSV con encoding {', '.join(encodings_to_try)}.")

        date_col_csv = st.session_state.date_col_name_csv
        if date_col_csv not in df.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")
        try: df[date_col_csv] = pd.to_datetime(df[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
            try: # Prova fallback inferenza
                 df[date_col_csv] = pd.to_datetime(df[date_col_csv], errors='coerce')
                 st.sidebar.warning("Formato data CSV non standard, tentata inferenza.")
            except Exception as e_date_csv:
                 raise ValueError(f"Errore conversione data CSV '{date_col_csv}': {e_date_csv}")

        df = df.dropna(subset=[date_col_csv])
        df = df.sort_values(by=date_col_csv).reset_index(drop=True)

        current_f_cols = st.session_state.feature_columns
        missing_features = [col for col in current_f_cols if col not in df.columns]
        if missing_features: raise ValueError(f"Colonne feature CSV mancanti: {', '.join(missing_features)}")

        # Pulizia colonne numeriche più robusta
        for col in current_f_cols:
             if df[col].dtype == 'object':
                  # 1. Rimuovi spazi bianchi iniziali/finali
                  df[col] = df[col].astype(str).str.strip()
                  # 2. Sostituisci 'N/A', '', '-', 'None', 'null' (case-insensitive) con NaN
                  df[col] = df[col].replace(['N/A', '', '-', 'None', 'null'], np.nan, regex=True)
                  # 3. Rimuovi separatori migliaia (punti), poi sostituisci virgola decimale
                  df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
             # 4. Converti in numerico, forzando errori a NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')


        n_nan = df[current_f_cols].isnull().sum().sum()
        if n_nan > 0:
              st.sidebar.caption(f"Trovati {n_nan} NaN/valori non numerici nel CSV. Eseguito ffill/bfill.")
              df[current_f_cols] = df[current_f_cols].fillna(method='ffill').fillna(method='bfill')
              if df[current_f_cols].isnull().sum().sum() > 0:
                  st.sidebar.error("NaN residui dopo fill. Controlla inizio/fine del file CSV.")
                  # Potrebbe essere utile mostrare quali colonne hanno ancora NaN
                  nan_cols = df[current_f_cols].isnull().sum()
                  st.sidebar.json(nan_cols[nan_cols > 0].to_dict())
                  # Per sicurezza, si potrebbe riempire con 0 o mediana
                  # df[current_f_cols] = df[current_f_cols].fillna(0)
                  # st.sidebar.warning("NaN residui riempiti con 0.")

        st.session_state.df = df
        st.sidebar.success(f"Dati CSV caricati ({len(df)} righe).")
    except Exception as e:
        df = None; st.session_state.df = None
        df_load_error = f'Errore dati CSV ({data_source_info}): {type(e).__name__} - {e}'
        st.sidebar.error(f"Errore CSV: {df_load_error}")
        # st.sidebar.code(traceback.format_exc()) # Per debug locale

df = st.session_state.get('df', None) # Recupera df aggiornato
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
# Resetta sempre lo stato attivo prima di caricare/selezionare
st.session_state.active_model_name = None; st.session_state.active_config = None
st.session_state.active_model = None; st.session_state.active_device = None
st.session_state.active_scaler_features = None; st.session_state.active_scaler_targets = None

if selected_model_display_name == MODEL_CHOICE_NONE:
    # st.sidebar.caption("Nessun modello selezionato.")
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
        targets_global = [col for col in st.session_state.feature_columns if 'Livello' in col]
        targets_up = st.multiselect("Target", targets_global, default=targets_global, key="up_targets")
        if m_f and sf_f and st_f and targets_up:
            # Assicurati che le feature columns siano quelle globali correnti
            current_model_features = st.session_state.feature_columns
            temp_cfg = {"input_window": iw, "output_window": ow, "hidden_size": hs,
                        "num_layers": nl, "dropout": dr, "target_columns": targets_up,
                        "feature_columns": current_model_features, # USA quelle correnti
                        "name": "uploaded"}
            model_to_load, device_to_load = load_specific_model(m_f, temp_cfg)
            scaler_f_to_load, scaler_t_to_load = load_specific_scalers(sf_f, st_f)
            if model_to_load and scaler_f_to_load and scaler_t_to_load: config_to_load = temp_cfg
            else: load_error_sidebar = True
        else: st.caption("Carica tutti i file e scegli i target.")
else: # Modello pre-addestrato
    model_info = available_models_dict[selected_model_display_name]
    st.session_state.active_model_name = selected_model_display_name
    # st.sidebar.caption(f"Caricamento: **{selected_model_display_name}**") # Meno verbose
    config_to_load = load_model_config(model_info["config_path"])
    if config_to_load:
        config_to_load["pth_path"] = model_info["pth_path"]
        config_to_load["scaler_features_path"] = model_info["scaler_features_path"]
        config_to_load["scaler_targets_path"] = model_info["scaler_targets_path"]
        config_to_load["name"] = model_info["config_name"]
        # IMPORTANTE: Se la config non ha le feature columns, usa quelle globali
        if "feature_columns" not in config_to_load or not config_to_load["feature_columns"]:
             st.warning(f"Config '{selected_model_display_name}' non specifica le feature_columns. Uso quelle globali.")
             config_to_load["feature_columns"] = st.session_state.feature_columns
        # Verifica consistenza tra feature config e quelle globali (se necessario)
        # ... (logica opzionale di verifica) ...

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
    st.session_state.active_model_name = selected_model_display_name # Salva il nome corretto

# Mostra feedback basato sullo stato sessione aggiornato
if st.session_state.active_model and st.session_state.active_config:
    cfg = st.session_state.active_config
    active_name = st.session_state.active_model_name
    st.sidebar.success(f"Modello ATTIVO: '{active_name}' (In:{cfg['input_window']}h, Out:{cfg['output_window']}h)")
elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]: st.sidebar.error("Caricamento modello fallito.")
elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not st.session_state.active_model: st.sidebar.info("Completa caricamento manuale modello.")
# Nessun messaggio se NESSUN modello è selezionato intenzionalmente


# --- Configurazione Soglie Dashboard (nella Sidebar) ---
st.sidebar.divider()
st.sidebar.subheader("Configurazione Soglie Dashboard")
with st.sidebar.expander("Modifica Soglie di Allerta", expanded=False):
    temp_thresholds = st.session_state.dashboard_thresholds.copy()
    # Colonne monitorabili (escludi data)
    monitorable_cols = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
    for col in monitorable_cols:
        # Estrai nome più breve per label nella sidebar usando la nuova funzione
        label_short = get_station_label(col, short=True) # Usa nuova funzione per label breve

        is_level = 'Livello' in col or '(m)' in col or '(mt)' in col
        step = 0.1 if is_level else 1.0
        fmt = "%.1f" if is_level else "%.0f"
        min_v = 0.0 # Le soglie non dovrebbero essere negative

        current_threshold = st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, 0.0)) # Usa default se non ancora in state

        # Usa st.number_input per permettere modifica
        new_threshold = st.number_input(
            label=f"Soglia {label_short}",
            value=current_threshold,
            min_value=min_v,
            step=step,
            format=fmt,
            key=f"thresh_{col}",
            help=f"Imposta la soglia di allerta per: {col}" # Tooltip con nome completo
        )
        # Aggiorna il dizionario temporaneo solo se il valore cambia
        if new_threshold != current_threshold:
             temp_thresholds[col] = new_threshold

    # Bottone per salvare le modifiche alle soglie
    if st.button("Salva Soglie", key="save_thresholds"):
        st.session_state.dashboard_thresholds = temp_thresholds.copy()
        st.success("Soglie aggiornate!")
        st.rerun() # CORRETTO: Ricarica per rendere effettive le nuove soglie


# --- Menu Navigazione ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
model_ready = st.session_state.active_model is not None and st.session_state.active_config is not None
data_ready_csv = df is not None # Dati CSV per analisi/training/simulazione CSV

# La dashboard ora non dipende più da modello/CSV
radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
# Definisci quali pagine richiedono cosa
requires_model = ['Simulazione']
requires_csv = ['Analisi Dati Storici', 'Allenamento Modello']

radio_captions = []
disabled_options = []
for opt in radio_options:
    caption = ""
    disabled = False
    if opt == 'Dashboard':
        caption = "Monitoraggio GSheet"
        # La dashboard richiede solo le credenziali GSheet (controllato all'interno)
    elif opt in requires_model and not model_ready:
        caption = "Richiede Modello attivo"
        disabled = True
    elif opt in requires_csv and not data_ready_csv:
        caption = "Richiede Dati CSV"
        disabled = True
    elif opt == 'Simulazione' and model_ready and not data_ready_csv: # Simulazione richiede modello, ma alcune opzioni usano CSV
         caption = "Esegui previsioni (CSV non disp.)" # Ok, ma con opzioni limitate
         disabled = False # Lascia accessibile
    elif opt == 'Simulazione' and model_ready and data_ready_csv:
         caption = "Esegui previsioni custom"
    elif opt == 'Analisi Dati Storici' and data_ready_csv:
         caption = "Esplora dati CSV caricati"
    elif opt == 'Allenamento Modello' and data_ready_csv:
         caption = "Allena un nuovo modello"
    else: # Caso di default o pagina OK
         if opt == 'Dashboard': caption = "Monitoraggio GSheet"
         elif opt == 'Simulazione': caption = "Esegui previsioni custom"
         elif opt == 'Analisi Dati Storici': caption = "Esplora dati CSV caricati"
         elif opt == 'Allenamento Modello': caption = "Allena un nuovo modello"

    radio_captions.append(caption)
    disabled_options.append(disabled)

# Logica selezione pagina (INVARIATA ma con nuove condizioni `disabled_options`)
current_page_index = 0
try:
     if 'current_page' in st.session_state:
          current_page_index = radio_options.index(st.session_state.current_page)
          if disabled_options[current_page_index]:
               st.sidebar.warning(f"Pagina '{st.session_state.current_page}' non disponibile. Reindirizzato a Dashboard.")
               current_page_index = 0 # Torna alla Dashboard se la pagina salvata è disabilitata
     else: # Seleziona la prima pagina disponibile
          current_page_index = next((i for i, disabled in enumerate(disabled_options) if not disabled), 0)
except ValueError: current_page_index = 0

page = st.sidebar.radio(
    'Scegli una funzionalità',
    options=radio_options,
    captions=radio_captions,
    index=current_page_index,
    # disabled=disabled_options, # Potrebbe dare problemi con captions < 1.28
    key='page_selector'
)
if not disabled_options[radio_options.index(page)]:
     st.session_state.current_page = page
else: # Se si seleziona manualmente una pagina disabilitata (non dovrebbe succedere con index logic)
     st.warning(f"La pagina '{page}' non è accessibile con le impostazioni correnti. Vai alla Dashboard.")
     st.session_state.current_page = radio_options[0] # Forza Dashboard
     # Forza il rerun per cambiare pagina visualizzata
     time.sleep(0.5) # Breve pausa per mostrare il messaggio
     st.rerun()


# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================

# Leggi stato attivo per le altre pagine
active_config = st.session_state.active_config
active_model = st.session_state.active_model
active_device = st.session_state.active_device
active_scaler_features = st.session_state.active_scaler_features
active_scaler_targets = st.session_state.active_scaler_targets
df_current_csv = st.session_state.get('df', None) # Dati CSV
# IMPORTANTE: Le feature columns per il modello devono essere quelle del MODELLO ATTIVO
feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns) if active_config else st.session_state.feature_columns
date_col_name_csv = st.session_state.date_col_name_csv

# --- PAGINA DASHBOARD (RIVISTA con MAPPA e LAYOUT MIGLIORATO) ---
if page == 'Dashboard':
    st.header(f'📊 Dashboard Monitoraggio Idrologico')

    # Verifica credenziali Google
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("🚨 **Errore Configurazione:** Credenziali Google ('GOOGLE_CREDENTIALS') non trovate nei secrets di Streamlit. Impossibile accedere al Google Sheet.")
        st.info("Aggiungi le credenziali del service account Google come secret 'GOOGLE_CREDENTIALS' per abilitare la dashboard.")
        st.stop() # Blocca l'esecuzione della pagina

    # --- Logica Fetch Dati con Gestione Cache ---
    # Crea una chiave per la cache basata sull'ora corrente arrotondata all'intervallo di refresh
    now_ts = time.time()
    cache_time_key = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)

    # Chiama la funzione cachata passando la chiave temporale
    latest_data, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
        cache_time_key, # Argomento per influenzare la cache
        GSHEET_ID,
        GSHEET_RELEVANT_COLS,
        GSHEET_DATE_COL,
        GSHEET_DATE_FORMAT
    )

    # Salva in session state
    st.session_state.last_dashboard_data = latest_data
    st.session_state.last_dashboard_error = error_msg
    # Salva il tempo dell'ULTIMA ESECUZIONE EFFETTIVA della funzione di fetch
    if latest_data is not None or error_msg is None : # Salva solo se il fetch è stato eseguito (anche se con errori di parsing)
        st.session_state.last_dashboard_fetch_time = actual_fetch_time

    # --- Visualizzazione Dati e Mappa ---
    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        if st.session_state.last_dashboard_fetch_time:
            last_fetch_dt = st.session_state.last_dashboard_fetch_time
            fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt
            fetch_secs_ago = int(fetch_time_ago.total_seconds())
            st.caption(f"Dati da GSheet recuperati l'ultima volta alle: {last_fetch_dt.strftime('%d/%m/%Y %H:%M:%S')} ({fetch_secs_ago}s fa). Refresh automatico ogni {DASHBOARD_REFRESH_INTERVAL_SECONDS}s.")
        else:
            st.caption("In attesa del primo recupero dati da Google Sheet...")

    with col_refresh_btn:
        if st.button("🔄 Forza Aggiorna", key="dash_refresh"):
            # Cancella esplicitamente la cache per questa funzione specifica
            fetch_gsheet_dashboard_data.clear()
            st.success("Cache GSheet pulita. Ricaricamento...")
            time.sleep(0.5) # Pausa per vedere messaggio
            st.rerun()

    # Mostra errore se presente
    if error_msg:
        if "API" in error_msg or "Foglio Google non trovato" in error_msg or "Credenziali" in error_msg:
             st.error(f"🚨 {error_msg}")
        else: # Errori di parsing o colonne mancanti sono warning
             st.warning(f"⚠️ {error_msg}")

    # Mostra dati se disponibili
    if latest_data is not None:
        # Timestamp ultimo rilevamento NEI DATI
        last_update_time = latest_data.get(GSHEET_DATE_COL)
        time_now_italy = datetime.now(italy_tz)

        if pd.notna(last_update_time):
             # Assicurati che sia timezone-aware (dovrebbe esserlo già dalla funzione fetch)
             if last_update_time.tzinfo is None:
                 last_update_time = italy_tz.localize(last_update_time)

             time_delta = time_now_italy - last_update_time
             minutes_ago = int(time_delta.total_seconds() // 60)
             time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
             if minutes_ago < 0: # Data nel futuro? Strano.
                 time_ago_str = "nel futuro?"
                 st.warning(f"⚠️ L'ultimo timestamp nei dati ({time_str}) sembra essere nel futuro!")
             elif minutes_ago < 2: time_ago_str = "pochi istanti fa"
             elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
             else: time_ago_str = f"{minutes_ago // 60}h {minutes_ago % 60}min fa"

             st.success(f"**Ultimo rilevamento nei dati:** {time_str} ({time_ago_str})")
             if minutes_ago > 30: # Soglia di avviso per dati vecchi
                  st.warning(f"⚠️ Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa.")
        else:
             st.warning("⚠️ Timestamp ultimo rilevamento non disponibile o non valido nei dati GSheet.")

        st.divider()

        # --- Layout Dashboard: Mappa a sinistra, Metriche a destra ---
        map_col, metrics_col = st.columns([2, 1]) # Mappa occupa 2/3, metriche 1/3

        # --- Pre-Processamento Dati per Mappa e Metriche ---
        locations_data = {} # Dizionario per aggregare dati per location_id
        current_alerts = [] # Lista di alert per questo ciclo
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]

        for col_name in cols_to_monitor:
            current_value = latest_data.get(col_name)
            threshold = st.session_state.dashboard_thresholds.get(col_name)
            alert_active = False

            if pd.notna(current_value) and threshold is not None and current_value >= threshold:
                alert_active = True
                current_alerts.append((col_name, current_value, threshold))

            # Controlla se la colonna ha coordinate definite
            if col_name in STATION_COORDS:
                coord_info = STATION_COORDS[col_name]
                loc_id = coord_info.get('location_id')
                lat = coord_info.get('lat')
                lon = coord_info.get('lon')
                sensor_type = coord_info.get('type', 'Sconosciuto')
                sensor_name = coord_info.get('name', col_name) # Nome specifico del sensore
                unit = '(mm)' if 'Pioggia' in col_name else ('(m)' if ('Livello' in col_name or '(m)' in col_name or '(mt)' in col_name) else '')

                if loc_id and lat is not None and lon is not None:
                    # Inizializza la location se non esiste
                    if loc_id not in locations_data:
                        locations_data[loc_id] = {
                            'lat': lat,
                            'lon': lon,
                            'location_name': loc_id, # Nome generico della location
                            'sensors': [],
                            'overall_alert': False, # Stato di allerta aggregato per la location
                            'has_data': False, # Flag se almeno un sensore ha dati validi
                            'types': set() # Tipi di sensori presenti (Pioggia, Livello)
                        }

                    # Aggiungi dati del sensore corrente alla location
                    sensor_info = {
                        'col_name': col_name,
                        'name': sensor_name,
                        'value': current_value,
                        'unit': unit,
                        'threshold': threshold,
                        'alert': alert_active,
                        'type': sensor_type
                    }
                    locations_data[loc_id]['sensors'].append(sensor_info)
                    locations_data[loc_id]['types'].add(sensor_type)

                    # Aggiorna stato aggregato della location
                    if alert_active:
                        locations_data[loc_id]['overall_alert'] = True
                    if pd.notna(current_value):
                        locations_data[loc_id]['has_data'] = True


        # --- Colonna Metriche ---
        with metrics_col:
             st.subheader("Valori Attuali")
             # Mostra metriche per ogni sensore monitorato
             for col_name in cols_to_monitor:
                 current_value = latest_data.get(col_name)
                 threshold = st.session_state.dashboard_thresholds.get(col_name)
                 alert_active = False
                 if pd.notna(current_value) and threshold is not None and current_value >= threshold:
                      alert_active = True

                 # Estrai nome stazione per metrica (usa la funzione helper)
                 label_metric = get_station_label(col_name, short=False) # Nome location
                 unit = '(mm)' if 'Pioggia' in col_name else ('(m)' if ('Livello' in col_name or '(m)' in col_name or '(mt)' in col_name) else '')

                 # Se ci sono più sensori nella stessa location, potremmo voler aggiungere il tipo
                 location_id_metric = STATION_COORDS.get(col_name, {}).get('location_id')
                 display_label = label_metric
                 if location_id_metric:
                     sensors_at_loc = [
                         sc['type'] for sc_name, sc in STATION_COORDS.items()
                         if sc.get('location_id') == location_id_metric
                     ]
                     if len(sensors_at_loc) > 1:
                          sensor_type_metric = STATION_COORDS[col_name].get('type', '')
                          type_abbr = 'Pioggia' if sensor_type_metric == 'Pioggia' else ('Livello' if sensor_type_metric == 'Livello' else '')
                          if type_abbr: display_label = f"{label_metric} ({type_abbr})"


                 if pd.isna(current_value):
                    st.metric(label=f"{display_label}", value="N/D", delta="Dato mancante", delta_color="off", help=col_name)
                 else:
                    # Formattazione valore
                    value_str = f"{current_value:.1f}{unit}" if unit == '(mm)' else f"{current_value:.2f}{unit}"

                    delta_str = None
                    delta_color = "off" # 'normal', 'inverse', 'off'

                    if alert_active:
                        delta_str = f"Sopra soglia ({threshold:.1f})"
                        delta_color = "inverse" # Rosso per superamento

                    st.metric(label=f"{display_label}", value=value_str, delta=delta_str, delta_color=delta_color, help=col_name) # Help mostra nome completo


        # --- Colonna Mappa ---
        with map_col:
            st.subheader("Mappa Stazioni")
            if not locations_data:
                 st.warning("Nessuna coordinata definita o stazione trovata per la mappa.")
            else:
                map_plot_data = []
                for loc_id, data in locations_data.items():
                    # Determina stato e icona aggregati
                    is_alert = data['overall_alert']
                    has_data = data['has_data']
                    types = data['types']

                    color = 'grey'
                    size = 8
                    symbol = 'circle' # Default

                    if has_data:
                         if is_alert:
                             color = 'red'
                             size = 15
                         else:
                             color = 'green'
                             size = 10

                    # Scegli simbolo base su tipi
                    if 'Pioggia' in types and 'Livello' in types:
                         symbol = 'star'
                    elif 'Pioggia' in types:
                         symbol = 'circle' #'cloud' non è standard in scattermapbox, usiamo circle
                    elif 'Livello' in types:
                         symbol = 'triangle-up'
                    else: # Caso senza tipo noto o solo tipi sconosciuti
                         symbol = 'square'


                    # Costruisci hover text aggregato
                    hover_lines = [f"<b>{data['location_name']}</b>"]
                    for sensor in data['sensors']:
                        sensor_label = get_station_label(sensor['col_name'], short=True) # Label breve del sensore
                        value_h = sensor['value']
                        unit_h = sensor['unit']
                        thr_h = sensor['threshold']
                        alert_h = sensor['alert']
                        val_str_h = "N/D"
                        if pd.notna(value_h):
                            val_str_h = f"{value_h:.1f}{unit_h}" if unit_h == '(mm)' else f"{value_h:.2f}{unit_h}"

                        status_h = ""
                        if alert_h: status_h = f" ⚠️ (Soglia: {thr_h:.1f})"
                        elif pd.notna(value_h): status_h = " (OK)"
                        # hover_lines.append(f"- {sensor['name']}: {val_str_h}{status_h}") # Usa nome specifico sensore
                        hover_lines.append(f"- {sensor_label}: {val_str_h}{status_h}") # Usa label breve sensore


                    hover_text = "<br>".join(hover_lines)

                    map_plot_data.append({
                        'lat': data['lat'],
                        'lon': data['lon'],
                        'location_id': loc_id,
                        'text': hover_text,
                        'color': color,
                        'size': size,
                        'symbol': symbol
                    })

                map_df = pd.DataFrame(map_plot_data)

                # Calcola centro mappa e zoom (semplice media)
                center_lat = map_df['lat'].mean()
                center_lon = map_df['lon'].mean()
                # Zoom approssimativo basato sulla distanza max (molto rudimentale)
                lat_range = map_df['lat'].max() - map_df['lat'].min()
                lon_range = map_df['lon'].max() - map_df['lon'].min()
                max_range = max(lat_range, lon_range) if not map_df.empty else 0
                zoom = 8 # Default
                if max_range < 0.01: zoom = 13
                elif max_range < 0.1: zoom = 12
                elif max_range < 0.5: zoom = 10
                elif max_range < 1.0: zoom = 9


                fig_map = go.Figure(go.Scattermapbox(
                    lat=map_df['lat'],
                    lon=map_df['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=map_df['size'],
                        color=map_df['color'],
                        symbol=map_df['symbol'], # Usa simbolo mappato
                        opacity=0.9
                    ),
                    text=map_df['text'],
                    hoverinfo='text',
                    customdata=map_df['location_id'] # Utile per futuri eventi click
                ))

                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    # mapbox_style="carto-positron", # Alternativa
                    autosize=True,
                    height=600, # Altezza mappa
                    hovermode='closest',
                    mapbox=dict(
                        center=go.layout.mapbox.Center(
                            lat=center_lat,
                            lon=center_lon
                        ),
                        zoom=zoom
                    ),
                    margin={"r":0,"t":0,"l":0,"b":0} # Margini ridotti
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("Verde: OK, Rosso: Allerta, Grigio: N/D. Dimensione indica stato. Simboli: ○ Pioggia, ▲ Livello, ★ Misto. **NOTA:** Verifica correttezza coordinate!")

        # Aggiorna lo stato degli alert attivi
        st.session_state.active_alerts = current_alerts

        # Mostra toast per *nuovi* alert o alert che persistono
        if current_alerts:
            alert_summary = f"{len(current_alerts)} sensori in allerta!"
            st.toast(alert_summary, icon="🚨")
            # Potresti voler rendere i toast meno invadenti o selettivi
            # for col, val, thr in current_alerts:
            #      label_alert_toast = get_station_label(col, short=True)
            #      val_fmt = f"{val:.1f}" if 'Pioggia' in col else f"{val:.2f}"
            #      thr_fmt = f"{thr:.1f}"
            #      st.toast(f"{label_alert_toast}: {val_fmt} ≥ {thr_fmt}", icon="⚠️")


        # Mostra box riepilogativo degli alert ATTIVI sotto mappa e metriche
        st.divider()
        if st.session_state.active_alerts:
            st.warning("**🚨 ALLERTE ATTIVE (per Sensore) 🚨**")
            alert_md = ""
            # Ordina gli alert per location per raggrupparli visivamente
            sorted_alerts = sorted(st.session_state.active_alerts, key=lambda x: get_station_label(x[0], short=False))
            for col, val, thr in sorted_alerts:
                label_alert = get_station_label(col, short=False) # Nome località
                sensor_type_alert = STATION_COORDS.get(col, {}).get('type', '')
                type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                val_fmt = f"{val:.1f}" if 'Pioggia' in col else f"{val:.2f}"
                thr_fmt = f"{thr:.1f}"
                unit = '(mm)' if 'Pioggia' in col else '(m)'
                alert_md += f"- **{label_alert}{type_str}**: Valore attuale **{val_fmt}{unit}** >= Soglia **{thr_fmt}** ({col})\n" # Aggiunto nome completo in fondo
            st.markdown(alert_md)
        else:
            st.success("✅ Nessuna soglia superata al momento.")

    else: # Se latest_data è None (fetch fallito all'inizio)
        st.error("Impossibile visualizzare i dati della dashboard al momento.")
        if not error_msg: # Se non c'è un messaggio d'errore specifico
             st.info("Controlla la connessione di rete o la configurazione del Google Sheet (ID, permessi).")

    # Meccanismo di refresh automatico
    # Usa una chiave univoca per evitare conflitti se questo componente viene riutilizzato
    component_key = "dashboard_auto_refresh"
    streamlit_js_eval(js_expressions=f"setInterval(function(){{streamlitHook.rerunScript(null)}}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});", key=component_key)


# --- PAGINA SIMULAZIONE ---
# (MODIFICATA LEGGERMENTE per usare df_current_csv e feature_columns_current_model)
elif page == 'Simulazione':
    st.header('🧪 Simulazione Idrologica')
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo per usare la Simulazione.")
    else:
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns_model = active_config["target_columns"] # Target del modello ML
        # feature_columns_current_model è già definito globalmente

        st.info(f"Simulazione con: **{st.session_state.active_model_name}** (Input: {input_window}h, Output: {output_window}h)")
        # Usa etichette brevi per i target
        target_labels = [get_station_label(t, short=True) for t in target_columns_model]
        st.caption(f"Target previsti dal modello: {', '.join(target_labels)}")
        # Mostra anche le feature richieste dal modello
        feature_labels = [get_station_label(f, short=True) for f in feature_columns_current_model]
        with st.expander("Feature richieste dal modello attivo"):
             st.caption(", ".join(feature_columns_current_model)) # Nomi completi
             # st.caption(", ".join(feature_labels)) # Nomi brevi


        sim_data_input = None
        sim_method_options = ['Manuale Costante', 'Importa da Google Sheet', 'Orario Dettagliato (Avanzato)']
        # Aggiungi opzione CSV solo se i dati CSV sono caricati
        if data_ready_csv:
             sim_method_options.append('Usa Ultime Ore da CSV Caricato')

        sim_method = st.radio(
            "Metodo preparazione dati simulazione",
            sim_method_options,
            key="sim_method_radio"
        )

        # --- Simulazione: Manuale Costante ---
        if sim_method == 'Manuale Costante':
            st.subheader(f'Inserisci valori costanti per {input_window} ore')
            temp_sim_values = {}
            cols_manual = st.columns(3)
            # Raggruppa per tipo per chiarezza UI
            feature_groups = {'Pioggia': [], 'Umidità': [], 'Livello': [], 'Altro': []}
            for feature in feature_columns_current_model:
                label_feat = get_station_label(feature, short=True) # Etichetta breve
                if 'Cumulata' in feature or 'Pioggia' in feature: feature_groups['Pioggia'].append((feature, label_feat))
                elif 'Umidita' in feature: feature_groups['Umidità'].append((feature, label_feat))
                elif 'Livello' in feature: feature_groups['Livello'].append((feature, label_feat))
                else: feature_groups['Altro'].append((feature, label_feat))

            col_idx = 0
            # Pioggia
            if feature_groups['Pioggia']:
                 with cols_manual[col_idx % 3]:
                      st.write("**Pioggia (mm/ora)**")
                      for feature, label_feat in feature_groups['Pioggia']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.0
                           temp_sim_values[feature] = st.number_input(label_feat, 0.0, value=round(default_val,1), step=0.5, format="%.1f", key=f"man_{feature}", help=feature)
                 col_idx += 1
            # Livelli
            if feature_groups['Livello']:
                 with cols_manual[col_idx % 3]:
                      st.write("**Livelli (m)**")
                      for feature, label_feat in feature_groups['Livello']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.5
                           temp_sim_values[feature] = st.number_input(label_feat, -5.0, 20.0, value=round(default_val,2), step=0.05, format="%.2f", key=f"man_{feature}", help=feature)
                 col_idx += 1
            # Umidità e Altro
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
                              # Usa un formato generico
                              temp_sim_values[feature] = st.number_input(label_feat, value=round(default_val,2), step=0.1, format="%.2f", key=f"man_{feature}", help=feature)

            sim_data_list = []
            try:
                # Ordina in base a feature_columns_current_model per consistenza
                ordered_values = [temp_sim_values[feature] for feature in feature_columns_current_model]
                sim_data_input = np.tile(ordered_values, (input_window, 1)).astype(float)
                #st.success(f"Dati costanti pronti ({sim_data_input.shape}).")
            except KeyError as ke: st.error(f"Errore: Feature '{ke}' richiesta dal modello ma non configurata nell'input manuale."); sim_data_input = None
            except Exception as e: st.error(f"Errore creazione dati costanti: {e}"); sim_data_input = None


        # --- Simulazione: Google Sheet ---
        elif sim_method == 'Importa da Google Sheet':
             st.subheader(f'Importa ultime {input_window} ore da Google Sheet')
             st.warning("⚠️ Funzionalità sperimentale. Verifica attentamente la mappatura tra colonne GSheet e colonne richieste dal modello!")
             st.caption("Questa funzione richiede che il Google Sheet contenga dati storici orari sufficienti e che le colonne siano mappabili.")

             sheet_url_sim = st.text_input("URL Foglio Google (con dati storici)", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url")

             # --- MAPPATURA CRITICA DA CONTROLLARE/ADATTARE ---
             # L'utente DEVE verificare che le chiavi (nomi colonne GSheet) e i valori (nomi colonne modello) siano corretti
             # Idealmente, questa mappatura dovrebbe essere specifica per il modello selezionato o configurabile
             column_mapping_gsheet_to_model_sim = {
                # Colonna GSheet : Colonna Modello (basato su feature_columns_current_model)
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)', # Assunzione Bettolelle
                # --- Livelli ---
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', # Assunzione Corinaldo/Nevola
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', # Assunzione Bettolelle
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
                # --- Umidità (SE PRESENTE nel modello e nel GSheet) ---
                # 'NomeColonnaUmiditaGSheet': 'Umidita\' Sensore 3452 (Montemurello)' # Esempio da DECOMMENTARE e ADATTARE
             }
             with st.expander("Mostra/Modifica Mappatura GSheet -> Modello (Avanzato)"):
                 # Permette all'utente di vedere e potenzialmente modificare la mappatura
                 # Usiamo json per visualizzare/modificare facilmente
                 try:
                     edited_mapping_str = st.text_area(
                         "Mappatura JSON (Chiave: Colonna GSheet, Valore: Colonna Modello)",
                         value=json.dumps(column_mapping_gsheet_to_model_sim, indent=2),
                         height=300,
                         key="gsheet_map_edit"
                     )
                     edited_mapping = json.loads(edited_mapping_str)
                     # Sostituisci la mappatura di default con quella editata se valida
                     if isinstance(edited_mapping, dict):
                          column_mapping_gsheet_to_model_sim = edited_mapping
                          # st.caption("Mappatura aggiornata.") # Meno verboso
                     else:
                          st.warning("Formato JSON non valido per la mappatura.")
                 except json.JSONDecodeError:
                     st.warning("Errore nel formato JSON della mappatura.")
                 except Exception as e_map:
                      st.error(f"Errore inatteso nella mappatura: {e_map}")


             # Verifica quali feature del modello NON sono coperte dalla mappatura
             model_features_set = set(feature_columns_current_model)
             mapped_model_features = set(column_mapping_gsheet_to_model_sim.values())
             missing_model_features_in_map = list(model_features_set - mapped_model_features)

             imputed_values_sim = {} # Per feature non mappate
             needs_imputation_input = False

             if missing_model_features_in_map:
                  st.warning(f"Le seguenti feature richieste dal modello non sono mappate da Google Sheet e necessitano di un valore costante:")
                  needs_imputation_input = True
                  for missing_f in missing_model_features_in_map:
                       label_missing = get_station_label(missing_f, short=True)
                       # Trova valore di default: mediana da CSV o 0.0
                       default_val = 0.0
                       fmt = "%.2f"; step = 0.1
                       if data_ready_csv and missing_f in df_current_csv and pd.notna(df_current_csv[missing_f].median()):
                           default_val = df_current_csv[missing_f].median()

                       if 'Umidita' in missing_f: fmt = "%.1f"; step = 1.0
                       elif 'Cumulata' in missing_f: fmt = "%.1f"; step = 0.5; default_val = max(0.0, default_val) # Pioggia non negativa
                       elif 'Livello' in missing_f: fmt = "%.2f"; step = 0.05

                       imputed_values_sim[missing_f] = st.number_input(
                           f"Valore costante per '{label_missing}'",
                           value=round(default_val, 2), # Arrotonda default
                           step=step,
                           format=fmt,
                           key=f"sim_gsheet_impute_{missing_f}",
                           help=f"Valore da usare per la feature non mappata: {missing_f}"
                           )

             # Funzione interna per importare N righe da GSheet (cachata)
             @st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet...")
             def fetch_historical_gsheet_data(sheet_id, n_rows, date_col, date_format, col_mapping, required_model_cols, impute_dict):
                # ... (codice funzione fetch_historical_gsheet_data INVARIATO) ...
                try:
                    if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti."
                    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
                    gc = gspread.authorize(credentials)
                    sh = gc.open_by_key(sheet_id)
                    worksheet = sh.sheet1
                    all_data = worksheet.get_all_values()
                    if not all_data or len(all_data) < (n_rows + 1): return None, f"Errore: Dati insufficienti nel GSheet (richieste {n_rows} righe, trovate {len(all_data)-1})."

                    headers = all_data[0]
                    # Prendi le ultime n_rows righe di dati (esclusa intestazione)
                    data_rows = all_data[-n_rows:]

                    df_gsheet = pd.DataFrame(data_rows, columns=headers)

                    # Seleziona e rinomina colonne in base alla mappatura
                    relevant_gsheet_cols = list(col_mapping.keys())
                    missing_gsheet_cols = [c for c in relevant_gsheet_cols if c not in df_gsheet.columns]
                    if missing_gsheet_cols: return None, f"Errore: Colonne GSheet mancanti nella mappatura: {', '.join(missing_gsheet_cols)}"

                    df_mapped = df_gsheet[relevant_gsheet_cols].rename(columns=col_mapping)

                    # Aggiungi colonne mancanti richieste dal modello con valori imputati
                    for model_col, impute_val in impute_dict.items():
                         if model_col not in df_mapped.columns:
                              df_mapped[model_col] = impute_val

                    # Verifica se tutte le colonne modello sono presenti ORA
                    final_missing = [c for c in required_model_cols if c not in df_mapped.columns]
                    if final_missing: return None, f"Errore: Colonne modello mancanti dopo mappatura e imputazione: {', '.join(final_missing)}"

                    # Pulisci dati numerici (virgola -> punto)
                    gsheet_date_col_in_mapping = None
                    for gsheet_c, model_c in col_mapping.items():
                         if gsheet_c == date_col:
                              gsheet_date_col_in_mapping = model_c # Nome modello della colonna data
                              break

                    for col in required_model_cols: # Pulisci solo le colonne modello
                        if col != gsheet_date_col_in_mapping: # Escludi colonna data (se mappata)
                             try:
                                 # Applica la pulizia a tutta la colonna
                                 df_mapped[col] = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                                 df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
                             except Exception as e_clean:
                                 st.warning(f"Problema pulizia colonna GSheet '{col}': {e_clean}")
                                 df_mapped[col] = np.nan # Imposta a NaN in caso di errore

                    # Converti colonna data/ora (se presente nei dati mappati)
                    if gsheet_date_col_in_mapping and gsheet_date_col_in_mapping in df_mapped.columns:
                         try:
                             df_mapped[gsheet_date_col_in_mapping] = pd.to_datetime(df_mapped[gsheet_date_col_in_mapping], format=date_format, errors='coerce')
                             if df_mapped[gsheet_date_col_in_mapping].isnull().any():
                                  st.warning(f"Alcune date/ore nella colonna '{gsheet_date_col_in_mapping}' non sono state convertite correttamente.")
                             # Ordina per data/ora per sicurezza, gestendo NaT
                             df_mapped = df_mapped.sort_values(by=gsheet_date_col_in_mapping, na_position='first')
                         except Exception as e_date:
                             return None, f"Errore conversione colonna data GSheet '{gsheet_date_col_in_mapping}': {e_date}"
                    # Se la colonna data non è mappata o non serve, non fare nulla

                    # Seleziona solo le colonne finali richieste dal modello nell'ordine corretto
                    try:
                        df_final = df_mapped[required_model_cols]
                    except KeyError as e_key:
                         return None, f"Errore selezione colonne finali: Colonna '{e_key}' non trovata dopo mappatura/imputazione."

                    # Gestisci NaN residui (ffill/bfill è spesso ragionevole per serie temporali)
                    nan_count_before = df_final.isnull().sum().sum()
                    if nan_count_before > 0:
                         st.warning(f"Trovati NaN nei dati importati/mappati da GSheet ({nan_count_before} valori). Applico ffill/bfill.")
                         df_final = df_final.fillna(method='ffill').fillna(method='bfill')
                         nan_count_after = df_final.isnull().sum().sum()
                         if nan_count_after > 0:
                              # Se rimangono NaN all'inizio, forse bfill non è bastato. Riempi con 0 o mediana?
                              st.error(f"Errore: NaN residui ({nan_count_after}) dopo ffill/bfill nei dati GSheet. Potrebbero mancare dati all'inizio.")
                              # Mostra colonne con NaN residui
                              nan_cols_resid = df_final.isnull().sum()
                              st.json(nan_cols_resid[nan_cols_resid > 0].to_dict())
                              return None, "Errore: NaN residui dopo ffill/bfill nei dati GSheet."


                    if len(df_final) != n_rows:
                         # Questo non dovrebbe accadere se il fetch iniziale ha avuto successo
                         return None, f"Errore: Numero di righe finale ({len(df_final)}) non corrisponde a quello richiesto ({n_rows})."

                    # Rimuovi la colonna data se era stata mappata, il modello vuole solo dati numerici
                    if gsheet_date_col_in_mapping and gsheet_date_col_in_mapping in df_final.columns:
                        df_final_numeric = df_final.drop(columns=[gsheet_date_col_in_mapping])
                        # Verifica che le colonne rimanenti siano esattamente quelle richieste dal modello
                        if set(df_final_numeric.columns) != set(required_model_cols) - {gsheet_date_col_in_mapping}:
                             # Questo controllo è ridondante se la selezione sopra funziona
                             return None, "Errore discrepanza colonne dopo rimozione data."
                        # Riordina le colonne per sicurezza
                        df_final_numeric = df_final_numeric[required_model_cols]
                        return df_final_numeric, None # Successo, restituisce solo dati numerici
                    else:
                        # Se la colonna data non era mappata o richiesta, df_final dovrebbe già essere corretto
                        # Verifica comunque le colonne
                        if set(df_final.columns) != set(required_model_cols):
                             return None, f"Errore discrepanza colonne finali. Attese: {required_model_cols}, Trovate: {list(df_final.columns)}"
                        # Riordina colonne
                        df_final = df_final[required_model_cols]
                        return df_final, None # Successo

                except Exception as e:
                    st.error(traceback.format_exc()) # Log completo per debug
                    return None, f"Errore imprevisto importazione storica GSheet: {type(e).__name__} - {e}"


             if st.button("Importa e Prepara da Google Sheet", key="sim_gsheet_import", disabled=needs_imputation_input and not imputed_values_sim):
                 sheet_id_sim = extract_sheet_id(sheet_url_sim)
                 if not sheet_id_sim: st.error("URL GSheet non valido.")
                 else:
                     # Chiama la funzione per importare dati storici
                     imported_df_numeric, import_err = fetch_historical_gsheet_data(
                         sheet_id_sim,
                         input_window,
                         GSHEET_DATE_COL, # Nome colonna data nel GSheet
                         GSHEET_DATE_FORMAT,
                         column_mapping_gsheet_to_model_sim, # Mappatura (potenzialmente editata)
                         feature_columns_current_model, # Colonne richieste dal modello
                         imputed_values_sim # Valori da usare per colonne non mappate
                     )

                     if import_err:
                          st.error(f"Importazione GSheet fallita: {import_err}")
                          st.session_state.imported_sim_data_gs = None
                          sim_data_input = None
                     elif imported_df_numeric is not None:
                          st.success(f"Importate e mappate {len(imported_df_numeric)} righe da Google Sheet.")
                          # Verifica shape finale
                          if imported_df_numeric.shape == (input_window, len(feature_columns_current_model)):
                              # Salva il DataFrame pronto per la simulazione nello stato sessione
                              st.session_state.imported_sim_data_gs = imported_df_numeric
                              sim_data_input = imported_df_numeric.values # Prepara per la simulazione
                              with st.expander("Mostra Dati Numerici Importati da GSheet (pronti per modello)"):
                                   st.dataframe(imported_df_numeric.round(3))
                          else:
                               st.error(f"Errore: Shape dati importati ({imported_df_numeric.shape}) non corrisponde a quella attesa ({input_window}, {len(feature_columns_current_model)}).")
                               st.session_state.imported_sim_data_gs = None
                               sim_data_input = None
                     else:
                          st.error("Importazione GSheet non riuscita (nessun dato restituito).")
                          st.session_state.imported_sim_data_gs = None
                          sim_data_input = None

             # Se dati importati sono nello stato sessione (da run precedente o appena importati)
             # E il metodo selezionato è ancora GSheet
             elif sim_method == 'Importa da Google Sheet' and 'imported_sim_data_gs' in st.session_state and st.session_state.imported_sim_data_gs is not None:
                 imported_df_state = st.session_state.imported_sim_data_gs
                 # Verifica che corrisponda alle impostazioni correnti
                 if isinstance(imported_df_state, pd.DataFrame) and imported_df_state.shape == (input_window, len(feature_columns_current_model)):
                     sim_data_input = imported_df_state.values
                     st.info("Utilizzo dati precedentemente importati da Google Sheet.")
                     with st.expander("Mostra Dati Numerici Importati da GSheet (cache)"):
                          st.dataframe(imported_df_state.round(3))
                 else:
                     st.warning("I dati GSheet importati non corrispondono più alla configurazione attuale o non sono validi. Rieseguire importazione.")
                     st.session_state.imported_sim_data_gs = None # Invalida cache stato
                     sim_data_input = None


        # --- Simulazione: Orario Dettagliato ---
        elif sim_method == 'Orario Dettagliato (Avanzato)':
            # Logica INVARIATA, usa feature_columns_current_model
            st.subheader(f'Inserisci dati orari per le {input_window} ore precedenti')
            session_key_hourly = f"sim_hourly_data_{input_window}_{'_'.join(sorted(feature_columns_current_model))}" # Chiave più specifica

            needs_reinit = (
                session_key_hourly not in st.session_state or
                not isinstance(st.session_state[session_key_hourly], pd.DataFrame) or
                st.session_state[session_key_hourly].shape[0] != input_window or
                list(st.session_state[session_key_hourly].columns) != feature_columns_current_model
            )
            if needs_reinit:
                 st.caption("Inizializzazione tabella dati orari...")
                 init_vals = {}
                 for col in feature_columns_current_model:
                      med_val = 0.0
                      if data_ready_csv and col in df_current_csv and pd.notna(df_current_csv[col].median()):
                           med_val = df_current_csv[col].median()
                      elif col in DEFAULT_THRESHOLDS: # Fallback a soglia/default se no CSV
                           med_val = DEFAULT_THRESHOLDS.get(col, 0.0) * 0.2 # Usa una frazione della soglia come guess
                           if 'Cumulata' in col: med_val = max(0.0, med_val) # Pioggia >= 0
                      init_vals[col] = float(med_val)
                 # Assicura ordine colonne corretto
                 init_df = pd.DataFrame(np.repeat([list(init_vals.values())], input_window, axis=0), columns=feature_columns_current_model)
                 # Forza riordinamento colonne per sicurezza
                 init_df = init_df[feature_columns_current_model]
                 st.session_state[session_key_hourly] = init_df.fillna(0.0)

            df_for_editor = st.session_state[session_key_hourly].copy()
            # Assicura ordine colonne prima dell'editor
            df_for_editor = df_for_editor[feature_columns_current_model]

            if df_for_editor.isnull().sum().sum() > 0: df_for_editor = df_for_editor.fillna(0.0)
            try: df_for_editor = df_for_editor.astype(float)
            except Exception as e_cast:
                 st.error(f"Errore conversione tabella in float: {e_cast}. Reset.")
                 if session_key_hourly in st.session_state: del st.session_state[session_key_hourly]
                 st.rerun()

            column_config_editor = {}
            for col in feature_columns_current_model: # Itera nell'ordine corretto
                 label_edit = get_station_label(col, short=True) # Etichetta breve per editor
                 fmt = "%.3f"; step = 0.01; min_v=None; max_v=None
                 if 'Cumulata' in col or 'Pioggia' in col: fmt = "%.1f"; step = 0.5; min_v=0.0
                 elif 'Umidita' in col: fmt = "%.1f"; step = 1.0; min_v=0.0; max_v=100.0
                 elif 'Livello' in col: fmt = "%.3f"; step = 0.01; min_v=-5.0; max_v=20.0
                 # Configurazione specifica per la colonna corrente
                 column_config_editor[col] = st.column_config.NumberColumn(
                     label=label_edit,
                     help=col,
                     format=fmt,
                     step=step,
                     min_value=min_v,
                     max_value=max_v,
                     required=True # Forza l'utente a non lasciare vuoto
                     )

            edited_df = st.data_editor(
                 df_for_editor,
                 height=(input_window + 1) * 35 + 3,
                 use_container_width=True,
                 column_config=column_config_editor,
                 key=f"editor_{session_key_hourly}",
                 num_rows="fixed" # Impedisce aggiunta/rimozione righe
                 )

            validation_passed = False
            if edited_df.shape[0] != input_window: st.error(f"Tabella deve avere {input_window} righe."); sim_data_input = None
            elif list(edited_df.columns) != feature_columns_current_model: st.error("Ordine/nomi colonne tabella non corrispondono più al modello."); sim_data_input = None
            elif edited_df.isnull().sum().sum() > 0: st.warning("Valori mancanti in tabella. Compilare tutti i campi."); sim_data_input = None
            else:
                 try:
                      # Assicura ordine colonne finale prima di convertire in numpy
                      sim_data_input_edit = edited_df[feature_columns_current_model].astype(float).values
                      if sim_data_input_edit.shape == (input_window, len(feature_columns_current_model)):
                           sim_data_input = sim_data_input_edit; validation_passed = True
                           #st.success(f"Dati orari pronti ({sim_data_input.shape}).") # Messo dopo bottone
                      else: st.error("Errore shape dati tabella."); sim_data_input = None
                 except Exception as e_edit: st.error(f"Errore conversione dati tabella: {e_edit}"); sim_data_input = None

            if validation_passed and not st.session_state[session_key_hourly].equals(edited_df):
                 st.session_state[session_key_hourly] = edited_df # Aggiorna stato solo se valido e cambiato
                 # st.caption("Dati orari aggiornati nello stato.") # Meno verboso


        # --- Simulazione: Ultime Ore da CSV ---
        elif sim_method == 'Usa Ultime Ore da CSV Caricato':
             st.subheader(f"Usa le ultime {input_window} ore dai dati CSV caricati")
             if not data_ready_csv:
                  st.error("Dati CSV non caricati. Seleziona un altro metodo o carica un file CSV.")
             elif len(df_current_csv) < input_window:
                  st.error(f"Dati CSV ({len(df_current_csv)} righe) insufficienti per l'input richiesto ({input_window} ore).")
             else:
                  try:
                       # Seleziona le colonne richieste DAL MODELLO nell'ordine corretto
                       latest_csv_data_df = df_current_csv.iloc[-input_window:][feature_columns_current_model]
                       # Verifica se ci sono NaN nei dati selezionati
                       if latest_csv_data_df.isnull().sum().sum() > 0:
                            st.warning(f"Trovati NaN nelle ultime {input_window} ore del CSV. Controlla i dati originali.")
                            st.dataframe(latest_csv_data_df[latest_csv_data_df.isnull().any(axis=1)])
                            # Opzione: fermarsi o tentare fill? Per ora fermiamoci.
                            st.error("Impossibile usare dati CSV con valori mancanti per la simulazione.")
                            sim_data_input = None
                       else:
                            latest_csv_data = latest_csv_data_df.values
                            if latest_csv_data.shape == (input_window, len(feature_columns_current_model)):
                                sim_data_input = latest_csv_data
                                #st.success(f"Dati dalle ultime {input_window} ore del CSV pronti ({sim_data_input.shape}).") # Messo dopo bottone
                                last_ts_csv = df_current_csv.iloc[-1][date_col_name_csv]
                                st.caption(f"Basato su dati CSV fino a: {last_ts_csv.strftime('%d/%m/%Y %H:%M')}")
                                with st.expander("Mostra dati CSV usati"):
                                     display_cols_csv = [date_col_name_csv] + feature_columns_current_model
                                     st.dataframe(df_current_csv.iloc[-input_window:][display_cols_csv].round(3))
                            else:
                                st.error("Errore nella forma dei dati estratti dal CSV.")
                                sim_data_input = None
                  except KeyError as ke:
                      st.error(f"Errore: Colonna '{ke}' richiesta dal modello non trovata nel file CSV caricato.")
                      sim_data_input = None
                  except Exception as e_csv_sim:
                       st.error(f"Errore durante l'estrazione dati dal CSV: {e_csv_sim}")
                       sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE ---
        st.divider()
        # Mostra se i dati sono pronti PRIMA del bottone
        if sim_data_input is not None:
             st.success(f"Dati di input ({sim_method}) pronti ({sim_data_input.shape}).")

        run_simulation = st.button('Esegui simulazione', type="primary", disabled=(sim_data_input is None), key="sim_run")

        if run_simulation and sim_data_input is not None:
             # Validazione finale input (già fatta in parte sopra, ma ricontrolla)
             valid_input = False
             if not isinstance(sim_data_input, np.ndarray):
                 st.error("Errore: Input simulazione non è un array NumPy.")
             elif sim_data_input.shape[0] != input_window:
                 st.error(f"Errore righe input simulazione. Atteso:{input_window}, Ottenuto:{sim_data_input.shape[0]}")
             elif sim_data_input.shape[1] != len(feature_columns_current_model):
                 st.error(f"Errore colonne input simulazione. Atteso:{len(feature_columns_current_model)}, Ottenuto:{sim_data_input.shape[1]}")
             elif np.isnan(sim_data_input).any():
                 st.error(f"Errore: Rilevati NaN nell'input simulazione ({np.isnan(sim_data_input).sum()} valori). Controlla i dati inseriti/importati.")
             else:
                 valid_input = True

             if valid_input:
                  with st.spinner('Simulazione in corso...'):
                       predictions_sim = predict(active_model, sim_data_input, active_scaler_features, active_scaler_targets, active_config, active_device)
                       if predictions_sim is not None:
                           st.subheader(f'Risultato Simulazione: Previsione per {output_window} ore')
                           # Determina timestamp inizio previsione
                           start_pred_time = datetime.now(italy_tz) # Ora attuale come riferimento default

                           # Logica per determinare start_pred_time basata sul metodo di input
                           last_input_time_found = None
                           if sim_method == 'Usa Ultime Ore da CSV Caricato' and data_ready_csv:
                                try:
                                     last_csv_time = df_current_csv.iloc[-1][date_col_name_csv]
                                     # Assicurati che sia timezone-aware
                                     if isinstance(last_csv_time, pd.Timestamp):
                                         if last_csv_time.tzinfo is None:
                                              last_input_time_found = italy_tz.localize(last_csv_time)
                                         else:
                                              last_input_time_found = last_csv_time.tz_convert(italy_tz)
                                except Exception as e_time_csv:
                                    st.warning(f"Impossibile determinare l'ora dall'ultimo dato CSV: {e_time_csv}")

                           elif sim_method == 'Importa da Google Sheet':
                                # Qui è più complesso recuperare l'ultimo timestamp,
                                # perché la funzione di fetch restituisce solo i dati numerici.
                                # Potremmo modificare fetch_historical_gsheet_data per restituire anche
                                # l'ultimo timestamp, ma per ora usiamo il default (now).
                                # Oppure, se avessimo salvato il DF *con* la data... ma non l'abbiamo fatto.
                                st.caption("Nota: Timestamp iniziale previsione basato sull'ora corrente per import da GSheet.")
                                pass # Usa now() come fallback

                           # Se abbiamo trovato un timestamp valido dall'input, usalo
                           if last_input_time_found:
                               start_pred_time = last_input_time_found
                               st.caption(f"Previsione a partire da: {start_pred_time.strftime('%d/%m/%Y %H:%M %Z')}")

                           pred_times_sim = [start_pred_time + timedelta(hours=i+1) for i in range(output_window)]
                           results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns_model) # Usa target modello
                           results_df_sim.insert(0, 'Ora previsione', [t.strftime('%d/%m %H:%M') for t in pred_times_sim])

                           # Rinomina colonne DataFrame risultati per chiarezza
                           results_df_sim.columns = ['Ora previsione'] + [get_station_label(col, short=True) + f" ({col.split('(')[-1].split(')')[0]})" for col in target_columns_model]

                           st.dataframe(results_df_sim.round(3))
                           st.markdown(get_table_download_link(results_df_sim, f"simulazione_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                           st.subheader('Grafici Previsioni Simulate')
                           # Usa target_columns_model qui
                           figs_sim = plot_predictions(predictions_sim, active_config, start_pred_time)
                           for i, fig_sim in enumerate(figs_sim):
                               # Nome file basato sulla colonna target originale
                               s_name_file = target_columns_model[i].replace('[','').replace(']','').replace('(','').replace(')','').replace('/','_').replace(' ','_').strip()
                               st.plotly_chart(fig_sim, use_container_width=True)
                               st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                       else: st.error("Predizione simulazione fallita.")
        elif run_simulation and sim_data_input is None: st.error("Dati input simulazione non pronti o non validi. Prepara i dati prima.")


# --- PAGINA ANALISI DATI STORICI ---
# (MODIFICATA per usare df_current_csv e feature_columns_current_model)
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
             # Localize start/end datetimes if df datetimes are localized (assumed italy_tz here)
            # Questo passaggio è importante se si confrontano datetime aware con naive
            if pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]) and df_current_csv[date_col_name_csv].dt.tz is not None:
                 tz_csv = df_current_csv[date_col_name_csv].dt.tz
                 start_dt = tz_csv.localize(start_dt)
                 end_dt = tz_csv.localize(end_dt)
            elif pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]):
                 # Se CSV è naive, lascia start/end naive
                 pass
            else: # Se la colonna data non è datetime (non dovrebbe accadere)
                 st.error("Colonna data CSV non è del tipo datetime.")
                 st.stop()


            mask = (df_current_csv[date_col_name_csv] >= start_dt) & (df_current_csv[date_col_name_csv] <= end_dt)
            filtered_df = df_current_csv.loc[mask]
            if len(filtered_df) == 0: st.warning("Nessun dato nel periodo selezionato.")
            else:
                 st.success(f"Trovati {len(filtered_df)} record ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}).")
                 tab1, tab2, tab3 = st.tabs(["Andamento Temporale", "Statistiche/Distribuzione", "Correlazione"])

                 # Ottieni opzioni feature con etichette brevi per i widget
                 # Usa TUTTE le colonne numeriche disponibili nel DF filtrato, non solo quelle del modello
                 # Escludi la colonna data
                 potential_features_analysis = filtered_df.select_dtypes(include=np.number).columns.tolist()
                 # Rimuovi eventuali colonne non desiderate (es. indici se presenti)
                 potential_features_analysis = [f for f in potential_features_analysis if f not in ['index', 'level_0']] # Esempio

                 feature_labels_analysis = {get_station_label(f, short=True): f for f in potential_features_analysis}

                 if not feature_labels_analysis:
                      st.warning("Nessuna feature numerica trovata nei dati filtrati per l'analisi.")
                      st.stop()

                 with tab1:
                      st.subheader("Andamento Temporale Features CSV")
                      # Default sensato: primi 2 livelli o prime 2 feature
                      default_labels_ts = []
                      level_labels = [lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f]
                      if len(level_labels) >= 2: default_labels_ts = level_labels[:2]
                      elif len(level_labels) == 1: default_labels_ts = level_labels
                      elif len(feature_labels_analysis) >= 2: default_labels_ts = list(feature_labels_analysis.keys())[:2]
                      elif len(feature_labels_analysis) == 1: default_labels_ts = list(feature_labels_analysis.keys())

                      # Usa etichette brevi nel multiselect, ma plotta usando nomi originali
                      selected_labels_ts = st.multiselect(
                          "Seleziona feature",
                          options=list(feature_labels_analysis.keys()),
                          default=default_labels_ts,
                          key="analisi_ts"
                          )
                      features_plot = [feature_labels_analysis[lbl] for lbl in selected_labels_ts] # Mappa label -> nome colonna originale

                      if features_plot:
                           fig_ts = go.Figure()
                           for feature in features_plot:
                                # Usa etichetta breve nella legenda
                                legend_name = get_station_label(feature, short=True)
                                fig_ts.add_trace(go.Scatter(
                                    x=filtered_df[date_col_name_csv],
                                    y=filtered_df[feature],
                                    mode='lines',
                                    name=legend_name,
                                    hovertemplate=f'<b>{legend_name}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'
                                    ))
                           fig_ts.update_layout(title='Andamento Temporale Selezionato', xaxis_title='Data e Ora', yaxis_title='Valore', height=500, hovermode="x unified")
                           st.plotly_chart(fig_ts, use_container_width=True)
                           st.markdown(get_plotly_download_link(fig_ts, f"andamento_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)
                      else: st.info("Seleziona almeno una feature da visualizzare.")

                 with tab2:
                      st.subheader("Statistiche e Distribuzione")
                      # Default sensato per selectbox: primo livello o prima feature
                      default_stat_label = next((lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f), list(feature_labels_analysis.keys())[0])
                      selected_label_stat = st.selectbox(
                          "Seleziona feature",
                          options=list(feature_labels_analysis.keys()),
                          index=list(feature_labels_analysis.keys()).index(default_stat_label), # Usa index del label default
                          key="analisi_stat"
                          )
                      feature_stat = feature_labels_analysis[selected_label_stat] # Mappa label -> nome colonna originale

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
                      # Default: tutte le feature o un sottoinsieme ragionevole (es. livelli e piogge principali)
                      default_corr_labels = list(feature_labels_analysis.keys()) # Default a tutte

                      selected_labels_corr = st.multiselect(
                          "Seleziona feature per correlazione",
                          options=list(feature_labels_analysis.keys()),
                          default=default_corr_labels,
                          key="analisi_corr"
                          )
                      features_corr = [feature_labels_analysis[lbl] for lbl in selected_labels_corr] # Mappa label -> nome colonna originale

                      if len(features_corr) > 1:
                           corr_matrix = filtered_df[features_corr].corr()
                           # Usa etichette brevi per la heatmap
                           heatmap_labels = [get_station_label(f, short=True) for f in features_corr]
                           fig_hm = go.Figure(data=go.Heatmap(
                               z=corr_matrix.values,
                               x=heatmap_labels,
                               y=heatmap_labels,
                               colorscale='RdBu', zmin=-1, zmax=1, colorbar=dict(title='Corr'),
                               text=corr_matrix.round(2).values, # Mostra valore sulla cella
                               texttemplate="%{text}",
                               hoverongaps = False
                               ))
                           fig_hm.update_layout(
                               title='Matrice di Correlazione',
                               height=max(400, len(heatmap_labels)*30), # Altezza dinamica
                               xaxis_tickangle=-45,
                               yaxis_autorange='reversed' # Matrice standard
                               )
                           st.plotly_chart(fig_hm, use_container_width=True)
                           st.markdown(get_plotly_download_link(fig_hm, f"correlazione_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)

                           # Scatter Plot solo se selezionate poche feature per evitare confusione
                           if len(selected_labels_corr) <= 10: # Limite arbitrario
                                st.subheader("Scatter Plot Correlazione (2 Feature)")
                                cs1, cs2 = st.columns(2)
                                # Usa le label selezionate per la correlazione come opzioni qui
                                label_x = cs1.selectbox("Feature X", selected_labels_corr, index=0, key="scat_x")
                                # Default Y diverso da X se possibile
                                default_y_index = 1 if len(selected_labels_corr) > 1 else 0
                                label_y = cs2.selectbox("Feature Y", selected_labels_corr, index=default_y_index, key="scat_y")

                                fx = feature_labels_analysis.get(label_x)
                                fy = feature_labels_analysis.get(label_y)

                                if fx and fy:
                                    fig_sc = go.Figure(data=[go.Scatter(
                                        x=filtered_df[fx], y=filtered_df[fy],
                                        mode='markers', marker=dict(size=5, opacity=0.6),
                                        name=f'{label_x} vs {label_y}' # Usa label brevi
                                        )])
                                    fig_sc.update_layout(title=f'Correlazione: {label_x} vs {label_y}', xaxis_title=label_x, yaxis_title=label_y, height=500)
                                    st.plotly_chart(fig_sc, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sc, f"scatter_{label_x.replace(' ','_')}_vs_{label_y.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"), unsafe_allow_html=True)
                           else:
                                st.info("Troppe feature selezionate per mostrare lo scatter plot interattivo. Riduci la selezione nella matrice di correlazione.")

                      else: st.info("Seleziona almeno due feature per calcolare la correlazione.")
                 st.divider()
                 st.subheader('Download Dati Filtrati CSV')
                 st.markdown(get_table_download_link(filtered_df, f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)


# --- PAGINA ALLENAMENTO MODELLO ---
# (MODIFICATA per usare df_current_csv e feature_columns globali per input)
elif page == 'Allenamento Modello':
    st.header('🎓 Allenamento Nuovo Modello LSTM')
    if not data_ready_csv:
        st.warning("⚠️ Dati storici CSV non caricati. Carica un file CSV valido.")
    else:
        st.success(f"Dati CSV disponibili per addestramento: {len(df_current_csv)} righe.")
        st.subheader('Configurazione Addestramento')
        # Usa nome file suggerito, ma valida caratteri non permessi
        default_save_name = f"modello_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
        save_name_input = st.text_input("Nome base per salvare modello (a-z, A-Z, 0-9, _, -)", default_save_name, key="train_save_name")
        # Pulisci nome file da caratteri non validi
        save_name = re.sub(r'[^a-zA-Z0-9_-]', '_', save_name_input)
        if save_name != save_name_input:
            st.caption(f"Nome file corretto in: `{save_name}`")

        st.write("**1. Seleziona Feature Input (Default: tutte quelle definite globalmente):**")
        # Permetti all'utente di DESELEZIONARE feature dall'input se lo desidera
        all_available_features = st.session_state.feature_columns # Base di partenza
        with st.expander("Seleziona Feature Input Modello", expanded=False):
            cols_feat = st.columns(4)
            selected_features_train = []
            for i, feat in enumerate(all_available_features):
                 label_feat_train = get_station_label(feat, short=True)
                 with cols_feat[i % len(cols_feat)]:
                      if st.checkbox(label_feat_train, value=True, key=f"train_feat_{feat}", help=feat):
                          selected_features_train.append(feat)

        if not selected_features_train:
            st.warning("Seleziona almeno una feature di input per il modello.")

        st.write("**2. Seleziona Target Output (Livelli Idrometrici):**")
        selected_targets_train = []
        # Opzioni target: SOLO i livelli presenti nelle FEATURE DI INPUT SELEZIONATE
        hydro_features_options_train = [f for f in selected_features_train if 'Livello' in f]

        if not hydro_features_options_train:
            st.warning("Nessuna colonna 'Livello Idrometrico' trovata tra le feature di input selezionate. Impossibile selezionare target.")
        else:
            # Default: target del modello attivo (se validi e presenti tra le opzioni) o il primo livello trovato
            default_targets_train = []
            if model_ready and active_config.get("target_columns"):
                 active_targets = active_config["target_columns"]
                 # Verifica che i target del modello attivo siano tra le OPZIONI disponibili ORA
                 valid_active_targets = [t for t in active_targets if t in hydro_features_options_train]
                 if valid_active_targets:
                      default_targets_train = valid_active_targets
            if not default_targets_train and hydro_features_options_train: # Se ancora vuoto, prendi il primo
                 default_targets_train = hydro_features_options_train[:1]

            # Usa checkbox con etichette brevi
            cols_t = st.columns(min(len(hydro_features_options_train), 5))
            for i, feat in enumerate(hydro_features_options_train):
                with cols_t[i % len(cols_t)]:
                     lbl = get_station_label(feat, short=True) # Etichetta breve
                     if st.checkbox(lbl, value=(feat in default_targets_train), key=f"train_target_{feat}", help=feat):
                         selected_targets_train.append(feat)

        st.write("**3. Imposta Parametri:**")
        with st.expander("Parametri Modello e Training", expanded=True):
             c1t, c2t, c3t = st.columns(3)
             # Usa parametri modello attivo come default se disponibile e se le finestre sono valide
             default_iw = active_config["input_window"] if model_ready else 24
             default_ow = active_config["output_window"] if model_ready else 12
             default_hs = active_config["hidden_size"] if model_ready else 128
             default_nl = active_config["num_layers"] if model_ready else 2
             default_dr = active_config["dropout"] if model_ready else 0.2
             default_bs = active_config.get("batch_size", 32) if model_ready else 32 # Usa 32 se non in config
             default_vs = active_config.get("val_split_percent", 20) if model_ready else 20
             default_lr = active_config.get("learning_rate", 0.001) if model_ready else 0.001
             default_ep = active_config.get("epochs_run", 50) if model_ready else 50


             iw_t = c1t.number_input("Input Win (h)", 6, 168, default_iw, 6, key="t_in")
             ow_t = c1t.number_input("Output Win (h)", 1, 72, default_ow, 1, key="t_out")
             vs_t = c1t.slider("% Validazione", 0, 50, default_vs, 1, key="t_val", help="% dati finali per validazione")
             hs_t = c2t.number_input("Hidden Size", 16, 1024, default_hs, 16, key="t_hid")
             nl_t = c2t.number_input("Num Layers", 1, 8, default_nl, 1, key="t_lay")
             dr_t = c2t.slider("Dropout", 0.0, 0.7, default_dr, 0.05, key="t_drop")
             lr_t = c3t.number_input("Learning Rate", 1e-5, 1e-2, default_lr, format="%.5f", step=1e-4, key="t_lr")
             bs_t = c3t.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], default_bs, key="t_batch")
             ep_t = c3t.number_input("Epoche", 5, 500, default_ep, 5, key="t_epochs")

        st.write("**4. Avvia Addestramento:**")
        valid_name = bool(save_name) # Già pulito sopra
        valid_features = bool(selected_features_train)
        valid_targets = bool(selected_targets_train)
        ready_to_train = valid_name and valid_features and valid_targets

        if not valid_features: st.warning("Seleziona almeno una feature di input.")
        if not valid_targets: st.warning("Seleziona almeno un target.")
        if not valid_name: st.warning("Inserisci un nome valido per il modello.")


        train_button = st.button("Addestra Nuovo Modello", type="primary", disabled=not ready_to_train, key="train_run")
        if train_button and ready_to_train:
             st.info(f"Avvio addestramento per '{save_name}'...")
             with st.spinner('Preparazione dati...'):
                  # Usa le feature SELEZIONATE dall'utente per l'input
                  training_features_selected = selected_features_train
                  st.caption(f"Feature usate per input modello: {len(training_features_selected)}")
                  st.caption(f"Target selezionati per output: {', '.join(selected_targets_train)}")

                  # Verifica che i target siano un sottoinsieme delle feature (necessario?) - No.
                  # Verifica che tutte le feature/target esistano nel df
                  cols_to_check_df = training_features_selected + selected_targets_train
                  missing_in_df = [c for c in cols_to_check_df if c not in df_current_csv.columns]
                  if missing_in_df:
                       st.error(f"Errore: Le seguenti colonne selezionate non esistono nel DataFrame CSV: {', '.join(missing_in_df)}")
                       st.stop()

                  # Prepara i dati usando le feature selezionate e i target selezionati
                  X_tr, y_tr, X_v, y_v, sc_f_tr, sc_t_tr = prepare_training_data(
                      df_current_csv.copy(), # Usa copia per sicurezza
                      training_features_selected, # Feature INPUT selezionate
                      selected_targets_train, # Target OUTPUT selezionati
                      iw_t, ow_t, vs_t
                  )
                  if X_tr is None: st.error("Preparazione dati fallita."); st.stop()
                  st.success(f"Dati pronti: {len(X_tr)} seq. train, {len(X_v)} seq. val.")

             st.subheader("Addestramento...")
             input_size_train = len(training_features_selected) # Basato su feature selezionate
             output_size_train = len(selected_targets_train)
             trained_model = None
             try:
                 trained_model, train_losses, val_losses = train_model(
                     X_tr, y_tr, X_v, y_v, input_size_train, output_size_train, ow_t,
                     hs_t, nl_t, ep_t, bs_t, lr_t, dr_t
                 )
             except Exception as e_train: st.error(f"Errore training: {e_train}"); st.error(traceback.format_exc())

             if trained_model:
                 st.success("Addestramento completato!")
                 st.subheader("Salvataggio Risultati")
                 os.makedirs(MODELS_DIR, exist_ok=True)
                 base_path = os.path.join(MODELS_DIR, save_name)
                 m_path = f"{base_path}.pth"; c_path = f"{base_path}.json"
                 sf_path = f"{base_path}_features.joblib"; st_path = f"{base_path}_targets.joblib"
                 # Determina la validation loss finale
                 final_val_loss = None
                 if val_losses and vs_t > 0:
                      valid_val_losses = [v for v in val_losses if v is not None]
                      if valid_val_losses: final_val_loss = min(valid_val_losses)

                 # SALVA le feature effettivamente usate nel training nella config
                 config_save = {
                     "input_window": iw_t, "output_window": ow_t, "hidden_size": hs_t,
                     "num_layers": nl_t, "dropout": dr_t,
                     "target_columns": selected_targets_train, # Target usati
                     "feature_columns": training_features_selected, # Feature INPUT usate!
                     "training_date": datetime.now(italy_tz).isoformat(), # Usa timezone
                     "final_val_loss": final_val_loss,
                     "epochs_run": ep_t,
                     "batch_size": bs_t,
                     "val_split_percent": vs_t,
                     "learning_rate": lr_t,
                     "display_name": save_name, # Usa il nome file come display name default
                     "source_data_info": data_source_info # Info sul file CSV usato
                 }
                 try:
                     torch.save(trained_model.state_dict(), m_path)
                     with open(c_path, 'w') as f: json.dump(config_save, f, indent=4)
                     joblib.dump(sc_f_tr, sf_path); joblib.dump(sc_t_tr, st_path)
                     st.success(f"Modello '{save_name}' salvato in '{MODELS_DIR}/'")
                     st.caption(f"Salvati: {os.path.basename(m_path)}, {os.path.basename(c_path)}, {os.path.basename(sf_path)}, {os.path.basename(st_path)}")
                     st.subheader("Download File Modello")
                     col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                     with col_dl1: st.markdown(get_download_link_for_file(m_path, "Modello (.pth)"), unsafe_allow_html=True)
                     with col_dl2: st.markdown(get_download_link_for_file(c_path, "Config (.json)"), unsafe_allow_html=True)
                     with col_dl3: st.markdown(get_download_link_for_file(sf_path, "Scaler Feat (.joblib)"), unsafe_allow_html=True)
                     with col_dl4: st.markdown(get_download_link_for_file(st_path, "Scaler Targ (.joblib)"), unsafe_allow_html=True)

                     # Offri di ricaricare l'app per vedere il nuovo modello nella lista
                     if st.button("Ricarica App per aggiornare lista modelli"):
                          # Potrebbe essere utile pulire lo stato del modello attivo prima di ricaricare
                          st.session_state.active_model_name = None
                          st.session_state.active_config = None
                          st.session_state.active_model = None
                          st.rerun()

                 except Exception as e_save: st.error(f"Errore salvataggio file: {e_save}"); st.error(traceback.format_exc())
             elif not train_button: pass # Non fare nulla se il bottone non è stato premuto
             else: st.error("Addestramento fallito o interrotto. Impossibile salvare.")


# --- Footer ---
st.sidebar.divider()
st.sidebar.info('App Idrologica Dashboard & Predict © 2025 tutti i diritti riservati a Alberto Bussaglia')
