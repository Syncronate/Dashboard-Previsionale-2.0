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
import random # Assicurati che questo import sia presente all'inizio del file

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
# --- MODIFICATO: Aumentato numero di righe per coprire 24h con intervalli di 30min ---
DASHBOARD_HISTORY_ROWS = 48 # MODIFICATO: Numero di righe storiche da recuperare (48 righe = 24 ore @ 30 min/riga)

# --- MODIFICATO: DEFAULT_THRESHOLDS con i nuovi valori richiesti ---
DEFAULT_THRESHOLDS = { # Soglie predefinite (l'utente può modificarle)
    # Nome Sensore: {'alert': value, 'attention': value}
    # Pioggia (tutte le stazioni)
    'Arcevia - Pioggia Ora (mm)':          {'alert': 5.0, 'attention': 2.0},
    'Barbara - Pioggia Ora (mm)':          {'alert': 5.0, 'attention': 2.0},
    'Corinaldo - Pioggia Ora (mm)':        {'alert': 5.0, 'attention': 2.0},
    'Misa - Pioggia Ora (mm)':             {'alert': 5.0, 'attention': 2.0}, # Bettolelle Pioggia?
    # Livelli
    'Serra dei Conti - Livello Misa (mt)': {'alert': 1.7,  'attention': 1.2},
    'Pianello di Ostra - Livello Misa (m)':{'alert': 2.0,  'attention': 1.5},
    'Nevola - Livello Nevola (mt)':        {'alert': 2.5,  'attention': 2.0}, # Corinaldo Livello Nevola
    'Misa - Livello Misa (mt)':            {'alert': 2.0,  'attention': 1.7}, # Bettolelle Livello Misa
    'Ponte Garibaldi - Livello Misa 2 (mt)':{'alert': 2.0, 'attention': 1.8}
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
# (prepare_training_data rimane invariata, usata solo per training)
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
         # Non è un errore fatale, ma un warning se val_split > 0
         if val_split > 0:
             st.warning(f"Set di Validazione vuoto dopo lo split (val_split={val_split}%). Training procederà senza validazione.")
         X_val = np.empty((0, seq_len_in, num_features), dtype=np.float32)
         y_val = np.empty((0, seq_len_out, num_targets), dtype=np.float32)
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

# --- OTTIMIZZAZIONE: Cache per evitare scansione disco ad ogni rerun ---
@st.cache_data(show_spinner="Ricerca modelli disponibili...")
def find_available_models(models_dir=MODELS_DIR):
    """ Trova modelli validi (pth, json, scaler) nella directory specificata. """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE find_available_models") # Debug cache
    available = {}
    if not os.path.isdir(models_dir):
        # st.warning(f"Directory modelli '{models_dir}' non trovata.") # Potrebbe essere normale
        return available
    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth_path in pth_files:
        base = os.path.splitext(os.path.basename(pth_path))[0]
        cfg_p = os.path.join(models_dir, f"{base}.json")
        scf_p = os.path.join(models_dir, f"{base}_features.joblib")
        sct_p = os.path.join(models_dir, f"{base}_targets.joblib")

        # Verifica esistenza di tutti i file necessari
        if os.path.exists(cfg_p) and os.path.exists(scf_p) and os.path.exists(sct_p):
            try:
                 # Legge il display_name dal JSON, fallback al nome file
                 with open(cfg_p, 'r', encoding='utf-8') as f: # Aggiunto encoding
                     config_data = json.load(f)
                     name = config_data.get("display_name", base)
                     # Verifica base minima della config per evitare errori dopo
                     required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
                     if not all(k in config_data for k in required_keys):
                          raise ValueError("Config JSON incompleta")
            except Exception as e_cfg:
                 st.warning(f"Modello '{base}' ignorato: errore lettura o config JSON ({cfg_p}) non valida: {e_cfg}")
                 name = None # Segnala errore

            if name: # Se il nome è stato letto correttamente (nessun errore)
                available[name] = {"config_name": base, "pth_path": pth_path, "config_path": cfg_p,
                                   "scaler_features_path": scf_p, "scaler_targets_path": sct_p}
        else:
             # Avvisa solo se il file .pth esiste ma mancano gli altri, potrebbe essere un modello incompleto
             # Non avvisare se mancano tutti, la cartella potrebbe contenere altro
             # st.warning(f"Modello '{base}' ignorato: file associati mancanti (.json, .joblib).")
             pass
    return available

# --- Le funzioni sottostanti usano già caching di Streamlit dove appropriato ---
@st.cache_data
def load_model_config(_config_path):
    try:
        with open(_config_path, 'r', encoding='utf-8') as f: config = json.load(f) # Aggiunto encoding
        required = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        if not all(k in config for k in required): st.error(f"Config '{os.path.basename(_config_path)}' incompleto."); return None # Nome file più breve
        return config
    except Exception as e: st.error(f"Errore caricamento config '{os.path.basename(_config_path)}': {e}"); return None


@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    if not config: st.error("Config non valida per caricamento modello."); return None, None # Messaggio più specifico
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        f_cols_model = config.get("feature_columns", [])
        # Fallback alle feature globali se non presenti nel config
        if not f_cols_model:
            f_cols_model = st.session_state.get("feature_columns", [])
            if f_cols_model: # Aggiunto check se anche globali non definite
                st.warning(f"Config modello '{config.get('display_name', 'N/A')}' non specifica 'feature_columns'. Uso le {len(f_cols_model)} globali.")
            else:
                 st.error("Impossibile determinare 'feature_columns' da config o globali per caricare il modello.")
                 return None, None
        if not f_cols_model: # Doppio check dopo fallback
             st.error("Impossibile determinare input_size: 'feature_columns' non definite.")
             return None, None

        input_size_model = len(f_cols_model)
        output_size_model = len(config["target_columns"]) # Aggiunto per chiarezza
        output_window_model = config["output_window"]

        model = HydroLSTM(input_size_model, config["hidden_size"], output_size_model,
                          output_window_model, config["num_layers"], config["dropout"]).to(device)

        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'): # Gestisce BytesIO o simili
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError(f"Percorso modello non valido: tipo {type(_model_path)}")
        model.eval()
        # st.success(f"Modello '{config.get('display_name', 'N/A')}' caricato su {device}.") # Meno verbose
        return model, device
    except FileNotFoundError as fnf:
        st.error(f"Errore File Modello: {fnf}")
        return None, None
    except Exception as e:
        model_name_err = config.get('display_name', 'N/A')
        st.error(f"Errore caricamento modello '{model_name_err}': {type(e).__name__} - {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(_scaler_features_path, _scaler_targets_path):
    try:
        def _load_joblib(path):
             if isinstance(path, str):
                  if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
                  return joblib.load(path)
             elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
             else: raise TypeError(f"Percorso scaler non valido: tipo {type(path)}")
        sf = _load_joblib(_scaler_features_path)
        st = _load_joblib(_scaler_targets_path)
        # Verifica base degli scaler caricati
        if not hasattr(sf, 'transform') or not hasattr(st, 'inverse_transform'):
            raise TypeError("Uno o entrambi gli oggetti caricati non sembrano scaler validi (mancano metodi).")
        # st.success(f"Scaler caricati.") # Meno verbose
        return sf, st
    except FileNotFoundError as fnf:
        st.error(f"Errore File Scaler: {fnf}"); return None, None
    except Exception as e:
        st.error(f"Errore caricamento scaler: {type(e).__name__} - {e}");
        st.error(traceback.format_exc()); return None, None

# --- predict rimane invariata, dipende da modello e scaler cachati ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict: Modello, scaler o config mancanti."); return None
    input_w = config["input_window"]; output_w = config["output_window"]
    target_cols = config["target_columns"]; f_cols_cfg = config.get("feature_columns", [])

    # Aggiunto controllo più robusto su n_features_in_ se disponibile
    try:
        expected_features = scaler_features.n_features_in_
    except AttributeError:
        expected_features = len(f_cols_cfg) if f_cols_cfg else None
        if expected_features is None:
            st.warning("Predict: Impossibile verificare numero colonne input (scaler features non fittato o config mancante).")

    if input_data.shape[0] != input_w:
        st.error(f"Predict: Input righe {input_data.shape[0]} != Finestra richiesta {input_w}."); return None
    if expected_features is not None and input_data.shape[1] != expected_features:
        st.error(f"Predict: Input colonne {input_data.shape[1]} != Features attese dallo scaler/config {expected_features}."); return None

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(inp_tens)
        out_np = output.cpu().numpy().reshape(output_w, len(target_cols))

        # Aggiunto controllo più robusto su n_features_in_ per scaler target
        try:
            expected_targets = scaler_targets.n_features_in_
        except AttributeError:
            st.error("Predict: Scaler targets non sembra essere fittato (manca n_features_in_)."); return None

        if expected_targets != len(target_cols):
            st.error(f"Predict: Output targets modello ({len(target_cols)}) != Targets attesi dallo scaler ({expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except ValueError as ve:
        # Errore comune se le colonne non corrispondono allo scaler
        st.error(f"Errore durante scaling/predict: {ve}")
        st.error(f"Shape input_data: {input_data.shape}, Feature attese: {expected_features}")
        st.error(traceback.format_exc()); return None
    except Exception as e:
        st.error(f"Errore imprevisto durante predict: {type(e).__name__} - {e}");
        st.error(traceback.format_exc()); return None

# --- MODIFICATA: plot_predictions aggiunge soglie e migliora titolo ---
def plot_predictions(predictions, config, thresholds, start_time=None):
    """
    Genera grafici Plotly per le previsioni del modello, includendo le soglie.
    Utilizza intervalli di 30 minuti per l'asse temporale.
    Il titolo mostra il nome della stazione più chiaramente.
    """
    if config is None or predictions is None or thresholds is None: return []
    output_w = config["output_window"]; target_cols = config["target_columns"]
    figs = []
    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        if start_time:
            # Calcola step temporali ogni 30 minuti
            steps = [start_time + timedelta(minutes=30*(h+1)) for h in range(output_w)]
            x_axis, x_title = steps, "Data e Ora Previste (intervalli 30 min)"
        else:
            # Usa indici come fallback se non c'è start_time
            steps_idx = np.arange(1, output_w + 1)
            x_axis, x_title = steps_idx, f"Passi Futuri ({output_w} x 30 min)"

        # --- MODIFICATO: Estrai nome stazione per titolo grafico ---
        station_name_graph = get_station_label(sensor, short=False) # Usa la funzione helper (nome completo località)

        # Aggiungi linea previsione
        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsto'))

        # --- NUOVO: Aggiungi linee soglia dal dizionario ---
        sensor_thresholds = thresholds.get(sensor, {}) # Prendi dizionario soglie per questo sensore
        threshold_alert = sensor_thresholds.get('alert')
        threshold_attention = sensor_thresholds.get('attention')

        # Aggiungi linea soglia ALLERTA (Rossa)
        if threshold_alert is not None and isinstance(threshold_alert, (int, float)):
            fig.add_hline(
                y=threshold_alert, line_dash="dash", line_color="red",
                annotation_text=f"Allerta ({threshold_alert:.1f})",
                annotation_position="bottom right"
            )
        # Aggiungi linea soglia ATTENZIONE (Gialla/Arancione)
        if threshold_attention is not None and isinstance(threshold_attention, (int, float)):
             fig.add_hline(
                y=threshold_attention, line_dash="dash", line_color="orange", # Orange più visibile di yellow
                annotation_text=f"Attenzione ({threshold_attention:.1f})",
                annotation_position="top right" # Posizione diversa per evitare sovrapposizioni
            )

        fig.update_layout(
            # --- MODIFICATO: Titolo più chiaro ---
            title=f'Previsione Simulazione: {station_name_graph}',
            xaxis_title=x_title,
            yaxis_title=f'{sensor.split("(")[-1].split(")")[0].strip()}', # Estrae unità
            height=400,
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        figs.append(fig)
    return figs

# --- fetch_gsheet_dashboard_data utilizza già caching, logica interna ottimizzata per robustezza ---
# --- Assicurarsi che DASHBOARD_HISTORY_ROWS sia il valore desiderato ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format, num_rows_to_fetch=DASHBOARD_HISTORY_ROWS):
    """
    Importa le ultime 'num_rows_to_fetch' righe di dati dal Google Sheet.
    Restituisce un DataFrame pulito o None in caso di errore grave.
    Usa num_rows_to_fetch definito globalmente (o passato come argomento).
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
        start_index = max(1, len(all_values) - num_rows_to_fetch)
        data_rows = all_values[start_index:]

        if not data_rows:
             return None, f"Errore: Nessuna riga di dati trovata nelle ultime {num_rows_to_fetch} posizioni.", actual_fetch_time

        # Verifica che tutte le colonne richieste siano presenti negli header
        headers_set = set(headers)
        missing_cols = [col for col in relevant_columns if col not in headers_set]
        if missing_cols:
            return None, f"Errore: Colonne GSheet richieste mancanti: {', '.join(missing_cols)}", actual_fetch_time

        # Crea DataFrame
        df = pd.DataFrame(data_rows, columns=headers)

        # Seleziona solo le colonne rilevanti PRIMA della pulizia per efficienza
        df = df[relevant_columns]

        # Converti e pulisci i dati
        error_parsing = []
        for col in relevant_columns: # Ora itera solo sulle colonne rilevanti
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
                        # Potrebbe essere utile loggare/tracciare le date originali con errori
                        error_parsing.append(f"Formato data non valido per '{col}' in alcune righe (atteso: {date_format})")

                except Exception as e_date:
                    error_parsing.append(f"Errore conversione data '{col}': {e_date}")
                    df[col] = pd.NaT # Forza NaT su tutta la colonna in caso di errore grave
            else: # Colonne numeriche
                try:
                    # Ottimizzazione: Usa regex=False dove possibile
                    # Assicura sia stringa prima di usare .str
                    df_col_str = df[col].astype(str)
                    # Sostituisci virgola, spazi, etc. e converti in numerico
                    df_col_str = df_col_str.str.replace(',', '.', regex=False).str.strip()
                    # Gestisci 'N/A' e stringhe vuote comuni più efficientemente
                    df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Controlla se ci sono NaN dopo la conversione (non blocca, solo informativo)
                    # if df[col].isnull().any():
                    #     pass

                except Exception as e_num:
                    error_parsing.append(f"Errore conversione numerica '{col}': {e_num}")
                    df[col] = np.nan # Forza NaN in caso di errore grave

        # Gestisci NaT nella colonna data
        if df[date_col].isnull().any():
             # L'avviso è già presente, non serve fare altro qui a meno di non voler rimuovere le righe
             # df = df.dropna(subset=[date_col]) # Opzione se le righe con data errata sono inutilizzabili
             pass

        # Ordina per data/ora per sicurezza (importante per grafici TS)
        df = df.sort_values(by=date_col, na_position='first').reset_index(drop=True)

        # Gestisci NaN numerici (ffill/bfill opzionale, può mascherare problemi)
        # L'info message attuale è sufficiente, evitiamo ffill/bfill che potrebbe introdurre dati errati
        nan_numeric_count = df.drop(columns=[date_col]).isnull().sum().sum()
        if nan_numeric_count > 0:
            # st.info(...) # L'info message è già presente nel codice originale, va bene
            pass

        if error_parsing:
            # Restituisce il DataFrame ma segnala gli errori
            error_message = "Attenzione: Errori durante la conversione dei dati GSheet. " + " | ".join(error_parsing)
            return df, error_message, actual_fetch_time

        # Restituisce il DataFrame pulito e ordinato
        return df, None, actual_fetch_time # Nessun errore grave

    except gspread.exceptions.APIError as api_e:
        # Gestione errori API invariata
        try:
            error_details = api_e.response.json()
            error_message = error_details.get('error', {}).get('message', str(api_e))
            status_code = error_details.get('error', {}).get('code', 'N/A')
            if status_code == 403: error_message += " Verifica condivisione foglio con email service account."
            elif status_code == 429: error_message += f" Limite API Google superato. Riprova tra un po'. Intervallo attuale: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
            else: error_message = f"Codice {status_code}: {error_message}"
        except: error_message = str(api_e)
        return None, f"Errore API Google Sheets: {error_message}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound:
        return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}'). Verifica ID e permessi.", actual_fetch_time
    except Exception as e:
        return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time


# --- MODIFICATA/SOSTITUISCI QUESTA FUNZIONE: train_model per ottimizzazione ---
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                is_optimization_trial=False, # NUOVO: Flag per trial di ottimizzazione
                opt_trial_num=None, opt_total_trials=None): # NUOVO: Info per status
    """
    Allena il modello LSTM.
    Se is_optimization_trial=True, esegue meno output e restituisce best_val_loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    # Gestione più robusta per X_val/y_val potenzialmente None o vuoti
    val_dataset = None
    if X_val is not None and y_val is not None and X_val.size > 0 and y_val.size > 0:
         try:
             val_dataset = TimeSeriesDataset(X_val, y_val)
         except Exception as e_val_ds:
             st.warning(f"Impossibile creare Validation Dataset: {e_val_ds}") # Avvisa ma non blocca

    # Aggiungi controllo per evitare DataLoader con dataset vuoto (anche se prepare_training_data dovrebbe gestirlo)
    if len(train_dataset) == 0:
        st.error("Errore Critico: Training dataset è vuoto. Impossibile procedere con l'addestramento.")
        return float('inf') if is_optimization_trial else (None, [], []) # Segnala errore

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset and len(val_dataset) > 0 else None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state_dict = None # Per salvare lo stato migliore (solo in training finale)

    # --- Gestione Output Streamlit differenziato ---
    progress_bar = None
    status_text_global = None # Rinominato per evitare clash con quello interno al trial
    loss_chart_placeholder = None

    # Funzione helper per grafico loss (definita qui per chiarezza, era globale prima)
    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        valid_v_loss = [v for v in v_loss if v is not None and np.isfinite(v)] if v_loss else []
        if valid_v_loss:
             fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0))
        if placeholder: placeholder.plotly_chart(fig, use_container_width=True)


    if not is_optimization_trial:
        progress_bar = st.progress(0)
        status_text_global = st.empty()
        loss_chart_placeholder = st.empty()
        st.write(f"Inizio training per {epochs} epoche su {device}...")
    else:
        # Durante l'ottimizzazione, potremmo non voler usare st.empty globalmente
        # Lo gestiamo all'interno del loop di ottimizzazione
        pass

    start_training_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Gestione train_loss NaN/inf
        if not np.isfinite(train_loss):
            if not is_optimization_trial:
                st.warning(f"Epoch {epoch+1}: Train loss è NaN/inf. Interruzione training.")
            # Se siamo in trial, restituisci 'inf' per segnalare fallimento
            # Se in training finale, potresti voler restituire None o sollevare errore
            return float('inf') if is_optimization_trial else (model, train_losses, val_losses) # Modello fino a questo punto

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        current_val_loss = None # Default se non c'è validation
        if val_loader:
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    epoch_val_loss += loss.item()

            # Gestione val_loss NaN/inf
            if not np.isfinite(epoch_val_loss):
                 if not is_optimization_trial: # Mostra warning solo in training finale
                      st.warning(f"Epoch {epoch+1}: Validation loss è NaN/inf.")
                 current_val_loss = float('inf') # Segnala come infinito per lo scheduler/best_loss
            else:
                 # Verifica se len(val_loader) è > 0 prima della divisione
                 if len(val_loader) > 0:
                     current_val_loss = epoch_val_loss / len(val_loader)
                 else: # Caso improbabile se val_loader è stato creato
                     current_val_loss = float('inf')
                     if not is_optimization_trial:
                        st.warning(f"Epoch {epoch+1}: Validation loader ha lunghezza 0, Val Loss impostata a inf.")


            val_losses.append(current_val_loss)
            # Passa a scheduler solo se la loss è finita
            if np.isfinite(current_val_loss):
                 scheduler.step(current_val_loss)

                 # Aggiorna best_val_loss (importante per return) e salva stato SE non è trial
                 if current_val_loss < best_val_loss:
                     best_val_loss = current_val_loss
                     if not is_optimization_trial:
                         # Copia state_dict solo per il training finale per salvare il modello migliore
                         try:
                             best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
                         except Exception as e_clone:
                              st.warning(f"Errore durante la copia dello stato del modello migliore: {e_clone}. Potrebbe non essere salvato correttamente.")
                              best_model_state_dict = None # Resetta per sicurezza
            # Se current_val_loss è inf, non aggiornare best_val_loss (rimane 'inf' o il valore buono precedente)
        else:
            val_losses.append(None) # Aggiungi None se non c'è validation

        # --- Aggiornamento UI ---
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{current_val_loss:.6f}" if current_val_loss is not None and np.isfinite(current_val_loss) else ("NaN/Inf" if current_val_loss is not None else "N/A")
        epoch_time = time.time() - epoch_start_time

        if not is_optimization_trial:
            progress_percentage = (epoch + 1) / epochs
            if progress_bar: progress_bar.progress(progress_percentage)
            if status_text_global: status_text_global.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) - Train Loss: {train_loss:.6f}, Val Loss: {val_loss_str} - LR: {current_lr:.6f}')
            if loss_chart_placeholder: update_loss_chart(train_losses, val_losses, loss_chart_placeholder)
        # Non mostrare output per epoca durante ottimizzazione HP per non intasare

    total_training_time = time.time() - start_training_time

    if not is_optimization_trial:
        if status_text_global: status_text_global.text(f"Training completato in {total_training_time:.1f} secondi.") # Aggiorna status finale
        else: st.write(f"Training completato in {total_training_time:.1f} secondi.") # Fallback se status_text non inizializzato
        final_model_to_return = model # Restituisci l'ultimo modello di default

        if best_model_state_dict: # Se abbiamo salvato uno stato migliore (solo in training finale con validation)
            try:
                # Assicurati che i tensori siano sul device giusto prima di caricarli
                best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
                model.load_state_dict(best_model_state_dict_on_device)
                st.success(f"Caricato modello migliore dall'epoca con Val Loss: {best_val_loss:.6f}")
                final_model_to_return = model # Ora è il modello migliore
            except Exception as e_load_best:
                st.error(f"Errore caricamento best model state: {e_load_best}. Uso modello ultima epoca.")
        elif not val_loader:
            st.warning("Nessun set di validazione fornito, usato modello dell'ultima epoca.")
        # elif val_loader and not best_model_state_dict: # Rimosso controllo ridondante
        #     st.warning("Nessun miglioramento rilevato sulla validation loss o errore nel salvataggio stato migliore, usato modello dell'ultima epoca.")
        elif val_loader: # Se c'era validation ma non è stato salvato uno stato (raro se non ci sono errori)
            if best_val_loss == float('inf'):
                 st.warning("Validation loss non è mai migliorata (rimasta inf/NaN?), usato modello dell'ultima epoca.")
            else: # Caso generico se non c'è best_model_state_dict ma c'era val_loader
                 st.warning("Usato modello dell'ultima epoca (best_val_loss non trovata o errore salvataggio stato).")


        return final_model_to_return, train_losses, val_losses # Return per training normale
    else:
        # Return per trial di ottimizzazione
        # Restituisci la miglior validation loss finita trovata durante questo trial
        # Se non c'era validation o la loss è sempre stata inf, restituisce 'inf'
        final_metric_trial = best_val_loss if np.isfinite(best_val_loss) else float('inf')
        # print(f"Trial {opt_trial_num}/{opt_total_trials}: Best Val Loss = {final_metric_trial}") # Debug opzionale
        return final_metric_trial


# --- Funzioni Helper Download invariate ---
def get_table_download_link(df, filename="data.csv"):
    try:
        csv = df.to_csv(index=False, sep=';', decimal=',')
        b64 = base64.b64encode(csv.encode('utf-8')).decode()
        # Aggiunta media type per robustezza
        return f'<a href="data:text/csv;charset=utf-8;base64,{b64}" download="{filename}">Scarica CSV</a>'
    except Exception as e:
        st.error(f"Errore generazione link download CSV ({filename}): {e}")
        return "<i>Errore link CSV</i>"

def get_binary_file_download_link(file_object, filename, text):
    try:
        file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None: mime_type = 'application/octet-stream' # Fallback generico
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e:
        st.error(f"Errore generazione link download binario ({filename}): {e}")
        return f"<i>Errore link {filename}</i>"


def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    href_html = ""
    href_png = ""
    try:
        # Considera aggiunta try-except attorno a write_html
        buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn') # Usa CDN per ridurre dimensione file
        buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'
    except Exception as e_html:
         st.warning(f"Errore generazione HTML per {filename_base}: {e_html}")
         href_html = f"<i>Errore HTML</i>"

    try:
        # Verifica se kaleido è installato in modo sicuro
        import importlib
        kaleido_spec = importlib.util.find_spec("kaleido")
        if kaleido_spec is not None:
            buf_png = io.BytesIO(); fig.write_image(buf_png, format="png", scale=2) # Aumenta scala per qualità PNG
            buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode()
            href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
        else:
             pass # Kaleido non installato, non mostrare link PNG
    # except ImportError: pass # Gestito da find_spec
    except Exception as e_png:
         st.warning(f"Errore generazione PNG per {filename_base} (kaleido potrebbe mancare o essere non funzionante): {e_png}", icon="🖼️")
         href_png = f"<i>Errore PNG</i>" # Segnala errore nel link

    return f"{href_html} {href_png}".strip()


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
        # Usa f-string per chiarezza
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e:
        st.error(f"Errore generazione link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet invariata ---
def extract_sheet_id(url):
    if not isinstance(url, str): return None # Gestione input non stringa
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    # Fallback: se l'URL è solo l'ID stesso
    if re.fullmatch(r'[a-zA-Z0-9-_]+', url): return url
    return None

# --- Funzione Estrazione Etichetta Stazione invariata ---
def get_station_label(col_name, short=False):
    if not isinstance(col_name, str): return "N/A" # Gestione input non stringa
    if col_name in STATION_COORDS:
        location_id = STATION_COORDS[col_name].get('location_id')
        if location_id:
            if short:
                sensor_type = STATION_COORDS[col_name].get('type', '')
                # Check if multiple sensors share the same location_id
                sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == location_id]
                if len(sensors_at_loc) > 1:
                    # Abbreviate type if multiple sensors at the same location
                    type_abbr = 'P' if sensor_type == 'Pioggia' else ('L' if sensor_type == 'Livello' else '')
                    label = f"{location_id} ({type_abbr})"
                    return label[:25] + ('...' if len(label) > 25 else '') # Troncamento più esplicito
                else:
                    # Only one sensor type at this location, just use location ID
                    return location_id[:25] + ('...' if len(location_id) > 25 else '')
            else:
                # Return full location ID if not short
                return location_id
    # Fallback if not in STATION_COORDS
    parts = col_name.split(' - ')
    if len(parts) > 1:
        location = parts[0].strip()
        measurement = parts[1].split(' (')[0].strip()
        if short:
             label = f"{location} - {measurement}"
             return label[:25] + ('...' if len(label) > 25 else '')
        else: return location # Return just the location part for non-short version
    else: # Fallback even further if format is unexpected
        label = col_name.split(' (')[0].strip()
        return label[:25] + ('...' if len(label) > 25 else '')


# --- Inizializzazione Session State ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scaler_features' not in st.session_state: st.session_state.active_scaler_features = None
if 'active_scaler_targets' not in st.session_state: st.session_state.active_scaler_targets = None
if 'df' not in st.session_state: st.session_state.df = None # Dati CSV storici

# Default feature columns (usate se non specificate nel config del modello)
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [
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

# --- MODIFICATO: Inizializza st.session_state.dashboard_thresholds con la nuova struttura ---
if 'dashboard_thresholds' not in st.session_state:
    # Deep copy per evitare modifiche accidentali al default
    st.session_state.dashboard_thresholds = json.loads(json.dumps(DEFAULT_THRESHOLDS))

if 'last_dashboard_data' not in st.session_state: st.session_state.last_dashboard_data = None # DataFrame dashboard
if 'last_dashboard_error' not in st.session_state: st.session_state.last_dashboard_error = None
if 'last_dashboard_fetch_time' not in st.session_state: st.session_state.last_dashboard_fetch_time = None
if 'active_alerts' not in st.session_state: st.session_state.active_alerts = []
# --- NUOVO: Aggiungi stato per allerte di attenzione ---
if 'active_attentions' not in st.session_state: st.session_state.active_attentions = []
# --- NUOVO: Stato per Ottimizzazione HP ---
if 'hp_optimize_enabled' not in st.session_state: st.session_state.hp_optimize_enabled = False
if 'best_hp_params' not in st.session_state: st.session_state.best_hp_params = None
if 'hp_n_trials' not in st.session_state: st.session_state.hp_n_trials = 30
if 'hp_epochs_per_trial' not in st.session_state: st.session_state.hp_epochs_per_trial = 15
if 'hp_lr_range' not in st.session_state: st.session_state.hp_lr_range = (1e-4, 5e-3)
if 'hp_bs_range' not in st.session_state: st.session_state.hp_bs_range = (32, 128)
if 'hp_hs_range' not in st.session_state: st.session_state.hp_hs_range = (64, 256)
if 'hp_nl_range' not in st.session_state: st.session_state.hp_nl_range = (1, 4)
if 'hp_dr_range' not in st.session_state: st.session_state.hp_dr_range = (0.1, 0.4)


# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('🌊 Dashboard e Modello Predittivo Idrologico')

# --- Sidebar ---
st.sidebar.header('Impostazioni')

# --- Caricamento Dati Storici (per Analisi/Training) ---
st.sidebar.subheader('Dati Storici (per Analisi/Training)')
uploaded_data_file = st.sidebar.file_uploader('Carica CSV Dati Storici (Opzionale)', type=['csv'], key="data_uploader")

# --- Logica Caricamento DF ---
df = None; df_load_error = None; data_source_info = ""
data_path_to_load = None; is_uploaded = False
# Determina cosa caricare
if uploaded_data_file is not None:
    data_path_to_load = uploaded_data_file; is_uploaded = True
    data_source_info = f"File caricato: **{uploaded_data_file.name}**"
elif os.path.exists(DEFAULT_DATA_PATH):
    # Carica da default solo se non c'è già in sessione (o se è stato caricato un file nuovo)
    if 'df' not in st.session_state or st.session_state.df is None:
        data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
    else: # Usa dati già in sessione
        data_path_to_load = None # Non ricaricare
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (usando cache sessione)"
# Se non è stato caricato file e il default non esiste, controlla se c'è qualcosa in sessione
elif 'df' in st.session_state and st.session_state.df is not None:
     data_source_info = f"Dati CSV da cache sessione (file originale sconosciuto)."
     data_path_to_load = None # Non ricaricare
else: # Nessun file caricato, default non trovato, sessione vuota
     df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato e nessun file caricato. Carica un CSV per Analisi/Training."

# Esegui caricamento/processamento SE necessario
# Ricarica sempre se è stato caricato un nuovo file
if data_path_to_load and ('df' not in st.session_state or st.session_state.df is None or is_uploaded):
    with st.sidebar.spinner(f"Caricamento e processamento CSV da {data_source_info}..."):
        try:
            read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
            df_temp = None
            for enc in encodings_to_try:
                try:
                    # Riavvolgi file object se necessario
                    if hasattr(data_path_to_load, 'seek'): data_path_to_load.seek(0)
                    df_temp = pd.read_csv(data_path_to_load, encoding=enc, **read_args)
                    # st.sidebar.caption(f"Letto con encoding {enc}")
                    break # Successo, esci dal loop encoding
                except UnicodeDecodeError:
                    continue # Prova prossimo encoding
                except Exception as read_e:
                    # Rilancia altri errori di lettura
                    raise read_e
            if df_temp is None:
                raise ValueError(f"Impossibile leggere CSV con encoding {', '.join(encodings_to_try)}.")

            date_col_csv = st.session_state.date_col_name_csv
            if date_col_csv not in df_temp.columns:
                raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante nel file.")

            # --- Pulizia Date ---
            original_date_dtype = df_temp[date_col_csv].dtype
            try:
                # Prova formato standard prima
                df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
            except ValueError:
                try:
                     # Fallback: prova inferenza (più lento, meno affidabile)
                     df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce') # Coerce invalids to NaT
                     if df_temp[date_col_csv].isnull().any():
                          st.sidebar.warning(f"Formato data CSV non standard ('{date_col_csv}'). Alcune righe contengono date non valide (convertite in NaT).")
                     else:
                          st.sidebar.warning(f"Formato data CSV non standard ('{date_col_csv}'). Tentata inferenza automatica.")
                except Exception as e_date_csv_infer:
                     raise ValueError(f"Errore conversione data CSV '{date_col_csv}' anche con inferenza: {e_date_csv_infer}")

            df_temp = df_temp.dropna(subset=[date_col_csv]) # Rimuovi righe con data non valida
            if df_temp.empty:
                raise ValueError("Nessuna riga valida dopo la conversione/pulizia della data CSV.")

            # --- Ordinamento ---
            df_temp = df_temp.sort_values(by=date_col_csv).reset_index(drop=True)

            # --- Pulizia Feature Numeriche ---
            # Usa le feature globali come riferimento per la pulizia
            current_f_cols_state = st.session_state.get('feature_columns', [])
            features_actually_in_df = [col for col in current_f_cols_state if col in df_temp.columns]
            missing_features_in_df = [col for col in current_f_cols_state if col not in df_temp.columns]

            if missing_features_in_df:
                 st.sidebar.warning(f"Attenzione: Le seguenti feature globali non sono nel CSV: {', '.join(missing_features_in_df)}. Saranno ignorate per Analisi/Training basato su questo CSV.")

            cols_to_clean = features_actually_in_df # Pulisci solo le colonne definite come feature
            if not cols_to_clean:
                 st.sidebar.warning("Nessuna feature globale trovata nel CSV. Pulizia numerica limitata.")
                 # Potresti decidere di pulire tutte le colonne numeriche, ma è rischioso
                 # cols_to_clean = df_temp.select_dtypes(include='object').columns # Esempio alternativo

            for col in cols_to_clean:
                 if col in df_temp.columns: # Doppio check
                     # Forza a stringa per pulizia robusta
                     df_temp[col] = df_temp[col].astype(str)
                     # Rimuovi spazi bianchi
                     df_temp[col] = df_temp[col].str.strip()
                     # Sostituisci comuni placeholder NaN
                     df_temp[col] = df_temp[col].replace(['N/A', '', '-', 'None', 'null', 'NaN', 'nan', '#N/D', '#DIV/0!'], np.nan, regex=False)
                     # Sostituisci separatore decimale (virgola -> punto)
                     # Applica solo se la colonna contiene effettivamente virgole (ottimizzazione)
                     if df_temp[col].astype(str).str.contains(',').any():
                         df_temp[col] = df_temp[col].str.replace(',', '.', regex=False)
                     # Rimuovi separatori migliaia (punto) - attenzione, fare DOPO sostituzione virgola
                     if df_temp[col].astype(str).str.count('\.').max() > 1: # Se ci sono più punti, probabile separatore migliaia
                          df_temp[col] = df_temp[col].str.replace('.', '', regex=False) # Rimuove TUTTI i punti

                     # Converti in numerico, errori -> NaN
                     df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

            # --- Gestione NaN dopo pulizia ---
            cols_to_check_nan = features_actually_in_df # Controlla solo le feature rilevanti
            n_nan_before_fill = df_temp[cols_to_check_nan].isnull().sum().sum()
            if n_nan_before_fill > 0:
                  st.sidebar.caption(f"Trovati {n_nan_before_fill} NaN/non numerici nelle colonne feature del CSV dopo pulizia. Applico forward-fill e backward-fill.")
                  df_temp[cols_to_check_nan] = df_temp[cols_to_check_nan].fillna(method='ffill').fillna(method='bfill')
                  n_nan_after_fill = df_temp[cols_to_check_nan].isnull().sum().sum()
                  if n_nan_after_fill > 0:
                      st.sidebar.error(f"NaN residui ({n_nan_after_fill}) dopo fill. Controlla inizio/fine file CSV o colonne con tutti NaN.")
                      nan_cols_report = df_temp[cols_to_check_nan].isnull().sum()
                      st.sidebar.caption("Colonne con NaN residui:")
                      st.sidebar.json(nan_cols_report[nan_cols_report > 0].to_dict())
                      # Non bloccare necessariamente, ma segnala chiaramente il problema

            # --- Aggiorna Session State ---
            st.session_state.df = df_temp
            st.sidebar.success(f"Dati CSV caricati e processati ({len(st.session_state.df)} righe valide).")
            df = st.session_state.df # Aggiorna variabile locale 'df'

        except Exception as e:
            df = None; st.session_state.df = None # Resetta df in caso di errore
            df_load_error = f'Errore caricamento/processamento dati CSV ({data_source_info}): {type(e).__name__} - {e}'
            st.sidebar.error(f"Errore CSV: {df_load_error}")
            st.sidebar.info("Le funzionalità 'Analisi Dati Storici' e 'Allenamento Modello' non saranno disponibili.")
            st.sidebar.code(traceback.format_exc()) # Mostra traceback per debug

# Se non c'è stato caricamento e non c'è nulla in sessione, usa l'errore pre-calcolato
df = st.session_state.get('df', None) # Assicurati che 'df' rifletta lo stato attuale
if df is None and df_load_error:
     # Mostra errore solo se non è già stato mostrato sopra
     if 'Errore CSV:' not in st.sidebar.getvalue(): # Evita duplicati
         st.sidebar.error(f"Errore CSV: {df_load_error}")
         st.sidebar.info("Le funzionalità 'Analisi Dati Storici' e 'Allenamento Modello' non saranno disponibili.")
elif df is None and not uploaded_data_file and not os.path.exists(DEFAULT_DATA_PATH):
      st.sidebar.info(f"Nessun dato CSV caricato o default ('{DEFAULT_DATA_PATH}') trovato. Funzionalità Analisi/Training disabilitate.")


# --- Selezione Modello ---
st.sidebar.divider()
st.sidebar.subheader("Modello Predittivo (per Simulazione)")

# --- OTTIMIZZAZIONE: find_available_models è ora cachata ---
available_models_dict = find_available_models(MODELS_DIR)
model_display_names = sorted(list(available_models_dict.keys())) # Ordina per consistenza
MODEL_CHOICE_UPLOAD = "Carica File Manualmente..."
MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"
selection_options = [MODEL_CHOICE_NONE] + model_display_names + [MODEL_CHOICE_UPLOAD]
current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

# Gestisce caso in cui modello attivo non sia più nella lista (es. file eliminato)
if current_selection_name not in selection_options:
    st.session_state.active_model_name = MODEL_CHOICE_NONE # Resetta a None
    current_selection_name = MODEL_CHOICE_NONE
    # Resetta anche modello/config/scaler associati
    st.session_state.active_config = None
    st.session_state.active_model = None
    st.session_state.active_device = None
    st.session_state.active_scaler_features = None
    st.session_state.active_scaler_targets = None

try: current_index = selection_options.index(current_selection_name)
except ValueError: current_index = 0 # Fallback a None

selected_model_display_name = st.sidebar.selectbox(
    "Modello:", selection_options, index=current_index,
    key="model_selector", # Aggiunto key per stabilità
    help="Scegli un modello pre-addestrato dalla cartella 'models' o carica i file manualmente."
)

# --- Logica Caricamento Modello ---
config_to_load = None; model_to_load = None; device_to_load = None
scaler_f_to_load = None; scaler_t_to_load = None; load_error_sidebar = False

# Resetta stato attivo prima di tentare il caricamento (necessario se si cambia selezione)
active_model_changed = (selected_model_display_name != st.session_state.get('active_model_name', MODEL_CHOICE_NONE))

# Logica pulizia stato precedente SE il modello è cambiato
if active_model_changed:
    # Pulisci le risorse cachate specifiche del vecchio modello (se esistono)
    # NOTA: Questo non è strettamente necessario con @st.cache_resource se le chiavi dipendono dal path,
    # ma può essere utile per liberare memoria esplicitamente.
    # load_specific_model.clear() # Svuota cache modello
    # load_specific_scalers.clear() # Svuota cache scaler
    # load_model_config.clear() # Svuota cache config

    # Resetta le variabili di stato relative al modello
    st.session_state.active_model_name = None
    st.session_state.active_config = None
    st.session_state.active_model = None
    st.session_state.active_device = None
    st.session_state.active_scaler_features = None
    st.session_state.active_scaler_targets = None
    # print(f"Modello cambiato da '{current_selection_name}' a '{selected_model_display_name}', stato resettato.") # Debug


# Procede al caricamento solo se la selezione è valida e diversa da None
if selected_model_display_name != MODEL_CHOICE_NONE:
    if selected_model_display_name == MODEL_CHOICE_UPLOAD:
        with st.sidebar.expander("Carica File Modello Manualmente", expanded=False):
            m_f = st.file_uploader('.pth', type=['pth'], key="up_pth")
            sf_f = st.file_uploader('.joblib (Features)', type=['joblib'], key="up_scf")
            st_f = st.file_uploader('.joblib (Target)', type=['joblib'], key="up_sct")
            st.caption("Configura parametri modello:")
            c1, c2 = st.columns(2)
            # Usa valori di fallback sensati o da stato globale
            fallback_up_iw = st.session_state.get('active_config', {}).get('input_window', 24)
            fallback_up_ow = st.session_state.get('active_config', {}).get('output_window', 12)
            fallback_up_hs = st.session_state.get('active_config', {}).get('hidden_size', 128)
            fallback_up_nl = st.session_state.get('active_config', {}).get('num_layers', 2)
            fallback_up_dr = st.session_state.get('active_config', {}).get('dropout', 0.2)

            iw = c1.number_input("In Win", 1, 168, fallback_up_iw, 6, key="up_in")
            ow = c1.number_input("Out Win", 1, 72, fallback_up_ow, 1, key="up_out")
            hs = c2.number_input("Hidden", 16, 1024, fallback_up_hs, 16, key="up_hid")
            nl = c2.number_input("Layers", 1, 8, fallback_up_nl, 1, key="up_lay")
            dr = c2.slider("Dropout", 0.0, 0.7, fallback_up_dr, 0.05, key="up_drop")

            # Le feature globali sono la base per la selezione dei target qui
            # Filtra per colonne Livello dalle feature globali
            targets_global_options = [col for col in st.session_state.get('feature_columns', []) if 'Livello' in col]
            # Default: prova a usare target del modello attivo, altrimenti seleziona tutti i livelli
            default_targets_up = st.session_state.get('active_config', {}).get('target_columns', targets_global_options)
            # Assicura che i default siano tra le opzioni disponibili
            default_targets_up = [t for t in default_targets_up if t in targets_global_options]

            targets_up = st.multiselect("Target", targets_global_options, default=default_targets_up, key="up_targets")

            if m_f and sf_f and st_f and targets_up:
                # Le feature per il modello caricato SARANNO quelle globali correnti
                current_model_features = st.session_state.get('feature_columns', [])
                if not current_model_features:
                     st.error("Errore: Le feature globali ('feature_columns' in session_state) non sono definite. Impossibile caricare modello manuale.")
                else:
                    temp_cfg = {"input_window": iw, "output_window": ow, "hidden_size": hs,
                                "num_layers": nl, "dropout": dr, "target_columns": targets_up,
                                "feature_columns": current_model_features, # USA quelle globali
                                "name": "uploaded_model",
                                "display_name": "Modello Caricato Manualmente"} # Aggiunto display name

                    # Chiama le funzioni di caricamento (che sono cachate)
                    model_to_load, device_to_load = load_specific_model(m_f, temp_cfg)
                    scaler_f_to_load, scaler_t_to_load = load_specific_scalers(sf_f, st_f)

                    if model_to_load and scaler_f_to_load and scaler_t_to_load:
                        config_to_load = temp_cfg
                        # Se caricato con successo, aggiorna nome attivo SOLO ALLA FINE
                        # st.session_state.active_model_name = selected_model_display_name # Fatto sotto
                    else:
                        load_error_sidebar = True # Segnala errore nel caricamento
            elif not (m_f and sf_f and st_f):
                 st.caption("Carica tutti e tre i file (.pth, .joblib x2) e scegli i target.")
            elif not targets_up:
                 st.caption("Seleziona almeno un target per il modello.")

    else: # Modello pre-addestrato selezionato dalla lista
        if selected_model_display_name in available_models_dict:
            model_info = available_models_dict[selected_model_display_name]
            # Carica config usando la funzione cachata
            config_to_load = load_model_config(model_info["config_path"])
            if config_to_load:
                # Aggiungi info percorso e nome file per riferimento
                config_to_load["pth_path"] = model_info["pth_path"]
                config_to_load["scaler_features_path"] = model_info["scaler_features_path"]
                config_to_load["scaler_targets_path"] = model_info["scaler_targets_path"]
                config_to_load["config_name"] = model_info["config_name"] # Nome file senza estensione
                config_to_load["display_name"] = selected_model_display_name # Nome visualizzato

                # Gestione feature_columns (se mancano nel JSON, usa quelle globali)
                if "feature_columns" not in config_to_load or not config_to_load["feature_columns"]:
                     global_features = st.session_state.get('feature_columns', [])
                     if global_features:
                         st.warning(f"Config '{selected_model_display_name}' non specifica 'feature_columns'. Uso le {len(global_features)} feature globali definite.")
                         config_to_load["feature_columns"] = global_features # Usa quelle globali
                     else:
                         st.error(f"Errore: Config '{selected_model_display_name}' non specifica 'feature_columns' e le feature globali non sono definite.")
                         config_to_load = None # Invalida config
                         load_error_sidebar = True

                # Carica modello e scaler solo se config è valido
                if config_to_load:
                    # Chiama le funzioni di caricamento (che sono cachate)
                    model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
                    scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])

                    if not (model_to_load and scaler_f_to_load and scaler_t_to_load):
                        load_error_sidebar = True; config_to_load = None # Reset config se caricamento fallisce
                    # else: # Successo: nome attivo aggiornato sotto
                        # st.session_state.active_model_name = selected_model_display_name # Fatto sotto
            else:
                load_error_sidebar = True # Errore caricamento config
        else:
             # Questo caso non dovrebbe accadere se available_models_dict è aggiornato
             st.sidebar.error(f"Modello '{selected_model_display_name}' selezionato ma non trovato nella lista disponibile.")
             load_error_sidebar = True


# Salva nello stato sessione SOLO se tutto è caricato correttamente E la selezione non è 'None'
# Questo blocco viene eseguito DOPO il tentativo di caricamento
if config_to_load and model_to_load and device_to_load and scaler_f_to_load and scaler_t_to_load and selected_model_display_name != MODEL_CHOICE_NONE:
    # Aggiorna lo stato sessione solo se il caricamento è andato a buon fine
    st.session_state.active_config = config_to_load
    st.session_state.active_model = model_to_load
    st.session_state.active_device = device_to_load
    st.session_state.active_scaler_features = scaler_f_to_load
    st.session_state.active_scaler_targets = scaler_t_to_load
    st.session_state.active_model_name = selected_model_display_name # Aggiorna il nome attivo QUI
    # print(f"Modello '{selected_model_display_name}' caricato con successo nello stato sessione.") # Debug
elif active_model_changed and selected_model_display_name == MODEL_CHOICE_NONE:
     # Se l'utente ha selezionato esplicitamente "Nessun Modello", pulisci lo stato
     st.session_state.active_model_name = MODEL_CHOICE_NONE
     st.session_state.active_config = None
     st.session_state.active_model = None
     st.session_state.active_device = None
     st.session_state.active_scaler_features = None
     st.session_state.active_scaler_targets = None
     # print("Nessun modello selezionato, stato pulito.") # Debug


# Mostra feedback basato sullo stato sessione AGGIORNATO
active_model_in_state = st.session_state.get('active_model')
active_config_in_state = st.session_state.get('active_config')
active_name_in_state = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

if active_model_in_state and active_config_in_state:
    cfg = active_config_in_state
    # Gestisci nome per modello caricato manualmente
    display_feedback_name = cfg.get("display_name", active_name_in_state if active_name_in_state != MODEL_CHOICE_UPLOAD else "Modello Caricato")
    # MODIFICATO: Aggiunto il tempo equivalente in ore nella sidebar
    try: # Aggiunto try-except per robustezza se config non è completo
        input_win_fb = cfg.get('input_window', '?')
        output_win_fb = cfg.get('output_window', '?')
        input_h_fb = input_win_fb / 2.0 if isinstance(input_win_fb, int) else '?'
        output_h_fb = output_win_fb / 2.0 if isinstance(output_win_fb, int) else '?'
        st.sidebar.success(f"Modello ATTIVO: '{display_feedback_name}' (In:{input_win_fb} [~{input_h_fb}h], Out:{output_win_fb} [~{output_h_fb}h])")
    except Exception as e_feedback:
         st.sidebar.warning(f"Modello attivo, ma errore visualizzazione dettagli: {e_feedback}")

elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]:
     # Mostra errore solo se il caricamento di un modello specifico è fallito
     st.sidebar.error(f"Caricamento modello '{selected_model_display_name}' fallito.")
elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not active_model_in_state:
     # Se l'utente sta tentando di caricare manualmente ma non è ancora completo
     st.sidebar.info("Completa caricamento manuale modello.")
elif active_name_in_state == MODEL_CHOICE_NONE:
     # Se nessun modello è selezionato
     st.sidebar.info("Nessun modello selezionato per la simulazione.")


# --- Configurazione Soglie Dashboard ---
st.sidebar.divider()
st.sidebar.subheader("Configurazione Soglie Dashboard")
with st.sidebar.expander("Modifica Soglie di Allerta (Rosso) e Attenzione (Giallo)", expanded=False):
    temp_thresholds_update = json.loads(json.dumps(st.session_state.dashboard_thresholds)) # Deep copy per modifica sicura

    monitorable_cols = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]

    for col in monitorable_cols:
        label_short = get_station_label(col, short=True)
        st.markdown(f"**{label_short}** (`{col}`)") # Nome colonna come sottotitolo

        # Recupera soglie correnti o default (con gestione chiave mancante)
        current_sensor_thresholds = st.session_state.dashboard_thresholds.get(col, {})
        # Usa i default globali come fallback finale
        default_sensor_thresholds = DEFAULT_THRESHOLDS.get(col, {})
        current_alert = current_sensor_thresholds.get('alert', default_sensor_thresholds.get('alert', 0.0))
        current_attention = current_sensor_thresholds.get('attention', default_sensor_thresholds.get('attention', 0.0))

        is_level = 'Livello' in col or '(m)' in col or '(mt)' in col
        step = 0.1 if is_level else 0.5 # Aggiustato step pioggia
        # --- MODIFICATO: Usare %.2f per livelli per maggiore precisione come richiesto nelle soglie default ---
        fmt = "%.2f" if is_level else "%.1f"
        min_v = 0.0

        c1_th, c2_th = st.columns(2) # Colonne per le due soglie

        # Input Soglia Allerta (Rossa)
        with c1_th:
            new_alert = st.number_input(
                label=f"🔴 Allerta", value=float(current_alert), min_value=min_v, step=step, format=fmt, # Cast a float per sicurezza
                key=f"thresh_alert_{col}", help=f"Soglia di ALLERTA (Rossa) per: {label_short}"
            )
            # Assicurati che new_alert non sia None per la logica successiva
            if new_alert is None: new_alert = min_v # Fallback a minimo se l'input diventa vuoto

            # Aggiorna dizionario temporaneo solo se c'è una modifica
            if new_alert != current_alert:
                if col not in temp_thresholds_update: temp_thresholds_update[col] = {}
                temp_thresholds_update[col]['alert'] = new_alert

        # Input Soglia Attenzione (Gialla/Arancione)
        with c2_th:
            # Calcola il massimo valore consentito per la soglia di attenzione (basato sul valore ATTUALE del widget allerta)
            max_attention_value = new_alert if new_alert is not None else float('inf')

            # Imposta il valore di default ('value') per il widget Attention
            # prendendo il minimo tra il valore attuale in sessione e il massimo consentito ora
            default_attention_value = min(float(current_attention), max_attention_value)

            new_attention = st.number_input(
                label=f"🟠 Attenzione",
                value=default_attention_value, # Usa il valore default aggiustato
                min_value=min_v,
                max_value=max_attention_value, # Il massimo rimane new_alert
                step=step,
                format=fmt,
                key=f"thresh_attention_{col}",
                help=f"Soglia di ATTENZIONE (Gialla/Arancione) per: {label_short}. Non può superare la soglia di Allerta."
            )
            # Assicurati che new_attention non sia None
            if new_attention is None: new_attention = min_v

            # Aggiorna dizionario temporaneo solo se c'è una modifica
            if new_attention != current_attention:
                # Assicura che il valore inserito non superi l'allerta (il widget dovrebbe già farlo, ma doppia sicurezza)
                adjusted_new_attention = min(new_attention, max_attention_value)
                if col not in temp_thresholds_update: temp_thresholds_update[col] = {}
                temp_thresholds_update[col]['attention'] = adjusted_new_attention
                if adjusted_new_attention < new_attention: # Se il valore è stato limitato
                     st.warning(f"Soglia Attenzione per '{label_short}' limitata al valore di Allerta ({max_attention_value}).")

        st.caption("") # Piccolo spazio tra i sensori


    if st.button("Salva Soglie", key="save_thresholds", type="primary"):
        # Validazione finale: assicura attention <= alert per ogni sensore prima di salvare
        validation_ok = True
        final_thresholds_to_save = {}
        for col in monitorable_cols: # Itera su tutte le colonne monitorabili
            thresholds_dict = temp_thresholds_update.get(col, {}) # Prendi valori modificati o default
            # Recupera valori finali, fallback su stato attuale o default globale se non modificati
            alert_val = thresholds_dict.get('alert', st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, {})).get('alert', 0.0))
            attention_val = thresholds_dict.get('attention', st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, {})).get('attention', 0.0))

            # Gestisci valori None (non dovrebbero esserci con i fallback sopra, ma per sicurezza)
            if alert_val is None: alert_val = 0.0
            if attention_val is None: attention_val = 0.0

            # Converti in float per confronto sicuro
            try:
                 alert_val_f = float(alert_val)
                 attention_val_f = float(attention_val)
            except (ValueError, TypeError):
                 st.error(f"Errore: Valori soglia non numerici per '{get_station_label(col, short=True)}'. Impossibile salvare.")
                 validation_ok = False
                 break

            if attention_val_f > alert_val_f:
                 st.error(f"Errore validazione per '{get_station_label(col, short=True)}': Soglia Attenzione ({attention_val_f}) non può essere maggiore della Soglia Allerta ({alert_val_f}).")
                 validation_ok = False
                 break # Ferma alla prima violazione

            final_thresholds_to_save[col] = {'alert': alert_val_f, 'attention': attention_val_f}


        if validation_ok:
            # Assicurati che tutti i sensori monitorabili siano presenti nel dizionario finale
            # (già fatto dal loop sopra che itera su monitorable_cols)
            st.session_state.dashboard_thresholds = json.loads(json.dumps(final_thresholds_to_save)) # Salva deep copy
            st.success("Soglie aggiornate!")
            time.sleep(0.5)
            st.rerun()
        else:
             st.warning("Modifiche soglie NON salvate a causa di errori di validazione.")


# --- !!! CORREZIONE POSIZIONAMENTO VARIABILI GLOBALI !!! ---
# Definisci queste variabili QUI, dopo il caricamento/gestione di df e modello nella sidebar,
# ma PRIMA del menu di navigazione che le utilizza.

# Recupera variabili necessarie da session state (più pulito)
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scaler_features = st.session_state.get('active_scaler_features')
active_scaler_targets = st.session_state.get('active_scaler_targets')
df_current_csv = st.session_state.get('df', None) # Dati CSV (possono essere None)
data_ready_csv = df_current_csv is not None # Definisci data_ready_csv qui
# Feature columns specifiche del modello attivo, fallback a quelle globali
feature_columns_current_model = active_config.get("feature_columns", st.session_state.get('feature_columns', [])) if active_config else st.session_state.get('feature_columns', [])
date_col_name_csv = st.session_state.date_col_name_csv # Nome colonna data per CSV
model_ready = active_model is not None and active_config is not None # Definisci model_ready qui

# --- Menu Navigazione ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
# Le variabili model_ready e data_ready_csv sono definite sopra

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
requires_model = ['Simulazione']
requires_csv = ['Analisi Dati Storici', 'Allenamento Modello'] # Anche simulazione può usare CSV

radio_captions = []
disabled_options = []
default_page_idx = 0 # Default alla Dashboard

# Determina stato disabled e captions
for i, opt in enumerate(radio_options):
    caption = ""; disabled = False
    if opt == 'Dashboard': caption = "Monitoraggio GSheet (30 min)" # MODIFICATO: Aggiunto riferimento frequenza
    elif opt == 'Simulazione':
        if not model_ready: caption = "Richiede Modello attivo"; disabled = True
        else: caption = "Esegui previsioni" # Testo più conciso
    elif opt == 'Analisi Dati Storici':
        # Usa la variabile data_ready_csv definita sopra
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Esplora dati CSV"
    elif opt == 'Allenamento Modello':
        # Usa la variabile data_ready_csv definita sopra
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Allena un nuovo modello"

    radio_captions.append(caption); disabled_options.append(disabled)
    if opt == 'Dashboard': default_page_idx = i # Assicura che il default sia Dashboard

# Logica selezione pagina (migliorata gestione default/redirect)
current_page_key = 'current_page'
selected_page = st.session_state.get(current_page_key, radio_options[default_page_idx]) # Default Dashboard

# Se la pagina salvata non è valida o è disabilitata, resetta al default
try:
    current_page_index_in_options = radio_options.index(selected_page)
    if disabled_options[current_page_index_in_options]:
        st.sidebar.warning(f"Pagina '{selected_page}' non disponibile. Reindirizzato a Dashboard.")
        selected_page = radio_options[default_page_idx]
        st.session_state[current_page_key] = selected_page # Aggiorna session state
        # Forzare rerun qui può causare loop, meglio lasciar aggiornare il widget sotto
        # st.rerun()
except ValueError:
    # La pagina salvata non esiste più nelle opzioni, resetta al default
    selected_page = radio_options[default_page_idx]
    st.session_state[current_page_key] = selected_page

# Trova l'indice della pagina selezionata (che ora è sicuramente valida)
try:
    current_idx_display = radio_options.index(selected_page)
except ValueError:
     current_idx_display = default_page_idx # Fallback ulteriore

# Crea il widget radio
chosen_page = st.sidebar.radio(
    'Scegli una funzionalità:', # Label più corta
    options=radio_options,
    captions=radio_captions,
    index=current_idx_display,
    key='page_selector_radio', # Key diversa da quella di session state
    # disabled non è un argomento diretto di st.radio, la logica sopra gestisce la selezione/redirect
)

# Aggiorna session state e forza rerun SE la scelta cambia
if chosen_page != selected_page:
    st.session_state[current_page_key] = chosen_page
    st.rerun() # Forza rerun per caricare la nuova pagina

# La pagina da visualizzare è ora in chosen_page
page = chosen_page


# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================
# Le variabili globali (active_config, df_current_csv, etc.) sono già state recuperate prima del menu

# --- PAGINA DASHBOARD ---
if page == 'Dashboard':
    # MODIFICATO: Titolo per chiarezza sulla frequenza
    st.header(f'📊 Dashboard Monitoraggio Idrologico (Aggiornamento @ 30 min)')

    # Controllo credenziali Google
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("🚨 **Errore Configurazione:** Credenziali Google ('GOOGLE_CREDENTIALS') non trovate nei secrets di Streamlit.")
        st.info("Aggiungi le credenziali del tuo service account Google Cloud come secret per abilitare la dashboard. Assicurati che l'account abbia accesso in lettura al Google Sheet.")
        st.stop() # Blocca esecuzione pagina Dashboard

    # --- Logica Fetch Dati (usa TTL e chiave cache basata sul tempo) ---
    now_ts = time.time()
    # Chiave cache cambia ogni intervallo di refresh
    cache_time_key = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)

    # Chiama la funzione cachata (passando esplicitamente num_rows)
    df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
        cache_time_key,
        GSHEET_ID,
        GSHEET_RELEVANT_COLS,
        GSHEET_DATE_COL,
        GSHEET_DATE_FORMAT,
        num_rows_to_fetch=DASHBOARD_HISTORY_ROWS # Usa la costante (modificata a 48)
    )

    # Salva risultati in session state per riferimento (es. ultimo fetch time)
    st.session_state.last_dashboard_data = df_dashboard # Salva il DataFrame (o None)
    st.session_state.last_dashboard_error = error_msg
    if df_dashboard is not None or error_msg is None: # Aggiorna tempo solo se fetch OK o nessun errore grave
        st.session_state.last_dashboard_fetch_time = actual_fetch_time

    # --- Visualizzazione Dati e Grafici ---
    col_status, col_refresh_btn = st.columns([3, 1]) # Più spazio per status
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
            try:
                fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess
                fetch_secs_ago = int(fetch_time_ago.total_seconds())
                hours_represented = DASHBOARD_HISTORY_ROWS / 2 # Calcola ore rappresentate
                status_text = (
                    f"Dati GSheet recuperati (ultime ~{hours_represented:.0f} ore, {DASHBOARD_HISTORY_ROWS} rilevazioni @ 30 min) alle: "
                    f"{last_fetch_dt_sess.strftime('%d/%m/%Y %H:%M:%S')} "
                    f"({fetch_secs_ago}s fa). "
                    f"Refresh auto: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
                )
                st.caption(status_text)
            except Exception as e_status:
                 st.caption(f"Errore calcolo status tempo fetch: {e_status}")
        else:
            st.caption("In attesa del primo recupero dati da Google Sheet...")

    with col_refresh_btn:
        # Bottone per forzare pulizia cache e rerun
        if st.button("🔄 Forza Aggiorna", key="dash_refresh_button", help="Pulisce la cache dei dati GSheet e ricarica."):
            # Pulisce la cache specifica della funzione
            fetch_gsheet_dashboard_data.clear()
            st.success("Cache GSheet pulita. Ricaricamento in corso...")
            time.sleep(0.5) # Pausa per mostrare messaggio
            st.rerun() # Ricarica l'app

    # Mostra errori fetch GSheet (se presenti)
    if error_msg:
        # Differenzia errori gravi da warning
        if "API" in error_msg or "Foglio Google non trovato" in error_msg or "Credenziali" in error_msg:
            st.error(f"🚨 Errore Grave GSheet: {error_msg}")
        else: # Errori di parsing dati, ecc.
            st.warning(f"⚠️ Attenzione Dati GSheet: {error_msg}")

    # Mostra dati e grafici SE il DataFrame è stato caricato correttamente ed non è vuoto
    if df_dashboard is not None and not df_dashboard.empty:
        # --- Logica stato ultimo rilevamento ---
        try:
            latest_row_data = df_dashboard.iloc[-1]
            last_update_time = latest_row_data.get(GSHEET_DATE_COL)
            time_now_italy = datetime.now(italy_tz)

            if pd.notna(last_update_time):
                 # Assicurati sia timezone-aware (dovrebbe esserlo dalla funzione fetch)
                 if last_update_time.tzinfo is None: last_update_time = italy_tz.localize(last_update_time)
                 else: last_update_time = last_update_time.tz_convert(italy_tz)

                 time_delta = time_now_italy - last_update_time
                 minutes_ago = int(time_delta.total_seconds() // 60)
                 time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')

                 # Calcola stringa "quanto tempo fa"
                 if minutes_ago < 0: time_ago_str = "nel futuro?" # Controllo sanità
                 elif minutes_ago < 2: time_ago_str = "pochi istanti fa"
                 elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
                 elif minutes_ago < 120: time_ago_str = f"circa 1 ora fa" # Migliore leggibilità
                 else: time_ago_str = f"circa {minutes_ago // 60} ore fa"

                 st.success(f"**Ultimo rilevamento nei dati:** {time_str} ({time_ago_str})")
                 # Warning se i dati sono troppo vecchi (oltre 90 min?)
                 if minutes_ago > 90:
                     st.warning(f"⚠️ Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa (atteso ogni 30 min).")
                 # Info se l'aggiornamento è molto recente
                 elif minutes_ago <= 35: # Entro ~30 min
                     st.info(f"L'ultimo aggiornamento sembra recente (rilevato {minutes_ago} min fa).")

            else:
                 st.warning("⚠️ Timestamp dell'ultimo rilevamento non disponibile o non valido nei dati GSheet più recenti.")
        except Exception as e_latest_ts:
             st.warning(f"Errore lettura timestamp ultimo rilevamento: {e_latest_ts}")

        st.divider()

        # --- Tabella Valori Attuali e Soglie ---
        st.subheader("Tabella Valori Attuali")
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
        table_rows = []
        current_alerts = [] # Ricalcola alert ad ogni rerun basato sui dati correnti
        current_attentions = [] # Lista separata per stato attenzione

        try:
            latest_row_data_table = df_dashboard.iloc[-1] # Usa l'ultima riga disponibile
            for col_name in cols_to_monitor:
                current_value = latest_row_data_table.get(col_name)
                # Recupera entrambe le soglie
                sensor_thresholds = st.session_state.dashboard_thresholds.get(col_name, {})
                threshold_alert = sensor_thresholds.get('alert')
                threshold_attention = sensor_thresholds.get('attention')

                alert_active = False
                attention_active = False
                value_numeric = np.nan # Usato per confronto e styling
                value_display = "N/D"  # Stringa mostrata nella tabella
                unit = ""

                # Determina unità basata sul nome colonna
                if 'Pioggia' in col_name and '(mm)' in col_name: unit = '(mm)'
                elif 'Livello' in col_name and ('(m)' in col_name or '(mt)' in col_name): unit = '(m)'

                if pd.notna(current_value) and isinstance(current_value, (int, float)):
                     value_numeric = current_value
                     # Formattazione per display
                     if unit == '(mm)': value_display = f"{current_value:.1f} {unit}"
                     elif unit == '(m)': value_display = f"{current_value:.2f} {unit}"
                     else: value_display = f"{current_value:.2f}" # Fallback

                     # Controllo soglie (Alert ha priorità su Attention)
                     if threshold_alert is not None and isinstance(threshold_alert, (int, float)) and current_value >= threshold_alert:
                          alert_active = True
                          current_alerts.append((col_name, current_value, threshold_alert))
                     elif threshold_attention is not None and isinstance(threshold_attention, (int, float)) and current_value >= threshold_attention:
                          attention_active = True
                          current_attentions.append((col_name, current_value, threshold_attention)) # Aggiungi a lista attention

                elif pd.notna(current_value):
                     # Se il valore non è numerico ma non è NaN (es. stringa residua)
                     value_display = f"{current_value} (?)" # Segnala potenziale problema

                # Stato per la tabella (con priorità Alert > Attention > OK)
                if alert_active: status = "🔴 ALLERTA"
                elif attention_active: status = "🟠 ATTENZIONE" # NUOVO STATO
                elif pd.notna(value_numeric): status = "✅ OK"
                else: status = "⚪ N/D" # Non disponibile

                # Soglie per display (mostra entrambe se disponibili, format corretto)
                fmt_thresh = "%.2f" if unit == '(m)' else "%.1f"
                alert_display = fmt_thresh % threshold_alert if threshold_alert is not None else "-"
                attention_display = fmt_thresh % threshold_attention if threshold_attention is not None else "-"
                threshold_display_combined = f"Att: {attention_display} / All: {alert_display}"

                table_rows.append({
                    "Sensore": get_station_label(col_name, short=True),
                    "Nome Completo": col_name,
                    "Valore Numerico": value_numeric,
                    "Valore Attuale": value_display,
                    "Soglie (Att/All)": threshold_display_combined, # Mostra entrambe
                    "Soglia Alert": threshold_alert, # Per stile
                    "Soglia Attention": threshold_attention, # Per stile
                    "Stato": status
                })
        except IndexError:
            st.warning("Impossibile recuperare l'ultima riga di dati dal DataFrame della dashboard.")
            df_display = pd.DataFrame(columns=["Sensore", "Valore Attuale", "Soglie (Att/All)", "Stato"]) # Tabella vuota
        except Exception as e_table:
             st.error(f"Errore durante la creazione della tabella valori attuali: {e_table}")
             df_display = pd.DataFrame(columns=["Sensore", "Valore Attuale", "Soglie (Att/All)", "Stato"]) # Tabella vuota

        if not table_rows and df_dashboard is not None and not df_dashboard.empty: # Se non ci sono righe ma il df non era vuoto
             st.warning("Nessuna riga generata per la tabella dei valori attuali.")
             df_display = pd.DataFrame(columns=["Sensore", "Valore Attuale", "Soglie (Att/All)", "Stato"]) # Tabella vuota
        elif table_rows:
            df_display = pd.DataFrame(table_rows)

            # --- Funzione di stile per evidenziare righe (Alert e Attention) ---
            def highlight_thresholds(row):
                style = [''] * len(row)
                alert_thresh = row['Soglia Alert']
                attention_thresh = row['Soglia Attention']
                current_val = row['Valore Numerico']
                text_color = 'black' # Testo nero di default

                # Applica stile solo se i valori sono numerici validi
                if pd.notna(current_val):
                    # Conversione soglie a float per confronto sicuro
                    try: alert_thresh_f = float(alert_thresh)
                    except (ValueError, TypeError): alert_thresh_f = None
                    try: attention_thresh_f = float(attention_thresh)
                    except (ValueError, TypeError): attention_thresh_f = None

                    if alert_thresh_f is not None and current_val >= alert_thresh_f:
                        # Allerta (Rosso)
                        background = 'rgba(255, 0, 0, 0.25)' # Rosso più intenso
                        style = [f'background-color: {background}; color: white; font-weight: bold;'] * len(row) # Testo bianco per contrasto
                    elif attention_thresh_f is not None and current_val >= attention_thresh_f:
                        # Attenzione (Giallo/Arancione)
                        background = 'rgba(255, 165, 0, 0.25)' # Arancione più visibile, leggermente più intenso
                        style = [f'background-color: {background}; color: {text_color};'] * len(row) # Testo nero ok con arancione
                return style

            # Mostra tabella con stile aggiornato
            cols_to_show_in_table = ["Sensore", "Valore Attuale", "Soglie (Att/All)", "Stato"]
            st.dataframe(
                df_display.style.apply(highlight_thresholds, axis=1, subset=pd.IndexSlice[:, ["Valore Numerico", "Soglia Alert", "Soglia Attention"]]), # Applica stile basato su colonne numeriche
                column_order=cols_to_show_in_table,
                hide_index=True,
                use_container_width=True,
                column_config={ # Configurazione colonne per tooltip e tipo
                    "Sensore": st.column_config.TextColumn("Sensore", help="Nome breve del sensore/località"),
                    "Valore Attuale": st.column_config.TextColumn("Valore Attuale", help="Ultimo valore misurato con unità"),
                    "Soglie (Att/All)": st.column_config.TextColumn("Soglie (Att/All)", help="Soglie di Attenzione e Allerta configurate"),
                    "Stato": st.column_config.TextColumn("Stato", help="Stato rispetto alle soglie (OK, ATTENZIONE, ALLERTA, N/D)"),
                }
            )

            # Aggiorna alert globali in session state
            st.session_state.active_alerts = current_alerts
            st.session_state.active_attentions = current_attentions # Salva anche stato attenzione
        # Fine blocco if table_rows

        st.divider()

        # --- Grafico Comparativo Configurabile ---
        st.subheader("Grafico Comparativo Storico")
        # Opzioni selezione basate su nomi brevi
        sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
        # Default: Seleziona i primi 2-3 sensori di livello, se presenti
        default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3]
        if not default_selection_labels and len(sensor_options_compare) > 0: # Fallback se non ci sono livelli
             default_selection_labels = list(sensor_options_compare.keys())[:min(len(sensor_options_compare), 2)] # Max 2 di default

        selected_labels_compare = st.multiselect(
            "Seleziona sensori da confrontare:",
            options=list(sensor_options_compare.keys()),
            default=default_selection_labels,
            key="compare_select_multi" # Key widget
        )

        # Mappa le label selezionate ai nomi colonna originali
        selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare]

        if selected_cols_compare:
            fig_compare = go.Figure()
            # Usa date reali come asse x
            x_axis_data = df_dashboard[GSHEET_DATE_COL]
            for col in selected_cols_compare:
                label = get_station_label(col, short=True) # Label breve per legenda
                fig_compare.add_trace(go.Scatter(
                    x=x_axis_data,
                    y=df_dashboard[col],
                    mode='lines', # Solo linee per più pulizia? 'lines+markers' se preferito
                    name=label,
                    hovertemplate=f'<b>{label}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'
                ))

            hours_represented_compare = DASHBOARD_HISTORY_ROWS / 2
            fig_compare.update_layout(
                title=f"Andamento Storico Comparato (ultime ~{hours_represented_compare:.0f} ore)",
                xaxis_title='Data e Ora',
                yaxis_title='Valore Misurato',
                height=500,
                hovermode="x unified",
                legend_title_text='Sensori',
                margin=dict(t=50, b=40, l=40, r=10) # Margini
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            # Link download
            compare_filename_base = f"compare_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)
        else:
            st.info("Seleziona almeno un sensore per visualizzare il grafico comparativo.")

        st.divider()

        # --- Grafici Individuali ---
        st.subheader("Grafici Individuali Storici")
        num_cols_individual = 3 # Quanti grafici per riga
        graph_cols = st.columns(num_cols_individual)
        col_idx_graph = 0

        x_axis_data_indiv = df_dashboard[GSHEET_DATE_COL] # Date reali per asse x

        for col_name in cols_to_monitor:
            with graph_cols[col_idx_graph % num_cols_individual]:
                try: # Try-except per grafico individuale
                    # Recupera entrambe le soglie
                    sensor_thresholds_indiv = st.session_state.dashboard_thresholds.get(col_name, {})
                    threshold_alert_indiv = sensor_thresholds_indiv.get('alert')
                    threshold_attention_indiv = sensor_thresholds_indiv.get('attention')

                    label_individual = get_station_label(col_name, short=True)
                    unit_individual = ''
                    if 'Pioggia' in col_name and '(mm)' in col_name: unit_individual = '(mm)'
                    elif 'Livello' in col_name and ('(m)' in col_name or '(mt)' in col_name): unit_individual = '(m)'
                    yaxis_title_individual = f"Valore {unit_individual}".strip()

                    fig_individual = go.Figure()
                    fig_individual.add_trace(go.Scatter(
                        x=x_axis_data_indiv,
                        y=df_dashboard[col_name],
                        mode='lines', name=label_individual, # 'lines+markers' se preferisci punti visibili
                        line=dict(color='royalblue'),
                        hovertemplate=f'<b>{label_individual}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'
                    ))

                    # --- Aggiungi entrambe le linee soglia ---
                    fmt_thresh_graph = "%.2f" if unit_individual == '(m)' else "%.1f"
                    # Aggiungi linea soglia ALLERTA (Rossa)
                    if threshold_alert_indiv is not None and isinstance(threshold_alert_indiv, (int, float)):
                        fig_individual.add_hline(
                            y=threshold_alert_indiv, line_dash="dash", line_color="red",
                            annotation_text=f"Allerta ({fmt_thresh_graph % threshold_alert_indiv})",
                            annotation_position="bottom right"
                        )
                    # Aggiungi linea soglia ATTENZIONE (Gialla/Arancione)
                    if threshold_attention_indiv is not None and isinstance(threshold_attention_indiv, (int, float)):
                         fig_individual.add_hline(
                            y=threshold_attention_indiv, line_dash="dash", line_color="orange",
                            annotation_text=f"Attenzione ({fmt_thresh_graph % threshold_attention_indiv})",
                            annotation_position="top right" # Posizione diversa
                        )

                    fig_individual.update_layout(
                        title=f"{label_individual}", # Titolo grafico
                        xaxis_title=None, # Nasconde titolo asse x per compattezza
                        xaxis_showticklabels=True, # Ma mostra le etichette
                        yaxis_title=yaxis_title_individual,
                        height=300, # Grafici più piccoli
                        hovermode="x unified",
                        showlegend=False, # Legenda non necessaria per grafico singolo
                        margin=dict(t=40, b=30, l=50, r=10) # Aggiustato margini
                    )
                    fig_individual.update_yaxes(rangemode='tozero') # Asse Y parte da zero
                    st.plotly_chart(fig_individual, use_container_width=True)

                    # Link download grafico individuale
                    ind_filename_base = f"sensor_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)

                except Exception as e_graph_indiv:
                    st.warning(f"Errore generazione grafico per {label_individual}: {e_graph_indiv}")

            col_idx_graph += 1


        # Riepilogo Alert Attivi (mostra solo Allerta rossa)
        st.divider()
        active_alerts_sess = st.session_state.get('active_alerts', []) # Prende solo allerta rossa
        if active_alerts_sess:
            st.warning("**🚨 ALLERTE ATTIVE (Valori Attuali >= Soglia Rossa) 🚨**")
            alert_md = ""
            # Ordina alert per nome località per consistenza
            sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
            for col, val, thr in sorted_alerts:
                label_alert = get_station_label(col, short=False) # Nome completo località
                sensor_info = STATION_COORDS.get(col, {})
                sensor_type_alert = sensor_info.get('type', '') # Es. 'Pioggia' o 'Livello'
                type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""

                # Formattazione valore e soglia (usa format corretto)
                unit = '(mm)' if 'Pioggia' in col else ('(m)' if 'Livello' in col else '')
                fmt_val_rep = "%.2f" if unit == '(m)' else "%.1f"
                fmt_thr_rep = "%.2f" if unit == '(m)' else "%.1f"

                try: # Try-except per formattazione
                    val_fmt = fmt_val_rep % val
                    thr_fmt = fmt_thr_rep % thr if isinstance(thr, (int, float)) else str(thr)
                    alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit}** >= Soglia Allerta **{thr_fmt}{unit}**\n"
                except Exception as e_fmt_alert:
                     alert_md += f"- **{label_alert}{type_str}**: Errore formattazione valori ({val}, {thr})\n"

            st.markdown(alert_md)
        else:
            st.success("✅ Nessuna soglia di Allerta (Rossa) superata nell'ultimo rilevamento.")

        # Nota sullo stato di Attenzione (Giallo)
        active_attentions_sess = st.session_state.get('active_attentions', [])
        if active_attentions_sess:
            st.info(f"ℹ️ Rilevati {len(active_attentions_sess)} sensori in stato di Attenzione (Giallo). Controlla la tabella e i grafici per i dettagli.")

    elif df_dashboard is not None and df_dashboard.empty: # Fetch OK ma nessun dato
        st.warning("Il recupero dati da Google Sheet ha restituito un set di dati vuoto.")
        if not error_msg: st.info("Controlla che ci siano dati recenti nel foglio Google e che le colonne richieste esistano.")

    else: # Se df_dashboard è None (fetch fallito gravemente)
        st.error("Impossibile visualizzare i dati della dashboard al momento.")
        if not error_msg: st.info("Controlla la connessione internet, le credenziali Google e l'ID del foglio.")

    # --- Meccanismo di refresh automatico ---
    component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
    js_code = f"""
    (function() {{
        const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}';
        // Clear previous interval if it exists
        if (window[intervalIdKey]) {{
            clearInterval(window[intervalIdKey]);
            // console.log('Cleared previous auto-refresh interval: ' + window[intervalIdKey]);
        }}
        // Set new interval
        window[intervalIdKey] = setInterval(function() {{
            // Check if streamlitHook and rerunScript function exist
            if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') {{
                console.log('Auto-refreshing dashboard via streamlitHook...');
                try {{
                    window.streamlitHook.rerunScript(null);
                }} catch (e) {{
                    console.error('Error calling rerunScript:', e);
                    // Optionally clear interval if rerun fails persistently
                    // clearInterval(window[intervalIdKey]);
                }}
            }} else {{
                // Fallback for older Streamlit versions or if hook is not ready
                // console.warn('streamlitHook.rerunScript not available for auto-refresh. Attempting location.reload().');
                // Consider removing reload as it's a full page refresh
                // location.reload();
            }}
        }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});
        // console.log('Set new auto-refresh interval: ' + window[intervalIdKey]);
        return window[intervalIdKey]; // Return interval ID (optional)
    }})();
    """
    try:
        # Use want_output=False as we don't need the interval ID back in Python
        streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
    except Exception as e_js:
         # Avoid showing error in UI, just log it server-side
         print(f"Warning: Impossibile impostare auto-refresh js: {e_js}")
         # st.warning(f"Impossibile impostare auto-refresh: {e_js}")


# --- PAGINA SIMULAZIONE ---
elif page == 'Simulazione':
    st.header('🧪 Simulazione Idrologica')
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo (dalla sidebar) per usare la Simulazione.")
        st.info("Puoi scegliere un modello pre-addestrato dalla cartella 'models' o caricare i file manualmente.")
    else:
        # Recupera config del modello attivo (già definito globalmente)
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns_model = active_config["target_columns"]
        # feature_columns_current_model definito globalmente

        input_hours = input_window / 2.0
        output_hours = output_window / 2.0
        model_display_name_sim = active_config.get("display_name", st.session_state.active_model_name)
        st.info(f"Simulazione con Modello Attivo: **{model_display_name_sim}**")
        st.caption(f"Finestra Input: {input_window} rilevazioni (~{input_hours:.1f} ore) | Finestra Output: {output_window} rilevazioni (~{output_hours:.1f} ore)")
        target_labels = [get_station_label(t, short=True) for t in target_columns_model]
        st.caption(f"Target previsti ({len(target_labels)}): {', '.join(target_labels)}")
        with st.expander(f"Feature richieste dal modello ({len(feature_columns_current_model)})"):
             st.caption(", ".join(feature_columns_current_model))

        sim_data_input = None # Numpy array che conterrà i dati di input (input_window x num_features)
        sim_start_time_info = None # Timestamp per l'inizio della previsione

        # --- Selezione Metodo Input Simulazione ---
        sim_method_options = ['Manuale (Valori Costanti)', 'Importa da Google Sheet (Ultime Rilevazioni)', 'Dettagliato per Intervallo (Tabella)']
        # Aggiunge opzione CSV solo se i dati CSV sono disponibili
        if data_ready_csv: sim_method_options.append('Usa Ultime Rilevazioni da CSV Caricato')

        sim_method = st.radio("Metodo preparazione dati input simulazione:", sim_method_options, key="sim_method_radio_select", horizontal=True)

        # --- Simulazione: Manuale Costante ---
        if sim_method == 'Manuale (Valori Costanti)':
            st.subheader(f'Inserisci valori costanti per le {input_window} rilevazioni di input (~{input_hours:.1f} ore)')
            st.caption("Il modello userà questi valori ripetuti per tutta la finestra di input.")
            temp_sim_values = {}
            cols_manual = st.columns(3)
            # Raggruppa feature per tipo per migliore organizzazione
            feature_groups = {'Pioggia': [], 'Umidità': [], 'Livello': [], 'Altro': []}
            for feature in feature_columns_current_model:
                label_feat = get_station_label(feature, short=True)
                if 'Cumulata' in feature or 'Pioggia' in feature: feature_groups['Pioggia'].append((feature, label_feat))
                elif 'Umidita' in feature: feature_groups['Umidità'].append((feature, label_feat))
                elif 'Livello' in feature: feature_groups['Livello'].append((feature, label_feat))
                else: feature_groups['Altro'].append((feature, label_feat))

            col_idx_man = 0
            # Funzione helper per creare input numerico
            def create_num_input(feature, label, default, step, fmt, key_suffix, help_text):
                 # Assicura che default sia float
                 try: default_f = float(default)
                 except (ValueError, TypeError): default_f = 0.0
                 return st.number_input(label, value=default_f, step=step, format=fmt, key=f"man_{key_suffix}", help=help_text)

            # Popola colonne con i gruppi
            if feature_groups['Pioggia']:
                 with cols_manual[col_idx_man % 3]:
                      st.markdown("**Pioggia (mm/30min)**")
                      for feature, label_feat in feature_groups['Pioggia']:
                           # Cerca mediana nel CSV se disponibile e numerica
                           default_val = 0.0
                           if data_ready_csv and feature in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[feature]):
                               med = df_current_csv[feature].median()
                               if pd.notna(med): default_val = med
                           temp_sim_values[feature] = create_num_input(feature, label_feat, round(max(0.0, default_val),1), 0.5, "%.1f", feature, feature)
                 col_idx_man += 1
            if feature_groups['Livello']:
                 with cols_manual[col_idx_man % 3]:
                      st.markdown("**Livelli (m)**")
                      for feature, label_feat in feature_groups['Livello']:
                           default_val = 0.5 # Fallback generico
                           if data_ready_csv and feature in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[feature]):
                               med = df_current_csv[feature].median()
                               if pd.notna(med): default_val = med
                           temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,2), 0.05, "%.2f", feature, feature)
                 col_idx_man += 1
            # Metti Umidità e Altro insieme se presenti
            if feature_groups['Umidità'] or feature_groups['Altro']:
                with cols_manual[col_idx_man % 3]:
                     if feature_groups['Umidità']:
                          st.markdown("**Umidità (%)**")
                          for feature, label_feat in feature_groups['Umidità']:
                               default_val = 70.0
                               if data_ready_csv and feature in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[feature]):
                                   med = df_current_csv[feature].median()
                                   if pd.notna(med): default_val = med
                               temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,1), 1.0, "%.1f", feature, feature)
                     if feature_groups['Altro']:
                          st.markdown("**Altre Feature**")
                          for feature, label_feat in feature_groups['Altro']:
                              default_val = 0.0
                              if data_ready_csv and feature in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[feature]):
                                   med = df_current_csv[feature].median()
                                   if pd.notna(med): default_val = med
                              temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,2), 0.1, "%.2f", feature, feature)

            # Crea l'array numpy di input ripetendo i valori costanti
            try:
                ordered_values = [temp_sim_values[feature] for feature in feature_columns_current_model]
                # Verifica che tutti i valori siano numerici
                if any(v is None for v in ordered_values):
                     st.error("Errore: Uno o più valori manuali non sono validi (None).")
                     sim_data_input = None
                else:
                    sim_data_input = np.tile(ordered_values, (input_window, 1)).astype(float)
                    sim_start_time_info = datetime.now(italy_tz)
            except KeyError as ke:
                st.error(f"Errore: Feature modello '{ke}' mancante nell'input manuale fornito. Verifica la configurazione.")
                sim_data_input = None
            except Exception as e:
                st.error(f"Errore creazione dati input costanti: {type(e).__name__} - {e}")
                sim_data_input = None

        # --- Simulazione: Google Sheet ---
        elif sim_method == 'Importa da Google Sheet (Ultime Rilevazioni)':
             st.subheader(f'Importa le ultime {input_window} rilevazioni (~{input_hours:.1f} ore) da Google Sheet')
             st.warning("⚠️ Assicurati che le colonne GSheet e la mappatura siano corrette e che i dati nel foglio siano numerici e puliti!")
             # Usa ID GSheet di default (quello della dashboard) come suggerimento
             sheet_url_sim = st.text_input("URL o ID Foglio Google da cui importare", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url_input")
             sheet_id_sim = extract_sheet_id(sheet_url_sim)

             # Mappatura di default (da aggiornare se i nomi cambiano)
             column_mapping_gsheet_to_model_sim_default = {
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
                # 'NomeColonnaUmidita_nel_GSheet': 'Umidita\' Sensore 3452 (Montemurello)', # Esempio se ci fosse Umidità
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
             }
             # Mantieni solo le mappature rilevanti per le feature del modello corrente
             column_mapping_gsheet_to_model_sim = {
                 gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_model_sim_default.items()
                 if model_col in feature_columns_current_model
             }

             with st.expander("Mostra/Modifica Mappatura GSheet -> Modello (Avanzato)"):
                 try:
                     edited_mapping_str = st.text_area("Mappatura Colonne (JSON: {'gsheet_col': 'model_col'})",
                                                     value=json.dumps(column_mapping_gsheet_to_model_sim, indent=2),
                                                     height=300, key="sim_gsheet_map_edit_area")
                     edited_mapping = json.loads(edited_mapping_str)
                     if isinstance(edited_mapping, dict):
                         column_mapping_gsheet_to_model_sim = edited_mapping
                         # st.caption("Mappatura aggiornata.") # Rimuoviamo per pulizia UI
                     else: st.warning("Formato JSON mappatura non valido. Modifiche ignorate.")
                 except json.JSONDecodeError: st.warning("Errore parsing JSON mappatura. Modifiche ignorate.")
                 except Exception as e_map: st.error(f"Errore applicazione mappatura: {e_map}")

             # Verifica quali feature modello MANCANO nella mappatura
             model_features_set = set(feature_columns_current_model)
             mapped_model_features_target = set(column_mapping_gsheet_to_model_sim.values())
             missing_model_features_in_map = list(model_features_set - mapped_model_features_target)

             # Chiedi valori costanti per le feature mancanti
             imputed_values_sim = {}
             needs_imputation_input = False
             if missing_model_features_in_map:
                  st.warning(f"Le seguenti feature del modello non hanno una colonna GSheet corrispondente nella mappatura. Verrà usato un valore costante:")
                  needs_imputation_input = True
                  cols_impute = st.columns(3)
                  col_idx_imp = 0
                  for missing_f in missing_model_features_in_map:
                       with cols_impute[col_idx_imp % 3]:
                            label_missing = get_station_label(missing_f, short=True)
                            default_val = 0.0; fmt = "%.2f"; step = 0.1
                            if data_ready_csv and missing_f in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[missing_f]):
                                 med = df_current_csv[missing_f].median()
                                 if pd.notna(med): default_val = med
                            if 'Umidita' in missing_f: fmt = "%.1f"; step = 1.0
                            elif 'Cumulata' in missing_f or 'Pioggia' in missing_f: fmt = "%.1f"; step = 0.5; default_val = max(0.0, default_val)
                            elif 'Livello' in missing_f: fmt = "%.2f"; step = 0.05

                            imputed_values_sim[missing_f] = st.number_input(f"Valore per '{label_missing}'", value=round(float(default_val), 2), step=step, format=fmt, key=f"sim_gsheet_impute_val_{missing_f}", help=f"Valore costante per {missing_f}")
                       col_idx_imp += 1

             # Definizione funzione fetch specifica per simulazione (NON cachata per essere sicuri di prendere gli ultimi dati)
             # @st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet...") # Rimosso caching qui
             def fetch_sim_gsheet_data_live(sheet_id_fetch, n_rows, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict):
                 """ Recupera, mappa, pulisce e ordina dati da GSheet per simulazione. """
                 print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data_live (Rows: {n_rows})")
                 try:
                     if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", None
                     credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
                     gc = gspread.authorize(credentials)
                     sh = gc.open_by_key(sheet_id_fetch)
                     worksheet = sh.sheet1
                     all_data_gs = worksheet.get_all_values()

                     if not all_data_gs or len(all_data_gs) < (n_rows + 1):
                         return None, f"Errore: Dati GSheet insufficienti (trovate {len(all_data_gs)-1} righe dati, richieste {n_rows}).", None

                     headers_gs = all_data_gs[0]
                     start_index_gs = max(1, len(all_data_gs) - n_rows) # Prendi le ultime n_rows di DATI
                     data_rows_gs = all_data_gs[start_index_gs:]

                     df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)

                     required_gsheet_cols_from_mapping = list(col_mapping.keys())
                     missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
                     if missing_gsheet_cols_in_sheet:
                         return None, f"Errore: Colonne GSheet specificate nella mappatura ma mancanti nel foglio: {', '.join(missing_gsheet_cols_in_sheet)}", None

                     # Seleziona e rinomina
                     df_mapped = df_gsheet_raw[required_gsheet_cols_from_mapping].rename(columns=col_mapping)

                     # Aggiungi colonne mancanti con valori imputati
                     for model_col, impute_val in impute_dict.items():
                          if model_col not in df_mapped.columns:
                              df_mapped[model_col] = impute_val # Usa valore da input utente

                     # Verifica se mancano ancora colonne modello richieste
                     final_missing_model_cols = [c for c in required_model_cols_fetch if c not in df_mapped.columns]
                     if final_missing_model_cols:
                         return None, f"Errore: Colonne modello mancanti dopo mappatura e imputazione: {', '.join(final_missing_model_cols)}", None

                     # --- Pulizia Dati Mappati ---
                     last_valid_timestamp = None
                     # Trova il nome mappato della colonna data GSheet originale
                     date_col_model_name_gs = None
                     if date_col_gs in col_mapping: date_col_model_name_gs = col_mapping[date_col_gs]

                     for col in required_model_cols_fetch: # Itera su tutte le colonne richieste dal modello
                         if col == date_col_model_name_gs: # Pulisci colonna data (se presente e mappata)
                             try:
                                 df_mapped[col] = pd.to_datetime(df_mapped[col], format=date_format_gs, errors='coerce')
                                 if df_mapped[col].isnull().any():
                                     st.warning(f"Date non valide trovate in GSheet ('{col}').")
                                 # Localizza
                                 if df_mapped[col].dt.tz is None: df_mapped[col] = df_mapped[col].dt.tz_localize(italy_tz)
                                 else: df_mapped[col] = df_mapped[col].dt.tz_convert(italy_tz)
                             except Exception as e_date_clean:
                                 return None, f"Errore conversione/pulizia data GSheet '{col}': {e_date_clean}", None
                         else: # Pulisci colonne numeriche
                              try:
                                  # Forza a stringa per pulizia robusta
                                  df_mapped[col] = df_mapped[col].astype(str)
                                  df_mapped[col] = df_mapped[col].str.strip()
                                  # Sostituisci comuni placeholder NaN
                                  df_mapped[col] = df_mapped[col].replace(['N/A', '', '-', 'None', 'null', 'NaN', 'nan', '#N/D', '#DIV/0!'], np.nan, regex=False)
                                  # Sostituisci virgola con punto
                                  if df_mapped[col].astype(str).str.contains(',').any():
                                      df_mapped[col] = df_mapped[col].str.replace(',', '.', regex=False)
                                  # Converti in numerico
                                  df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
                              except Exception as e_clean_num:
                                  st.warning(f"Problema pulizia GSheet colonna '{col}': {e_clean_num}. Verrà trattata come NaN.")
                                  df_mapped[col] = np.nan # Forza NaN in caso di errore

                     # Ordina per data (se presente e valida)
                     if date_col_model_name_gs and date_col_model_name_gs in df_mapped.columns and pd.api.types.is_datetime64_any_dtype(df_mapped[date_col_model_name_gs]):
                          df_mapped = df_mapped.sort_values(by=date_col_model_name_gs, na_position='first')
                          if not df_mapped[date_col_model_name_gs].dropna().empty:
                             last_valid_timestamp = df_mapped[date_col_model_name_gs].dropna().iloc[-1]

                     # --- Selezione Colonne Finali e Gestione NaN ---
                     try:
                         # Seleziona ESATTAMENTE le colonne richieste dal modello NELL'ORDINE CORRETTO
                         df_final = df_mapped[required_model_cols_fetch]
                     except KeyError as e_key:
                         return None, f"Errore selezione/ordine colonne finali: '{e_key}' non trovata dopo mappatura/imputazione.", None

                     numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns
                     nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
                     if nan_count_before > 0:
                          st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet importati. Applico forward-fill e backward-fill.")
                          df_final[numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
                          nan_count_after = df_final[numeric_cols_to_fill].isnull().sum().sum()
                          if nan_count_after > 0:
                              st.error(f"Errore: {nan_count_after} NaN residui dopo fillna. Controlla dati GSheet (colonne vuote all'inizio?).")
                              st.dataframe(df_final[df_final.isnull().any(axis=1)]) # Mostra righe con NaN
                              return None, "Errore: NaN residui dopo fillna.", None

                     # Verifica numero righe finale
                     if len(df_final) != n_rows:
                         return None, f"Errore: Numero righe finali ({len(df_final)}) diverso da richiesto ({n_rows}).", None

                     # Rimuovi colonna data se era stata inclusa per ordinamento ma non è una feature
                     if date_col_model_name_gs and date_col_model_name_gs not in required_model_cols_fetch and date_col_model_name_gs in df_final.columns:
                          df_final = df_final.drop(columns=[date_col_model_name_gs])

                     # Verifica finale delle colonne
                     if list(df_final.columns) != required_model_cols_fetch:
                          return None, f"Errore: Colonne finali ({list(df_final.columns)}) non corrispondono a quelle richieste ({required_model_cols_fetch}).", None


                     return df_final, None, last_valid_timestamp # Successo

                 except gspread.exceptions.APIError as api_e_sim:
                     error_message_sim = str(api_e_sim)
                     try:
                         error_details = api_e_sim.response.json().get('error', {})
                         error_message_sim = error_details.get('message', str(api_e_sim))
                         status_code = error_details.get('code', 'N/A')
                         if status_code == 403: error_message_sim += " Verifica condivisione foglio GSheet."
                         elif status_code == 429: error_message_sim += f" Limite API Google superato."
                     except: pass
                     return None, f"Errore API Google Sheets: {error_message_sim}", None
                 except gspread.exceptions.SpreadsheetNotFound:
                     return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id_fetch}').", None
                 except Exception as e_sim_fetch:
                     st.error(traceback.format_exc())
                     return None, f"Errore imprevisto importazione GSheet per simulazione: {type(e_sim_fetch).__name__} - {e_sim_fetch}", None

             # Bottone per avviare importazione GSheet
             if st.button("Importa e Prepara da Google Sheet", key="sim_run_gsheet_import", disabled=(not sheet_id_sim)):
                 if not sheet_id_sim: st.error("URL o ID del Foglio Google non valido.")
                 else:
                     with st.spinner("Importazione e preparazione dati da Google Sheet..."):
                         imported_df_numeric, import_err, last_ts_gs = fetch_sim_gsheet_data_live(
                             sheet_id_sim,
                             input_window, # n_rows = input_window
                             GSHEET_DATE_COL, # Nome colonna data nel GSheet
                             GSHEET_DATE_FORMAT, # Formato data nel GSheet
                             column_mapping_gsheet_to_model_sim, # Mappatura GSheet -> Modello
                             feature_columns_current_model, # Lista colonne richieste dal modello
                             imputed_values_sim # Valori per colonne mancanti
                         )

                     if import_err:
                         st.error(f"Importazione GSheet fallita: {import_err}")
                         st.session_state.imported_sim_data_gs_df = None
                         sim_data_input = None
                     elif imported_df_numeric is not None:
                          st.success(f"Importate e processate {len(imported_df_numeric)} rilevazioni da GSheet.")
                          if imported_df_numeric.shape == (input_window, len(feature_columns_current_model)):
                              st.session_state.imported_sim_data_gs_df = imported_df_numeric # Salva in sessione
                              sim_data_input = imported_df_numeric.values # Prepara per la predizione
                              sim_start_time_info = last_ts_gs if last_ts_gs else datetime.now(italy_tz) # Usa timestamp o ora corrente
                              with st.expander("Mostra Dati Numerici Importati (pronti per modello)"):
                                   st.dataframe(imported_df_numeric.round(3))
                          else:
                               st.error(f"Errore Shape dati GSheet post-processamento ({imported_df_numeric.shape}) vs atteso ({input_window}, {len(feature_columns_current_model)}).")
                               st.session_state.imported_sim_data_gs_df = None
                               sim_data_input = None
                     else:
                          st.error("Importazione GSheet non riuscita per motivi sconosciuti (DataFrame None).")
                          st.session_state.imported_sim_data_gs_df = None
                          sim_data_input = None

             # Logica per usare dati GSheet già in sessione (se bottone non premuto)
             elif 'imported_sim_data_gs_df' in st.session_state and st.session_state.imported_sim_data_gs_df is not None:
                 imported_df_state = st.session_state.imported_sim_data_gs_df
                 if isinstance(imported_df_state, pd.DataFrame) and imported_df_state.shape == (input_window, len(feature_columns_current_model)):
                     sim_data_input = imported_df_state.values
                     # TODO: Salvare e recuperare anche il timestamp 'last_ts_gs' in session state se si vuole essere precisi
                     sim_start_time_info = st.session_state.get('sim_start_time_gs', datetime.now(italy_tz)) # Usa ora salvata o corrente
                     st.info("Utilizzo dati importati precedentemente da Google Sheet (da cache sessione). Premi 'Importa' per aggiornare.")
                     with st.expander("Mostra Dati Importati (da cache sessione)"):
                         st.dataframe(imported_df_state.round(3))
                 else: # Dati in sessione non validi
                      st.session_state.imported_sim_data_gs_df = None # Pulisci stato invalido


        # --- Simulazione: Dettagliato per Intervallo (Tabella) ---
        elif sim_method == 'Dettagliato per Intervallo (Tabella)':
            st.subheader(f'Inserisci dati dettagliati per le {input_window} rilevazioni di input (~{input_hours:.1f} ore)')
            st.caption("Modifica la tabella sottostante con i valori desiderati per ogni intervallo di 30 minuti.")

            # Chiave univoca per lo stato della tabella basata sui parametri correnti
            session_key_hourly = f"sim_hourly_data_{input_window}_{'_'.join(sorted(feature_columns_current_model))}"

            # Inizializza o recupera il DataFrame per l'editor
            if session_key_hourly not in st.session_state:
                 st.caption("Inizializzazione tabella dati con valori mediani (da CSV se disponibile)...")
                 init_vals = {}
                 for col in feature_columns_current_model:
                      med_val = 0.0
                      if data_ready_csv and col in df_current_csv and pd.api.types.is_numeric_dtype(df_current_csv[col]):
                          med = df_current_csv[col].median()
                          if pd.notna(med): med_val = med
                      # Fallback se CSV non disponibile o colonna non numerica
                      elif 'Pioggia' in col or 'Cumulata' in col: med_val = 0.0
                      elif 'Livello' in col: med_val = 0.5
                      elif 'Umidita' in col: med_val = 70.0

                      init_vals[col] = float(med_val) # Assicura sia float

                 # Crea DataFrame con valori iniziali ripetuti
                 init_df_data = np.repeat([list(init_vals.values())], input_window, axis=0)
                 init_df = pd.DataFrame(init_df_data, columns=feature_columns_current_model)
                 st.session_state[session_key_hourly] = init_df.astype(float) # Assicura tipi float
            else: # Recupera da session state
                init_df = st.session_state[session_key_hourly]
                # Validazione rapida del DataFrame in sessione
                if not isinstance(init_df, pd.DataFrame) or init_df.shape != (input_window, len(feature_columns_current_model)) or list(init_df.columns) != feature_columns_current_model:
                     st.warning("Dati tabella in sessione non validi. Reinizializzazione.")
                     del st.session_state[session_key_hourly] # Rimuovi stato errato
                     st.rerun() # Forza reinizializzazione

            # Prepara DataFrame per l'editor (copia per sicurezza)
            df_for_editor = init_df.copy()

            # Configurazione colonne per data_editor
            column_config_editor = {}
            for col in feature_columns_current_model:
                 label_edit = get_station_label(col, short=True)
                 fmt = "%.3f"; step = 0.01; min_v=None; max_v=None # Default generico
                 if 'Pioggia' in col or 'Cumulata' in col: fmt = "%.1f"; step = 0.5; min_v=0.0; label_edit += " (mm/30m)"
                 elif 'Umidita' in col: fmt = "%.1f"; step = 1.0; min_v=0.0; max_v=100.0; label_edit += " (%)"
                 elif 'Livello' in col: fmt = "%.3f"; step = 0.01; min_v = 0.0; label_edit += " (m)" # Assumi livelli >= 0

                 column_config_editor[col] = st.column_config.NumberColumn(
                     label=label_edit, help=col, format=fmt, step=step,
                     min_value=min_v, max_value=max_v, required=True # Rende la colonna obbligatoria
                 )

            # Mostra data editor
            edited_df = st.data_editor(
                df_for_editor,
                height=min(600, (input_window + 1) * 35 + 3), # Limita altezza massima
                use_container_width=True,
                column_config=column_config_editor,
                key=f"data_editor_{session_key_hourly}",
                num_rows="fixed" # Impedisce aggiunta/rimozione righe
            )

            # Validazione dati editati
            validation_passed = False
            if not isinstance(edited_df, pd.DataFrame):
                st.error("Errore interno: L'editor non ha restituito un DataFrame.")
            elif edited_df.shape[0] != input_window:
                st.error(f"Errore: La tabella deve avere esattamente {input_window} righe (intervalli).")
            elif list(edited_df.columns) != feature_columns_current_model:
                 st.error("Errore: Colonne della tabella modificate in modo imprevisto.")
            elif edited_df.isnull().values.any(): # Controllo più efficiente per NaN
                 nan_cols = edited_df.columns[edited_df.isnull().any()].tolist()
                 st.warning(f"Attenzione: Valori mancanti rilevati nella tabella nelle colonne: {', '.join(nan_cols)}. Compilare tutte le celle.")
            else:
                 try:
                      # Tenta conversione a float (dovrebbe già esserlo)
                      sim_data_input_edit = edited_df[feature_columns_current_model].astype(float).values
                      # Verifica finale shape
                      if sim_data_input_edit.shape == (input_window, len(feature_columns_current_model)):
                          sim_data_input = sim_data_input_edit
                          sim_start_time_info = datetime.now(italy_tz)
                          validation_passed = True
                          # Aggiorna session state solo se ci sono state modifiche
                          if not init_df.equals(edited_df):
                              st.session_state[session_key_hourly] = edited_df
                              # st.caption("Modifiche tabella salvate in sessione.") # Meno verbose
                      else: st.error("Errore shape dati tabella dopo conversione.")
                 except Exception as e_edit_final:
                      st.error(f"Errore finale conversione dati tabella: {type(e_edit_final).__name__} - {e_edit_final}")


        # --- Simulazione: Ultime Rilevazioni da CSV ---
        elif sim_method == 'Usa Ultime Rilevazioni da CSV Caricato':
             st.subheader(f"Usa le ultime {input_window} rilevazioni (~{input_hours:.1f} ore) dai dati CSV caricati")
             st.warning("⚠️ Assicurati che l'intervallo temporale dei dati CSV sia coerente con quello su cui il modello è stato addestrato!")
             if not data_ready_csv:
                 st.error("Dati CSV non caricati. Carica un file CSV nella sidebar.")
             elif len(df_current_csv) < input_window:
                 st.error(f"Dati CSV ({len(df_current_csv)} righe) insufficienti per la finestra di input richiesta ({input_window} rilevazioni).")
             else:
                  try:
                       # Seleziona le ultime righe e le colonne richieste dal modello NELL'ORDINE CORRETTO
                       latest_csv_data_df = df_current_csv.iloc[-input_window:][feature_columns_current_model]

                       if latest_csv_data_df.isnull().values.any():
                            nan_cols_csv = latest_csv_data_df.columns[latest_csv_data_df.isnull().any()].tolist()
                            st.error(f"Trovati valori mancanti (NaN) nelle ultime {input_window} rilevazioni delle colonne richieste nel CSV ({', '.join(nan_cols_csv)}). Impossibile usare per simulazione.")
                            with st.expander("Mostra righe CSV con NaN (ultime rilevazioni)"):
                                 st.dataframe(latest_csv_data_df[latest_csv_data_df.isnull().any(axis=1)])
                            sim_data_input = None
                       else:
                            latest_csv_data_np = latest_csv_data_df.astype(float).values
                            if latest_csv_data_np.shape == (input_window, len(feature_columns_current_model)):
                                sim_data_input = latest_csv_data_np
                                try:
                                    last_ts_csv_used = df_current_csv.iloc[-1][date_col_name_csv]
                                    if pd.notna(last_ts_csv_used):
                                         # Gestione timezone (assume naive -> localizza, altrimenti converte)
                                         if not isinstance(last_ts_csv_used, pd.Timestamp): # Converti se non è già Timestamp
                                              last_ts_csv_used = pd.to_datetime(last_ts_csv_used)
                                         if last_ts_csv_used.tzinfo is None: sim_start_time_info = italy_tz.localize(last_ts_csv_used)
                                         else: sim_start_time_info = last_ts_csv_used.tz_convert(italy_tz)
                                         st.caption(f"Simulazione basata su dati CSV fino a: {sim_start_time_info.strftime('%d/%m/%Y %H:%M %Z')}")
                                    else: sim_start_time_info = datetime.now(italy_tz); st.caption("Ora inizio previsione: Ora corrente (timestamp CSV non valido).")
                                except KeyError: sim_start_time_info = datetime.now(italy_tz); st.caption(f"Ora inizio previsione: Ora corrente (colonna data '{date_col_name_csv}' non trovata).")
                                except Exception as e_ts_csv: sim_start_time_info = datetime.now(italy_tz); st.caption(f"Ora inizio previsione: Ora corrente (errore lettura timestamp CSV: {e_ts_csv}).")

                                with st.expander("Mostra dati CSV usati per l'input"):
                                     cols_to_show_csv = [date_col_name_csv] + feature_columns_current_model
                                     st.dataframe(df_current_csv.iloc[-input_window:][cols_to_show_csv].round(3))
                            else:
                                st.error(f"Errore shape dati CSV estratti ({latest_csv_data_np.shape}) vs atteso ({input_window}, {len(feature_columns_current_model)}).")
                                sim_data_input = None
                  except KeyError as ke:
                       st.error(f"Errore: Colonna modello '{ke}' non trovata nel DataFrame CSV caricato. Verifica il file CSV o le feature del modello.")
                       sim_data_input = None
                  except Exception as e_csv_sim_extract:
                       st.error(f"Errore imprevisto durante estrazione dati CSV per simulazione: {type(e_csv_sim_extract).__name__} - {e_csv_sim_extract}")
                       st.error(traceback.format_exc())
                       sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE ---
        st.divider()
        # Verifica finale se i dati di input sono pronti
        input_ready = (
            sim_data_input is not None and
            isinstance(sim_data_input, np.ndarray) and
            sim_data_input.shape == (input_window, len(feature_columns_current_model)) and
            not np.isnan(sim_data_input).any()
        )

        if sim_data_input is not None and not input_ready:
             # Mostra errori di validazione più specifici se l'input è stato generato ma non è valido
             if not isinstance(sim_data_input, np.ndarray): st.error("Errore interno: Dati input non sono un array NumPy.")
             elif sim_data_input.shape != (input_window, len(feature_columns_current_model)): st.error(f"Errore shape dati input: Atteso ({input_window}, {len(feature_columns_current_model)}), Ottenuto {sim_data_input.shape}.")
             elif np.isnan(sim_data_input).any(): st.error(f"Errore: Trovati valori NaN ({np.isnan(sim_data_input).sum()}) nei dati di input finali pronti per la simulazione.")
             else: st.error("Errore sconosciuto nei dati di input per la simulazione.")

        run_simulation_button = st.button('🚀 Esegui Simulazione', type="primary", disabled=(not input_ready), key="sim_run_exec")

        if run_simulation_button:
             if input_ready:
                  with st.spinner('Simulazione in corso...'):
                       predictions_sim = predict(active_model, sim_data_input, active_scaler_features, active_scaler_targets, active_config, active_device)

                  if predictions_sim is not None and isinstance(predictions_sim, np.ndarray) and predictions_sim.shape == (output_window, len(target_columns_model)):
                       st.subheader(f'📊 Risultato Simulazione: Previsione per le prossime {output_window} rilevazioni (~{output_hours:.1f} ore)')

                       start_pred_time = sim_start_time_info if sim_start_time_info else datetime.now(italy_tz)
                       st.caption(f"Previsione calcolata a partire da: {start_pred_time.strftime('%d/%m/%Y %H:%M %Z')}")

                       pred_times_sim = [start_pred_time + timedelta(minutes=30*(i+1)) for i in range(output_window)]
                       # 1. Crea DataFrame iniziale dai risultati numerici
                       results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns_model)
                       # 2. Inserisci la colonna 'Ora Prevista' come stringa formattata
                       results_df_sim.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_sim])

                       # 3. Rinomina colonne per display
                       rename_dict = {'Ora Prevista': 'Ora Prevista'}
                       original_to_renamed_map = {}
                       for col in target_columns_model:
                           unit = '(m)' if 'Livello' in col else '' # Solo per livelli (m) per ora
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip()
                           # Gestisci nomi duplicati (improbabile ma possibile)
                           count = 1
                           final_name = new_name
                           while final_name in rename_dict.values():
                               count += 1
                               final_name = f"{new_name}_{count}"
                           rename_dict[col] = final_name
                           original_to_renamed_map[col] = final_name

                       results_df_sim_renamed = results_df_sim.rename(columns=rename_dict)

                       # 4. Prepara DataFrame per display (arrotondamento e stile)
                       df_to_display = results_df_sim_renamed.copy()

                       # --- Funzione Stile Valori ---
                       def style_alert_value(val, original_col_name, thresholds_dict):
                           """ Applica stile rosso se val >= soglia alert, arancione se >= attenzione. """
                           alert_thresh = thresholds_dict.get(original_col_name, {}).get('alert')
                           attention_thresh = thresholds_dict.get(original_col_name, {}).get('attention')
                           style = ''
                           # Converte in float per confronto sicuro
                           try: val_f = float(val)
                           except (ValueError, TypeError): return style # Non applicare stile se non numerico
                           try: alert_thresh_f = float(alert_thresh)
                           except (ValueError, TypeError): alert_thresh_f = None
                           try: attention_thresh_f = float(attention_thresh)
                           except (ValueError, TypeError): attention_thresh_f = None

                           if alert_thresh_f is not None and val_f >= alert_thresh_f:
                               style = 'color: red; font-weight: bold;'
                           elif attention_thresh_f is not None and val_f >= attention_thresh_f:
                                style = 'color: orange; font-weight: bold;' # Arancione per attenzione
                           return style

                       # Applica lo stile e l'arrotondamento
                       styler = df_to_display.style
                       format_dict = {}
                       numeric_cols_renamed = [original_to_renamed_map[col] for col in target_columns_model if col in original_to_renamed_map]

                       for original_col, renamed_col in original_to_renamed_map.items():
                            if renamed_col in df_to_display.columns: # Sicurezza
                                # Applica stile cella per cella
                                styler = styler.applymap(
                                    lambda x: style_alert_value(x, original_col, st.session_state.dashboard_thresholds),
                                    subset=[renamed_col]
                                )
                                # Imposta formato numerico (es. 3 decimali)
                                format_dict[renamed_col] = "{:.3f}"

                       styler = styler.format(format_dict)

                       # Mostra il DataFrame STILIZZATO e FORMATTATO
                       st.dataframe(styler)

                       # Link per il download (usa DataFrame originale NON stilizzato/formattato)
                       st.markdown(get_table_download_link(results_df_sim, f"simulazione_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                       # --- Grafici ---
                       st.subheader('📈 Grafici Previsioni Simulate')
                       # Passa il dizionario delle soglie alla funzione plot_predictions
                       figs_sim = plot_predictions(
                           predictions_sim,
                           active_config,
                           st.session_state.dashboard_thresholds, # Passa le soglie correnti
                           start_pred_time # Passa l'ora di inizio per l'asse X
                       )
                       # Layout colonne per i grafici
                       num_graph_cols = min(len(figs_sim), 2) # Max 2 grafici per riga
                       sim_cols = st.columns(num_graph_cols)
                       for i, fig_sim in enumerate(figs_sim):
                           with sim_cols[i % num_graph_cols]:
                                try: # Try-except per plot individuale
                                    target_col_name = target_columns_model[i]
                                    # Nome file sicuro per download
                                    s_name_file = re.sub(r'[^\w-]', '_', get_station_label(target_col_name, short=False))
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                                except IndexError:
                                    st.warning(f"Errore: Indice grafico {i} fuori range.")
                                except Exception as e_plot_sim:
                                    st.warning(f"Errore durante la visualizzazione del grafico {i}: {e_plot_sim}")

                  else:
                       st.error("Predizione simulazione fallita o risultato non valido (shape o tipo errati).")
                       if predictions_sim is not None: # Debug info se predizione non è None ma invalida
                           st.write("Shape ricevuto:", predictions_sim.shape if isinstance(predictions_sim, np.ndarray) else type(predictions_sim))
                           st.write("Shape atteso:", (output_window, len(target_columns_model)))
             else:
                  st.error("Impossibile eseguire la simulazione: dati input non pronti o non validi. Controlla i passaggi precedenti.")


# --- PAGINA ANALISI DATI STORICI ---
elif page == 'Analisi Dati Storici':
    st.header('🔎 Analisi Dati Storici (da file CSV)')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili. Carica un file CSV valido nella sidebar per usare questa funzionalità.")
    else:
        st.info(f"Dataset CSV caricato: {len(df_current_csv)} righe.")
        try:
            min_dt_csv = df_current_csv[date_col_name_csv].min()
            max_dt_csv = df_current_csv[date_col_name_csv].max()
            st.caption(f"Periodo dati: dal {min_dt_csv.strftime('%d/%m/%Y %H:%M')} al {max_dt_csv.strftime('%d/%m/%Y %H:%M')}")
            min_date = min_dt_csv.date()
            max_date = max_dt_csv.date()
        except Exception as e_date_range:
            st.error(f"Errore lettura range date dal CSV: {e_date_range}. Impossibile filtrare.")
            st.stop() # Blocca analisi se le date non sono valide

        col1, col2 = st.columns(2)

        if min_date >= max_date: # Gestisce caso di un solo giorno o date invertite
             start_date = min_date
             end_date = max_date
             col1.info(f"Disponibile solo il giorno: {min_date.strftime('%d/%m/%Y')}")
        else:
             # Widget selezione date
             start_date = col1.date_input('Data inizio', min_date, min_value=min_date, max_value=max_date, key="analisi_start_date")
             # Assicura che end_date non sia prima di start_date
             end_date_value = max(start_date, max_date)
             end_date = col2.date_input('Data fine', end_date_value, min_value=start_date, max_value=max_date, key="analisi_end_date")

        # Validazione date (redundante con min/max_value, ma per sicurezza)
        if start_date > end_date:
             st.error("Data inizio non può essere successiva alla data fine.")
             st.stop()
        else:
            try:
                # Creazione timestamp per il filtro (inizio giorno / fine giorno)
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.max.time())

                # Gestione Timezone (importante per confronto corretto)
                df_date_col = df_current_csv[date_col_name_csv]
                if pd.api.types.is_datetime64_any_dtype(df_date_col) and df_date_col.dt.tz is not None:
                     tz_csv = df_date_col.dt.tz
                     # Rendi start/end dt aware con lo stesso timezone dei dati
                     start_dt = tz_csv.localize(start_dt)
                     end_dt = tz_csv.localize(end_dt)
                elif pd.api.types.is_datetime64_any_dtype(df_date_col) and df_date_col.dt.tz is None:
                     # Se i dati sono naive, lascia start/end dt naive (warning opzionale?)
                     # st.warning("Dati CSV non hanno timezone. Comparazione date potrebbe essere imprecisa se i dati attraversano cambi ora legale.")
                     pass
                elif not pd.api.types.is_datetime64_any_dtype(df_date_col):
                     st.error(f"Errore interno: La colonna data CSV '{date_col_name_csv}' non è di tipo datetime."); st.stop()

                # Filtra DataFrame
                mask = (df_current_csv[date_col_name_csv] >= start_dt) & (df_current_csv[date_col_name_csv] <= end_dt)
                filtered_df = df_current_csv.loc[mask]

            except Exception as e_filter:
                st.error(f"Errore durante il filtraggio per data: {type(e_filter).__name__} - {e_filter}")
                st.stop()


            if filtered_df.empty:
                 st.warning(f"Nessun dato trovato nel periodo selezionato ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}).")
            else:
                 st.success(f"Trovati {len(filtered_df)} record nel periodo selezionato.")
                 tab1, tab2, tab3 = st.tabs(["📊 Andamento Temporale", "📈 Statistiche/Distribuzione", "🔗 Correlazione"])

                 # Feature numeriche disponibili nei dati filtrati
                 potential_features_analysis = filtered_df.select_dtypes(include=np.number).columns.tolist()
                 potential_features_analysis = [f for f in potential_features_analysis if f not in ['index', 'level_0']] # Rimuovi eventuali indici
                 # Mappa nomi brevi a nomi colonna
                 feature_labels_analysis = {get_station_label(f, short=True): f for f in potential_features_analysis}
                 if not feature_labels_analysis:
                     st.warning("Nessuna colonna numerica valida trovata nei dati filtrati per l'analisi."); st.stop()

                 with tab1:
                      st.subheader("Andamento Temporale Features")
                      # Default: primi 2 livelli, o prime 2 features
                      default_labels_ts = [lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f][:2]
                      if not default_labels_ts: default_labels_ts = list(feature_labels_analysis.keys())[:min(2, len(feature_labels_analysis))]
                      selected_labels_ts = st.multiselect("Seleziona feature da visualizzare:", options=list(feature_labels_analysis.keys()), default=default_labels_ts, key="analisi_ts_multi")
                      features_plot = [feature_labels_analysis[lbl] for lbl in selected_labels_ts]

                      if features_plot:
                           fig_ts = go.Figure()
                           for feature in features_plot:
                                legend_name = get_station_label(feature, short=True) # Usa label breve per legenda
                                fig_ts.add_trace(go.Scatter(
                                    x=filtered_df[date_col_name_csv],
                                    y=filtered_df[feature],
                                    mode='lines', name=legend_name,
                                    hovertemplate=f'<b>{legend_name}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>' # Format hover
                                ))
                           fig_ts.update_layout(
                                title='Andamento Temporale Selezionato',
                                xaxis_title='Data e Ora',
                                yaxis_title='Valore', height=500, hovermode="x unified",
                                margin=dict(t=50, b=40, l=40, r=10) # Margini ridotti
                            )
                           st.plotly_chart(fig_ts, use_container_width=True)
                           ts_filename = f"andamento_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                           st.markdown(get_plotly_download_link(fig_ts, ts_filename), unsafe_allow_html=True)
                      else: st.info("Seleziona almeno una feature per visualizzare l'andamento temporale.")

                 with tab2:
                      st.subheader("Statistiche Descrittive e Distribuzione")
                      # Default: primo livello o prima feature
                      default_stat_label = next((lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f), list(feature_labels_analysis.keys())[0])
                      try: # Trova indice del default
                          default_stat_idx = list(feature_labels_analysis.keys()).index(default_stat_label)
                      except ValueError:
                          default_stat_idx = 0 # Fallback se non trovato

                      selected_label_stat = st.selectbox("Seleziona feature per statistiche:", options=list(feature_labels_analysis.keys()), index=default_stat_idx, key="analisi_stat_select")
                      feature_stat = feature_labels_analysis.get(selected_label_stat)

                      if feature_stat:
                           st.markdown(f"**Statistiche per: {selected_label_stat}** (`{feature_stat}`)")
                           try:
                               st.dataframe(filtered_df[[feature_stat]].describe().round(3))
                           except Exception as e_describe:
                                st.error(f"Errore calcolo statistiche per {selected_label_stat}: {e_describe}")

                           st.markdown(f"**Distribuzione per: {selected_label_stat}**")
                           try:
                               fig_hist = go.Figure(data=[go.Histogram(x=filtered_df[feature_stat], name=selected_label_stat)])
                               fig_hist.update_layout(
                                   title=f'Distribuzione di {selected_label_stat}',
                                   xaxis_title='Valore', yaxis_title='Frequenza', height=400,
                                   margin=dict(t=50, b=40, l=40, r=10)
                               )
                               st.plotly_chart(fig_hist, use_container_width=True)
                               hist_filename = f"distrib_{selected_label_stat.replace(' ','_').replace('(','').replace(')','')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                               st.markdown(get_plotly_download_link(fig_hist, hist_filename), unsafe_allow_html=True)
                           except Exception as e_hist:
                                st.error(f"Errore generazione istogramma per {selected_label_stat}: {e_hist}")

                 with tab3:
                      st.subheader("Matrice di Correlazione e Scatter Plot")
                      # Default: tutte le feature disponibili
                      default_corr_labels = list(feature_labels_analysis.keys())
                      selected_labels_corr = st.multiselect("Seleziona feature per correlazione:", options=list(feature_labels_analysis.keys()), default=default_corr_labels, key="analisi_corr_multi")
                      features_corr = [feature_labels_analysis[lbl] for lbl in selected_labels_corr]

                      if len(features_corr) > 1:
                           try:
                               corr_matrix = filtered_df[features_corr].corr()
                               heatmap_labels = [get_station_label(f, short=True) for f in features_corr] # Usa nomi brevi per heatmap

                               fig_hm = go.Figure(data=go.Heatmap(
                                   z=corr_matrix.values, x=heatmap_labels, y=heatmap_labels,
                                   colorscale='RdBu', zmin=-1, zmax=1, # Scala colori rosso/blu
                                   colorbar=dict(title='Corr'),
                                   text=corr_matrix.round(2).values, texttemplate="%{text}", # Mostra valori sulla mappa
                                   hoverongaps=False
                               ))
                               fig_hm.update_layout(
                                   title='Matrice di Correlazione',
                                   height=max(400, len(heatmap_labels)*35), # Altezza dinamica
                                   xaxis_tickangle=-45, yaxis_autorange='reversed',
                                   margin=dict(t=50, b=40, l=60, r=10)
                                )
                               st.plotly_chart(fig_hm, use_container_width=True)
                               hm_filename = f"correlazione_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                               st.markdown(get_plotly_download_link(fig_hm, hm_filename), unsafe_allow_html=True)
                           except Exception as e_corr:
                                st.error(f"Errore calcolo/visualizzazione matrice correlazione: {e_corr}")

                           # --- Scatter Plot ---
                           if len(selected_labels_corr) <= 15: # Limite più alto per scatter
                                st.markdown("**Scatter Plot Correlazione (Coppia di Feature)**")
                                cs1, cs2 = st.columns(2)
                                # Usa label brevi per selezione
                                label_x = cs1.selectbox("Feature Asse X:", selected_labels_corr, index=0, key="scatter_x_select")
                                # Default Y: seconda feature se disponibile, altrimenti la prima
                                default_y_index = 1 if len(selected_labels_corr) > 1 else 0
                                label_y = cs2.selectbox("Feature Asse Y:", selected_labels_corr, index=default_y_index, key="scatter_y_select")

                                # Recupera nomi colonna originali
                                fx = feature_labels_analysis.get(label_x)
                                fy = feature_labels_analysis.get(label_y)

                                if fx and fy and fx != fy:
                                    try:
                                        fig_sc = go.Figure(data=[go.Scatter(
                                            x=filtered_df[fx], y=filtered_df[fy], mode='markers',
                                            marker=dict(size=5, opacity=0.6),
                                            name=f'{label_x} vs {label_y}',
                                            text=filtered_df[date_col_name_csv].dt.strftime('%d/%m %H:%M'), # Mostra data/ora su hover
                                            hovertemplate=f'<b>{label_x}</b>: %{{x:.2f}}<br><b>{label_y}</b>: %{{y:.2f}}<br>%{{text}}<extra></extra>' # Format hover
                                        )])
                                        fig_sc.update_layout(
                                            title=f'Correlazione: {label_x} vs {label_y}',
                                            xaxis_title=label_x, yaxis_title=label_y, height=500,
                                            margin=dict(t=50, b=40, l=40, r=10)
                                        )
                                        st.plotly_chart(fig_sc, use_container_width=True)
                                        sc_filename = f"scatter_{label_x.replace(' ','_')}_vs_{label_y.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                                        st.markdown(get_plotly_download_link(fig_sc, sc_filename), unsafe_allow_html=True)
                                    except Exception as e_scatter:
                                         st.error(f"Errore generazione scatter plot per {label_x} vs {label_y}: {e_scatter}")
                                elif fx and fy and fx == fy:
                                     st.info("Seleziona due feature diverse per lo scatter plot.")
                           else:
                                st.info("Troppe feature selezionate (>15) per visualizzare lo scatter plot interattivo.")
                      elif len(features_corr) == 1:
                           st.info("Seleziona almeno due feature per calcolare la correlazione.")
                      else:
                           st.info("Seleziona almeno due feature per visualizzare la matrice di correlazione.")

                 st.divider()
                 st.subheader('Download Dati Filtrati (CSV)')
                 download_filename_filtered = f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                 st.markdown(get_table_download_link(filtered_df, download_filename_filtered), unsafe_allow_html=True)


# --- PAGINA ALLENAMENTO MODELLO ---
elif page == 'Allenamento Modello':
    st.header('🎓 Allenamento Nuovo Modello LSTM')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili. Carica un file CSV valido nella sidebar per allenare un nuovo modello.")
    else:
        st.success(f"Dati CSV disponibili per l'allenamento: {len(df_current_csv)} righe.")
        st.subheader('Configurazione Addestramento')

        # --- Nome Modello ---
        default_save_name = f"modello_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
        save_name_input = st.text_input("Nome base per salvare il modello (file .pth, .json, .joblib)", default_save_name, key="train_save_filename")
        # Pulisci nome file
        save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_')
        if not save_name: save_name = "modello_default" # Fallback
        if save_name != save_name_input: st.caption(f"Nome file corretto in: `{save_name}`")

        # --- Selezione Feature Input ---
        st.markdown("**1. Seleziona Feature Input per il Modello:**")
        all_global_features = st.session_state.get('feature_columns', [])
        # Feature effettivamente presenti nel CSV caricato
        features_present_in_csv = [f for f in all_global_features if f in df_current_csv.columns]
        missing_from_csv = [f for f in all_global_features if f not in df_current_csv.columns]
        if missing_from_csv:
            st.caption(f"Nota: Le seguenti feature globali non sono nel CSV caricato e non possono essere usate: {', '.join(missing_from_csv)}")

        if not features_present_in_csv:
             st.error("Errore: Nessuna delle feature definite globalmente è presente nel file CSV caricato. Impossibile procedere.")
             st.stop()

        selected_features_train = []
        # Default: seleziona tutte le feature presenti nel CSV
        default_features = features_present_in_csv

        with st.expander(f"Seleziona Feature Input (Disponibili: {len(features_present_in_csv)})", expanded=False):
            num_cols_feat = 4 # Colonne per i checkbox
            cols_feat_train = st.columns(num_cols_feat)
            col_idx_feat = 0
            for i, feat in enumerate(features_present_in_csv):
                 with cols_feat_train[col_idx_feat % num_cols_feat]:
                     label_feat_train = get_station_label(feat, short=True)
                     is_checked = st.checkbox(label_feat_train, value=(feat in default_features), key=f"train_feat_check_{feat}", help=feat)
                     if is_checked:
                          selected_features_train.append(feat)
                 col_idx_feat += 1

        if not selected_features_train: st.warning("Seleziona almeno una feature di input.")
        else: st.caption(f"{len(selected_features_train)} feature di input selezionate.")

        # --- Selezione Target Output ---
        st.markdown("**2. Seleziona Target Output (solo sensori di Livello tra gli input selezionati):**")
        selected_targets_train = []
        # Opzioni: solo colonne 'Livello' tra le feature SELEZIONATE sopra
        hydro_level_features_options = [f for f in selected_features_train if 'Livello' in f]

        if not hydro_level_features_options:
             st.warning("Nessuna colonna 'Livello' tra le feature di input selezionate. Impossibile selezionare target.")
        else:
            # Default: target del modello attivo (se presenti tra le opzioni) o primo livello disponibile
            default_targets_train = []
            active_config_targets = active_config.get("target_columns", []) if active_config else []
            valid_active_targets = [t for t in active_config_targets if t in hydro_level_features_options]
            if valid_active_targets:
                default_targets_train = valid_active_targets
            elif hydro_level_features_options: # Se non ci sono target validi dal modello attivo, prendi il primo
                default_targets_train = [hydro_level_features_options[0]]

            # Layout checkbox target
            num_cols_target = min(len(hydro_level_features_options), 5) # Max 5 colonne
            cols_target_train = st.columns(num_cols_target)
            for i, feat_target in enumerate(hydro_level_features_options):
                with cols_target_train[i % num_cols_target]:
                     lbl_target = get_station_label(feat_target, short=True)
                     is_target_checked = st.checkbox(lbl_target, value=(feat_target in default_targets_train), key=f"train_target_check_{feat_target}", help=f"Seleziona come output: {feat_target}")
                     if is_target_checked:
                         selected_targets_train.append(feat_target)

        if not selected_targets_train: st.warning("Seleziona almeno un target di output (Livello).")
        else: st.caption(f"{len(selected_targets_train)} target di output selezionati.")

        # ===========================================================
        # --- Sezione Parametri Principali (MODIFICATA per usare HP Opt results) ---
        # ===========================================================
        st.markdown("**3. Imposta Parametri Modello e Addestramento Principale:**")
        with st.expander("Parametri Modello e Training", expanded=True):
             c1t, c2t, c3t = st.columns(3)

             # Prendi i parametri migliori trovati (se ci sono), altrimenti usa i default del modello attivo o globali
             best_hp = st.session_state.get('best_hp_params', {})

             # Valori di fallback (dal modello attivo o globali)
             fallback_iw = active_config.get("input_window", 24) if active_config else 24
             fallback_ow = active_config.get("output_window", 12) if active_config else 12
             fallback_hs = active_config.get("hidden_size", 128) if active_config else 128
             fallback_nl = active_config.get("num_layers", 2) if active_config else 2
             fallback_dr = active_config.get("dropout", 0.2) if active_config else 0.2
             fallback_bs = active_config.get("batch_size", 32) if active_config else 32
             fallback_vs = active_config.get("val_split_percent", 20) if active_config else 20
             fallback_lr = active_config.get("learning_rate", 0.001) if active_config else 0.001
             fallback_ep = 50 # Default epoche training finale (non prese da modello attivo o HP opt)

             # --- Widget Parametri ---
             # Finestre NON sono ottimizzate
             iw_t = c1t.number_input("Finestra Input (n. rilevazioni)", 6, 168, fallback_iw, 6, key="t_param_in_win")
             ow_t = c1t.number_input("Finestra Output (n. rilevazioni)", 1, 72, fallback_ow, 1, key="t_param_out_win")
             # Split Validazione NON è ottimizzato, MA è importante per HP opt
             vs_t = c1t.slider("% Dati per Validazione", 0, 50, fallback_vs, 1, key="t_param_val_split", help="0% = nessun set di validazione. Richiesto > 0 per Ottimizzazione HP.")

             # Usa i valori da best_hp se presenti, altrimenti usa i fallback
             hs_t = c2t.number_input("Neuroni Nascosti (Hidden Size)", 16, 1024,
                                     value=int(best_hp.get("hidden_size", fallback_hs)), # Usa best_hp o fallback, cast a int
                                     step=16, key="t_param_hidden")
             nl_t = c2t.number_input("Numero Livelli LSTM (Layers)", 1, 8,
                                     value=int(best_hp.get("num_layers", fallback_nl)), # Usa best_hp o fallback, cast a int
                                     step=1, key="t_param_layers")
             dr_t = c2t.slider("Dropout", 0.0, 0.7,
                               value=float(best_hp.get("dropout", fallback_dr)), # Usa best_hp o fallback, cast a float
                               step=0.05, key="t_param_dropout")

             lr_t = c3t.number_input("Learning Rate", 1e-5, 1e-2,
                                     value=float(best_hp.get("learning_rate", fallback_lr)), # Usa best_hp o fallback, cast a float
                                     format="%.5f", step=1e-4, key="t_param_lr")
             bs_options = [8, 16, 32, 64, 128, 256] # Opzioni batch size
             bs_val = best_hp.get("batch_size", fallback_bs) # Usa best_hp o fallback
             # Assicura che il valore sia tra le opzioni, altrimenti usa fallback
             if bs_val not in bs_options: bs_val = fallback_bs
             bs_t = c3t.select_slider("Batch Size", options=bs_options,
                                      value=int(bs_val), # Cast a int
                                      key="t_param_batch")
             # Epoche: usa il default, non quelle dei trial HP
             ep_t = c3t.number_input("Numero Epoche (Training Finale)", 5, 1000, # Aumentato max epoche finale
                                     value=fallback_ep,
                                     step=5, key="t_param_epochs")

        # ===========================================================
        # --- NUOVA SEZIONE: Ottimizzazione Iperparametri (Random Search) ---
        # ===========================================================
        st.markdown("**4. Ottimizzazione Iperparametri con Random Search (Opzionale)**")

        # Usa il valore in session_state per il checkbox
        optimize_hp = st.checkbox("Abilita Ottimizzazione Iperparametri", value=st.session_state.hp_optimize_enabled, key="enable_hp_opt_cb", help="Esegue una ricerca casuale per trovare buoni iperparametri iniziali (LR, Hidden Size, Layers, Dropout, Batch Size). Richiede un set di validazione > 0%.")
        st.session_state.hp_optimize_enabled = optimize_hp # Aggiorna lo stato

        if optimize_hp:
            # Controlla se la validazione è attiva (necessaria per l'ottimizzazione)
            if vs_t <= 0:
                st.error("L'ottimizzazione degli iperparametri richiede una percentuale di dati per la Validazione > 0%. Imposta '% Dati per Validazione' (sopra) ad un valore > 0 e riavvia l'ottimizzazione.")
            else:
                st.info("Verranno eseguiti allenamenti brevi con parametri casuali per trovare una buona configurazione iniziale.")
                n_trials_hp = st.number_input("Numero di Trial Random Search da Eseguire", min_value=5, max_value=200, value=st.session_state.get('hp_n_trials', 30), step=5, key="hp_n_trials_input")
                st.session_state.hp_n_trials = n_trials_hp # Salva in sessione
                epochs_per_trial = st.number_input("Numero Epoche per ciascun Trial HP", min_value=3, max_value=50, value=st.session_state.get('hp_epochs_per_trial', 15), step=1, key="hp_epochs_per_trial_input", help="Meno epoche per velocizzare la ricerca.")
                st.session_state.hp_epochs_per_trial = epochs_per_trial # Salva in sessione

                st.markdown("**Definisci i Range di Ricerca:**")
                c1_hp, c2_hp, c3_hp = st.columns(3)

                with c1_hp:
                    lr_range_default = st.session_state.get('hp_lr_range', (1e-4, 5e-3))
                    lr_range_hp = st.slider("Range Learning Rate", 1e-5, 1e-2, lr_range_default, format="%.5f", key="hp_lr_range_slider")
                    st.session_state.hp_lr_range = lr_range_hp

                    bs_choices_hp = [16, 32, 64, 128] # Batch size più comuni
                    bs_range_default = st.session_state.get('hp_bs_range', (32, 128))
                    # Assicurati che i default siano nella lista choices e validi
                    bs_range_default_corrected = (max(bs_choices_hp[0], bs_range_default[0]), min(bs_choices_hp[-1], bs_range_default[1]))
                    # Verifica che min <= max
                    if bs_range_default_corrected[0] > bs_range_default_corrected[1]: bs_range_default_corrected = (bs_choices_hp[0], bs_choices_hp[-1])

                    bs_range_hp = st.select_slider("Range Batch Size", options=bs_choices_hp, value=bs_range_default_corrected, key="hp_bs_range_slider")
                    st.session_state.hp_bs_range = bs_range_hp
                    # Calcola le opzioni possibili basate sul range selezionato
                    bs_indices = [i for i, b in enumerate(bs_choices_hp) if b >= bs_range_hp[0] and b <= bs_range_hp[1]]
                    possible_batch_sizes = [bs_choices_hp[i] for i in bs_indices] if bs_indices else [bs_range_hp[0]] # Fallback al minimo se range non valido

                with c2_hp:
                    hs_range_default = st.session_state.get('hp_hs_range', (64, 256))
                    hs_range_hp = st.slider("Range Hidden Size (multipli 16)", 32, 512, hs_range_default, step=16, key="hp_hs_range_slider")
                    st.session_state.hp_hs_range = hs_range_hp

                    nl_range_default = st.session_state.get('hp_nl_range', (1, 4))
                    nl_range_hp = st.slider("Range Numero Layers", 1, 6, nl_range_default, step=1, key="hp_nl_range_slider")
                    st.session_state.hp_nl_range = nl_range_hp

                with c3_hp:
                    dr_range_default = st.session_state.get('hp_dr_range', (0.1, 0.4))
                    dr_range_hp = st.slider("Range Dropout", 0.0, 0.6, dr_range_default, step=0.05, key="hp_dr_range_slider")
                    st.session_state.hp_dr_range = dr_range_hp

                # --- Pulsante per AVVIARE SOLO l'ottimizzazione ---
                # Disabilita se vs_t è 0 (perché l'ottimizzazione fallirebbe)
                start_optimization = st.button("🚀 Avvia Ottimizzazione HP", key="hp_opt_start_button", disabled=(vs_t <= 0))

                if start_optimization:
                    # Validazione input prima di procedere
                    if not selected_features_train: st.error("Seleziona almeno una feature di input (Sezione 1)."); st.stop()
                    if not selected_targets_train: st.error("Seleziona almeno un target di output (Sezione 2)."); st.stop()
                    if vs_t <= 0: st.error("Imposta '% Dati per Validazione' > 0 (Sezione 3)."); st.stop() # Doppio controllo

                    data_prepared_ok = False
                    X_tr_hp, y_tr_hp, X_v_hp, y_v_hp = None, None, None, None
                    input_size_hp, output_size_hp = None, None

                    # --- Preparazione Dati (una sola volta per tutti i trial) ---
                    st.info(f"Preparazione dati per ottimizzazione HP (Finestre: In={iw_t}, Out={ow_t}, Val={vs_t}%)...")
                    with st.spinner('Preparazione dati...'):
                        cols_needed_hp = list(set(selected_features_train + selected_targets_train))
                        missing_in_df_final_hp = [c for c in cols_needed_hp if c not in df_current_csv.columns]

                        if missing_in_df_final_hp:
                             st.error(f"Errore Critico HP: Le colonne selezionate {', '.join(missing_in_df_final_hp)} non sono nel CSV. Impossibile procedere.")
                             st.stop()

                        # Usa una copia del DataFrame
                        temp_X_tr, temp_y_tr, temp_X_v, temp_y_v, sc_f_hp, sc_t_hp = prepare_training_data(
                              df_current_csv.copy(),
                              selected_features_train,
                              selected_targets_train,
                              iw_t, # Usa finestre correnti dai widget
                              ow_t,
                              vs_t # Usa split validazione corrente
                          )

                        # Verifica che i dati siano stati creati correttamente
                        if temp_X_tr is not None and temp_y_tr is not None and temp_X_v is not None and temp_y_v is not None and sc_f_hp is not None and sc_t_hp is not None:
                            # Verifica aggiuntiva se X_v è vuoto nonostante vs_t > 0 (già gestito in prepare_training_data con warning)
                            if vs_t > 0 and temp_X_v.shape[0] == 0:
                                st.error(f"Errore Preparazione Dati HP: % Validazione è {vs_t}% ma il set di validazione risulta vuoto (0 sequenze). Dataset troppo piccolo o split non valido?")
                                st.stop()
                            else:
                                X_tr_hp, y_tr_hp, X_v_hp, y_v_hp = temp_X_tr, temp_y_tr, temp_X_v, temp_y_v
                                input_size_hp = X_tr_hp.shape[2]
                                output_size_hp = y_tr_hp.shape[2]
                                data_prepared_ok = True
                                st.success(f"Dati pronti per HP: {len(X_tr_hp)} train, {len(X_v_hp)} validation.")
                        else:
                           st.error("Preparazione dati per HP fallita. Controlla i log e i parametri (finestre vs lunghezza dati).")
                           st.stop()

                    # --- Procedi solo se i dati sono pronti ---
                    if data_prepared_ok:
                        st.info(f"Avvio Ottimizzazione Random Search con {n_trials_hp} trials ({epochs_per_trial} epoche/trial)...")
                        progress_bar_opt = st.progress(0)
                        results_hp_list = []
                        trial_status_placeholder = st.empty() # Placeholder per status trial

                        # --- Loop Random Search ---
                        start_opt_time = time.time()
                        for i in range(n_trials_hp):
                            # 1. Campiona parametri
                            current_lr = random.uniform(lr_range_hp[0], lr_range_hp[1])
                            # Arrotonda hidden size a multipli di 16
                            current_hidden = random.randint(hs_range_hp[0] // 16, hs_range_hp[1] // 16) * 16
                            # Assicura che sia almeno 16
                            current_hidden = max(16, current_hidden)
                            current_layers = random.randint(nl_range_hp[0], nl_range_hp[1])
                            current_dropout = random.uniform(dr_range_hp[0], dr_range_hp[1])
                            current_batch_size = random.choice(possible_batch_sizes)

                            current_params = {
                                "learning_rate": current_lr, "hidden_size": current_hidden,
                                "num_layers": current_layers, "dropout": current_dropout,
                                "batch_size": current_batch_size,
                            }

                            trial_status_placeholder.info(f"⏳ Trial {i+1}/{n_trials_hp}: Params = LR:{current_lr:.5f}, Hidden:{current_hidden}, Layers:{current_layers}, Dropout:{current_dropout:.3f}, Batch:{current_batch_size}")

                            try:
                                # 2. Chiama train_model in modalità ottimizzazione
                                best_val_loss_trial = train_model(
                                             X_tr_hp, y_tr_hp, X_v_hp, y_v_hp,
                                             input_size=input_size_hp, output_size=output_size_hp, output_window=ow_t,
                                             hidden_size=current_params["hidden_size"], num_layers=current_params["num_layers"],
                                             epochs=epochs_per_trial, # Usa epochs per trial qui
                                             batch_size=current_params["batch_size"],
                                             learning_rate=current_params["learning_rate"], dropout=current_params["dropout"],
                                             is_optimization_trial=True, # Flag importante
                                             opt_trial_num=i+1, opt_total_trials=n_trials_hp
                                           )

                                # Registra risultato se valido
                                if np.isfinite(best_val_loss_trial):
                                    results_hp_list.append((current_params, float(best_val_loss_trial)))
                                    trial_status_placeholder.write(f"✅ Trial {i+1}/{n_trials_hp}: Completato. Best Val Loss = {best_val_loss_trial:.6f}")
                                else:
                                    trial_status_placeholder.warning(f"⚠️ Trial {i+1}/{n_trials_hp}: Val Loss non valida (inf/NaN). Trial scartato.")

                            except Exception as e_opt_trial:
                                 st.warning(f"Trial HP {i+1} fallito con errore: {type(e_opt_trial).__name__} - {e_opt_trial}")
                                 # Non aggiungere a results_hp_list
                                 trial_status_placeholder.error(f"❌ Trial {i+1}/{n_trials_hp}: Errore durante l'esecuzione.")
                                 # Stampa traceback per debug solo se necessario
                                 # print(f"--- Traceback Trial HP {i+1} ---")
                                 # traceback.print_exc()
                                 # print("-------------------------------")

                            # Aggiorna progress bar
                            progress_bar_opt.progress((i + 1) / n_trials_hp)
                            time.sleep(0.05) # Pausa UI minima

                        # --- Fine Loop ---
                        total_opt_time = time.time() - start_opt_time
                        progress_bar_opt.empty() # Rimuovi barra progresso
                        trial_status_placeholder.empty() # Rimuovi status ultimo trial

                        # 3. Trova e mostra i migliori risultati
                        st.info(f"Ottimizzazione HP completata in {total_opt_time:.1f} secondi.")
                        if results_hp_list:
                            # Filtra nuovamente per sicurezza (anche se train_model dovrebbe restituire inf)
                            valid_results = [r for r in results_hp_list if np.isfinite(r[1])]
                            if valid_results:
                                valid_results.sort(key=lambda x: x[1]) # Ordina per loss crescente
                                best_params_found_hp, best_metric_found_hp = valid_results[0]

                                st.success(f"🏁 Miglior risultato trovato (Validation Loss): {best_metric_found_hp:.6f}")
                                st.write("Migliori Parametri Trovati:")
                                # Formatta parametri per migliore leggibilità
                                formatted_best_params = {
                                    "learning_rate": f"{best_params_found_hp['learning_rate']:.5f}",
                                    "hidden_size": best_params_found_hp['hidden_size'],
                                    "num_layers": best_params_found_hp['num_layers'],
                                    "dropout": f"{best_params_found_hp['dropout']:.3f}",
                                    "batch_size": best_params_found_hp['batch_size']
                                }
                                st.json(formatted_best_params)

                                # Salva i migliori parametri in session state (valori numerici originali)
                                st.session_state['best_hp_params'] = best_params_found_hp

                                st.info("💡 I campi dei parametri principali (Sezione 3) sono stati aggiornati con i valori trovati. Puoi ora avviare l'allenamento finale o modificarli ulteriormente.")
                                # Bottone per forzare rerun e vedere i widget aggiornati
                                # Usare on_click=st.rerun è più pulito
                                st.button("Applica e Ricarica Parametri", key="force_reload_params_button", on_click=st.rerun, help="Ricarica la pagina per applicare i parametri trovati ai widget.")
                            else:
                                st.error("Ottimizzazione completata, ma nessun trial ha prodotto una metrica valida (tutti inf/NaN?).")
                                if 'best_hp_params' in st.session_state: del st.session_state['best_hp_params'] # Rimuovi vecchi parametri se presenti
                        else:
                            st.error("Ottimizzazione fallita o nessun trial completato con successo.")
                            if 'best_hp_params' in st.session_state: del st.session_state['best_hp_params'] # Rimuovi vecchi parametri
        # --- Fine del blocco if optimize_hp: ---

        # ===========================================================
        # --- Fine Sezione Ottimizzazione HP ---
        # ===========================================================

        st.divider() # Separatore prima del training finale

        # --- Avvio Addestramento Principale ---
        st.markdown("**5. Avvia Addestramento Principale:**") # *** CORRETTO NUMERO SEZIONE ***

        # Validazione finale prima di abilitare il bottone
        valid_name = bool(save_name)
        valid_features = bool(selected_features_train)
        valid_targets = bool(selected_targets_train)
        # Verifica che le finestre siano valide
        valid_windows = (iw_t > 0 and ow_t > 0 and (iw_t + ow_t) <= len(df_current_csv)) if data_ready_csv else False
        # Verifica che vs_t sia valido (può essere 0 per training finale)
        valid_vs = (vs_t >= 0 and vs_t <= 50)

        ready_to_train = valid_name and valid_features and valid_targets and valid_windows and valid_vs

        # Messaggi di errore specifici per disabilitare il bottone
        disabled_reason = ""
        if not valid_features: disabled_reason += "❌ Seleziona feature input.\n"
        if not valid_targets: disabled_reason += "❌ Seleziona target output.\n"
        if not valid_name: disabled_reason += "❌ Inserisci nome modello.\n"
        if not valid_windows and data_ready_csv: disabled_reason += f"❌ Finestre ({iw_t}+{ow_t}) > Dati ({len(df_current_csv)}).\n"
        elif not data_ready_csv: disabled_reason += "❌ Dati CSV non pronti.\n"
        if not valid_vs: disabled_reason += "❌ % Validazione non valido.\n"


        train_button = st.button("▶️ Addestra Nuovo Modello (Finale)", type="primary", disabled=(not ready_to_train), key="train_run_button", help=disabled_reason if not ready_to_train else "Avvia l'addestramento finale con i parametri correnti.")

        if train_button and ready_to_train:
             st.info(f"Avvio addestramento finale per il modello '{save_name}'...")
             # Usa i valori attuali dei widget (iw_t, ow_t, vs_t, hs_t, nl_t, ep_t, bs_t, lr_t, dr_t)
             with st.spinner('Preparazione dati per training finale...'):
                  cols_needed = list(set(selected_features_train + selected_targets_train)) # Usa set per evitare duplicati
                  missing_in_df_final = [c for c in cols_needed if c not in df_current_csv.columns]
                  if missing_in_df_final:
                       st.error(f"Errore Critico: Le colonne selezionate {', '.join(missing_in_df_final)} non sono state trovate nel DataFrame CSV finale. Impossibile procedere.")
                       st.stop()

                  # Chiamata a prepare_training_data con i parametri finali dai widget
                  X_tr, y_tr, X_v, y_v, sc_f_tr, sc_t_tr = prepare_training_data(
                      df_current_csv.copy(),
                      selected_features_train, # Usa le feature selezionate
                      selected_targets_train, # Usa i target selezionati
                      iw_t, # Usa valore widget finestra input
                      ow_t, # Usa valore widget finestra output
                      vs_t  # Usa valore widget split validazione
                  )
                  # Verifica risultato preparazione dati
                  if X_tr is None or y_tr is None or sc_f_tr is None or sc_t_tr is None:
                       st.error("Preparazione dati per training finale fallita. Controlla i log e i parametri (es. finestre vs lunghezza dati).")
                       st.stop()
                  # Controllo se validation è vuoto ma dovrebbe esserci (già gestito in prepare_training_data)
                  val_set_size = 0
                  if X_v is not None: val_set_size = len(X_v)
                  # Non bloccare qui se val set è vuoto, train_model lo gestisce
                  st.success(f"Dati pronti per training finale: {len(X_tr)} train, {val_set_size} validation.")

             st.subheader("⏳ Addestramento Finale in corso...")
             input_size_train = X_tr.shape[2] # Dimensione da dati preparati
             output_size_train = y_tr.shape[2] # Dimensione da dati preparati
             trained_model = None
             train_start_time = time.time()
             try:
                 # Chiama train_model in modalità NON ottimizzazione (default)
                 # Usa i parametri dai widget (hs_t, nl_t, ep_t, bs_t, lr_t, dr_t)
                 trained_model, train_losses, val_losses = train_model(
                     X_tr, y_tr, X_v, y_v,
                     input_size_train, output_size_train, ow_t, # Passa ow_t
                     hs_t, nl_t, ep_t, bs_t, lr_t, dr_t,
                     is_optimization_trial=False # Esplicito per chiarezza
                 )
                 train_end_time = time.time()
                 if trained_model: # Controlla se il modello è stato restituito
                    st.info(f"Tempo di addestramento finale: {train_end_time - train_start_time:.2f} secondi.")
                 else: # Se train_model restituisce None
                    st.error("Addestramento finale non ha restituito un modello valido.")

             except Exception as e_train_run:
                 st.error(f"Errore durante l'addestramento finale: {type(e_train_run).__name__} - {e_train_run}")
                 st.error(traceback.format_exc())
                 trained_model = None # Assicura sia None in caso di eccezione

             # --- Salvataggio Modello ---
             if trained_model:
                 st.success("Addestramento finale completato con successo!")
                 st.subheader("💾 Salvataggio Risultati Modello")
                 os.makedirs(MODELS_DIR, exist_ok=True)
                 base_path = os.path.join(MODELS_DIR, save_name)
                 m_path = f"{base_path}.pth"
                 c_path = f"{base_path}.json"
                 sf_path = f"{base_path}_features.joblib"
                 st_path = f"{base_path}_targets.joblib"

                 # Calcola final val loss (migliore trovata durante il training finale)
                 final_val_loss = None
                 if val_losses and vs_t > 0:
                      valid_val_losses = [v for v in val_losses if v is not None and np.isfinite(v)]
                      if valid_val_losses: final_val_loss = min(valid_val_losses)

                 # Prepara config da salvare, includendo info HP Opt SE è stata eseguita
                 hp_opt_info = {}
                 if st.session_state.get('hp_optimize_enabled', False) and 'best_hp_params' in st.session_state and st.session_state.best_hp_params is not None:
                      hp_opt_info = {
                         "hp_optimization_enabled": True,
                         "hp_optimization_best_params_found": st.session_state.best_hp_params,
                         "hp_optimization_num_trials": st.session_state.get('hp_n_trials'),
                         "hp_optimization_epochs_per_trial": st.session_state.get('hp_epochs_per_trial'),
                      }
                 else:
                      hp_opt_info = {"hp_optimization_enabled": False}


                 config_save = {
                     # Parametri usati per il training finale
                     "input_window": iw_t, "output_window": ow_t, "hidden_size": hs_t,
                     "num_layers": nl_t, "dropout": dr_t, "batch_size": bs_t,
                     "learning_rate": lr_t, "epochs_run": ep_t, "val_split_percent": vs_t,
                     # Info su features/target
                     "feature_columns": selected_features_train,
                     "target_columns": selected_targets_train,
                     # Metadati
                     "training_date": datetime.now(italy_tz).isoformat(),
                     "final_val_loss": final_val_loss if final_val_loss is not None else 'N/A',
                     "display_name": save_name, # Nome per UI
                     "source_data_info": data_source_info, # Info su file CSV usato
                     # Info Ottimizzazione HP
                     **hp_opt_info
                 }

                 try:
                     # Salva stato modello
                     torch.save(trained_model.state_dict(), m_path)
                     # Salva config JSON
                     with open(c_path, 'w', encoding='utf-8') as f:
                         json.dump(config_save, f, indent=4, ensure_ascii=False)
                     # Salva scaler
                     joblib.dump(sc_f_tr, sf_path)
                     joblib.dump(sc_t_tr, st_path)

                     st.success(f"Modello '{save_name}' e file associati salvati in '{MODELS_DIR}/'")
                     st.subheader("⬇️ Download File Modello Addestrato")
                     col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                     with col_dl1: st.markdown(get_download_link_for_file(m_path, "Modello (.pth)"), unsafe_allow_html=True)
                     with col_dl2: st.markdown(get_download_link_for_file(c_path, "Config (.json)"), unsafe_allow_html=True)
                     with col_dl3: st.markdown(get_download_link_for_file(sf_path, "Scaler Feat (.joblib)"), unsafe_allow_html=True)
                     with col_dl4: st.markdown(get_download_link_for_file(st_path, "Scaler Targ (.joblib)"), unsafe_allow_html=True)

                     # Bottone per ricaricare e vedere nuovo modello
                     st.info("Dopo il download, potresti voler ricaricare l'app per vedere il nuovo modello nella lista.")
                     if st.button("Pulisci Cache Modelli e Ricarica App", key="train_reload_app"):
                         find_available_models.clear() # Pulisce cache modelli
                         # Resetta stato attivo per forzare ricaricamento
                         st.session_state.pop('active_model_name', None)
                         st.session_state.pop('active_config', None)
                         st.session_state.pop('active_model', None)
                         st.session_state.pop('active_device', None)
                         st.session_state.pop('active_scaler_features', None)
                         st.session_state.pop('active_scaler_targets', None)
                         # Resetta anche HP params per il prossimo training
                         st.session_state.pop('best_hp_params', None)
                         st.session_state.hp_optimize_enabled = False # Disabilita checkbox HP opt
                         st.success("Cache modelli pulita. Ricaricamento...")
                         time.sleep(1)
                         st.rerun()

                 except Exception as e_save_files:
                     st.error(f"Errore durante il salvataggio dei file del modello: {type(e_save_files).__name__} - {e_save_files}")
                     st.error(traceback.format_exc())

             elif not train_button:
                 pass # L'utente non ha cliccato il bottone (o non era ready)
             else: # train_button era True e ready_to_train era True, ma trained_model è None
                 st.error("Addestramento finale fallito o interrotto prima di produrre un modello. Impossibile salvare.")


# --- Footer ---
st.sidebar.divider()
st.sidebar.info('App Idrologica Dashboard & Predict © 2024 Alberto Bussaglia')
