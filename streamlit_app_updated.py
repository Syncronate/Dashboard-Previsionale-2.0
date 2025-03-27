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
DASHBOARD_HISTORY_ROWS = 48 # Numero di righe storiche da recuperare (48 righe = 24 ore @ 30 min/riga)


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
         st.warning("Set di Validazione vuoto dopo lo split.")
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
                 with open(cfg_p, 'r', encoding='utf-8') as f: # Specifica encoding
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
    # ... (codice invariato) ...
    try:
        with open(_config_path, 'r', encoding='utf-8') as f: config = json.load(f) # Specifica encoding
        required = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        if not all(k in config for k in required): st.error(f"Config '{_config_path}' incompleto."); return None
        return config
    except Exception as e: st.error(f"Errore caricamento config '{_config_path}': {e}"); return None


@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    # ... (codice invariato) ...
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
    # ... (codice invariato) ...
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

# --- predict rimane invariata, dipende da modello e scaler cachati ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    # ... (codice invariato) ...
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict: Modello, scaler o config mancanti."); return None
    input_w = config["input_window"]; output_w = config["output_window"]
    target_cols = config["target_columns"]; f_cols_cfg = config.get("feature_columns", [])
    # Aggiunto controllo più robusto su n_features_in_ se disponibile
    expected_features = len(f_cols_cfg) if f_cols_cfg else getattr(scaler_features, 'n_features_in_', None)

    if input_data.shape[0] != input_w:
        st.error(f"Predict: Input righe {input_data.shape[0]} != Finestra {input_w}."); return None
    if expected_features is not None and input_data.shape[1] != expected_features:
        st.error(f"Predict: Input colonne {input_data.shape[1]} != Features attese {expected_features}."); return None
    elif expected_features is None and not f_cols_cfg:
        st.warning("Predict: Impossibile verificare numero colonne input rispetto a config/scaler.")

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(inp_tens)
        out_np = output.cpu().numpy().reshape(output_w, len(target_cols))

        # Aggiunto controllo più robusto su n_features_in_ per scaler target
        expected_targets = getattr(scaler_targets, 'n_features_in_', None)
        if expected_targets is None:
            st.error("Predict: Scaler targets non sembra essere fittato (manca n_features_in_)."); return None
        if expected_targets != len(target_cols):
            st.error(f"Predict: Output targets modello ({len(target_cols)}) != Scaler Targets ({expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except ValueError as ve:
        # Errore comune se le colonne non corrispondono allo scaler
        st.error(f"Errore durante scaling/predict: {ve}")
        st.error(f"Shape input_data: {input_data.shape}, Feature attese: {expected_features}")
        st.error(traceback.format_exc()); return None
    except Exception as e:
        st.error(f"Errore imprevisto durante predict: {e}");
        st.error(traceback.format_exc()); return None
# --- NUOVO: Mappatura Nomi Target Modello (JSON) -> Nomi Soglie (GSheet) ---
MODEL_TARGET_TO_GSHEET_MAP = {
    # "Nome nel JSON Target": "Nome nel Dizionario Soglie (GSheet)"
    "Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)": "Serra dei Conti - Livello Misa (mt)",
    "Livello Idrometrico Sensore 1112 [m] (Bettolelle)":      "Misa - Livello Misa (mt)",
    "Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)": "Nevola - Livello Nevola (mt)",
    "Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)": "Pianello di Ostra - Livello Misa (m)",
    # --- Aggiungi questa riga SE Ponte Garibaldi è un target nel tuo modello ---
    "Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)": "Ponte Garibaldi - Livello Misa 2 (mt)"
    # Aggiungi altre eventuali mappature se ci sono altri target di livello
}
# --- MODIFICATA: plot_predictions usa mappatura per cercare soglie ---
def plot_predictions(predictions, config, thresholds, start_time=None):
    """
    Genera grafici Plotly per le previsioni del modello, includendo le soglie.
    Usa una mappatura per trovare le soglie corrette se i nomi target del modello
    non corrispondono alle chiavi del dizionario soglie.
    """
    figs = []
    # --- Controlli input robusti (come prima) ---
    if config is None or not isinstance(config, dict):
        st.error("Errore grafico: Configurazione modello mancante o non valida.")
        return figs
    if predictions is None or not isinstance(predictions, np.ndarray) or predictions.ndim != 2:
        st.error("Errore grafico: Predizioni mancanti o formato non valido.")
        return figs
    if thresholds is None or not isinstance(thresholds, dict):
        st.warning("Attenzione grafico: Dizionario soglie mancante o non valido. Grafici senza soglie.")
        thresholds = {}

    output_w = config.get("output_window")
    target_cols = config.get("target_columns") # Nomi dal JSON del modello

    # --- Controlli validità config e predizioni (come prima) ---
    if not isinstance(output_w, int) or output_w <= 0:
        st.error(f"Errore grafico: Finestra output ('output_window'={output_w}) non valida nel config.")
        return figs
    if not isinstance(target_cols, list) or not target_cols:
        st.error("Errore grafico: Colonne target ('target_columns') mancanti o non valide nel config.")
        return figs
    if predictions.shape != (output_w, len(target_cols)):
        st.error(f"Errore grafico: Shape predizioni {predictions.shape} non corrisponde a output ({output_w}, {len(target_cols)}).")
        return figs

    # --- Gestione asse X (come prima) ---
    if start_time and isinstance(start_time, datetime):
        if start_time.tzinfo is None: start_time = italy_tz.localize(start_time)
        else: start_time = start_time.tz_convert(italy_tz)
        steps = [start_time + timedelta(minutes=30*(h+1)) for h in range(output_w)]
        x_axis, x_title = steps, "Data e Ora Previste (intervalli 30 min)"
    else:
        if start_time: st.warning("Formato start_time non valido per asse X temporale. Uso indici.")
        steps_idx = np.arange(1, output_w + 1)
        x_axis, x_title = steps_idx, f"Passi Futuri ({output_w} x 30 min)"

    # --- Ciclo sui target del modello ---
    for i, sensor_json_name in enumerate(target_cols): # Nome dal JSON
        if i >= predictions.shape[1]: continue # Sicurezza

        fig = go.Figure()
        # Usa get_station_label con il nome JSON per il titolo principale
        station_name_graph = get_station_label(sensor_json_name, short=False)

        # --- Aggiungi traccia previsione (come prima) ---
        try:
            fig.add_trace(go.Scatter(
                x=x_axis, y=predictions[:, i], mode='lines+markers', name='Previsto',
                line=dict(color='blue'),
                hovertemplate=f'<b>{station_name_graph}</b><br>%{{x|%d/%m %H:%M}} (%{{customdata[0]}})<br>Prev: %{{y:.2f}}<extra></extra>',
                customdata=[(f"Step {h+1}",) for h in range(output_w)]
            ))
        except Exception as e_trace:
            st.error(f"Errore aggiunta trace previsione per {sensor_json_name}: {e_trace}")
            continue

        # --- NUOVO/MODIFICATO: Ricerca soglie tramite mappatura ---
        gsheet_key_for_threshold = MODEL_TARGET_TO_GSHEET_MAP.get(sensor_json_name) # Trova nome GSheet
        sensor_thresholds = None # Inizializza a None
        thresholds_lookup_successful = False # Flag per il titolo

        if gsheet_key_for_threshold:
            # Se abbiamo trovato un nome GSheet corrispondente nella mappa...
            # ...cerca le soglie usando QUEL nome GSheet nel dizionario thresholds
            sensor_thresholds = thresholds.get(gsheet_key_for_threshold)
            if sensor_thresholds is not None:
                thresholds_lookup_successful = True # Trovato il dict delle soglie!
            else:
                # Il nome mappato esiste, ma non c'è nel dizionario soglie attuale? (Meno probabile)
                st.warning(f"⚠️ Soglie non definite per '{station_name_graph}' (target JSON: `{sensor_json_name}`, chiave GSheet cercata: `{gsheet_key_for_threshold}`).\n"
                           f"   La mappatura esiste, ma la chiave GSheet non ha soglie nel dizionario attuale.", icon="📊")
        else:
            # Se il nome JSON non è stato trovato nella mappatura
            st.warning(f"⚠️ **Mappatura mancante per '{station_name_graph}' (target JSON: `{sensor_json_name}`).**\n"
                       f"   Aggiungere la corrispondenza in `MODEL_TARGET_TO_GSHEET_MAP` per visualizzare le soglie.", icon="🔗")

        # Assicura che sensor_thresholds sia un dizionario per sicurezza nelle chiamate .get()
        if sensor_thresholds is None:
            sensor_thresholds = {}

        # Recupera i valori (saranno None se non definiti o se la ricerca è fallita)
        threshold_alert = sensor_thresholds.get('alert')
        threshold_attention = sensor_thresholds.get('attention')

        # --- Aggiungi linee soglia (logica invariata, usa threshold_alert/attention) ---
        # Aggiungi linea soglia ALLERTA (Rossa)
        if threshold_alert is not None and isinstance(threshold_alert, (int, float)):
            try:
                fig.add_hline( y=threshold_alert, line_dash="dash", line_color="red",
                    annotation_text=f"Allerta ({threshold_alert:.1f})", annotation_position="bottom right",
                    annotation_font=dict(color="red", size=10) )
            except Exception as e_hline_a: st.warning(f"Errore aggiunta hline Allerta per {sensor_json_name}: {e_hline_a}")

        # Aggiungi linea soglia ATTENZIONE (Arancione)
        if threshold_attention is not None and isinstance(threshold_attention, (int, float)):
             if threshold_alert is not None and isinstance(threshold_alert, (int, float)) and threshold_attention > threshold_alert:
                  st.warning(f"Nota: Soglia Attenzione ({threshold_attention}) per '{station_name_graph}' supera Allerta ({threshold_alert}).", icon="⚠️")
             try:
                 fig.add_hline( y=threshold_attention, line_dash="dash", line_color="orange",
                    annotation_text=f"Attenzione ({threshold_attention:.1f})", annotation_position="top right",
                    annotation_font=dict(color="orange", size=10) )
             except Exception as e_hline_att: st.warning(f"Errore aggiunta hline Attenzione per {sensor_json_name}: {e_hline_att}")

        # --- Configura Layout grafico (titolo aggiornato) ---
        try:
            yaxis_unit_suffix = ""
            if '(mm)' in sensor_json_name: yaxis_unit_suffix = ' (mm)'
            elif '(m)' in sensor_json_name or '(mt)' in sensor_json_name: yaxis_unit_suffix = ' (m)'

            # Aggiorna titolo in base al successo della ricerca soglie
            title_text = f'Previsione: {station_name_graph}'
            if not thresholds_lookup_successful:
                title_text += ' (Soglie non trovate/mappate)'

            fig.update_layout(
                title=title_text, xaxis_title=x_title, yaxis_title=f'Valore Previsto{yaxis_unit_suffix}',
                height=400, hovermode="x unified", margin=dict(t=50, b=40, l=50, r=10)
            )
            fig.update_yaxes(rangemode='tozero')
        except Exception as e_layout:
            st.error(f"Errore configurazione layout grafico per {sensor_json_name}: {e_layout}")

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
                         df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer') # Aggiunto ambiguous='infer'
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
                    # --- MODIFICATO: Aggiunta gestione per '#DIV/0!' o errori simili di GSheet ---
                    df_col_str = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null', '#DIV/0!', '#N/A', '#VALUE!'], np.nan, regex=False)
                    df[col] = pd.to_numeric(df_col_str, errors='coerce')

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


# --- train_model invariata, usata solo per training ---
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2):
    # ... (codice invariato) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val) if X_val.size > 0 else None # Gestione val vuoto
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 è spesso più stabile in ambienti semplici
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset else None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) # verbose=False è già ok
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state_dict = None
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        # Plotta val loss solo se ci sono dati validi (non None)
        valid_v_loss = [v for v in v_loss if v is not None] if v_loss else []
        if valid_v_loss:
             fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0))
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training per {epochs} epoche su {device}...")
    start_training_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train(); train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch); loss = criterion(outputs, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
            # Aggiorna meno frequentemente per ridurre overhead Streamlit? No, l'utente vuole vedere il progresso.
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = None # Default se non c'è validation
        if val_loader:
            model.eval(); current_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch); loss = criterion(outputs, y_batch)
                    current_val_loss += loss.item()
            val_loss = current_val_loss / len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss) # Scheduler step basato su val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Copia lo state_dict sulla CPU per evitare problemi se si esaurisce la VRAM
                best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else:
            val_losses.append(None) # Aggiungi None se non c'è validation

        progress_percentage = (epoch + 1) / epochs
        progress_bar.progress(progress_percentage)
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        epoch_time = time.time() - epoch_start_time
        status_text.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) - Train Loss: {train_loss:.6f}, Val Loss: {val_loss_str} - LR: {current_lr:.6f}')
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)
        # time.sleep(0.01) # Rimosso, plotly update è sufficiente per dare respiro

    total_training_time = time.time() - start_training_time
    st.write(f"Training completato in {total_training_time:.1f} secondi.")

    # Carica il miglior modello trovato (se esiste e validation era attiva)
    if best_model_state_dict:
        try:
            # Riporta i pesi sul device corretto prima di caricarli
            best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
            model.load_state_dict(best_model_state_dict_on_device)
            st.success(f"Caricato modello migliore (Epoca con Val Loss: {best_val_loss:.6f})")
        except Exception as e_load_best:
            st.error(f"Errore caricamento best model state: {e_load_best}. Uso modello ultima epoca.")
    elif not val_loader:
        st.warning("Nessun set di validazione, usato modello dell'ultima epoca.")
    else: # C'era validation, ma non è stato trovato un best_model_state_dict (improbabile se val_loss era calcolato)
        st.warning("Nessun miglioramento rilevato o errore nel salvataggio stato migliore, usato modello dell'ultima epoca.")

    return model, train_losses, val_losses


# --- Funzioni Helper Download invariate ---
def get_table_download_link(df, filename="data.csv"):
    # ... (codice invariato) ...
    csv = df.to_csv(index=False, sep=';', decimal=',')
    b64 = base64.b64encode(csv.encode('utf-8')).decode()
    # Aggiunta media type per robustezza
    return f'<a href="data:text/csv;charset=utf-8;base64,{b64}" download="{filename}">Scarica CSV</a>'


def get_binary_file_download_link(file_object, filename, text):
    # ... (codice invariato) ...
    file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None: mime_type = 'application/octet-stream' # Fallback generico
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'


def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    # ... (codice invariato) ...
    # Considera aggiunta try-except attorno a write_html
    try:
        buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn') # Usa CDN per ridurre dimensione file
        buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'
    except Exception as e_html:
         st.warning(f"Errore generazione HTML per {filename_base}: {e_html}")
         href_html = ""

    href_png = ""
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
         pass # Silenzia altri errori PNG

    return f"{href_html} {href_png}".strip()


def get_download_link_for_file(filepath, link_text=None):
    # ... (codice invariato) ...
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
    # ... (codice invariato) ...
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- Funzione Estrazione Etichetta Stazione (VERIFICA INDENTAZIONE) ---
def get_station_label(col_name, short=False):
# Inizio funzione (livello 0 indentazione)
    if col_name in STATION_COORDS:
    # Primo Blocco IF (livello 1 indentazione)
        location_id = STATION_COORDS[col_name].get('location_id')
        if location_id:
        # Secondo Blocco IF (livello 2 indentazione)
            if short:
            # Terzo Blocco IF (livello 3 indentazione)
                sensor_type = STATION_COORDS[col_name].get('type', '')
                sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == location_id]
                if len(sensors_at_loc) > 1:
                # Quarto Blocco IF (livello 4 indentazione)
                    type_abbr = 'P' if sensor_type == 'Pioggia' else ('L' if sensor_type == 'Livello' else '')
                    label = f"{location_id} ({type_abbr})"
                    return label[:25] + ('...' if len(label) > 25 else '')
                else:
                # Blocco ELSE per Quarto IF (livello 4 indentazione)
                    return location_id[:25] + ('...' if len(location_id) > 25 else '')
            else:
            # Blocco ELSE per Terzo IF (livello 3 indentazione)
                return location_id
    # --- Fallback se col_name NON è in STATION_COORDS ---
    # <<< QUESTA PARTE DEVE ESSERE A LIVELLO 0 INDENTAZIONE, come la primissima "if" >>>
    parts = col_name.split(' - ') # Livello 0 indentazione
    if len(parts) > 1:
    # IF del fallback (livello 1 indentazione)
        location = parts[0].strip()
        measurement = parts[1].split(' (')[0].strip()
        if short:
        # IF dentro fallback (livello 2 indentazione)
             label = f"{location} - {measurement}"
             return label[:25] + ('...' if len(label) > 25 else '')
        else:
        # ELSE dentro fallback (livello 2 indentazione)
             return location
    else:
    # ELSE del fallback (livello 1 indentazione)
        label = col_name.split(' (')[0].strip()
        return label[:25] + ('...' if len(label) > 25 else '')
# Fine funzione (livello 0 indentazione)
# --- NUOVO: Mappatura Nomi Target Modello (JSON) -> Nomi Soglie (GSheet) ---
MODEL_TARGET_TO_GSHEET_MAP = {
    # "Nome nel JSON Target": "Nome nel Dizionario Soglie (GSheet)"
    "Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)": "Serra dei Conti - Livello Misa (mt)",
    "Livello Idrometrico Sensore 1112 [m] (Bettolelle)":      "Misa - Livello Misa (mt)",
    "Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)": "Nevola - Livello Nevola (mt)",
    "Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)": "Pianello di Ostra - Livello Misa (m)",
    # --- Aggiungi questa riga SE Ponte Garibaldi è un target nel tuo modello ---
    "Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)": "Ponte Garibaldi - Livello Misa 2 (mt)"
    # Aggiungi altre eventuali mappature se ci sono altri target di livello
}
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


# --- Inizializzazione Session State invariata ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
# ... (altre chiavi session state invariate) ...
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

# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('🌊 Dashboard e Modello Predittivo Idrologico')

# --- Sidebar ---
st.sidebar.header('Impostazioni')

# --- Caricamento Dati Storici (per Analisi/Training) ---
# --- COMMENTO: Se questa sezione e le pagine dipendenti sono inutilizzate, ---
# --- la loro rimozione è la maggiore ottimizzazione possibile. ---
st.sidebar.subheader('Dati Storici (per Analisi/Training)')
uploaded_data_file = st.sidebar.file_uploader('Carica CSV Dati Storici (Opzionale)', type=['csv'], key="data_uploader")

# --- Logica Caricamento DF (INVARIATA) ---
# Questa logica viene eseguita solo se un file viene caricato o se DEFAULT_DATA_PATH esiste
# Non impatta la dashboard se non si interagisce con essa.
df = None; df_load_error = None; data_source_info = ""
data_path_to_load = None; is_uploaded = False
# ... (logica caricamento CSV invariata) ...
if uploaded_data_file is not None:
    data_path_to_load = uploaded_data_file; is_uploaded = True
    data_source_info = f"File caricato: **{uploaded_data_file.name}**"
elif os.path.exists(DEFAULT_DATA_PATH):
    # Carica solo se non già presente in session state per evitare ricariche inutili
    if 'df' not in st.session_state or st.session_state.df is None:
        data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
    else:
        # Dati già caricati da sessione
        data_path_to_load = None
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (usando cache sessione)"
elif 'df' not in st.session_state or st.session_state.df is None:
     # Solo se non c'è file di default E non c'è niente in sessione
     df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato. Carica un CSV per Analisi/Training."


# Carica e processa solo se necessario
# Verifica se st.session_state.df non è già popolato da un run precedente
# e se c'è un file da caricare (nuovo upload o default non ancora in sessione)
if data_path_to_load and ('df' not in st.session_state or st.session_state.df is None or is_uploaded):
    try:
        # ... (resto della logica di caricamento e pulizia CSV invariata) ...
        read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
        df_loaded = False
        for enc in encodings_to_try:
            try:
                # Assicurati che il file object sia all'inizio se è stato caricato
                if hasattr(data_path_to_load, 'seek'): data_path_to_load.seek(0)
                df_temp = pd.read_csv(data_path_to_load, encoding=enc, **read_args)
                df_loaded = True; break
            except UnicodeDecodeError: continue
            except Exception as read_e: raise read_e # Rilancia altri errori di lettura
        if not df_loaded: raise ValueError(f"Impossibile leggere CSV con encoding {', '.join(encodings_to_try)}.")

        date_col_csv = st.session_state.date_col_name_csv
        if date_col_csv not in df_temp.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")

        # Tentativo conversione data più robusto
        try:
            # Prova il formato atteso prima
            df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
            try:
                 # Se fallisce, prova inferenza (con warning)
                 df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce')
                 if df_temp[date_col_csv].isnull().any():
                      st.sidebar.warning(f"Formato data CSV non standard in alcune righe di '{date_col_csv}'. Righe con data invalida saranno scartate.")
                 else:
                      st.sidebar.warning(f"Formato data CSV non standard ('{date_col_csv}'). Tentata inferenza automatica.")
            except Exception as e_date_csv_infer:
                 raise ValueError(f"Errore conversione data CSV '{date_col_csv}' anche con inferenza: {e_date_csv_infer}")

        # Rimuovi righe dove la data non è stata convertita correttamente
        df_temp = df_temp.dropna(subset=[date_col_csv])
        if df_temp.empty:
            raise ValueError("Nessuna riga valida dopo la conversione/pulizia della data CSV.")

        # Localizza le date CSV (assumendo siano naive e riferite all'Italia)
        try:
            if df_temp[date_col_csv].dt.tz is None:
                df_temp[date_col_csv] = df_temp[date_col_csv].dt.tz_localize(italy_tz, ambiguous='infer')
            else:
                df_temp[date_col_csv] = df_temp[date_col_csv].dt.tz_convert(italy_tz)
        except Exception as e_tz_csv:
            st.sidebar.error(f"Errore applicazione timezone ai dati CSV: {e_tz_csv}")
            # Decide se fermarsi o continuare con date naive (potrebbe dare problemi)
            # st.stop() # Opzione più sicura

        # Ordina per data
        df_temp = df_temp.sort_values(by=date_col_csv).reset_index(drop=True)

        # Usa le feature globali come riferimento, ma controlla esistenza nel df caricato
        current_f_cols_state = st.session_state.feature_columns
        features_actually_in_df = [col for col in current_f_cols_state if col in df_temp.columns]
        missing_features_in_df = [col for col in current_f_cols_state if col not in df_temp.columns]

        if missing_features_in_df:
             st.sidebar.warning(f"Attenzione: Le seguenti feature globali non sono nel CSV: {', '.join(missing_features_in_df)}. Saranno ignorate per Analisi/Training basato su questo CSV.")

        # Pulizia colonne numeriche (solo quelle presenti nel df e definite globalmente)
        for col in features_actually_in_df:
             # Verifica dtype prima di applicare metodi stringa
             if pd.api.types.is_object_dtype(df_temp[col]):
                  df_temp[col] = df_temp[col].astype(str).str.strip()
                  df_temp[col] = df_temp[col].replace(['N/A', '', '-', 'None', 'null', 'NaN', 'nan', '#DIV/0!', '#N/A', '#VALUE!'], np.nan, regex=False) # Aggiunto NaN/nan espliciti e errori sheet
                  # Sostituisci '.' come separatore migliaia (se presente) e poi ',' con '.' per decimali
                  # --- MODIFICATO: Gestione più attenta per non rimuovere il punto decimale vero ---
                  # Se contiene sia '.' che ',', è probabile che '.' sia migliaia
                  if df_temp[col].str.contains('.', regex=False).any() and df_temp[col].str.contains(',', regex=False).any():
                      df_temp[col] = df_temp[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                  # Se contiene solo ',', probabilmente è il decimale
                  elif df_temp[col].str.contains(',', regex=False).any():
                      df_temp[col] = df_temp[col].str.replace(',', '.', regex=False)
                  # Se contiene solo '.', si assume sia il decimale corretto

             # Converti in numerico, errori diventano NaN
             df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

        # Controllo NaN e fill (solo sulle colonne numeriche attese e presenti)
        cols_to_check_nan = features_actually_in_df # Già filtrato per esistenza
        n_nan_before_fill = df_temp[cols_to_check_nan].isnull().sum().sum()
        if n_nan_before_fill > 0:
              st.sidebar.caption(f"Trovati {n_nan_before_fill} NaN/non numerici nelle colonne feature/target del CSV. Eseguito ffill/bfill.")
              df_temp[cols_to_check_nan] = df_temp[cols_to_check_nan].fillna(method='ffill').fillna(method='bfill')
              n_nan_after_fill = df_temp[cols_to_check_nan].isnull().sum().sum()
              if n_nan_after_fill > 0:
                  st.sidebar.error(f"NaN residui ({n_nan_after_fill}) dopo fill. Controlla inizio/fine file CSV o colonne con tutti NaN.")
                  nan_cols_report = df_temp[cols_to_check_nan].isnull().sum()
                  st.sidebar.json(nan_cols_report[nan_cols_report > 0].to_dict())
                  # Considera fill con 0 come ultima risorsa se necessario
                  # df_temp[cols_to_check_nan] = df_temp[cols_to_check_nan].fillna(0)

        # Salva il DataFrame processato in session state
        st.session_state.df = df_temp
        st.sidebar.success(f"Dati CSV caricati e processati ({len(st.session_state.df)} righe valide).")
        df = st.session_state.df # Aggiorna variabile locale df

    except Exception as e:
        df = None; st.session_state.df = None # Assicura reset in caso di errore
        df_load_error = f'Errore caricamento/processamento dati CSV ({data_source_info}): {type(e).__name__} - {e}'
        st.sidebar.error(f"Errore CSV: {df_load_error}")
        st.sidebar.info("Le funzionalità 'Analisi Dati Storici' e 'Allenamento Modello' non saranno disponibili.")

# Recupera df da session state se non caricato in questo run
df = st.session_state.get('df', None)
if df is None and df_load_error:
     # Mostra errore solo se c'è stato un tentativo fallito di caricamento
     if 'data_uploader' in st.session_state and st.session_state.data_uploader is not None:
          pass # L'errore è già mostrato sopra
     elif os.path.exists(DEFAULT_DATA_PATH):
          st.sidebar.warning(f"Dati CSV default ('{DEFAULT_DATA_PATH}') non caricabili. {df_load_error}")
     else:
          st.sidebar.info("Nessun dato CSV caricato o trovato. Funzionalità Analisi/Training disabilitate.")


# --- Selezione Modello (usa find_available_models cachata) ---
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

# --- Logica Caricamento Modello (INVARIATA, usa funzioni cachate) ---
# ... (resto della logica caricamento modello invariata) ...
config_to_load = None; model_to_load = None; device_to_load = None
scaler_f_to_load = None; scaler_t_to_load = None; load_error_sidebar = False

# Resetta stato attivo prima di tentare il caricamento (necessario se si cambia selezione)
active_model_changed = (selected_model_display_name != st.session_state.get('active_model_name', MODEL_CHOICE_NONE))

if active_model_changed:
    st.session_state.active_model_name = None; st.session_state.active_config = None
    st.session_state.active_model = None; st.session_state.active_device = None
    st.session_state.active_scaler_features = None; st.session_state.active_scaler_targets = None

# Procede al caricamento solo se la selezione è valida e diversa da None
if selected_model_display_name != MODEL_CHOICE_NONE:
    if selected_model_display_name == MODEL_CHOICE_UPLOAD:
        # ... (logica upload manuale invariata) ...
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
                            "name": "uploaded_model",
                            "display_name": "Modello Caricato Manualmente"} # Aggiunto display name
                model_to_load, device_to_load = load_specific_model(m_f, temp_cfg)
                scaler_f_to_load, scaler_t_to_load = load_specific_scalers(sf_f, st_f)
                if model_to_load and scaler_f_to_load and scaler_t_to_load:
                    config_to_load = temp_cfg
                    # Se caricato con successo, aggiorna nome attivo
                    st.session_state.active_model_name = selected_model_display_name
                else: load_error_sidebar = True
            else: st.caption("Carica tutti i file e scegli i target.")

    else: # Modello pre-addestrato selezionato
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
                     st.warning(f"Config '{selected_model_display_name}' non specifica le feature_columns. Uso le {len(st.session_state.feature_columns)} feature globali definite.")
                     config_to_load["feature_columns"] = st.session_state.feature_columns # Usa quelle globali

                # Carica modello e scaler usando funzioni cachate
                model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
                scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])

                if not (model_to_load and scaler_f_to_load and scaler_t_to_load):
                    load_error_sidebar = True; config_to_load = None # Reset config se caricamento fallisce
                else:
                     # Se caricato con successo, aggiorna nome attivo
                     st.session_state.active_model_name = selected_model_display_name
            else:
                load_error_sidebar = True # Errore caricamento config
        else:
             # Questo caso non dovrebbe accadere se available_models_dict è aggiornato
             st.sidebar.error(f"Modello '{selected_model_display_name}' selezionato ma non trovato.")
             load_error_sidebar = True


# Salva nello stato sessione SOLO se tutto è caricato correttamente E la selezione non è 'None'
if config_to_load and model_to_load and device_to_load and scaler_f_to_load and scaler_t_to_load and selected_model_display_name != MODEL_CHOICE_NONE:
    st.session_state.active_config = config_to_load
    st.session_state.active_model = model_to_load
    st.session_state.active_device = device_to_load
    st.session_state.active_scaler_features = scaler_f_to_load
    st.session_state.active_scaler_targets = scaler_t_to_load
    # active_model_name è già stato aggiornato sopra nel blocco di caricamento


# Mostra feedback basato sullo stato sessione aggiornato
if st.session_state.active_model and st.session_state.active_config:
    cfg = st.session_state.active_config
    active_name = st.session_state.active_model_name # Dovrebbe essere aggiornato correttamente
    # Gestisci nome per modello caricato manualmente
    display_feedback_name = cfg.get("display_name", active_name if active_name != MODEL_CHOICE_UPLOAD else "Modello Caricato")
    # MODIFICATO: Aggiunto il tempo equivalente in ore nella sidebar
    st.sidebar.success(f"Modello ATTIVO: '{display_feedback_name}' (In:{cfg['input_window']} rilevazioni [~{cfg['input_window']/2:.1f}h], Out:{cfg['output_window']} rilevazioni [~{cfg['output_window']/2:.1f}h])")
elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]:
     st.sidebar.error(f"Caricamento modello '{selected_model_display_name}' fallito.")
elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not st.session_state.active_model:
     st.sidebar.info("Completa caricamento manuale modello.")
elif selected_model_display_name == MODEL_CHOICE_NONE:
     st.sidebar.info("Nessun modello selezionato per la simulazione.")


# --- MODIFICATA: Configurazione Soglie Dashboard (Alert + Attenzione) ---
st.sidebar.divider()
st.sidebar.subheader("Configurazione Soglie Dashboard")
with st.sidebar.expander("Modifica Soglie di Allerta (Rosso) e Attenzione (Giallo)", expanded=False):
    temp_thresholds_update = json.loads(json.dumps(st.session_state.dashboard_thresholds)) # Deep copy per modifica sicura

    monitorable_cols = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
    # cols_thresh = st.columns(2) # Due colonne per ridurre spazio verticale
    # col_idx_thresh = 0

    for col in monitorable_cols:
        label_short = get_station_label(col, short=True)
        st.markdown(f"**{label_short}** (`{col}`)") # Nome colonna come sottotitolo

        # Recupera soglie correnti o default (con gestione chiave mancante)
        current_sensor_thresholds = st.session_state.dashboard_thresholds.get(col, {})
        # --- NUOVO: Gestione più robusta dei default se colonna non esiste in DEFAULT_THRESHOLDS ---
        default_sensor_thresh = DEFAULT_THRESHOLDS.get(col, {'alert': 0.0, 'attention': 0.0})
        current_alert = current_sensor_thresholds.get('alert', default_sensor_thresh.get('alert', 0.0))
        current_attention = current_sensor_thresholds.get('attention', default_sensor_thresh.get('attention', 0.0))

        # Assicurati che siano float per number_input
        current_alert = float(current_alert) if current_alert is not None else 0.0
        current_attention = float(current_attention) if current_attention is not None else 0.0

        is_level = 'Livello' in col or '(m)' in col or '(mt)' in col
        step = 0.1 if is_level else 1.0
        fmt = "%.1f" if is_level else "%.0f"
        min_v = 0.0

        c1_th, c2_th = st.columns(2) # Colonne per le due soglie

        # Input Soglia Allerta (Rossa)
        with c1_th:
            new_alert = st.number_input(
                label=f"🔴 Allerta", value=current_alert, min_value=min_v, step=step, format=fmt,
                key=f"thresh_alert_{col}", help=f"Soglia di ALLERTA (Rossa) per: {col}"
            )
            if new_alert != current_alert:
                if col not in temp_thresholds_update: temp_thresholds_update[col] = {}
                temp_thresholds_update[col]['alert'] = new_alert

        # Input Soglia Attenzione (Gialla/Arancione)
        with c2_th:
            # Valore massimo per attenzione non può superare l'allerta
            max_attention_value = float(new_alert) if new_alert is not None else float('inf') # Usa inf se allerta non è definito
            # Gestisci caso in cui current_attention sia inizialmente maggiore di new_alert (reset)
            adjusted_current_attention = min(current_attention, max_attention_value)

            new_attention = st.number_input(
                label=f"🟠 Attenzione", value=adjusted_current_attention, min_value=min_v, max_value=max_attention_value, step=step, format=fmt,
                key=f"thresh_attention_{col}", help=f"Soglia di ATTENZIONE (Gialla/Arancione) per: {col}. Non può superare la soglia di Allerta."
            )

            # --- MODIFICATO: Logica validazione attenzione vs allerta ---
            # Assicura che attenzione <= allerta *dopo* l'input dell'utente
            if new_attention > new_alert:
                 # Non dovrebbe succedere con max_value, ma doppia sicurezza
                 st.warning(f"Soglia Attenzione ({new_attention:.1f}) per '{label_short}' supera Allerta ({new_alert:.1f}). È stata limitata.", icon="⚠️")
                 new_attention = new_alert # Limita al valore di allerta

            if new_attention != current_attention: # Confronta con valore originale prima dell'aggiustamento
                if col not in temp_thresholds_update: temp_thresholds_update[col] = {}
                temp_thresholds_update[col]['attention'] = new_attention

        # col_idx_thresh += 1
        st.divider() # Separatore tra sensori

    if st.button("Salva Soglie", key="save_thresholds", type="primary"):
        # Validazione finale: assicura attention <= alert per ogni sensore prima di salvare
        validation_ok = True
        for col, thresholds_dict in temp_thresholds_update.items():
            alert_val = thresholds_dict.get('alert')
            attention_val = thresholds_dict.get('attention')

            # Controlla che siano numerici prima del confronto
            is_alert_num = isinstance(alert_val, (int, float))
            is_attention_num = isinstance(attention_val, (int, float))

            if is_alert_num and is_attention_num and attention_val > alert_val:
                 st.error(f"Errore validazione per '{get_station_label(col, short=True)}': Soglia Attenzione ({attention_val}) non può essere maggiore della Soglia Allerta ({alert_val}).")
                 validation_ok = False
                 break # Ferma alla prima violazione

        if validation_ok:
            st.session_state.dashboard_thresholds = json.loads(json.dumps(temp_thresholds_update)) # Salva deep copy
            st.success("Soglie aggiornate!")
            time.sleep(0.5)
            # --- NUOVO: Forza rerun per applicare subito le soglie ---
            st.rerun()
        else:
             st.warning("Modifiche soglie NON salvate a causa di errori di validazione.")


# --- Menu Navigazione (logica disabilitazione invariata) ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
model_ready = st.session_state.active_model is not None and st.session_state.active_config is not None
data_ready_csv = df is not None # df ora riflette lo stato post-caricamento/errore

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
requires_model = ['Simulazione']
requires_csv = ['Analisi Dati Storici', 'Allenamento Modello'] # Anche simulazione può usare CSV

radio_captions = []
disabled_options = []
default_page_idx = 0 # Default alla Dashboard

for i, opt in enumerate(radio_options):
    caption = ""; disabled = False
    if opt == 'Dashboard': caption = "Monitoraggio GSheet (30 min)" # MODIFICATO: Aggiunto riferimento frequenza
    elif opt == 'Simulazione':
        if not model_ready: caption = "Richiede Modello attivo"; disabled = True
        else: caption = "Esegui previsioni" # Testo più conciso
    elif opt == 'Analisi Dati Storici':
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Esplora dati CSV"
    elif opt == 'Allenamento Modello':
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
        # Non serve rerun qui, il radio sotto userà il valore aggiornato
except ValueError:
    # La pagina salvata non esiste più nelle opzioni, resetta al default
    selected_page = radio_options[default_page_idx]
    st.session_state[current_page_key] = selected_page

# Trova l'indice della pagina selezionata (che ora è sicuramente valida)
current_idx_display = radio_options.index(selected_page)

# Crea il widget radio
chosen_page = st.sidebar.radio(
    'Scegli una funzionalità:', # Label più corta
    options=radio_options,
    captions=radio_captions,
    index=current_idx_display,
    key='page_selector_radio', # Key diversa da quella di session state
    # disabled non è un argomento diretto di st.radio, la logica sopra gestisce la selezione
)

# Aggiorna session state se la scelta cambia
if chosen_page != selected_page:
    st.session_state[current_page_key] = chosen_page
    # Forzare un rerun è utile qui per assicurare che il resto della pagina si aggiorni
    st.rerun()

# La pagina da visualizzare è ora in chosen_page
page = chosen_page

# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================

# Recupera variabili necessarie da session state (più pulito)
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scaler_features = st.session_state.get('active_scaler_features')
active_scaler_targets = st.session_state.get('active_scaler_targets')
df_current_csv = st.session_state.get('df', None) # Dati CSV (possono essere None)
# Feature columns specifiche del modello attivo, fallback a quelle globali
feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns) if active_config else st.session_state.feature_columns
date_col_name_csv = st.session_state.date_col_name_csv # Nome colonna data per CSV
# Recupera il dizionario delle soglie dalla session state
current_thresholds = st.session_state.get('dashboard_thresholds', {}) # Default a dict vuoto se manca

# --- PAGINA DASHBOARD (MODIFICATA per soglia attenzione) ---
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
            fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess
            fetch_secs_ago = int(fetch_time_ago.total_seconds())
            # MODIFICATO: Aggiorna testo status per riflettere le 48 righe / 24 ore / 30 min
            hours_represented = DASHBOARD_HISTORY_ROWS / 2 # Calcola ore rappresentate
            status_text = (
                f"Dati GSheet recuperati (ultime ~{hours_represented:.0f} ore, {DASHBOARD_HISTORY_ROWS} rilevazioni @ 30 min) alle: "
                f"{last_fetch_dt_sess.strftime('%d/%m/%Y %H:%M:%S')} "
                f"({fetch_secs_ago}s fa). "
                f"Refresh auto: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
            )
            st.caption(status_text)
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
        # --- Logica stato ultimo rilevamento (invariata) ---
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
             # Warning se i dati sono troppo vecchi (oltre 60 min?)
             # MODIFICATO: Aumentato warning a 90 min (3 intervalli mancati)
             if minutes_ago > 90:
                 st.warning(f"⚠️ Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa (atteso ogni 30 min).")
             # Aggiunto info se l'aggiornamento è molto recente
             elif minutes_ago <= 35: # Entro ~30 min
                 st.info(f"L'ultimo aggiornamento sembra recente (rilevato {minutes_ago} min fa).")

        else:
             st.warning("⚠️ Timestamp dell'ultimo rilevamento non disponibile o non valido nei dati GSheet più recenti.")

        st.divider()

        # --- Tabella Valori Attuali e Soglie (MODIFICATA per stato attenzione) ---
        st.subheader("Tabella Valori Attuali")
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
        table_rows = []
        current_alerts = [] # Ricalcola alert ad ogni rerun basato sui dati correnti
        current_attentions = [] # Lista separata per stato attenzione

        for col_name in cols_to_monitor:
            current_value = latest_row_data.get(col_name)
            # Recupera entrambe le soglie usando la variabile current_thresholds
            sensor_thresholds = current_thresholds.get(col_name, {})
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
                 else: value_display = f"{current_value:.2f}" # Fallback senza unità

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

            # Soglie per display (mostra entrambe se disponibili)
            alert_display = f"{threshold_alert:.1f}" if threshold_alert is not None else "-"
            attention_display = f"{threshold_attention:.1f}" if threshold_attention is not None else "-"
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

        df_display = pd.DataFrame(table_rows)

        # --- MODIFICATA: Funzione di stile per evidenziare righe (Alert e Attention) ---
        def highlight_thresholds(row):
            style = [''] * len(row)
            alert_thresh = row['Soglia Alert']
            attention_thresh = row['Soglia Attention']
            current_val = row['Valore Numerico']
            text_color = 'black' # Testo nero di default

            # Applica stile solo se i valori sono numerici validi
            if pd.notna(current_val):
                # Alert ha priorità
                if pd.notna(alert_thresh) and current_val >= alert_thresh:
                    # Allerta (Rosso più scuro per sfondo chiaro)
                    background = 'rgba(255, 0, 0, 0.3)' # Rosso sfondo leggermente più scuro
                    # text_color = 'white' # Testo bianco per contrasto? Opzionale
                    style = [f'background-color: {background}; color: {text_color}; font-weight: bold;'] * len(row)
                elif pd.notna(attention_thresh) and current_val >= attention_thresh:
                    # Attenzione (Arancione per sfondo chiaro)
                    background = 'rgba(255, 165, 0, 0.25)' # Arancione sfondo
                    style = [f'background-color: {background}; color: {text_color};'] * len(row)
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

        # Aggiorna alert globali in session state (solo allerta rossa per riepilogo)
        st.session_state.active_alerts = current_alerts
        # Salva anche stato attenzione per possibile uso futuro (non mostrato nel riepilogo attuale)
        st.session_state.active_attentions = current_attentions

        st.divider()

        # --- Grafico Comparativo Configurabile (usa dati ridotti DASHBOARD_HISTORY_ROWS) ---
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

            # MODIFICATO: Titolo aggiornato per riflettere le ore corrette
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
            # Link download (invariato)
            compare_filename_base = f"compare_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)
        else:
            st.info("Seleziona almeno un sensore per visualizzare il grafico comparativo.")

        st.divider()

        # --- Grafici Individuali (MODIFICATI per soglia attenzione) ---
        st.subheader("Grafici Individuali Storici")
        num_cols_individual = 3 # Quanti grafici per riga
        graph_cols = st.columns(num_cols_individual)
        col_idx_graph = 0

        x_axis_data_indiv = df_dashboard[GSHEET_DATE_COL] # Date reali per asse x

        for col_name in cols_to_monitor:
            with graph_cols[col_idx_graph % num_cols_individual]:
                # Recupera entrambe le soglie usando current_thresholds
                sensor_thresholds_indiv = current_thresholds.get(col_name, {})
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

                # --- NUOVO/MODIFICATO: Aggiungi entrambe le linee soglia ---
                # Aggiungi linea soglia ALLERTA (Rossa)
                if threshold_alert_indiv is not None and isinstance(threshold_alert_indiv, (int, float)):
                    fig_individual.add_hline(
                        y=threshold_alert_indiv, line_dash="dash", line_color="red",
                        annotation_text=f"Allerta ({threshold_alert_indiv:.1f})",
                        annotation_position="bottom right",
                        annotation_font=dict(color="red", size=10) # Rende annotation rossa
                    )
                # Aggiungi linea soglia ATTENZIONE (Gialla/Arancione)
                if threshold_attention_indiv is not None and isinstance(threshold_attention_indiv, (int, float)):
                     # Doppia verifica attenzione <= allerta (solo per warning, non impedisce plot)
                     if threshold_alert_indiv is not None and isinstance(threshold_alert_indiv, (int, float)) and threshold_attention_indiv > threshold_alert_indiv:
                          pass # Non fare nulla qui, le soglie sono visualizzate come sono state impostate

                     fig_individual.add_hline(
                        y=threshold_attention_indiv, line_dash="dash", line_color="orange",
                        annotation_text=f"Attenzione ({threshold_attention_indiv:.1f})",
                        annotation_position="top right", # Posizione diversa
                        annotation_font=dict(color="orange", size=10) # Rende annotation arancione
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

                # Link download grafico individuale (invariato)
                ind_filename_base = f"sensor_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)

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
                # Prova a ottenere tipo sensore da STATION_COORDS
                sensor_info = STATION_COORDS.get(col, {})
                sensor_type_alert = sensor_info.get('type', '') # Es. 'Pioggia' o 'Livello'
                type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""

                # Formattazione valore e soglia
                val_fmt = f"{val:.1f}" if 'Pioggia' in col else f"{val:.2f}"
                thr_fmt = f"{thr:.1f}" if isinstance(thr, (int, float)) else str(thr)
                unit = '(mm)' if 'Pioggia' in col else ('(m)' if 'Livello' in col else '') # Determina unità

                # Crea riga markdown
                alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit}** >= Soglia Allerta **{thr_fmt}{unit}**\n"
            st.markdown(alert_md)
        else:
            st.success("✅ Nessuna soglia di Allerta (Rossa) superata nell'ultimo rilevamento.")

        # Nota sullo stato di Attenzione (Giallo/Arancione)
        active_attentions_sess = st.session_state.get('active_attentions', [])
        if active_attentions_sess:
            st.info(f"ℹ️ Rilevati {len(active_attentions_sess)} sensori in stato di Attenzione (Arancione). Controlla la tabella e i grafici per i dettagli.")

    elif df_dashboard is not None and df_dashboard.empty: # Fetch OK ma nessun dato
        st.warning("Il recupero dati da Google Sheet ha restituito un set di dati vuoto.")
        if not error_msg: st.info("Controlla che ci siano dati recenti nel foglio Google e che le colonne richieste esistano.")

    else: # Se df_dashboard è None (fetch fallito gravemente)
        st.error("Impossibile visualizzare i dati della dashboard al momento.")
        if not error_msg: st.info("Controlla la connessione internet, le credenziali Google e l'ID del foglio.")

    # --- Meccanismo di refresh automatico (INVARIATO) ---
    # ... (codice refresh js invariato) ...
    component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
    js_code = f"""
    (function() {{
        const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}';
        if (window[intervalIdKey]) {{
            clearInterval(window[intervalIdKey]);
        }}
        window[intervalIdKey] = setInterval(function() {{
            if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') {{
                // console.log('Auto-refreshing dashboard...'); // Commentato per meno rumore in console
                window.streamlitHook.rerunScript(null);
            }} else {{
                // console.log('streamlitHook not ready for auto-refresh yet.'); // Commentato
            }}
        }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});
        return window[intervalIdKey];
    }})();
    """
    try:
        streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
    except Exception as e_js:
         st.warning(f"Impossibile impostare auto-refresh: {e_js}")


# --- PAGINA SIMULAZIONE (MODIFICATA per grafici, titolo, tabella) ---
elif page == 'Simulazione':
    st.header('🧪 Simulazione Idrologica')
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo (dalla sidebar) per usare la Simulazione.")
        st.info("Puoi scegliere un modello pre-addestrato dalla cartella 'models' o caricare i file manualmente.")
    else:
        # Recupera config del modello attivo
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns_model = active_config["target_columns"]
        # feature_columns_current_model definito globalmente all'inizio della sezione Pagine

        # MODIFICATO: Caption più chiare su 'rilevazioni' e tempo equivalente
        input_hours = input_window / 2.0
        output_hours = output_window / 2.0
        st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}**")
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
            # ... (codice invariato, usa feature_columns_current_model) ...
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
                 return st.number_input(label, value=default, step=step, format=fmt, key=f"man_{key_suffix}", help=help_text)

            # Popola colonne con i gruppi
            if feature_groups['Pioggia']:
                 with cols_manual[col_idx_man % 3]:
                      st.markdown("**Pioggia (mm/30min)**")
                      for feature, label_feat in feature_groups['Pioggia']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.0
                           temp_sim_values[feature] = create_num_input(feature, label_feat, round(max(0.0, default_val),1), 0.5, "%.1f", feature, feature)
                 col_idx_man += 1
            if feature_groups['Livello']:
                 with cols_manual[col_idx_man % 3]:
                      st.markdown("**Livelli (m)**")
                      for feature, label_feat in feature_groups['Livello']:
                           default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.5
                           temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,2), 0.05, "%.2f", feature, feature)
                 col_idx_man += 1
            # Metti Umidità e Altro insieme se presenti
            if feature_groups['Umidità'] or feature_groups['Altro']:
                with cols_manual[col_idx_man % 3]:
                     if feature_groups['Umidità']:
                          st.markdown("**Umidità (%)**")
                          for feature, label_feat in feature_groups['Umidità']:
                               default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 70.0
                               temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,1), 1.0, "%.1f", feature, feature)
                     if feature_groups['Altro']:
                          st.markdown("**Altre Feature**")
                          for feature, label_feat in feature_groups['Altro']:
                              default_val = df_current_csv[feature].median() if data_ready_csv and feature in df_current_csv and pd.notna(df_current_csv[feature].median()) else 0.0
                              temp_sim_values[feature] = create_num_input(feature, label_feat, round(default_val,2), 0.1, "%.2f", feature, feature)

            # Crea l'array numpy di input ripetendo i valori costanti
            try:
                ordered_values = [temp_sim_values[feature] for feature in feature_columns_current_model]
                sim_data_input = np.tile(ordered_values, (input_window, 1)).astype(float)
                sim_start_time_info = datetime.now(italy_tz) # Ora corrente per inizio previsione
            except KeyError as ke:
                st.error(f"Errore: Feature modello '{ke}' mancante nell'input manuale fornito. Verifica la configurazione.")
                sim_data_input = None
            except Exception as e:
                st.error(f"Errore creazione dati input costanti: {e}")
                sim_data_input = None

        # --- Simulazione: Google Sheet ---
        elif sim_method == 'Importa da Google Sheet (Ultime Rilevazioni)':
             # ... (codice invariato, usa feature_columns_current_model) ...
             st.subheader(f'Importa le ultime {input_window} rilevazioni (~{input_hours:.1f} ore) da Google Sheet')
             st.warning("⚠️ Funzionalità sperimentale: Assicurati che le colonne GSheet e la mappatura siano corrette!")
             # Usa ID GSheet di default (quello della dashboard) come suggerimento
             sheet_url_sim = st.text_input("URL Foglio Google da cui importare", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url_input")
             sheet_id_sim = extract_sheet_id(sheet_url_sim)

             # Mappatura di esempio (potrebbe necessitare aggiornamenti!)
             column_mapping_gsheet_to_model_sim = {
                # GSheet Col Name : Model Col Name
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
                # 'Colonna_GSheet_Umidita': 'Umidita\' Sensore 3452 (Montemurello)', # Esempio se esistesse
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
                # --- NUOVO: Aggiungi la colonna DATA per recuperarla ---
                GSHEET_DATE_COL : date_col_name_csv # Mappa la colonna data GSheet al nome colonna data atteso nel df storico
             }
             # Rimuovi dal mapping le colonne GSheet non richieste dal modello corrente (eccetto la data!)
             column_mapping_gsheet_to_model_sim_filtered = {
                 gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_model_sim.items()
                 if model_col in feature_columns_current_model or gs_col == GSHEET_DATE_COL # Mantieni data + feature richieste
             }

             with st.expander("Mostra/Modifica Mappatura GSheet -> Modello (Avanzato)"):
                 try:
                     edited_mapping_str = st.text_area("Mappatura Colonne (JSON: {'gsheet_col': 'model_col'})",
                                                     value=json.dumps(column_mapping_gsheet_to_model_sim_filtered, indent=2),
                                                     height=300, key="sim_gsheet_map_edit_area")
                     edited_mapping = json.loads(edited_mapping_str)
                     if isinstance(edited_mapping, dict):
                         column_mapping_gsheet_to_model_sim_filtered = edited_mapping
                         st.caption("Mappatura aggiornata.")
                     else: st.warning("Formato JSON mappatura non valido. Modifiche ignorate.")
                 except json.JSONDecodeError: st.warning("Errore parsing JSON mappatura. Modifiche ignorate.")
                 except Exception as e_map: st.error(f"Errore applicazione mappatura: {e_map}")

             # Verifica quali feature modello MANCANO nella mappatura (dopo averla filtrata)
             model_features_set = set(feature_columns_current_model)
             mapped_model_features_target = set(column_mapping_gsheet_to_model_sim_filtered.values())
             # Escludi colonna data dal check delle feature mancanti
             missing_model_features_in_map = list(model_features_set - mapped_model_features_target - {date_col_name_csv})

             # Chiedi valori costanti per le feature mancanti
             imputed_values_sim = {}
             needs_imputation_input = False
             if missing_model_features_in_map:
                  st.warning(f"Le seguenti feature del modello non sono state trovate nella mappatura GSheet. Verrà usato un valore costante:")
                  needs_imputation_input = True
                  cols_impute = st.columns(3)
                  col_idx_imp = 0
                  for missing_f in missing_model_features_in_map:
                       with cols_impute[col_idx_imp % 3]:
                            label_missing = get_station_label(missing_f, short=True)
                            default_val = 0.0; fmt = "%.2f"; step = 0.1
                            if data_ready_csv and missing_f in df_current_csv and pd.notna(df_current_csv[missing_f].median()): default_val = df_current_csv[missing_f].median()
                            if 'Umidita' in missing_f: fmt = "%.1f"; step = 1.0
                            elif 'Cumulata' in missing_f or 'Pioggia' in missing_f: fmt = "%.1f"; step = 0.5; default_val = max(0.0, default_val) # Include Pioggia
                            elif 'Livello' in missing_f: fmt = "%.2f"; step = 0.05

                            imputed_values_sim[missing_f] = st.number_input(f"Valore per '{label_missing}'", value=round(default_val, 2), step=step, format=fmt, key=f"sim_gsheet_impute_val_{missing_f}", help=f"Valore costante per {missing_f}")
                       col_idx_imp += 1

             # Definizione funzione fetch specifica per simulazione (cachata)
             @st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet...")
             def fetch_sim_gsheet_data(sheet_id_fetch, n_rows, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict, expected_date_col_name):
                 """ Recupera, mappa, pulisce e ordina dati da GSheet per simulazione. """
                 print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data (Rows: {n_rows})")
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
                     start_index_gs = max(1, len(all_data_gs) - n_rows)
                     data_rows_gs = all_data_gs[start_index_gs:]

                     df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)

                     # --- MODIFICATO: Prendi tutte le colonne GSheet necessarie dalla mappatura ---
                     required_gsheet_cols_from_mapping = list(col_mapping.keys())
                     missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
                     if missing_gsheet_cols_in_sheet:
                         return None, f"Errore: Colonne GSheet specificate nella mappatura ma mancanti nel foglio: {', '.join(missing_gsheet_cols_in_sheet)}", None

                     # --- MODIFICATO: Mappa e poi seleziona SOLO le colonne richieste (features + data) ---
                     df_mapped = df_gsheet_raw[required_gsheet_cols_from_mapping].rename(columns=col_mapping)

                     # Colonne richieste finali: feature modello + colonna data attesa
                     final_required_cols = required_model_cols_fetch + ([expected_date_col_name] if expected_date_col_name not in required_model_cols_fetch else [])

                     # Imputa valori mancanti (solo per feature, non per data)
                     for model_col, impute_val in impute_dict.items():
                          if model_col in final_required_cols and model_col not in df_mapped.columns:
                              df_mapped[model_col] = impute_val

                     final_missing_model_cols = [c for c in final_required_cols if c not in df_mapped.columns]
                     if final_missing_model_cols:
                         return None, f"Errore: Colonne modello/data mancanti dopo mappatura e imputazione: {', '.join(final_missing_model_cols)}", None

                     # Pulizia e conversione tipi (applicata a tutte le colonne selezionate)
                     last_valid_timestamp = None
                     for col in final_required_cols:
                         if col == expected_date_col_name:
                             try:
                                 df_mapped[col] = pd.to_datetime(df_mapped[col], format=date_format_gs, errors='coerce')
                                 if df_mapped[col].isnull().any():
                                     st.warning(f"Date non valide trovate in GSheet ('{col}').")
                                 if df_mapped[col].dt.tz is None: df_mapped[col] = df_mapped[col].dt.tz_localize(italy_tz, ambiguous='infer')
                                 else: df_mapped[col] = df_mapped[col].dt.tz_convert(italy_tz)
                             except Exception as e_date_clean:
                                 return None, f"Errore conversione/pulizia data GSheet '{col}': {e_date_clean}", None
                         else: # Feature numeriche
                              try:
                                  if pd.api.types.is_object_dtype(df_mapped[col]):
                                      col_str = df_mapped[col].astype(str)
                                      col_str = col_str.str.replace(',', '.', regex=False).str.strip()
                                      col_str = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan', '#DIV/0!', '#N/A', '#VALUE!'], np.nan, regex=False)
                                      df_mapped[col] = pd.to_numeric(col_str, errors='coerce')
                                  else: # Se è già numerico, converti solo per sicurezza (es. da int a float)
                                       df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
                              except Exception as e_clean_num:
                                  st.warning(f"Problema pulizia GSheet colonna '{col}': {e_clean_num}. Verrà trattata come NaN.")
                                  df_mapped[col] = np.nan

                     # Ordina per data (se la colonna data esiste ed è valida)
                     if expected_date_col_name in df_mapped.columns and pd.api.types.is_datetime64_any_dtype(df_mapped[expected_date_col_name]):
                          df_mapped = df_mapped.sort_values(by=expected_date_col_name, na_position='first')
                          if not df_mapped[expected_date_col_name].dropna().empty:
                             last_valid_timestamp = df_mapped[expected_date_col_name].dropna().iloc[-1]
                     else:
                          st.warning(f"Colonna data '{expected_date_col_name}' non trovata o non valida dopo il processo. Impossibile ordinare per data.")
                          last_valid_timestamp = datetime.now(italy_tz) # Fallback a ora corrente

                     # Seleziona e ordina le colonne come richiesto dal modello + data alla fine
                     try:
                         # Prendi prima le feature nell'ordine corretto
                         df_final = df_mapped[required_model_cols_fetch]
                         # Aggiungi la colonna data se esiste
                         if expected_date_col_name in df_mapped.columns:
                             df_final[expected_date_col_name] = df_mapped[expected_date_col_name]
                     except KeyError as e_key:
                         return None, f"Errore selezione/ordine colonne finali: '{e_key}' non trovata dopo mappatura/imputazione.", None

                     # Fill NaN DOPO selezione finale (solo sulle feature numeriche)
                     numeric_cols_to_fill = [col for col in required_model_cols_fetch if col in df_final.columns]
                     nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
                     if nan_count_before > 0:
                          st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet importati. Applico forward-fill e backward-fill.")
                          df_final[numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
                          if df_final[numeric_cols_to_fill].isnull().sum().sum() > 0:
                              st.error("Errore: NaN residui dopo fillna. Controlla dati GSheet (colonne vuote o inizio/fine file?).")
                              # Non restituire None, ma dataframe con NaN per ispezione? O restituire errore? Restituiamo errore.
                              return None, "Errore: NaN residui dopo fillna.", None

                     if len(df_final) != n_rows:
                         return None, f"Errore: Numero righe finali ({len(df_final)}) diverso da richiesto ({n_rows}).", None

                     # Restituisce DataFrame (feature + data) e ultimo timestamp valido
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

             if st.button("Importa e Prepara da Google Sheet", key="sim_run_gsheet_import", disabled=(not sheet_id_sim)):
                 if not sheet_id_sim: st.error("URL o ID del Foglio Google non valido.")
                 else:
                     # Passa expected_date_col_name a fetch_sim_gsheet_data
                     imported_df_gs, import_err, last_ts_gs = fetch_sim_gsheet_data(
                         sheet_id_sim,
                         input_window,
                         GSHEET_DATE_COL,
                         GSHEET_DATE_FORMAT,
                         column_mapping_gsheet_to_model_sim_filtered, # Usa la mappatura filtrata
                         feature_columns_current_model, # Lista delle sole feature modello
                         imputed_values_sim,
                         date_col_name_csv # Nome atteso colonna data nel df risultante
                     )

                     if import_err:
                         st.error(f"Importazione GSheet fallita: {import_err}")
                         st.session_state.imported_sim_data_gs_df = None
                         st.session_state.imported_sim_data_gs_ts = None
                         sim_data_input = None
                     elif imported_df_gs is not None:
                          st.success(f"Importate e processate {len(imported_df_gs)} rilevazioni da GSheet.")
                          # Verifica shape delle SOLE feature
                          if imported_df_gs[feature_columns_current_model].shape == (input_window, len(feature_columns_current_model)):
                              # Salva tutto il df (feature+data) per mostrarlo, ma usa solo feature per input
                              st.session_state.imported_sim_data_gs_df = imported_df_gs
                              st.session_state.imported_sim_data_gs_ts = last_ts_gs # Salva timestamp
                              sim_data_input = imported_df_gs[feature_columns_current_model].values # SOLO feature per il modello
                              sim_start_time_info = last_ts_gs if last_ts_gs else datetime.now(italy_tz)
                              with st.expander("Mostra Dati Importati da GSheet (pronti per modello)"):
                                   st.dataframe(imported_df_gs.round(3))
                          else:
                               st.error(f"Errore Shape dati feature GSheet post-processamento ({imported_df_gs[feature_columns_current_model].shape}) vs atteso ({input_window}, {len(feature_columns_current_model)}).")
                               st.session_state.imported_sim_data_gs_df = None
                               st.session_state.imported_sim_data_gs_ts = None
                               sim_data_input = None
                     else:
                          st.error("Importazione GSheet non riuscita per motivi sconosciuti.")
                          st.session_state.imported_sim_data_gs_df = None
                          st.session_state.imported_sim_data_gs_ts = None
                          sim_data_input = None

             elif sim_method == 'Importa da Google Sheet (Ultime Rilevazioni)' and 'imported_sim_data_gs_df' in st.session_state:
                 # Se i dati sono già in sessione, usali
                 imported_df_state = st.session_state.imported_sim_data_gs_df
                 last_ts_state = st.session_state.get('imported_sim_data_gs_ts')
                 if isinstance(imported_df_state, pd.DataFrame) and list(imported_df_state.columns) >= feature_columns_current_model:
                     # Verifica shape SOLE feature
                     if imported_df_state[feature_columns_current_model].shape == (input_window, len(feature_columns_current_model)):
                         sim_data_input = imported_df_state[feature_columns_current_model].values # SOLO feature
                         sim_start_time_info = last_ts_state if last_ts_state else datetime.now(italy_tz)
                         st.info("Utilizzo dati importati precedentemente da Google Sheet (da cache sessione).")
                         with st.expander("Mostra Dati Importati (da cache sessione)"):
                             st.dataframe(imported_df_state.round(3))
                     else:
                          st.warning("Dati GSheet in cache sessione non hanno la shape corretta. Riprova importazione.")
                          sim_data_input = None
                 else:
                     st.warning("Dati GSheet in cache sessione non validi. Riprova importazione.")
                     sim_data_input = None

        # --- Simulazione: Dettagliato per Intervallo (Tabella) ---
        elif sim_method == 'Dettagliato per Intervallo (Tabella)':
            # ... (codice invariato, usa feature_columns_current_model) ...
            st.subheader(f'Inserisci dati dettagliati per le {input_window} rilevazioni di input (~{input_hours:.1f} ore)')
            st.caption("Modifica la tabella sottostante con i valori desiderati per ogni intervallo di 30 minuti.")

            # --- MODIFICATO: Chiave sessione più robusta che include HASH delle feature columns ---
            feature_cols_tuple = tuple(sorted(feature_columns_current_model))
            feature_cols_hash = hash(feature_cols_tuple)
            session_key_hourly = f"sim_hourly_data_{input_window}_{feature_cols_hash}"

            needs_reinit = True
            if session_key_hourly in st.session_state:
                df_state = st.session_state[session_key_hourly]
                # Verifica più robusta: tipo, shape e nomi colonne nell'ordine atteso
                if isinstance(df_state, pd.DataFrame) and df_state.shape == (input_window, len(feature_columns_current_model)) and list(df_state.columns) == feature_columns_current_model:
                     needs_reinit = False

            if needs_reinit:
                 st.caption("Inizializzazione tabella dati...")
                 init_vals = {}
                 for col in feature_columns_current_model:
                      med_val = 0.0
                      if data_ready_csv and col in df_current_csv and pd.notna(df_current_csv[col].median()): med_val = df_current_csv[col].median()
                      elif col in DEFAULT_THRESHOLDS: # Usa soglia alert default come riferimento (se colonna esiste lì)
                           med_val = DEFAULT_THRESHOLDS.get(col, {}).get('alert', 0.0) * 0.2 # Prendi alert * 0.2
                      # Assicura float
                      med_val = float(med_val) if med_val is not None else 0.0
                      if 'Cumulata' in col or 'Pioggia' in col: med_val = max(0.0, med_val)
                      init_vals[col] = med_val

                 init_df = pd.DataFrame(np.repeat([list(init_vals.values())], input_window, axis=0), columns=list(init_vals.keys()))
                 # Assicura ordine corretto e fillna
                 st.session_state[session_key_hourly] = init_df[feature_columns_current_model].fillna(0.0).astype(float)

            # Lavora su una copia per l'editor
            df_for_editor = st.session_state[session_key_hourly].copy()
            # Assicura tipo float prima di passare all'editor
            try:
                df_for_editor = df_for_editor[feature_columns_current_model].astype(float)
            except Exception as e_cast_editor:
                 st.error(f"Errore preparazione tabella per modifica: {e_cast_editor}. Reset tabella.")
                 if session_key_hourly in st.session_state: del st.session_state[session_key_hourly]
                 st.rerun()

            # Configurazione colonne editor (come prima)
            column_config_editor = {}
            for col in feature_columns_current_model:
                 label_edit = get_station_label(col, short=True)
                 fmt = "%.3f"; step = 0.01; min_v=None; max_v=None
                 if 'Cumulata' in col or 'Pioggia' in col: fmt = "%.1f"; step = 0.5; min_v=0.0; label_edit += " (mm/30m)"
                 elif 'Umidita' in col: fmt = "%.1f"; step = 1.0; min_v=0.0; max_v=100.0
                 elif 'Livello' in col: fmt = "%.3f"; step = 0.01; label_edit += " (m)"; min_v=0.0 # Aggiunto min_v=0 per livelli

                 column_config_editor[col] = st.column_config.NumberColumn(label=label_edit, help=col, format=fmt, step=step, min_value=min_v, max_value=max_v, required=True)

            # Mostra l'editor dati
            edited_df = st.data_editor(
                df_for_editor,
                height=(input_window + 1) * 35 + 3,
                use_container_width=True,
                column_config=column_config_editor,
                key=f"data_editor_{session_key_hourly}",
                num_rows="fixed"
            )

            # Validazione dati post-modifica
            validation_passed = False
            if not isinstance(edited_df, pd.DataFrame):
                st.error("Errore interno: L'editor non ha restituito un DataFrame.")
            elif edited_df.shape[0] != input_window:
                st.error(f"Errore: La tabella deve avere esattamente {input_window} righe (intervalli).")
            elif list(edited_df.columns) != feature_columns_current_model:
                 st.error("Errore: Colonne della tabella modificate in modo imprevisto.")
            elif edited_df.isnull().sum().sum() > 0:
                 nan_cols = edited_df.isnull().sum()
                 st.warning(f"Attenzione: Valori mancanti rilevati nella tabella (colonne: {', '.join(nan_cols[nan_cols>0].index)}). Compilare tutte le celle.")
            else:
                 try:
                      # Assicura che siano float e nell'ordine giusto
                      sim_data_input_edit = edited_df[feature_columns_current_model].astype(float).values
                      if sim_data_input_edit.shape == (input_window, len(feature_columns_current_model)):
                          sim_data_input = sim_data_input_edit
                          sim_start_time_info = datetime.now(italy_tz) # Usa ora corrente per inizio
                          validation_passed = True
                          # Salva in sessione solo se modificato e valido
                          if not st.session_state[session_key_hourly].equals(edited_df):
                              st.session_state[session_key_hourly] = edited_df.copy() # Salva copia modificata
                              st.caption("Modifiche tabella salvate in sessione.")
                      else: st.error("Errore shape dati tabella dopo conversione.")
                 except Exception as e_edit_final:
                      st.error(f"Errore finale conversione dati tabella: {e_edit_final}")


        # --- Simulazione: Ultime Rilevazioni da CSV ---
        elif sim_method == 'Usa Ultime Rilevazioni da CSV Caricato':
             # ... (codice invariato, usa feature_columns_current_model) ...
             st.subheader(f"Usa le ultime {input_window} rilevazioni (~{input_hours:.1f} ore) dai dati CSV caricati")
             st.warning("⚠️ Assicurati che l'intervallo temporale dei dati CSV sia coerente con quello su cui il modello è stato addestrato!")
             if not data_ready_csv:
                 st.error("Dati CSV non caricati. Carica un file CSV nella sidebar.")
             elif len(df_current_csv) < input_window:
                 st.error(f"Dati CSV ({len(df_current_csv)} righe) insufficienti per la finestra di input richiesta ({input_window} rilevazioni).")
             else:
                  try:
                       # Seleziona le ultime righe e le colonne richieste NELL'ORDINE del modello
                       latest_csv_data_df = df_current_csv.iloc[-input_window:][feature_columns_current_model]

                       if latest_csv_data_df.isnull().sum().sum() > 0:
                            nan_report_csv = latest_csv_data_df.isnull().sum()
                            st.error(f"Trovati valori mancanti (NaN) nelle ultime {input_window} rilevazioni delle colonne richieste nel CSV. Impossibile usare per simulazione. Controlla: {', '.join(nan_report_csv[nan_report_csv>0].index)}")
                            with st.expander("Mostra righe CSV con NaN (ultime rilevazioni)"):
                                 st.dataframe(latest_csv_data_df[latest_csv_data_df.isnull().any(axis=1)])
                            sim_data_input = None
                       else:
                            # Converti in numpy float
                            latest_csv_data_np = latest_csv_data_df.astype(float).values
                            if latest_csv_data_np.shape == (input_window, len(feature_columns_current_model)):
                                sim_data_input = latest_csv_data_np
                                # Recupera timestamp dell'ultimo dato usato
                                try:
                                    last_ts_csv_used = df_current_csv.iloc[-1][date_col_name_csv]
                                    if pd.notna(last_ts_csv_used):
                                         # Assicurati sia timezone-aware (dovrebbe esserlo dal caricamento)
                                         if not isinstance(last_ts_csv_used, datetime): # Riconverti se necessario
                                             last_ts_csv_used = pd.to_datetime(last_ts_csv_used)
                                         if last_ts_csv_used.tzinfo is None: sim_start_time_info = italy_tz.localize(last_ts_csv_used)
                                         else: sim_start_time_info = last_ts_csv_used.tz_convert(italy_tz)
                                         st.caption(f"Simulazione basata su dati CSV fino a: {sim_start_time_info.strftime('%d/%m/%Y %H:%M %Z')}")
                                    else:
                                         sim_start_time_info = datetime.now(italy_tz); st.caption("Ora inizio previsione: Ora corrente (timestamp CSV non valido).")
                                except Exception as e_ts_csv:
                                     sim_start_time_info = datetime.now(italy_tz); st.caption(f"Ora inizio previsione: Ora corrente (errore lettura timestamp CSV: {e_ts_csv}).")

                                with st.expander("Mostra dati CSV usati per l'input"):
                                     # Mostra il df con le colonne nell'ordine corretto
                                     st.dataframe(latest_csv_data_df.round(3))
                            else:
                                st.error(f"Errore shape dati CSV estratti ({latest_csv_data_np.shape}) vs atteso ({input_window}, {len(feature_columns_current_model)}).")
                                sim_data_input = None
                  except KeyError as ke:
                       st.error(f"Errore: Colonna modello '{ke}' non trovata nel DataFrame CSV caricato. Verifica il file CSV o le feature del modello.")
                       sim_data_input = None
                  except Exception as e_csv_sim_extract:
                       st.error(f"Errore imprevisto durante estrazione dati CSV per simulazione: {e_csv_sim_extract}")
                       sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE (MODIFICATA per tabella e grafici) ---
        st.divider()
        # Verifica se i dati di input sono pronti
        input_ready = sim_data_input is not None and isinstance(sim_data_input, np.ndarray) and sim_data_input.shape == (input_window, len(feature_columns_current_model)) and not np.isnan(sim_data_input).any()

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
                       # MODIFICATO: Subheader
                       st.subheader(f'📊 Risultato Simulazione: Previsione per le prossime {output_window} rilevazioni (~{output_hours:.1f} ore)')

                       start_pred_time = sim_start_time_info if sim_start_time_info else datetime.now(italy_tz)
                       st.caption(f"Previsione calcolata a partire da: {start_pred_time.strftime('%d/%m/%Y %H:%M %Z')}")

                       pred_times_sim = [start_pred_time + timedelta(minutes=30*(i+1)) for i in range(output_window)]
                       # 1. Crea DataFrame iniziale dai risultati numerici
                       results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns_model)
                       # 2. Inserisci la colonna 'Ora Prevista' come stringa
                       results_df_sim.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_sim])

                       # 3. Rinomina colonne per display più leggibile
                       rename_dict = {'Ora Prevista': 'Ora Prevista'}
                       original_to_renamed_map = {}
                       for col in target_columns_model:
                           unit = '(m)' if 'Livello' in col else ('(mm)' if 'Pioggia' in col else '') # Aggiunta unità pioggia
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip()
                           count = 1
                           final_name = new_name
                           # Gestisci duplicati nel nome visualizzato
                           while final_name in rename_dict.values():
                               count += 1
                               final_name = f"{new_name}_{count}"
                           rename_dict[col] = final_name
                           original_to_renamed_map[col] = final_name # Mappa nome originale -> nome rinominato

                       results_df_sim_renamed = results_df_sim.rename(columns=rename_dict)

                       # 4. Arrotonda SOLO le colonne numeriche rinominate (prima dello styling)
                       df_to_display = results_df_sim_renamed.copy()
                       numeric_cols_renamed = [original_to_renamed_map[col] for col in target_columns_model if col in original_to_renamed_map]
                       try:
                           cols_in_df_to_round = [col for col in numeric_cols_renamed if col in df_to_display.columns]
                           if cols_in_df_to_round:
                               # Arrotonda a 3 decimali per livelli, 1 per pioggia? O usa round(3) per tutti
                               for col_r in cols_in_df_to_round:
                                   decimals = 3 if '(m)' in col_r else 1 # Usa nome rinominato per decidere
                                   df_to_display[col_r] = df_to_display[col_r].round(decimals)
                           else:
                               st.warning("Nessuna colonna numerica trovata per l'arrotondamento pre-visualizzazione.")
                       except Exception as e_round_display:
                           st.error(f"Errore durante l'arrotondamento selettivo: {e_round_display}")
                           # Fallback: usa il dataframe rinominato senza arrotondamento
                           df_to_display = results_df_sim_renamed

                       # --- NUOVO/MODIFICATO: Applicazione stile per valori sopra soglia ---
                       def style_threshold_value(val, original_col_name, thresholds_dict):
                           """ Funzione per applicare stile colore testo se val >= soglia. """
                           sensor_thresh = thresholds_dict.get(original_col_name, {})
                           alert_thresh = sensor_thresh.get('alert')
                           attention_thresh = sensor_thresh.get('attention')
                           style = '' # Stile default: nessun colore aggiuntivo

                           # Applica stile solo se val è numerico
                           if pd.notna(val) and isinstance(val, (int, float)):
                               # Priorità Allerta (Rosso)
                               if alert_thresh is not None and isinstance(alert_thresh, (int, float)) and val >= alert_thresh:
                                   style = 'color: red; font-weight: bold;'
                               # Attenzione (Arancione) solo se non già in allerta
                               elif attention_thresh is not None and isinstance(attention_thresh, (int, float)) and val >= attention_thresh:
                                   style = 'color: orange;'
                           return style

                       # Crea lo styler dal DataFrame arrotondato/preparato
                       styler = df_to_display.style

                       # Applica lo stile cella per cella alle colonne numeriche (usando la mappa)
                       for original_col, renamed_col in original_to_renamed_map.items():
                            if renamed_col in df_to_display.columns: # Sicurezza
                                # Usa lambda per passare il nome colonna ORIGINALE e le soglie alla funzione di stile
                                styler = styler.applymap(
                                    lambda x: style_threshold_value(x, original_col, current_thresholds),
                                    subset=[renamed_col] # Applica solo alla colonna rinominata corrente
                                )
                            else:
                                 st.warning(f"Colonna rinominata '{renamed_col}' (da '{original_col}') non trovata nel DataFrame per lo styling.")


                       # Mostra il DataFrame STILIZZATO
                       st.dataframe(styler, use_container_width=True, hide_index=True) # Aggiunto hide_index

                       # 6. Link per il download (usa DataFrame originale NON stilizzato)
                       st.markdown(get_table_download_link(results_df_sim, f"simulazione_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                       # --- Grafici (MODIFICATI per passare soglie) ---
                       st.subheader('📈 Grafici Previsioni Simulate')
                       # Passa il dizionario delle soglie alla funzione plot_predictions
                       figs_sim = plot_predictions(
                           predictions_sim,            # Array numpy delle previsioni
                           active_config,              # Config del modello attivo
                           current_thresholds,         # Dizionario delle soglie da session_state
                           start_pred_time             # Ora di inizio previsione
                       )

                       # Mostra i grafici
                       if figs_sim:
                           # Determina layout colonne (max 2 per riga)
                           num_sim_cols = min(len(figs_sim), 2)
                           sim_cols = st.columns(num_sim_cols)
                           for i, fig_sim in enumerate(figs_sim):
                               with sim_cols[i % num_sim_cols]:
                                    # Genera nome file per download
                                    target_col_name = target_columns_model[i]
                                    # Pulisci nome stazione per nome file
                                    s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False)).strip('_')
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                       else:
                           st.warning("Nessun grafico generato per la simulazione.")

                  else:
                       st.error("Predizione simulazione fallita o risultato non valido.")
             else:
                  st.error("Impossibile eseguire la simulazione: dati input non pronti o non validi.")


# --- PAGINA ANALISI DATI STORICI (INVARIATA nella logica principale) ---
# ... (codice invariato, usa df_current_csv e date_col_name_csv) ...
elif page == 'Analisi Dati Storici':
    st.header('🔎 Analisi Dati Storici (da file CSV)')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili. Carica un file CSV valido nella sidebar per usare questa funzionalità.")
    else:
        st.info(f"Dataset CSV caricato: {len(df_current_csv)} righe.")
        # Verifica se la colonna data è valida e timezone-aware
        if date_col_name_csv not in df_current_csv.columns or not pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]):
            st.error(f"Colonna data '{date_col_name_csv}' non trovata o non valida nel CSV. Impossibile procedere.")
            st.stop()
        elif df_current_csv[date_col_name_csv].dt.tz is None:
             st.warning(f"La colonna data '{date_col_name_csv}' non ha timezone. Sarà trattata come naive. Potrebbe causare problemi nel filtraggio.")
             # Potresti forzare una timezone qui se sei sicuro:
             # df_current_csv[date_col_name_csv] = df_current_csv[date_col_name_csv].dt.tz_localize(italy_tz, ambiguous='infer')

        # Periodo dati (gestione NaT)
        min_dt_csv = df_current_csv[date_col_name_csv].dropna().min()
        max_dt_csv = df_current_csv[date_col_name_csv].dropna().max()

        if pd.isna(min_dt_csv) or pd.isna(max_dt_csv):
            st.error("Impossibile determinare il range di date dai dati CSV.")
            st.stop()

        st.caption(f"Periodo dati: dal {min_dt_csv.strftime('%d/%m/%Y %H:%M')} al {max_dt_csv.strftime('%d/%m/%Y %H:%M')}")

        min_date_csv = min_dt_csv.date()
        max_date_csv = max_dt_csv.date()
        col1, col2 = st.columns(2)

        if min_date_csv == max_date_csv:
             start_date = min_date_csv
             end_date = max_date_csv
             col1.info(f"Disponibile solo il giorno: {min_date_csv.strftime('%d/%m/%Y')}")
        else:
             start_date = col1.date_input('Data inizio', min_date_csv, min_value=min_date_csv, max_value=max_date_csv, key="analisi_start_date")
             end_date = col2.date_input('Data fine', max_date_csv, min_value=start_date, max_value=max_date_csv, key="analisi_end_date")

        if start_date > end_date:
             st.error("Data inizio non può essere successiva alla data fine.")
        else:
            try:
                # Creazione timestamp inizio/fine con gestione timezone
                df_tz = df_current_csv[date_col_name_csv].dt.tz # Ottieni timezone dai dati (o None)
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.max.time())

                if df_tz is not None:
                    # Localizza start/end date alla stessa timezone dei dati
                    start_dt = df_tz.localize(start_dt)
                    end_dt = df_tz.localize(end_dt)
                # Se df_tz è None, start_dt/end_dt rimangono naive (confronto con naive)

                mask = (df_current_csv[date_col_name_csv] >= start_dt) & (df_current_csv[date_col_name_csv] <= end_dt)
                filtered_df = df_current_csv.loc[mask]
            except Exception as e_filter:
                st.error(f"Errore durante il filtraggio per data: {e_filter}"); st.stop()


            if filtered_df.empty:
                 st.warning(f"Nessun dato trovato nel periodo selezionato ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}).")
            else:
                 st.success(f"Trovati {len(filtered_df)} record nel periodo selezionato.")
                 tab1, tab2, tab3 = st.tabs(["📊 Andamento Temporale", "📈 Statistiche/Distribuzione", "🔗 Correlazione"])

                 potential_features_analysis = filtered_df.select_dtypes(include=np.number).columns.tolist()
                 potential_features_analysis = [f for f in potential_features_analysis if f not in ['index', 'level_0']]
                 feature_labels_analysis = {get_station_label(f, short=True): f for f in potential_features_analysis}
                 if not feature_labels_analysis:
                     st.warning("Nessuna colonna numerica valida trovata nei dati filtrati per l'analisi."); st.stop()

                 with tab1:
                      st.subheader("Andamento Temporale Features")
                      default_labels_ts = [lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f][:2]
                      if not default_labels_ts: default_labels_ts = list(feature_labels_analysis.keys())[:min(2, len(feature_labels_analysis))]
                      selected_labels_ts = st.multiselect("Seleziona feature da visualizzare:", options=list(feature_labels_analysis.keys()), default=default_labels_ts, key="analisi_ts_multi")
                      features_plot = [feature_labels_analysis[lbl] for lbl in selected_labels_ts]

                      if features_plot:
                           fig_ts = go.Figure()
                           for feature in features_plot:
                                legend_name = get_station_label(feature, short=True)
                                fig_ts.add_trace(go.Scatter(
                                    x=filtered_df[date_col_name_csv],
                                    y=filtered_df[feature],
                                    mode='lines', name=legend_name,
                                    hovertemplate=f'<b>{legend_name}</b><br>%{{x|%d/%m %H:%M}}<br>Val: %{{y:.2f}}<extra></extra>'
                                ))
                           fig_ts.update_layout(
                                title='Andamento Temporale Selezionato',
                                xaxis_title='Data e Ora',
                                yaxis_title='Valore', height=500, hovermode="x unified",
                                margin=dict(t=50, b=40, l=40, r=10)
                            )
                           st.plotly_chart(fig_ts, use_container_width=True)
                           ts_filename = f"andamento_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                           st.markdown(get_plotly_download_link(fig_ts, ts_filename), unsafe_allow_html=True)
                      else: st.info("Seleziona almeno una feature per visualizzare l'andamento temporale.")

                 with tab2:
                      st.subheader("Statistiche Descrittive e Distribuzione")
                      default_stat_label = next((lbl for lbl, f in feature_labels_analysis.items() if 'Livello' in f), list(feature_labels_analysis.keys())[0])
                      selected_label_stat = st.selectbox("Seleziona feature per statistiche:", options=list(feature_labels_analysis.keys()), index=list(feature_labels_analysis.keys()).index(default_stat_label), key="analisi_stat_select")
                      feature_stat = feature_labels_analysis.get(selected_label_stat)

                      if feature_stat:
                           st.markdown(f"**Statistiche per: {selected_label_stat}** (`{feature_stat}`)")
                           st.dataframe(filtered_df[[feature_stat]].describe().round(3))

                           st.markdown(f"**Distribuzione per: {selected_label_stat}**")
                           fig_hist = go.Figure(data=[go.Histogram(x=filtered_df[feature_stat], name=selected_label_stat)])
                           fig_hist.update_layout(
                               title=f'Distribuzione di {selected_label_stat}',
                               xaxis_title='Valore', yaxis_title='Frequenza', height=400,
                               margin=dict(t=50, b=40, l=40, r=10)
                           )
                           st.plotly_chart(fig_hist, use_container_width=True)
                           hist_filename = f"distrib_{selected_label_stat.replace(' ','_').replace('(','').replace(')','')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                           st.markdown(get_plotly_download_link(fig_hist, hist_filename), unsafe_allow_html=True)

                 with tab3:
                      st.subheader("Matrice di Correlazione e Scatter Plot")
                      default_corr_labels = list(feature_labels_analysis.keys())
                      selected_labels_corr = st.multiselect("Seleziona feature per correlazione:", options=list(feature_labels_analysis.keys()), default=default_corr_labels, key="analisi_corr_multi")
                      features_corr = [feature_labels_analysis[lbl] for lbl in selected_labels_corr]

                      if len(features_corr) > 1:
                           corr_matrix = filtered_df[features_corr].corr()
                           heatmap_labels = [get_station_label(f, short=True) for f in features_corr]

                           fig_hm = go.Figure(data=go.Heatmap(
                               z=corr_matrix.values, x=heatmap_labels, y=heatmap_labels,
                               colorscale='RdBu', zmin=-1, zmax=1,
                               colorbar=dict(title='Corr'),
                               text=corr_matrix.round(2).values, texttemplate="%{text}",
                               hoverongaps=False
                           ))
                           fig_hm.update_layout(
                               title='Matrice di Correlazione',
                               height=max(400, len(heatmap_labels)*35),
                               xaxis_tickangle=-45, yaxis_autorange='reversed',
                               margin=dict(t=50, b=40, l=60, r=10)
                            )
                           st.plotly_chart(fig_hm, use_container_width=True)
                           hm_filename = f"correlazione_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                           st.markdown(get_plotly_download_link(fig_hm, hm_filename), unsafe_allow_html=True)

                           if len(selected_labels_corr) <= 10:
                                st.markdown("**Scatter Plot Correlazione (Coppia di Feature)**")
                                cs1, cs2 = st.columns(2)
                                label_x = cs1.selectbox("Feature Asse X:", selected_labels_corr, index=0, key="scatter_x_select")
                                default_y_index = 1 if len(selected_labels_corr) > 1 else 0
                                label_y = cs2.selectbox("Feature Asse Y:", selected_labels_corr, index=default_y_index, key="scatter_y_select")

                                fx = feature_labels_analysis.get(label_x)
                                fy = feature_labels_analysis.get(label_y)

                                if fx and fy and fx != fy:
                                    fig_sc = go.Figure(data=[go.Scatter(
                                        x=filtered_df[fx], y=filtered_df[fy], mode='markers',
                                        marker=dict(size=5, opacity=0.6),
                                        name=f'{label_x} vs {label_y}',
                                        text=filtered_df[date_col_name_csv].dt.strftime('%d/%m %H:%M'),
                                        hovertemplate=f'<b>{label_x}</b>: %{{x:.2f}}<br><b>{label_y}</b>: %{{y:.2f}}<br>%{{text}}<extra></extra>'
                                    )])
                                    fig_sc.update_layout(
                                        title=f'Correlazione: {label_x} vs {label_y}',
                                        xaxis_title=label_x, yaxis_title=label_y, height=500,
                                        margin=dict(t=50, b=40, l=40, r=10)
                                    )
                                    st.plotly_chart(fig_sc, use_container_width=True)
                                    sc_filename = f"scatter_{label_x.replace(' ','_')}_vs_{label_y.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                                    st.markdown(get_plotly_download_link(fig_sc, sc_filename), unsafe_allow_html=True)
                                elif fx and fy and fx == fy:
                                     st.info("Seleziona due feature diverse per lo scatter plot.")
                           else:
                                st.info("Troppe feature selezionate (>10) per visualizzare lo scatter plot interattivo.")
                      elif len(features_corr) == 1:
                           st.info("Seleziona almeno due feature per calcolare la correlazione.")
                      else:
                           st.info("Seleziona almeno due feature per visualizzare la matrice di correlazione.")

                 st.divider()
                 st.subheader('Download Dati Filtrati (CSV)')
                 download_filename_filtered = f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                 st.markdown(get_table_download_link(filtered_df, download_filename_filtered), unsafe_allow_html=True)


# --- PAGINA ALLENAMENTO MODELLO (INVARIATA nella logica principale) ---
# ... (codice invariato) ...
elif page == 'Allenamento Modello':
    st.header('🎓 Allenamento Nuovo Modello LSTM')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili. Carica un file CSV valido nella sidebar per allenare un nuovo modello.")
    else:
        st.success(f"Dati CSV disponibili per l'allenamento: {len(df_current_csv)} righe.")
        st.subheader('Configurazione Addestramento')

        default_save_name = f"modello_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
        save_name_input = st.text_input("Nome base per salvare il modello (file .pth, .json, .joblib)", default_save_name, key="train_save_filename")
        # Pulisce il nome file
        save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_')
        if not save_name: save_name = "modello_default"
        if save_name != save_name_input: st.caption(f"Nome file corretto in: `{save_name}`")

        st.markdown("**1. Seleziona Feature Input per il Modello:**")
        all_global_features = st.session_state.feature_columns
        features_present_in_csv = [f for f in all_global_features if f in df_current_csv.columns]
        missing_from_csv = [f for f in all_global_features if f not in df_current_csv.columns]
        if missing_from_csv:
            st.caption(f"Nota: Le seguenti feature globali non sono nel CSV caricato e non possono essere usate: {', '.join(missing_from_csv)}")

        if not features_present_in_csv:
             st.error("Errore: Nessuna delle feature definite globalmente è presente nel file CSV caricato. Impossibile procedere.")
             st.stop()

        selected_features_train = []
        with st.expander(f"Seleziona Feature Input (Disponibili: {len(features_present_in_csv)})", expanded=False):
            # Organizza checkbox in colonne per compattezza
            num_feat_cols = 4
            cols_feat_train = st.columns(num_feat_cols)
            for i, feat in enumerate(features_present_in_csv):
                 with cols_feat_train[i % num_feat_cols]:
                     label_feat_train = get_station_label(feat, short=True)
                     is_checked = st.checkbox(label_feat_train, value=True, key=f"train_feat_check_{feat}", help=feat)
                     if is_checked:
                          selected_features_train.append(feat)
        if not selected_features_train: st.warning("Seleziona almeno una feature di input.")
        else: st.caption(f"{len(selected_features_train)} feature di input selezionate.")

        st.markdown("**2. Seleziona Target Output (solo sensori di Livello tra gli input selezionati):**")
        selected_targets_train = []
        # Opzioni target: solo feature 'Livello' che sono state SELEZIONATE come input
        hydro_level_features_options = [f for f in selected_features_train if 'Livello' in f]

        if not hydro_level_features_options:
             st.warning("Nessuna colonna 'Livello' tra le feature di input selezionate. Impossibile selezionare target.")
        else:
            # Default target: prova a usare i target del modello attivo (se validi e presenti) o il primo livello disponibile
            default_targets_train = []
            if active_config and active_config.get("target_columns"):
                 # Prendi i target del modello attivo che sono anche tra le opzioni disponibili
                 valid_active_targets = [t for t in active_config["target_columns"] if t in hydro_level_features_options]
                 if valid_active_targets: default_targets_train = valid_active_targets
            if not default_targets_train and hydro_level_features_options:
                 default_targets_train = [hydro_level_features_options[0]] # Fallback al primo livello

            # Checkbox per i target in colonne
            num_target_cols = min(len(hydro_level_features_options), 5)
            cols_target_train = st.columns(num_target_cols)
            for i, feat_target in enumerate(hydro_level_features_options):
                with cols_target_train[i % num_target_cols]:
                     lbl_target = get_station_label(feat_target, short=True)
                     is_target_checked = st.checkbox(lbl_target, value=(feat_target in default_targets_train), key=f"train_target_check_{feat_target}", help=f"Seleziona come output: {feat_target}")
                     if is_target_checked:
                         selected_targets_train.append(feat_target)

        if not selected_targets_train: st.warning("Seleziona almeno un target di output (Livello).")
        else: st.caption(f"{len(selected_targets_train)} target di output selezionati.")

        st.markdown("**3. Imposta Parametri Modello e Addestramento:**")
        with st.expander("Parametri Modello e Training", expanded=True):
             c1t, c2t, c3t = st.columns(3)
             # Usa valori modello attivo come default, altrimenti valori fissi
             default_iw = active_config["input_window"] if active_config else 24
             default_ow = active_config["output_window"] if active_config else 12
             default_hs = active_config["hidden_size"] if active_config else 128
             default_nl = active_config["num_layers"] if active_config else 2
             default_dr = active_config["dropout"] if active_config else 0.2
             default_bs = active_config.get("batch_size", 32) if active_config else 32
             default_vs = active_config.get("val_split_percent", 20) if active_config else 20
             default_lr = active_config.get("learning_rate", 0.001) if active_config else 0.001
             default_ep = active_config.get("epochs_run", 50) if active_config else 50

             iw_t = c1t.number_input("Finestra Input (n. rilevazioni)", 6, 168, default_iw, 6, key="t_param_in_win", help="Quante rilevazioni passate usa il modello (es. 24 = 12 ore)")
             ow_t = c1t.number_input("Finestra Output (n. rilevazioni)", 1, 72, default_ow, 1, key="t_param_out_win", help="Quante rilevazioni future predice il modello (es. 12 = 6 ore)")
             vs_t = c1t.slider("% Dati per Validazione", 0, 50, default_vs, 1, key="t_param_val_split", help="Percentuale finale dei dati usata per validare il modello durante il training (0% = nessun set di validazione)")

             hs_t = c2t.number_input("Neuroni Nascosti (Hidden Size)", 16, 1024, default_hs, 16, key="t_param_hidden", help="Numero di neuroni nei livelli LSTM")
             nl_t = c2t.number_input("Numero Livelli LSTM (Layers)", 1, 8, default_nl, 1, key="t_param_layers", help="Numero di livelli LSTM impilati")
             dr_t = c2t.slider("Dropout", 0.0, 0.7, default_dr, 0.05, key="t_param_dropout", help="Tasso di dropout per regolarizzazione (0 = disabilitato)")

             lr_t = c3t.number_input("Learning Rate", 1e-5, 1e-2, default_lr, format="%.5f", step=1e-4, key="t_param_lr", help="Tasso di apprendimento per l'ottimizzatore Adam")
             bs_t = c3t.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], default_bs, key="t_param_batch", help="Dimensione dei batch di dati usati in ogni step di training")
             ep_t = c3t.number_input("Numero Epoche", 5, 500, default_ep, 5, key="t_param_epochs", help="Numero di volte che il modello vedrà l'intero dataset di training")

        st.markdown("**4. Avvia Addestramento:**")
        valid_name = bool(save_name)
        valid_features = bool(selected_features_train)
        valid_targets = bool(selected_targets_train)
        ready_to_train = valid_name and valid_features and valid_targets

        if not valid_features: st.error("❌ Seleziona almeno una feature di input.")
        if not valid_targets: st.error("❌ Seleziona almeno un target di output (Livello).")
        if not valid_name: st.error("❌ Inserisci un nome valido per salvare il modello.")

        train_button = st.button("▶️ Addestra Nuovo Modello", type="primary", disabled=not ready_to_train, key="train_run_button")

        if train_button and ready_to_train:
             st.info(f"Avvio addestramento per il modello '{save_name}'...")
             with st.spinner('Preparazione dati per training...'):
                  # Assicurati che le colonne selezionate siano effettivamente nel df
                  cols_needed = list(set(selected_features_train + selected_targets_train)) # Uniche
                  missing_in_df_final = [c for c in cols_needed if c not in df_current_csv.columns]
                  if missing_in_df_final:
                       st.error(f"Errore Critico: Le colonne selezionate {', '.join(missing_in_df_final)} non sono state trovate nel DataFrame CSV finale. Impossibile procedere.")
                       st.stop()

                  # Prepara dati usando la funzione dedicata
                  X_tr, y_tr, X_v, y_v, sc_f_tr, sc_t_tr = prepare_training_data(
                      df_current_csv.copy(),       # Usa una copia del DF CSV
                      selected_features_train,   # Feature selezionate
                      selected_targets_train,    # Target selezionati
                      iw_t,                      # Finestra input
                      ow_t,                      # Finestra output
                      vs_t                       # Percentuale validation split
                  )
                  if X_tr is None or y_tr is None or sc_f_tr is None or sc_t_tr is None:
                       st.error("Preparazione dati fallita. Controlla i log e i parametri (finestre vs lunghezza dati).")
                       st.stop() # Blocca se la preparazione fallisce

                  val_set_size = len(X_v) if X_v is not None and X_v.size > 0 else 0 # Gestione X_v vuoto
                  st.success(f"Dati pronti: {len(X_tr)} sequenze di training, {val_set_size} sequenze di validazione.")

             st.subheader("⏳ Addestramento in corso...")
             input_size_train = len(selected_features_train)
             output_size_train = len(selected_targets_train)
             trained_model = None
             train_start_time = time.time()
             try:
                 # Chiama la funzione di training
                 trained_model, train_losses, val_losses = train_model(
                     X_tr, y_tr, X_v, y_v,
                     input_size_train, output_size_train, ow_t,
                     hs_t, nl_t, ep_t, bs_t, lr_t, dr_t
                 )
                 train_end_time = time.time()
                 st.info(f"Tempo di addestramento: {train_end_time - train_start_time:.2f} secondi.")

             except Exception as e_train_run:
                 st.error(f"Errore durante l'addestramento: {e_train_run}")
                 st.error(traceback.format_exc())
                 trained_model = None # Assicura che il modello non venga salvato

             # Se l'addestramento è completato con successo
             if trained_model:
                 st.success("Addestramento completato con successo!")
                 st.subheader("💾 Salvataggio Risultati Modello")
                 os.makedirs(MODELS_DIR, exist_ok=True) # Crea cartella se non esiste
                 base_path = os.path.join(MODELS_DIR, save_name)
                 m_path = f"{base_path}.pth"
                 c_path = f"{base_path}.json"
                 sf_path = f"{base_path}_features.joblib"
                 st_path = f"{base_path}_targets.joblib"

                 # Trova la migliore validation loss (se applicabile)
                 final_val_loss = None
                 if val_losses and vs_t > 0:
                      valid_val_losses = [v for v in val_losses if v is not None and np.isfinite(v)]
                      if valid_val_losses: final_val_loss = min(valid_val_losses)

                 # Crea dizionario di configurazione da salvare
                 config_save = {
                     "input_window": iw_t, "output_window": ow_t, "hidden_size": hs_t,
                     "num_layers": nl_t, "dropout": dr_t,
                     "feature_columns": selected_features_train, # Salva le feature usate
                     "target_columns": selected_targets_train, # Salva i target usati
                     "training_date": datetime.now(italy_tz).isoformat(),
                     "final_val_loss": final_val_loss if final_val_loss is not None else 'N/A',
                     "epochs_run": ep_t, "batch_size": bs_t, "val_split_percent": vs_t,
                     "learning_rate": lr_t,
                     "display_name": save_name, # Usa il nome file pulito come display name default
                     "source_data_info": data_source_info # Info su file CSV usato
                 }

                 # Salva i file
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

                     # Bottone per refreshare l'app e vedere il nuovo modello
                     st.info("Dopo il download, potresti voler ricaricare l'app per vedere il nuovo modello nella lista.")
                     if st.button("Pulisci Cache e Ricarica App", key="train_reload_app"):
                         # Pulisci cache specifiche e resetta stato modello attivo
                         find_available_models.clear() # Pulisce cache lista modelli
                         load_model_config.clear()     # Pulisce cache config
                         load_specific_model.clear()   # Pulisce cache modelli torch
                         load_specific_scalers.clear() # Pulisce cache scaler
                         st.session_state.pop('active_model_name', None)
                         st.session_state.pop('active_config', None)
                         st.session_state.pop('active_model', None)
                         st.session_state.pop('active_device', None)
                         st.session_state.pop('active_scaler_features', None)
                         st.session_state.pop('active_scaler_targets', None)
                         st.success("Cache pulita. Ricaricamento...")
                         time.sleep(1)
                         st.rerun() # Forza rerun completo

                 except Exception as e_save_files:
                     st.error(f"Errore durante il salvataggio dei file del modello: {e_save_files}")
                     st.error(traceback.format_exc())

             elif not train_button:
                 # Questo blocco non dovrebbe essere raggiunto se il bottone non è premuto
                 pass
             else: # Se trained_model è None dopo il tentativo di training
                 st.error("Addestramento fallito o interrotto. Impossibile salvare il modello.")


# --- Footer (invariato) ---
st.sidebar.divider()
st.sidebar.info('App Idrologica Dashboard & Predict © 2024 Alberto Bussaglia')
