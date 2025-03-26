import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
import math
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

# Configurazione della pagina
st.set_page_config(page_title="Modello Predittivo Idrologico", page_icon="🌊", layout="wide")

# --- Costanti ---
MODELS_DIR = "models" # Cartella dove risiedono i modelli pre-addestrati
DEFAULT_DATA_PATH = "dati_idro.csv" # Assumi sia nella stessa cartella dello script

# --- Definizioni Funzioni Core ML (Dataset, LSTM) ---

class TimeSeriesDataset(Dataset):
    """Dataset personalizzato per PyTorch."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class HydroLSTM(nn.Module):
    """Modello LSTM per previsioni idrologiche."""
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window)
        self.num_layers, self.hidden_size, self.output_window, self.output_size = num_layers, hidden_size, output_window, output_size
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0)); out = out[:, -1, :]; out = self.fc(out)
        return out.view(out.size(0), self.output_window, self.output_size)

# --- Funzioni Utilità ---

def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, val_split=20):
    """Prepara i dati per il training: crea sequenze, normalizza e splitta."""
    st.write(f"<small>Prepare Data: Input={input_window}h, Output={output_window}h, Val={val_split}%</small>", unsafe_allow_html=True)
    try:
        for col in feature_columns + target_columns:
            if col not in df.columns: raise ValueError(f"Colonna '{col}' richiesta non trovata.")
        if df[feature_columns + target_columns].isnull().sum().sum() > 0: st.warning("NaN residui prima creazione sequenze!")
    except ValueError as e: st.error(f"Errore colonne in prepare_training_data: {e}"); return [None]*6
    X, y = [], []; total_len, required_len = len(df), input_window + output_window
    if total_len < required_len: st.error(f"Dati insuff. ({total_len} righe) vs richiesti ({required_len})."); return [None]*6
    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_window][feature_columns].values)
        y.append(df.iloc[i + input_window : i + required_len][target_columns].values)
    if not X or not y: st.error("Errore: Nessuna sequenza X/y creata."); return [None]*6
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    scaler_features, scaler_targets = MinMaxScaler(), MinMaxScaler()
    if X.size == 0 or y.size == 0: st.error("Dati X o y vuoti prima normalizzazione."); return [None]*6
    n_seq, seq_in, n_feat = X.shape; n_seq_y, seq_out, n_targ = y.shape
    X_flat, y_flat = X.reshape(-1, n_feat), y.reshape(-1, n_targ)
    try: X_scaled_flat, y_scaled_flat = scaler_features.fit_transform(X_flat), scaler_targets.fit_transform(y_flat)
    except Exception as e: st.error(f"Errore scaling: {e}\nNaN X:{np.isnan(X_flat).sum()}, NaN y:{np.isnan(y_flat).sum()}"); return [None]*6
    X_scaled, y_scaled = X_scaled_flat.reshape(n_seq, seq_in, n_feat), y_scaled_flat.reshape(n_seq_y, seq_out, n_targ)
    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    if split_idx <= 0 or split_idx >= len(X_scaled):
        st.warning(f"Split indice ({split_idx}) non ideale.")
        if len(X_scaled) < 2: st.error("Dataset troppo piccolo per split."); return [None]*6
        split_idx = max(1, len(X_scaled) - 1) if split_idx >= len(X_scaled) else min(len(X_scaled) - 1, 1)
    X_train, y_train, X_val, y_val = X_scaled[:split_idx], y_scaled[:split_idx], X_scaled[split_idx:], y_scaled[split_idx:]
    st.write(f"<small>Split Dati: Train={len(X_train)}, Val={len(X_val)}</small>", unsafe_allow_html=True)
    if X_train.size == 0 or y_train.size == 0: st.error("Set Training vuoto."); return [None]*6
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set Validazione vuoto."); X_val, y_val = np.empty((0,seq_in,n_feat),dtype=np.float32), np.empty((0,seq_out,n_targ),dtype=np.float32)
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

@st.cache_data
def load_model_config(_config_path):
    """Carica config JSON modello."""
    try:
        with open(_config_path, 'r') as f: config = json.load(f)
        required = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        if not all(k in config for k in required): st.error(f"Config '{_config_path}' incompleta."); return None
        return config
    except Exception as e: st.error(f"Errore config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Carico modello...")
def load_specific_model(_model_path, config):
    """Carica modello .pth."""
    if not config: return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        features = config.get("feature_columns", st.session_state.get("feature_columns", []))
        if not features: st.error("Input size non determinabile ('feature_columns' mancanti)."); return None, None
        model = HydroLSTM(len(features), config["hidden_size"], len(config["target_columns"]), config["output_window"], config["num_layers"], config["dropout"]).to(device)
        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"'{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'): _model_path.seek(0); model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")
        model.eval(); st.success(f"Modello '{config.get('name', 'N/A')}' caricato ({device})."); return model, device
    except Exception as e: st.error(f"Errore caricamento modello '{config.get('name', 'N/A')}': {e}\n{traceback.format_exc()}"); return None, None

@st.cache_resource(show_spinner="Carico scaler...")
def load_specific_scalers(_scaler_f_path, _scaler_t_path):
    """Carica scaler .joblib."""
    try:
        def _load(p):
             if isinstance(p, str):
                  if not os.path.exists(p): raise FileNotFoundError(f"'{p}' non trovato.")
                  return joblib.load(p)
             elif hasattr(p, 'getvalue'): p.seek(0); return joblib.load(p)
             else: raise TypeError("Percorso scaler non valido.")
        scaler_f, scaler_t = _load(_scaler_f_path), _load(_scaler_t_path)
        st.success(f"Scaler caricati."); return scaler_f, scaler_t
    except Exception as e: st.error(f"Errore caricamento scaler: {e}"); return None, None

def find_available_models(models_dir=MODELS_DIR):
    """Trova set di file modello validi."""
    available = {};
    if not os.path.isdir(models_dir): return available
    for pth in glob.glob(os.path.join(models_dir, "*.pth")):
        base = os.path.splitext(os.path.basename(pth))[0]
        cfg_p, scf_p, sct_p = [os.path.join(models_dir, f"{base}{ext}") for ext in [".json", "_features.joblib", "_targets.joblib"]]
        if all(os.path.exists(p) for p in [cfg_p, scf_p, sct_p]):
            try:
                 with open(cfg_p, 'r') as f: cfg = json.load(f)
                 d_name = f"{cfg.get('display_name', base)} (In:{cfg.get('input_window','?')}, Out:{cfg.get('output_window','?')})"
            except: d_name = base
            available[d_name] = { "config_name": base, "pth_path": pth, "config_path": cfg_p, "scaler_features_path": scf_p, "scaler_targets_path": sct_p }
    return available

def predict(model, input_data, scaler_features, scaler_targets, config, device):
    """Esegue previsione."""
    if not all([model, scaler_features, scaler_targets, config]): st.error("Predict: Input mancanti."); return None
    iw, ow, tc = config["input_window"], config["output_window"], config["target_columns"]; fc_cfg = config.get("feature_columns", [])
    if input_data.shape[0] != iw: st.error(f"Predict: Righe {input_data.shape[0]} != Win {iw}."); return None
    if fc_cfg and input_data.shape[1] != len(fc_cfg): st.error(f"Predict: Colonne {input_data.shape[1]} != Feat {len(fc_cfg)}."); return None
    if not fc_cfg and hasattr(scaler_features, 'n_features_in_') and scaler_features.n_features_in_ != input_data.shape[1]: st.error(f"Predict: Colonne {input_data.shape[1]} != Scaler Feat {scaler_features.n_features_in_}."); return None
    model.eval()
    try:
        input_norm = scaler_features.transform(input_data); input_tensor = torch.FloatTensor(input_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(input_tensor)
        output_np = output.cpu().numpy().reshape(ow, len(tc))
        if not hasattr(scaler_targets, 'n_features_in_'): st.error("Predict: Scaler targets non fittato."); return None
        if scaler_targets.n_features_in_ != len(tc): st.error(f"Predict: Output {len(tc)} != Scaler Targets {scaler_targets.n_features_in_}."); return None
        return scaler_targets.inverse_transform(output_np)
    except Exception as e: st.error(f"Errore predict: {e}\n{traceback.format_exc()}"); return None

def plot_predictions(predictions, config, start_time=None):
    """Plotta previsioni con Plotly."""
    if config is None or predictions is None: return []
    ow, tc = config["output_window"], config["target_columns"]; figs = []
    for i, s_name in enumerate(tc):
        fig = go.Figure()
        if start_time: h = [start_time + timedelta(hours=j+1) for j in range(ow)]; x, xt = h, "Data/Ora Previste"
        else: h = np.arange(1, ow + 1); x, xt = h, "Ore Future"
        fig.add_trace(go.Scatter(x=x, y=predictions[:, i], mode='lines+markers', name=f'Prev. {s_name}'))
        fig.update_layout(title=f'Previsione - {s_name}', xaxis_title=xt, yaxis_title='Livello [m]', height=400, hovermode="x unified", margin=dict(l=0,r=0,b=0,t=40))
        fig.update_yaxes(rangemode='tozero'); figs.append(fig)
    return figs

def import_data_from_sheet(sheet_id, expected_cols, input_window, date_col_name='Data_Ora', date_format='%d/%m/%Y %H:%M'):
    """Importa dati da Google Sheet."""
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: st.error("Credenziali Google non trovate."); return None
        creds = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(creds); sh = gc.open_by_key(sheet_id); ws = sh.sheet1; data = ws.get_all_values()
        if not data or len(data) < 2: st.error("Foglio Google vuoto."); return None
        h, r = data[0], data[1:]; headers_set = set(h)
        missing = [c for c in expected_cols if c not in headers_set];
        if missing: st.error(f"Colonne GSheet mancanti: {', '.join(missing)}"); return None
        df = pd.DataFrame(r, columns=h); cols_k = [c for c in expected_cols if c in df.columns]
        if date_col_name not in cols_k and date_col_name in df.columns: cols_k.append(date_col_name)
        elif date_col_name not in df.columns: st.error(f"Colonna data GSheet '{date_col_name}' non trovata."); return None
        df = df[cols_k]; df[date_col_name] = pd.to_datetime(df[date_col_name], format=date_format, errors='coerce')
        df = df.dropna(subset=[date_col_name]).sort_values(by=date_col_name, ascending=True)
        num_cols = [c for c in df.columns if c != date_col_name]
        for col in num_cols: df[col] = pd.to_numeric(df[col].replace(['N/A', '', '-', ' '], np.nan, regex=False).astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df = df.tail(input_window)
        if len(df) < input_window: st.warning(f"GSheet: {len(df)}/{input_window} righe valide.")
        if len(df) == 0: st.error("GSheet: 0 righe valide."); return None
        st.success(f"Importate {len(df)} righe da GSheet."); return df
    except gspread.exceptions.APIError as e: st.error(f"Errore API GSheets: {e.response.json().get('error', {}).get('message', e)}."); return None
    except gspread.exceptions.SpreadsheetNotFound: st.error(f"Foglio Google non trovato (ID: '{sheet_id}')."); return None
    except ValueError as e: st.error(f"Errore conversione dati GSheet: {e}"); return None
    except Exception as e: st.error(f"Errore import GSheet: {e}\n{traceback.format_exc()}"); return None

def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window, hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2):
    """Allena il modello."""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)
    train_ds=TimeSeriesDataset(X_train, y_train); val_ds=TimeSeriesDataset(X_val, y_val)
    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True); val_loader=DataLoader(val_ds, batch_size=batch_size) if len(val_ds) > 0 else None
    criterion=nn.MSELoss(); optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate); scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    train_losses, val_losses = [], []; best_val_loss = float('inf'); best_state = None; p_bar=st.progress(0); status=st.empty(); chart=st.empty()
    def update_chart(tr, vl, plc): fig=go.Figure(); fig.add_trace(go.Scatter(y=tr,name='Train')); fig.add_trace(go.Scatter(y=vl,name='Val')); fig.update_layout(title='Loss',xaxis_title='Epoca',yaxis_title='MSE',height=350,margin=dict(l=0,r=0,b=0,t=40)); plc.plotly_chart(fig,use_container_width=True)
    status.write(f"<small>Training ({epochs} epoche) su {device}...</small>", unsafe_allow_html=True)
    for epoch in range(epochs):
        model.train(); tl = 0
        for Xb, yb in train_loader: Xb,yb=Xb.to(device),yb.to(device); o=model(Xb); l=criterion(o,yb); optimizer.zero_grad(); l.backward(); optimizer.step(); tl+=l.item()
        tl /= len(train_loader); train_losses.append(tl); vl = 0
        if val_loader:
             model.eval()
             with torch.no_grad():
                 for Xb, yb in val_loader: Xb,yb=Xb.to(device),yb.to(device); o=model(Xb); l=criterion(o,yb); vl+=l.item()
             if len(val_loader)>0: vl /= len(val_loader)
        val_losses.append(vl);
        if val_loader: scheduler.step(vl)
        p_bar.progress((epoch+1)/epochs); clr=optimizer.param_groups[0]['lr']; status.text(f"E:{epoch+1}/{epochs}|Tr L:{tl:.5f}|Val L:{vl:.5f}|LR:{clr:.6f}")
        update_chart(train_losses, val_losses, chart)
        if val_loader and vl < best_val_loss: best_val_loss = vl; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()}); st.success(f"Caricato best model (Val Loss: {best_val_loss:.5f})")
    else: st.warning("Usato modello ultima epoca.")
    return model, train_losses, val_losses

def get_table_download_link(df, filename="data.csv"):
    csv=df.to_csv(index=False, sep=';', decimal=','); b64=base64.b64encode(csv.encode('utf-8')).decode(); return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Scarica CSV</a>'
def get_binary_file_download_link(f_obj, fname, text): f_obj.seek(0); b64=base64.b64encode(f_obj.getvalue()).decode(); return f'<a href="data:application/octet-stream;base64,{b64}" download="{fname}">{text}</a>'
def get_plotly_download_link(fig, fbase, thtml="HTML", tpng="PNG"):
    buf_h=io.StringIO(); fig.write_html(buf_h); b64_h=base64.b64encode(buf_h.getvalue().encode()).decode(); href_h=f'<a href="data:text/html;base64,{b64_h}" download="{fbase}.html">{thtml}</a>"; href_p=""
    try: buf_p=io.BytesIO(); fig.write_image(buf_p,format="png"); buf_p.seek(0); b64_p=base64.b64encode(buf_p.getvalue()).decode(); href_p=f' <a href="data:image/png;base64,{b64_p}" download="{fbase}.png">{tpng}</a>'
    except Exception: pass
    return f"{href_h}{href_p}"
def extract_sheet_id(url):
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for p in patterns: m=re.search(p, url);
    if m: return m.group(1)
    return None

# --- Inizializzazione Session State ---
default_ss = { 'active_model_name': None, 'active_config': None, 'active_model': None, 'active_device': None, 'active_scaler_features': None, 'active_scaler_targets': None, 'df': None,
               'feature_columns': ['Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)', 'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)', 'Umidita\' Sensore 3452 (Montemurello)', 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)', 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'],
               'date_col_name_csv': 'Data e Ora' }
for key, val in default_ss.items():
    if key not in st.session_state: st.session_state[key] = val

# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('Dashboard Modello Predittivo Idrologico')

# --- Sidebar ---
with st.sidebar:
    st.header('Impostazioni Dati e Modello')
    st.write("--- Caricamento Dati Storici ---")
    uploaded_data_file = st.file_uploader('File CSV (Opzionale)', type=['csv'], help=f"Se non caricato, usa '{DEFAULT_DATA_PATH}'.", key="data_uploader")
    if uploaded_data_file: st.write(f"<small>File: `{uploaded_data_file.name}`</small>", unsafe_allow_html=True)
    else: st.write(f"<small>Nessun file caricato.</small>", unsafe_allow_html=True)

    df = None; df_load_error = None; data_source_info = ""; data_path_to_load = None; is_uploaded = False
    load_data_placeholder = st.empty()
    if uploaded_data_file: data_path_to_load, is_uploaded, data_source_info = uploaded_data_file, True, f"Upload: **{uploaded_data_file.name}**"
    else:
        load_data_placeholder.info("Controllo default...");
        if os.path.exists(DEFAULT_DATA_PATH): data_path_to_load, is_uploaded, data_source_info = DEFAULT_DATA_PATH, False, f"Default: **{DEFAULT_DATA_PATH}**"; load_data_placeholder.info(f"Trovato: {data_source_info}")
        else: df_load_error = f"Default '{DEFAULT_DATA_PATH}' non trovato."; data_source_info = "No Dati."; load_data_placeholder.warning(f"{df_load_error} Carica CSV.")

    if data_path_to_load:
        load_data_placeholder.info(f"Processo: {data_source_info}")
        try:
            read_args = {'sep': ';', 'decimal': ',', 'encoding': 'utf-8', 'low_memory': False}
            if is_uploaded: data_path_to_load.seek(0); df = pd.read_csv(data_path_to_load, **read_args)
            else: df = pd.read_csv(data_path_to_load, **read_args)
            date_col = st.session_state.date_col_name_csv
            if date_col not in df.columns: raise ValueError(f"Colonna data '{date_col}' mancante.")
            try: df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y %H:%M', errors='raise')
            except ValueError: df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
            if 'feature_columns' not in st.session_state: raise ValueError("`feature_columns` non in session_state.")
            current_features = st.session_state.feature_columns
            missing_f = [c for c in current_features if c not in df.columns]
            if missing_f: raise ValueError(f"Colonne feature mancanti: {', '.join(missing_f)}")
            for col in current_features:
                if col != date_col:
                    if df[col].dtype == 'object': df[col] = df[col].astype(str).str.replace('.','',regex=False).str.replace(',','.',regex=False).str.strip().replace(['N/A','','-'],np.nan,regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            n_nan = df[current_features].isnull().sum().sum()
            if n_nan > 0: df[current_features] = df[current_features].fillna(method='ffill').fillna(method='bfill')
            if df[current_features].isnull().sum().sum() > 0: raise ValueError("NaN residui dopo fill.")
            st.session_state.df = df; load_data_placeholder.success(f"Dati OK ({data_source_info}): {len(df)} righe.")
        except Exception as e: df = None; st.session_state.df = None; df_load_error = f'Errore dati: {e}'; load_data_placeholder.error(f"ERRORE DATI: {df_load_error}"); print(f"--ERRORE DATI--\n{traceback.format_exc()}\n--------------")
    df_current = st.session_state.get('df', None)

    st.divider(); st.header("Selezione Modello")
    available_models_dict = find_available_models(MODELS_DIR); model_display_names = list(available_models_dict.keys())
    MODEL_CHOICE_UPLOAD = "Carica File Manualmente..."; MODEL_CHOICE_NONE = "-- Nessun Modello --"
    selection_options = [MODEL_CHOICE_NONE] + model_display_names + [MODEL_CHOICE_UPLOAD]
    current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
    try: current_index = selection_options.index(current_selection_name)
    except ValueError: current_index = 0
    selected_model_display_name = st.selectbox("Modello Predittivo:", selection_options, index=current_index, help="Scegli modello o carica file.")

    st.session_state.active_model_name = None; st.session_state.active_config = None; st.session_state.active_model = None
    st.session_state.active_device = None; st.session_state.active_scaler_features = None; st.session_state.active_scaler_targets = None
    config_to_load, model_to_load, device_to_load, scaler_f_to_load, scaler_t_to_load = None, None, None, None, None
    load_error_sidebar = False; model_load_placeholder = st.empty()

    if selected_model_display_name == MODEL_CHOICE_NONE: model_load_placeholder.info("Nessun modello selezionato."); st.session_state.active_model_name = MODEL_CHOICE_NONE
    elif selected_model_display_name == MODEL_CHOICE_UPLOAD:
        st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
        with st.expander("Carica File Modello Manualmente", expanded=True):
            mfu=st.file_uploader('.pth',['pth'],key="up_pth"); sffu=st.file_uploader('Scaler Feat',['joblib'],key="up_scf"); stfu=st.file_uploader('Scaler Targ',['joblib'],key="up_sct")
            st.subheader("Configurazione"); c1,c2=st.columns(2)
            iwu=c1.number_input("In Win(h)",1,value=24,key="up_in"); owu=c1.number_input("Out Win(h)",1,value=12,key="up_out")
            hsu=c2.number_input("Hidden",16,value=128,step=16,key="up_hidden"); nlu=c2.number_input("Layers",1,value=2,key="up_layers"); dou=c2.slider("Dropout",0.0,0.7,0.2,0.05,key="up_dropout")
            hydro_g = [c for c in st.session_state.feature_columns if 'Livello' in c]; tcu = st.multiselect("Targets", hydro_g, default=hydro_g, key="up_targets")
            if mfu and sffu and stfu and tcu:
                model_load_placeholder.info("Carico file manuali...")
                temp_cfg = {"input_window":iwu, "output_window":owu, "hidden_size":hsu, "num_layers":nlu, "dropout":dou, "target_columns":tcu, "feature_columns":st.session_state.feature_columns, "name":"uploaded"}
                model_to_load, device_to_load = load_specific_model(mfu, temp_cfg)
                scaler_f_to_load, scaler_t_to_load = load_specific_scalers(sffu, stfu)
                if model_to_load and scaler_f_to_load and scaler_t_to_load: config_to_load = temp_cfg
                else: load_error_sidebar = True; model_load_placeholder.error("Errore caricamento file manuali.")
            else: model_load_placeholder.warning("Carica .pth, 2 .joblib e seleziona target.")
    else: # Modello pre-addestrato
        model_info = available_models_dict[selected_model_display_name]; model_base_name = model_info["config_name"]
        st.session_state.active_model_name = selected_model_display_name; model_load_placeholder.info(f"Carico: **{selected_model_display_name}**")
        config_to_load = load_model_config(model_info["config_path"])
        if config_to_load:
            config_to_load.update({"pth_path": model_info["pth_path"], "scaler_features_path": model_info["scaler_features_path"], "scaler_targets_path": model_info["scaler_targets_path"], "name": model_base_name})
            if "feature_columns" not in config_to_load: config_to_load["feature_columns"] = st.session_state.feature_columns
            model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
            scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])
            if not (model_to_load and scaler_f_to_load and scaler_t_to_load): load_error_sidebar = True; config_to_load = None; model_load_placeholder.error("Errore caricamento modello/scaler.")
        else: load_error_sidebar = True; model_load_placeholder.error("Errore caricamento config JSON.")

    if config_to_load and model_to_load and device_to_load and scaler_f_to_load and scaler_t_to_load:
        st.session_state.update({'active_config': config_to_load, 'active_model': model_to_load, 'active_device': device_to_load, 'active_scaler_features': scaler_f_to_load, 'active_scaler_targets': scaler_t_to_load})
        if selected_model_display_name != MODEL_CHOICE_UPLOAD: st.session_state.active_model_name = selected_model_display_name
        cfg = st.session_state.active_config; model_load_placeholder.success(f"Modello '{st.session_state.active_model_name}' ATTIVO (In:{cfg['input_window']}h, Out:{cfg['output_window']}h)")
    elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]: model_load_placeholder.error("Caricamento modello fallito.")
    elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not st.session_state.active_model: model_load_placeholder.info("Completa caricamento manuale.")

    st.divider(); st.header('Menu Navigazione')
    model_ready = st.session_state.active_model is not None and st.session_state.active_config is not None
    data_ready = df_current is not None
    radio_opts = ['Dashboard', 'Simulazione', 'Analisi', 'Training']
    radio_caps = ["No Dati/Modello" if not (data_ready and model_ready) else "Visualizza", "No Modello" if not model_ready else "Simula", "No Dati" if not data_ready else "Esplora", "No Dati" if not data_ready else "Allena"]
    page = st.radio('Scegli', options=radio_opts, captions=radio_caps, horizontal=True, label_visibility="collapsed")
    st.divider(); st.caption('App Idrologica LSTM © 2024')

# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================
active_cfg = st.session_state.active_config; active_mdl = st.session_state.active_model; active_dev = st.session_state.active_device
active_scf = st.session_state.active_scaler_features; active_sct = st.session_state.active_scaler_targets
df_curr = st.session_state.get('df', None); feat_cols_curr = active_cfg.get("feature_columns", st.session_state.feature_columns) if active_cfg else st.session_state.feature_columns
date_col = st.session_state.date_col_name_csv; model_ok = active_mdl is not None and active_cfg is not None; data_ok = df_curr is not None

if page == 'Dashboard':
    st.header('Dashboard Idrologica')
    if not (model_ok and data_ok): st.warning("⚠️ Seleziona Modello e carica Dati (CSV).")
    else:
        iw, ow, tc = active_cfg["input_window"], active_cfg["output_window"], active_cfg["target_columns"]
        st.info(f"Modello: **{st.session_state.active_model_name}** (In:{iw}h, Out:{ow}h)"); st.caption(f"Target: {', '.join(tc)}")
        try:
            last_d = df_curr.iloc[-1]; last_t = last_d[date_col]; c1,c2,c3 = st.columns([1,2,2])
            c1.metric("Ultimo Dato", last_t.strftime('%d/%m %H:%M'))
            c2.write('**Livelli [m]**'); c2.dataframe(last_d[tc].round(3).to_frame("V"), use_container_width=True)
            rain_f=[c for c in feat_cols_curr if 'Cumulata' in c]; hum_f=[c for c in feat_cols_curr if 'Umidita' in c]
            if rain_f: c3.write('**Pioggia [mm]**'); c3.dataframe(last_d[rain_f].round(2).to_frame("V"), use_container_width=True)
            if hum_f: c3.metric(f"**{hum_f[0].split('(')[0]}**", f"{last_d[hum_f[0]]:.1f}%")
            st.divider(); st.subheader('Previsione Ultimi Dati')
            if st.button('Genera previsione', type="primary", key="dash_pred"):
                with st.spinner('Previsione...'):
                    latest = df_curr.iloc[-iw:][feat_cols_curr].values
                    if latest.shape[0] < iw: st.error(f"Dati insuff. ({latest.shape[0]}/{iw}).")
                    else:
                        preds = predict(active_mdl, latest, active_scf, active_sct, active_cfg, active_dev)
                        if preds is not None:
                            pred_ts = [last_t + timedelta(hours=i+1) for i in range(ow)]; df_r = pd.DataFrame(preds, columns=tc); df_r.insert(0, 'Ora Prev', [t.strftime('%d/%m %H:%M') for t in pred_ts])
                            st.dataframe(df_r.round(3)); st.markdown(get_table_download_link(df_r, f"prev_{last_t.strftime('%y%m%d%H%M')}.csv"), unsafe_allow_html=True)
                            st.subheader('Grafici'); figs = plot_predictions(preds, active_cfg, last_t)
                            for i, fig in enumerate(figs): s_n=re.sub(r'[^a-zA-Z0-9_]','_',tc[i].split('(')[-1][:-1].strip()); st.plotly_chart(fig,use_container_width=True); st.markdown(get_plotly_download_link(fig,f"graf_{s_n}_{last_t.strftime('%y%m%d%H%M')}"),unsafe_allow_html=True)
        except Exception as e: st.error(f"Errore Dashboard: {e}\n{traceback.format_exc()}")

elif page == 'Simulazione':
    st.header('Simulazione Idrologica')
    if not model_ok: st.warning("⚠️ Seleziona Modello per Simulare.")
    else:
        iw, ow, tc = active_cfg["input_window"], active_cfg["output_window"], active_cfg["target_columns"]
        st.info(f"Modello: **{st.session_state.active_model_name}** (Input:{iw}h, Output:{ow}h)"); st.caption(f"Target: {', '.join(tc)}")
        sim_input = None; sim_m = st.radio("Prep. Dati:", ['Manuale Costante', 'Google Sheet', 'Orario Dettagliato'], horizontal=True)
        if sim_m == 'Manuale Costante':
            st.subheader(f'Valori costanti per {iw} ore'); temp_v={}; c1,c2,c3=st.columns(3); rain_f=[f for f in feat_cols_curr if 'Cumulata' in f]; hum_f=[f for f in feat_cols_curr if 'Umidita' in f]; hyd_f=[f for f in feat_cols_curr if 'Livello' in f]
            with c1: st.write("**Pioggia**"); [temp_v.update({f: st.number_input(f"{f.split('(')[0]}", 0.0, value=round(df_curr[f].median() if data_ok and f in df_curr else 0.0,1), step=0.5, format="%.1f", key=f"m_{i}")}) for i,f in enumerate(rain_f)]
            with c2: st.write("**Umidità**"); [temp_v.update({f: st.number_input(f"{f.split('(')[0]}", 0.0, 100.0, value=round(df_curr[f].median() if data_ok and f in df_curr else 70.0,1), step=1.0, format="%.1f", key=f"m_{len(rain_f)+i}")}) for i,f in enumerate(hum_f)]
            with c3: st.write("**Livelli**"); [temp_v.update({f: st.number_input(f"{f.split('[')[0]}", -2.0, 15.0, value=round(df_curr[f].median() if data_ok and f in df_curr else 0.5,2), step=0.05, format="%.2f", key=f"m_{len(rain_f)+len(hum_f)+i}")}) for i,f in enumerate(hyd_f)]
            try: sim_input = np.column_stack([np.repeat(temp_v[f], iw) for f in feat_cols_curr]); st.success("Dati costanti pronti.")
            except KeyError as e: st.error(f"Errore input: feature '{e}' mancante."); sim_input = None
        elif sim_m == 'Google Sheet':
            st.subheader(f'Importa {iw} ore da GSheet'); url_gs = st.text_input("URL Foglio Google", "https://docs.google.com/spreadsheets/d/...")
            map_gs = { 'Data_Ora': date_col, 'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', # etc. (DA PERSONALIZZARE!)
                      'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)'}
            exp_gs_cols=list(map_gs.keys()); date_gs=map_gs.get('Data_Ora','Data_Ora') # Usa nome mappato o default
            hum_m=next((f for f in feat_cols_curr if 'Umidita' in f),None); hum_map=any(mc==hum_m for mc in map_gs.values()) if hum_m else True; hum_man=None
            if hum_m and not hum_map: hum_man=st.number_input(f"Umidità (%) per '{hum_m}'", 0.0, 100.0, 75.0, 1.0, "%.1f")
            if st.button("Importa da GSheet"):
                sid=extract_sheet_id(url_gs);
                if not sid: st.error("URL GSheet non valido.")
                else:
                    with st.spinner("Importo GSheet..."):
                        gs_df = import_data_from_sheet(sid, exp_gs_cols, iw, date_gs)
                        if gs_df is not None:
                            map_df=pd.DataFrame(); ok_map=True
                            for gs_c, mdl_c in map_gs.items():
                                if gs_c in gs_df.columns: map_df[mdl_c] = gs_df[gs_c]
                            if hum_m and not hum_map and hum_man is not None: map_df[hum_m]=hum_man
                            elif hum_m and not hum_map: st.error(f"Umidità '{hum_m}' non fornita."); ok_map=False
                            if ok_map:
                                miss = [c for c in feat_cols_curr if c not in map_df.columns]
                                if miss: st.error(f"Errore GSheet Map: mancano {miss}")
                                else:
                                     try: sim_input=map_df[feat_cols_curr].values; st.session_state.update({'imported_sim_data':sim_input, 'imported_sim_df_preview':map_df}); st.success(f"Dati GSheet pronti ({len(sim_input)}r).")
                                     except Exception as e: st.error(f"Errore GSheet: {e}")
            if 'imported_sim_data' in st.session_state:
                 st.subheader("Anteprima GSheet"); prev_c = [date_col]+feat_cols_curr; prev_c=[c for c in prev_c if c in st.session_state.imported_sim_df_preview.columns]; st.dataframe(st.session_state.imported_sim_df_preview[prev_c].tail().round(3)); sim_input=st.session_state.imported_sim_data
        elif sim_m == 'Orario Dettagliato':
             st.subheader(f'Inserisci dati per {iw} ore'); skey=f"sim_hr_{iw}"
             if skey not in st.session_state: init_v={c:(df_curr[c].median() if data_ok and c in df_curr else 0.0) for c in feat_cols_curr}; st.session_state[skey]=pd.DataFrame(np.repeat([list(init_v.values())], iw, axis=0), columns=feat_cols_curr, index=[f"T-{iw-i}" for i in range(iw)])
             edited_df=st.data_editor(st.session_state[skey], height=(iw+1)*35+3, use_container_width=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in feat_cols_curr})
             if len(edited_df)!=iw: st.error(f"Tabella deve avere {iw} righe."); sim_input=None
             else:
                  try: sim_input=edited_df[feat_cols_curr].values; st.session_state[skey]=edited_df; st.success("Dati orari pronti.")
                  except Exception as e: st.error(f"Errore tabella: {e}"); sim_input=None
        st.divider(); run_sim = st.button('Esegui simulazione', type="primary", disabled=(sim_input is None), key="sim_run")
        if run_sim and sim_input is not None:
             if sim_input.shape[0]!=iw or sim_input.shape[1]!=len(feat_cols_curr): st.error(f"Errore shape input sim.")
             else:
                  with st.spinner('Simulazione...'):
                       preds=predict(active_mdl, sim_input, active_scf, active_sct, active_cfg, active_dev)
                       if preds is not None:
                           st.subheader(f'Risultato Simulazione ({ow} ore)'); now_s=datetime.now(); pred_t_s=[now_s+timedelta(hours=i+1) for i in range(ow)]
                           df_r_s=pd.DataFrame(preds, columns=tc); df_r_s.insert(0,'Ora Prev',[t.strftime('%d/%m %H:%M') for t in pred_t_s])
                           st.dataframe(df_r_s.round(3)); st.markdown(get_table_download_link(df_r_s,f"sim_{now_s.strftime('%y%m%d%H%M')}.csv"),unsafe_allow_html=True)
                           st.subheader('Grafici Sim'); figs_s=plot_predictions(preds, active_cfg, now_s)
                           for i, fig in enumerate(figs_s): s_n=re.sub(r'[^a-zA-Z0-9_]','_',tc[i].split('(')[-1][:-1].strip()); st.plotly_chart(fig,use_container_width=True); st.markdown(get_plotly_download_link(fig,f"graf_sim_{s_n}_{now_s.strftime('%y%m%d%H%M')}"),unsafe_allow_html=True)
        elif run_sim: st.error("Dati input simulazione non pronti.")

elif page == 'Analisi':
    st.header('Analisi Dati Storici')
    if not data_ok: st.warning("⚠️ Carica i Dati Storici (CSV).")
    else:
        st.info(f"Dataset: {len(df_curr)} righe ({df_curr[date_col].min().strftime('%d/%m/%y')} - {df_curr[date_col].max().strftime('%d/%m/%y')})")
        min_d, max_d = df_curr[date_col].min().date(), df_curr[date_col].max().date(); c1,c2=st.columns(2); sd=c1.date_input('Da:',min_d,min_d,max_d); ed=c2.date_input('A:',max_d,min_d,max_d)
        if sd > ed: st.error("Data inizio > Data fine.")
        else:
            mask = (df_curr[date_col].dt.date >= sd) & (df_curr[date_col].dt.date <= ed); df_filt = df_curr.loc[mask]
            if len(df_filt)==0: st.warning("Nessun dato nel periodo.")
            else:
                 st.success(f"{len(df_filt)} record ({sd.strftime('%d/%m/%y')}-{ed.strftime('%d/%m/%y')})."); t1,t2,t3=st.tabs(["📈 Andamento","📊 Stats","🔗 Corr"]); tc_curr=active_cfg["target_columns"] if active_cfg else []
                 with t1: feat_p=st.multiselect('Features',feat_cols_curr, default=tc_curr if tc_curr else feat_cols_curr[:1],key="an_fplot"); if feat_p: fig=go.Figure();[fig.add_trace(go.Scatter(x=df_filt[date_col],y=df_filt[f],mode='lines',name=f)) for f in feat_p]; fig.update_layout(title='Andamento',xaxis_title='Data',height=500,hovermode="x unified"); st.plotly_chart(fig,use_container_width=True)
                 with t2: feat_s=st.selectbox('Feature',feat_cols_curr,index=feat_cols_curr.index(tc_curr[0]) if tc_curr and tc_curr[0] in feat_cols_curr else 0, key="an_fstat"); st.write(f"**Stats: {feat_s}**"); st.dataframe(df_filt[feat_s].describe().round(3)); st.write(f"**Distribuzione: {feat_s}**"); fig_h=go.Figure(data=[go.Histogram(x=df_filt[feat_s])]); mv=df_filt[feat_s].mean(); fig_h.add_vline(x=mv,line_dash="dash",line_color="red",annotation_text=f"μ:{mv:.2f}"); fig_h.update_layout(title=f'Distribuzione {feat_s}',xaxis_title='Valore',yaxis_title='Freq.',height=400); st.plotly_chart(fig_h,use_container_width=True)
                 with t3: def_c=tc_curr+[f for f in feat_cols_curr if 'Cumulata' in f][:1] if tc_curr else feat_cols_curr[:min(5,len(feat_cols_curr))]; feat_c=st.multiselect('Feat Corr',feat_cols_curr,default=def_c,key="an_fcorr"); if len(feat_c)>1: corr_m=df_filt[feat_c].corr(); fig_hm=go.Figure(data=go.Heatmap(z=corr_m.values,x=corr_m.columns,y=corr_m.columns,colorscale='RdBu',zmin=-1,zmax=1,text=corr_m.round(2).values,texttemplate="%{text}")); fig_hm.update_layout(title='Correlazione',height=max(400,len(feat_c)*50)); st.plotly_chart(fig_hm,use_container_width=True)
                 st.divider(); st.subheader('Download Filtrati'); st.markdown(get_table_download_link(df_filt,f"filtrati_{sd.strftime('%y%m%d')}_{ed.strftime('%y%m%d')}.csv"),unsafe_allow_html=True)

elif page == 'Training':
    st.header('Allenamento Nuovo Modello LSTM')
    df_train = st.session_state.get('df', None)
    if not data_ok: st.warning("⚠️ Carica i Dati Storici (CSV).")
    else:
        st.success(f"Dati disponibili: {len(df_curr)} righe."); st.subheader('Configurazione'); save_name = st.text_input("Nome base file modello", value=f"modello_{datetime.now().strftime('%y%m%d_%H%M')}")
        st.write("**1. Target:**"); targets_train = []; hydro_f = [c for c in feat_cols_curr if 'Livello' in c]; cols_t = st.columns(min(len(hydro_f), 6))
        for i, f in enumerate(hydro_f):
            with cols_t[i % len(cols_t)]:
                 if st.checkbox(f.split('(')[-1][:-1].strip(), value=(f in active_cfg["target_columns"] if active_cfg else False), key=f"t_{i}"): targets_train.append(f)
        st.write("**2. Parametri:**");
        with st.expander("Parametri Modello & Training", expanded=True):
             c1,c2,c3=st.columns(3); iw=c1.number_input("In Win(h)", 6, 168, (active_cfg["input_window"] if active_cfg else 24), 6, key="t_in"); ow=c1.number_input("Out Win(h)", 1, 72, (active_cfg["output_window"] if active_cfg else 12), 1, key="t_out"); vs=c1.slider("% Val", 0, 40, 20, 1, key="t_val", help="0% per no validation")
             hs=c2.number_input("Hidden Size", 16, 1024, (active_cfg["hidden_size"] if active_cfg else 128), 16, key="t_hidden"); nl=c2.number_input("Layers", 1, 8, (active_cfg["num_layers"] if active_cfg else 2), 1, key="t_layers"); do=c2.slider("Dropout", 0.0, 0.7, (active_cfg["dropout"] if active_cfg else 0.2), 0.05, key="t_dropout")
             lr=c3.number_input("Learn Rate", 1e-5, 1e-2, 0.001, format="%.5f", step=1e-4, key="t_lr"); bs=c3.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_batch"); ep=c3.number_input("Epoche", 5, 500, 50, 5, key="t_epochs")
        st.write("**3. Avvia:**"); valid_n=bool(save_name and re.match(r'^[a-zA-Z0-9_-]+$', save_name)); valid_t=bool(targets_train); ready=valid_n and valid_t
        if not valid_t: st.warning("Seleziona target."); if not valid_n: st.warning("Inserisci nome valido.")
        train_btn = st.button("Addestra Nuovo Modello", type="primary", disabled=not ready, key="train_run")

        if train_btn:
             st.info(f"Avvio training '{save_name}'..."); train_ph=st.empty()
             with train_ph.container():
                 with st.spinner('Prep dati...'):
                      Xt,yt,Xv,yv,scf,sct = prepare_training_data(df_curr.copy(), feat_cols_curr, targets_train, iw, ow, vs)
                      if Xt is None: st.error("Prep dati fallita."); st.stop()
                 st.subheader("Training..."); istr, ostr = len(feat_cols_curr), len(targets_train)
                 try: model_t, loss_t, loss_v = train_model(Xt, yt, Xv, yv, istr, ostr, ow, hs, nl, ep, bs, lr, do)
                 except Exception as e: st.error(f"Errore training: {e}\n{traceback.format_exc()}"); model_t = None
                 if model_t:
                     st.success("Training OK!")
                     m_b, c_b, sf_b, st_b = io.BytesIO(), io.StringIO(), io.BytesIO(), io.BytesIO()
                     cfg_s = {"input_window":iw,"output_window":ow,"hidden_size":hs,"num_layers":nl,"dropout":do,"target_columns":targets_train,"feature_columns":feat_cols_curr,"training_date":datetime.now().isoformat(),"final_val_loss":min(loss_v) if loss_v and any(v!=0 for v in loss_v) else None,"display_name":save_name}
                     try:
                         torch.save(model_t.state_dict(), m_b); json.dump(cfg_s, c_b, indent=4); joblib.dump(scf, sf_b); joblib.dump(sct, st_b)
                         st.subheader("Download File"); st.info("Salva e carica i file su GitHub ('models'/).")
                         c1,c2,c3,c4 = st.columns(4)
                         c1.markdown(get_binary_file_download_link(m_b, f"{save_name}.pth", "⬇️ .pth"), unsafe_allow_html=True)
                         cfg_str=c_b.getvalue(); b64_j=base64.b64encode(cfg_str.encode('utf-8')).decode(); href_j=f'<a href="data:application/json;base64,{b64_j}" download="{save_name}.json">⬇️ .json</a>'
                         c2.markdown(href_j, unsafe_allow_html=True)
                         c3.markdown(get_binary_file_download_link(sf_b, f"{save_name}_features.joblib", "⬇️ Feat.joblib"), unsafe_allow_html=True)
                         c4.markdown(get_binary_file_download_link(st_b, f"{save_name}_targets.joblib", "⬇️ Targ.joblib"), unsafe_allow_html=True)
                         try: # Salva opzionale su disco
                              os.makedirs(MODELS_DIR,exist_ok=True); bp=os.path.join(MODELS_DIR,save_name)
                              with open(f"{bp}.pth",'wb') as f:f.write(m_b.getvalue()); with open(f"{bp}.json",'w') as f:f.write(cfg_str)
                              with open(f"{bp}_features.joblib",'wb') as f:f.write(sf_b.getvalue()); with open(f"{bp}_targets.joblib",'wb') as f:f.write(st_b.getvalue())
                              st.caption(f"File salvati anche in '{MODELS_DIR}/'.")
                         except Exception as e_dsk: st.warning(f"Errore salvataggio disco: {e_dsk}")
                     except Exception as e_dl: st.error(f"Errore prep download: {e_dl}\n{traceback.format_exc()}")
