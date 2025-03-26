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
# Rimosse costanti globali INPUT_WINDOW, OUTPUT_WINDOW

# --- Definizioni Funzioni Core ML (Dataset, LSTM) ---

# Dataset personalizzato per le serie temporali
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modello LSTM per serie temporali
class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
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
# --- Funzioni Utilità Modello/Dati ---

# ... (dopo la definizione di HydroLSTM e prima di load_model_config) ...

# Funzione per preparare i dati per l'addestramento
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, val_split=20):
    """
    Prepara i dati per il training: crea sequenze, normalizza e splitta.
    Args:
        df (pd.DataFrame): DataFrame con i dati storici.
        feature_columns (list): Lista nomi colonne input.
        target_columns (list): Lista nomi colonne output (target).
        input_window (int): Lunghezza sequenza input.
        output_window (int): Lunghezza sequenza output.
        val_split (int): Percentuale dati per validazione (dal fondo).

    Returns:
        tuple: (X_train, y_train, X_val, y_val, scaler_features, scaler_targets) o None se errore.
    """
    st.write(f"Prepare Data: Input Window={input_window}, Output Window={output_window}, Val Split={val_split}%")
    st.write(f"Prepare Data: Feature Cols ({len(feature_columns)}): {', '.join(feature_columns[:3])}...")
    st.write(f"Prepare Data: Target Cols ({len(target_columns)}): {', '.join(target_columns)}")

    # Assicurati che le colonne siano numeriche (già fatto nel caricamento df, ma doppia sicurezza)
    try:
        for col in feature_columns + target_columns:
            if col not in df.columns:
                 raise ValueError(f"Colonna '{col}' richiesta per training non trovata nel DataFrame.")
            # La conversione numerica e gestione NaN è già fatta nel blocco caricamento df
            # df[col] = pd.to_numeric(df[col], errors='coerce')
        # df = df.fillna(method='ffill').fillna(method='bfill') # Già fatto
        if df[feature_columns + target_columns].isnull().sum().sum() > 0:
             st.warning("NaN residui rilevati prima della creazione sequenze! Controlla caricamento dati.")
             # Potresti decidere di fermare qui o procedere con cautela
             # return None, None, None, None, None, None

    except ValueError as e:
        st.error(f"Errore colonne in prepare_training_data: {e}")
        return None, None, None, None, None, None

    # Creazione delle sequenze di input (X) e output (y)
    X, y = [], []
    total_len = len(df)
    required_len = input_window + output_window

    if total_len < required_len:
         st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe per input+output).")
         return None, None, None, None, None, None

    st.write(f"Creazione sequenze da {total_len - required_len + 1} punti possibili...")
    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_window][feature_columns].values)
        y.append(df.iloc[i + input_window : i + required_len][target_columns].values)

    if not X or not y:
        st.error("Errore: Nessuna sequenza X/y creata. Controlla finestre e lunghezza dati.")
        return None, None, None, None, None, None

    X = np.array(X, dtype=np.float32) # Specifica dtype per evitare problemi dopo
    y = np.array(y, dtype=np.float32)
    st.write(f"Sequenze create: X shape={X.shape}, y shape={y.shape}") # Log shape

    # Normalizzazione dei dati
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    # Verifica dimensioni prima del reshape
    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima della normalizzazione.")
        return None, None, None, None, None, None

    # Il reshape deve avvenire prima del fit_transform
    num_sequences, seq_len_in, num_features = X.shape
    num_sequences_y, seq_len_out, num_targets = y.shape # y ha già 3 dimensioni

    X_flat = X.reshape(-1, num_features)
    y_flat = y.reshape(-1, num_targets) # y è (num_seq, output_window, num_targets), quindi reshape a (num_seq*output_window, num_targets)

    st.write(f"Shape per scaling: X_flat={X_flat.shape}, y_flat={y_flat.shape}") # Log shape

    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except Exception as e_scale:
         st.error(f"Errore durante scaling: {e_scale}")
         st.error(f"NaN in X_flat: {np.isnan(X_flat).sum()}, NaN in y_flat: {np.isnan(y_flat).sum()}")
         return None, None, None, None, None, None


    # Riporta alla forma originale 3D
    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets)
    st.write(f"Shape post scaling: X_scaled={X_scaled.shape}, y_scaled={y_scaled.shape}") # Log shape


    # Divisione in set di addestramento e validazione
    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    if split_idx == 0 or split_idx == len(X_scaled):
         st.warning(f"Split indice ({split_idx}) non valido per divisione train/val. Controlla val_split e lunghezza dati.")
         # Potresti usare tutto per training o dare errore
         if len(X_scaled) < 2: # Non si può splittare
              st.error("Dataset troppo piccolo per creare set di validazione.")
              return None, None, None, None, None, None
         # Fallback: usa almeno 1 campione per validazione se possibile
         split_idx = max(1, len(X_scaled) - 1) if split_idx == len(X_scaled) else split_idx
         split_idx = min(len(X_scaled) - 1, split_idx) if split_idx == 0 else split_idx


    X_train = X_scaled[:split_idx]
    y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    y_val = y_scaled[split_idx:]

    st.write(f"Split Dati: Train={len(X_train)}, Validation={len(X_val)}")

    # Controlla che i set non siano vuoti
    if X_train.size == 0 or y_train.size == 0:
         st.error("Set di Training vuoto dopo lo split.")
         return None, None, None, None, None, None
    # Il set di validazione può essere vuoto se val_split è 0 o quasi
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set di Validazione vuoto dopo lo split.")
         # Imposta a array vuoti con shape corretta se necessario per evitare errori dopo
         X_val = np.empty((0, seq_len_in, num_features), dtype=np.float32)
         y_val = np.empty((0, seq_len_out, num_targets), dtype=np.float32)


    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

# ... (resto delle funzioni di utilità come load_model_config, load_specific_model, etc.)

@st.cache_data # Cache basata sugli argomenti (percorsi file)
def load_model_config(_config_path): # Aggiunto underscore per evitare conflitti nome cache
    """Carica la configurazione JSON di un modello."""
    try:
        with open(_config_path, 'r') as f:
            config = json.load(f)
        # Validazione base della config
        required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        # feature_columns è opzionale nella config, verrà preso dal default se non presente
        if not all(key in config for key in required_keys):
            st.error(f"File config '{_config_path}' incompleto. Mancano chiavi.")
            return None
        return config
    except FileNotFoundError:
        st.error(f"File config '{_config_path}' non trovato.")
        return None
    except json.JSONDecodeError:
        st.error(f"Errore parsing JSON '{_config_path}'.")
        return None
    except Exception as e:
        st.error(f"Errore caricamento config '{_config_path}': {e}")
        return None

# Cache per caricare il modello specifico
@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config): # Aggiunto underscore per evitare conflitti nome cache
    """Carica un modello .pth dati il percorso e la sua configurazione."""
    if not config:
        st.error("Configurazione non valida per caricare il modello.")
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Assumi feature_columns sia nella config o usa un default vuoto se non c'è
        feature_columns_model = config.get("feature_columns", [])
        if not feature_columns_model:
             st.warning(f"Chiave 'feature_columns' non trovata nella config del modello '{config.get('name', 'N/A')}'. La dimensione input sarà inferita dallo scaler.")
             # L'inferenza dallo scaler avverrà in un secondo momento o causerà errore se non matchano
             # Per creare il modello serve comunque una dimensione input. Se non c'è, usiamo un valore placeholder
             # MA questo è rischioso. È FONDAMENTALE che feature_columns sia nella config o definita globalmente.
             # Usiamo le feature globali da session_state come fallback robusto
             feature_columns_model = st.session_state.get("feature_columns", [])
             if not feature_columns_model:
                   st.error("Impossibile determinare input_size: 'feature_columns' non in config né in session_state.")
                   return None, None

        input_size_model = len(feature_columns_model)

        model = HydroLSTM(
            input_size=input_size_model,
            hidden_size=config["hidden_size"],
            output_size=len(config["target_columns"]),
            output_window=config["output_window"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        ).to(device)

        # Caricamento pesi
        if isinstance(_model_path, str):
             if not os.path.exists(_model_path):
                  raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'):
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else:
             raise TypeError("Percorso modello non valido.")

        model.eval()
        st.success(f"Modello '{config.get('name', os.path.basename(str(_model_path)))}' caricato su {device}.")
        return model, device
    except FileNotFoundError as e:
         st.error(f"Errore caricamento modello: {e}")
         return None, None
    except Exception as e:
        st.error(f"Errore caricamento pesi modello '{config.get('name', os.path.basename(str(_model_path)))}': {e}")
        st.error(traceback.format_exc())
        return None, None

# Cache per caricare gli scaler specifici
@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(_scaler_features_path, _scaler_targets_path): # Aggiunto underscore
    """Carica gli scaler .joblib dati i percorsi."""
    scaler_features = None
    scaler_targets = None
    try:
        def _load_joblib(path):
             if isinstance(path, str):
                  if not os.path.exists(path):
                       raise FileNotFoundError(f"File scaler '{path}' non trovato.")
                  return joblib.load(path)
             elif hasattr(path, 'getvalue'):
                  path.seek(0)
                  return joblib.load(path)
             else:
                  raise TypeError("Percorso scaler non valido.")

        scaler_features = _load_joblib(_scaler_features_path)
        scaler_targets = _load_joblib(_scaler_targets_path)

        st.success(f"Scaler caricati.")
        return scaler_features, scaler_targets
    except FileNotFoundError as e:
         st.error(f"Errore caricamento scaler: {e}")
         return None, None
    except Exception as e:
        st.error(f"Errore caricamento scaler: {e}")
        return None, None

# Funzione per trovare modelli nella cartella
def find_available_models(models_dir=MODELS_DIR):
    """Scansiona la cartella dei modelli e restituisce un dizionario di modelli validi."""
    available = {}
    if not os.path.isdir(models_dir):
        # Non mostrare warning se la cartella semplicemente non esiste
        # st.warning(f"Cartella modelli '{models_dir}' non trovata.")
        return available

    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth_path in pth_files:
        base_name = os.path.splitext(os.path.basename(pth_path))[0]
        config_path = os.path.join(models_dir, f"{base_name}.json")
        scaler_f_path = os.path.join(models_dir, f"{base_name}_features.joblib")
        scaler_t_path = os.path.join(models_dir, f"{base_name}_targets.joblib")

        if os.path.exists(config_path) and os.path.exists(scaler_f_path) and os.path.exists(scaler_t_path):
            # Tenta di leggere la config per avere un nome più descrittivo (opzionale)
            try:
                 with open(config_path, 'r') as f_cfg:
                      cfg_data = json.load(f_cfg)
                      display_name = cfg_data.get("display_name", base_name) # Usa display_name se presente
                      # Potresti aggiungere info su finestre nel nome:
                      # display_name = f"{base_name} (In:{cfg_data['input_window']} Out:{cfg_data['output_window']})"
            except:
                 display_name = base_name # Fallback al nome base

            available[display_name] = { # Usa display_name come chiave
                "config_name": base_name, # Salva il nome base originale
                "pth_path": pth_path,
                "config_path": config_path,
                "scaler_features_path": scaler_f_path,
                "scaler_targets_path": scaler_t_path
            }
        else:
            st.warning(f"Modello '{base_name}' ignorato: file mancanti (json o joblib).")
    return available

# --- Funzione Predict Modificata ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict: Modello, scaler o config mancanti.")
        return None

    input_window = config["input_window"]
    output_window = config["output_window"]
    target_columns = config["target_columns"]
    feature_columns_cfg = config.get("feature_columns", [])

    # Validazione input_data shape
    if input_data.shape[0] != input_window:
         st.error(f"Predict: Input righe {input_data.shape[0]} != Finestra Modello {input_window}.")
         return None
    if feature_columns_cfg and input_data.shape[1] != len(feature_columns_cfg):
         st.error(f"Predict: Input colonne {input_data.shape[1]} != Features Modello {len(feature_columns_cfg)}.")
         return None
    # Controllo vs scaler (se feature_columns non erano in config)
    if not feature_columns_cfg and hasattr(scaler_features, 'n_features_in_') and scaler_features.n_features_in_ != input_data.shape[1]:
        st.error(f"Predict: Input colonne {input_data.shape[1]} != Scaler Features {scaler_features.n_features_in_}.")
        return None

    model.eval()
    try:
        input_normalized = scaler_features.transform(input_data)
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        output_np = output.cpu().numpy().reshape(output_window, len(target_columns))

        if not hasattr(scaler_targets, 'n_features_in_'):
             st.error("Predict: Scaler targets non sembra fittato (manca n_features_in_).")
             return None
        if scaler_targets.n_features_in_ != len(target_columns):
             st.error(f"Predict: Output targets {len(target_columns)} != Scaler Targets {scaler_targets.n_features_in_}.")
             return None

        predictions = scaler_targets.inverse_transform(output_np)
        return predictions

    except Exception as e:
        st.error(f"Errore durante predict: {e}")
        st.error(traceback.format_exc())
        return None

# --- Funzione Plot Modificata ---
def plot_predictions(predictions, config, start_time=None):
    if config is None or predictions is None:
        return []

    output_window = config["output_window"]
    target_columns = config["target_columns"]
    figures = []
    for i, sensor_name in enumerate(target_columns):
        fig = go.Figure()
        if start_time:
            hours = [start_time + timedelta(hours=h+1) for h in range(output_window)]
            x_axis, x_title = hours, "Data e Ora Previste"
        else:
            hours = np.arange(1, output_window + 1)
            x_axis, x_title = hours, "Ore Future"

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Prev. {sensor_name}'))
        fig.update_layout(
            title=f'Previsione - {sensor_name}',
            xaxis_title=x_title, yaxis_title='Livello idrometrico [m]', height=400, hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        figures.append(fig)
    return figures

# --- Import Data from Sheet (Modificata per Input Window) ---
def import_data_from_sheet(sheet_id, expected_cols, input_window, date_col_name='Data_Ora', date_format='%d/%m/%Y %H:%M'):
    """Importa dati da Google Sheet, pulisce e restituisce le ultime `input_window` righe valide."""
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets:
             st.error("Credenziali Google non trovate nei secrets.")
             return None
        credentials = Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"],
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        data = worksheet.get_all_values()

        if not data or len(data) < 2:
            st.error("Foglio Google vuoto o solo intestazione.")
            return None

        headers = data[0]
        rows = data[1:]
        actual_headers_set = set(headers)
        missing_cols = [col for col in expected_cols if col not in actual_headers_set]
        if missing_cols:
             st.error(f"Colonne GSheet mancanti: {', '.join(missing_cols)}")
             return None

        df_sheet = pd.DataFrame(rows, columns=headers)
        # Seleziona solo colonne attese per evitare errori dopo
        columns_to_keep = [col for col in expected_cols if col in df_sheet.columns]
        if date_col_name not in columns_to_keep and date_col_name in df_sheet.columns:
             columns_to_keep.append(date_col_name)
        elif date_col_name not in df_sheet.columns:
             st.error(f"Colonna data GSheet '{date_col_name}' non trovata.")
             return None
        df_sheet = df_sheet[columns_to_keep]


        # Pulizia e Conversione
        df_sheet[date_col_name] = pd.to_datetime(df_sheet[date_col_name], format=date_format, errors='coerce')
        df_sheet = df_sheet.dropna(subset=[date_col_name])
        df_sheet = df_sheet.sort_values(by=date_col_name, ascending=True)

        numeric_cols = [col for col in df_sheet.columns if col != date_col_name]
        for col in numeric_cols:
            df_sheet[col] = df_sheet[col].replace(['N/A', '', '-', ' '], np.nan, regex=False)
            df_sheet[col] = df_sheet[col].astype(str).str.replace(',', '.', regex=False)
            df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')

        # Prendi ultime `input_window` righe
        df_sheet = df_sheet.tail(input_window)

        if len(df_sheet) < input_window:
             st.warning(f"GSheet: Trovate solo {len(df_sheet)} righe valide ({input_window} richieste).")
             if len(df_sheet) == 0: return None # Errore se 0 righe
        elif len(df_sheet) > input_window: # Non dovrebbe succedere
             df_sheet = df_sheet.tail(input_window)

        st.success(f"Importate e pulite {len(df_sheet)} righe da GSheet.")
        return df_sheet

    except gspread.exceptions.APIError as api_e:
        error_details = api_e.response.json()
        error_message = error_details.get('error', {}).get('message', str(api_e))
        st.error(f"Errore API Google Sheets: {error_message}.")
        return None
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Foglio Google non trovato (ID: '{sheet_id}').")
        return None
    except ValueError as ve:
         if "time data" in str(ve) and "does not match format" in str(ve):
             st.error(f"Errore formato data GSheet: {ve}. Atteso: '{date_format}'.")
         else:
             st.error(f"Errore conversione dati GSheet: {ve}")
         return None
    except Exception as e:
        st.error(f"Errore importazione GSheet: {type(e).__name__} - {e}")
        st.error(traceback.format_exc())
        return None

# --- Funzione Allenamento Modificata (Corretta e con Salvataggio Config) ---
# Definizione corretta della funzione
def train_model(
    X_train, y_train, X_val, y_val, input_size, output_size, output_window,
    hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state_dict = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()

    def update_loss_chart(train_losses, val_losses, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, mode='lines+markers', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines+markers', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss (Train vs Validation)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=400)
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        if len(val_loader) > 0: val_loss /= len(val_loader)
        else: val_loss = 0
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        progress_percentage = (epoch + 1) / epochs
        progress_bar.progress(progress_percentage)
        current_lr = optimizer.param_groups[0]['lr']
        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}')
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            # st.text(f"Nuovo miglior modello (epoca {epoch+1}), Val Loss: {best_val_loss:.6f}")

        time.sleep(0.05)

    if best_model_state_dict:
        best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
        model.load_state_dict(best_model_state_dict_on_device)
        st.success(f"Caricato modello migliore (Val Loss: {best_val_loss:.6f})")
    else:
        st.warning("Nessun miglioramento, usato modello ultima epoca.")

    return model, train_losses, val_losses

# --- Funzioni Helper Download ---
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False, sep=';', decimal=',')
    b64 = base64.b64encode(csv.encode('utf-8')).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Scarica CSV</a>'

def get_binary_file_download_link(file_object, filename, text):
    file_object.seek(0)
    b64 = base64.b64encode(file_object.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    buffer_html = io.StringIO()
    fig.write_html(buffer_html)
    b64_html = base64.b64encode(buffer_html.getvalue().encode()).decode()
    href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'
    href_png = ""
    try:
        buffer_png = io.BytesIO()
        fig.write_image(buffer_png, format="png") # Richiede kaleido
        buffer_png.seek(0)
        b64_png = base64.b64encode(buffer_png.getvalue()).decode()
        href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
    except Exception: pass # Ignora se kaleido non c'è
    return f"{href_html} {href_png}"

# --- Funzione Estrazione ID GSheet ---
def extract_sheet_id(url):
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- Inizializzazione Session State ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scaler_features' not in st.session_state: st.session_state.active_scaler_features = None
if 'active_scaler_targets' not in st.session_state: st.session_state.active_scaler_targets = None
if 'df' not in st.session_state: st.session_state.df = None # Per salvare il df caricato

# Definisci feature_columns globalmente o leggile da una config/modello di default
# È FONDAMENTALE che siano definite prima del caricamento dati o del caricamento modello
if 'feature_columns' not in st.session_state:
     # Definisci qui l'elenco COMPLETO delle feature che i tuoi modelli potrebbero usare
     # Se diversi modelli usano diverse feature, questo DEVE essere specificato nel .json
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
# Definisci anche la colonna data globale
if 'date_col_name_csv' not in st.session_state:
     st.session_state.date_col_name_csv = 'Data e Ora'


# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('Dashboard Modello Predittivo Idrologico')

# --- Sidebar ---
st.sidebar.header('Impostazioni Dati e Modello')

# --- Caricamento Dati Storici con Feedback Migliorato ---
st.sidebar.write("--- Caricamento Dati Storici ---")
uploaded_data_file = st.sidebar.file_uploader(
    'File CSV Dati Storici (Opzionale: sovrascrive default)',
    type=['csv'],
    help=f"Se non caricato, verrà usato '{DEFAULT_DATA_PATH}' se presente.",
    key="data_uploader"
)

if uploaded_data_file is not None:
    st.sidebar.write(f"File selezionato: `{uploaded_data_file.name}` ({uploaded_data_file.size} bytes)")
else:
    st.sidebar.write("Nessun file CSV caricato manualmente.")

# Tenta caricamento df (da upload o default)
df = None
df_load_error = None
data_source_info = ""
data_path_to_load = None
is_uploaded = False

if uploaded_data_file is not None:
    data_path_to_load = uploaded_data_file
    is_uploaded = True
    data_source_info = f"File caricato: **{uploaded_data_file.name}**"
    st.sidebar.write(f"-> Tentativo di caricare da: {data_source_info}")
else:
    st.sidebar.write("-> Nessun file caricato, controllo default...")
    if os.path.exists(DEFAULT_DATA_PATH):
        data_path_to_load = DEFAULT_DATA_PATH
        is_uploaded = False
        data_source_info = f"File di default: **{DEFAULT_DATA_PATH}**"
        st.sidebar.write(f"-> Trovato default: {data_source_info}")
    else:
        df_load_error = f"File default '{DEFAULT_DATA_PATH}' non trovato. Carica un file CSV."
        data_source_info = "Nessun dato disponibile."
        st.sidebar.warning(f"-> Default non trovato: {df_load_error}")

if data_path_to_load:
    st.sidebar.write(f"-> Inizio lettura e processamento da: {data_source_info}")
    try:
        if is_uploaded:
            data_path_to_load.seek(0)
            df = pd.read_csv(data_path_to_load, sep=';', decimal=',', encoding='utf-8', low_memory=False) # Aggiunto low_memory=False
        else:
            df = pd.read_csv(data_path_to_load, sep=';', decimal=',', encoding='utf-8', low_memory=False)

        st.sidebar.write(f"-> CSV letto, {len(df)} righe iniziali.")

        # Pulizia e Validazione Dati
        date_col_name_csv = st.session_state.date_col_name_csv
        if date_col_name_csv not in df.columns:
             raise ValueError(f"Colonna data '{date_col_name_csv}' mancante.")

        try:
             df[date_col_name_csv] = pd.to_datetime(df[date_col_name_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
             st.sidebar.warning("Formato data standard fallito, provo inferenza...")
             df[date_col_name_csv] = pd.to_datetime(df[date_col_name_csv], errors='coerce')

        df = df.dropna(subset=[date_col_name_csv])
        st.sidebar.write(f"-> Dopo dropna date: {len(df)} righe.")
        df = df.sort_values(by=date_col_name_csv).reset_index(drop=True)

        # Verifica feature_columns
        if 'feature_columns' not in st.session_state or not st.session_state.feature_columns:
             raise ValueError("Definizione 'feature_columns' mancante in session_state.")
        current_feature_columns = st.session_state.feature_columns

        missing_features = [col for col in current_feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Colonne feature mancanti nel CSV: {', '.join(missing_features)}")

        st.sidebar.write("-> Colonne data e feature verificate.")

        # Conversione numerica e gestione NaN
        for col in current_feature_columns:
              if col != date_col_name_csv:
                    if df[col].dtype == 'object':
                         # Tentativo più robusto di pulizia stringhe prima della conversione
                         df[col] = df[col].astype(str).str.replace('.', '', regex=False) # Rimuove punti migliaia
                         df[col] = df[col].astype(str).str.replace(',', '.', regex=False) # Virgola -> Punto decimali
                         # Rimuovi spazi bianchi extra
                         df[col] = df[col].astype(str).str.strip()
                         # Gestisci specificamente N/A e stringhe vuote
                         df[col] = df[col].replace(['N/A', '', '-'], np.nan, regex=False)
                    # Ora converti, forzando errori a NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        n_nan_before = df[current_feature_columns].isnull().sum().sum()
        if n_nan_before > 0:
              st.sidebar.warning(f"{n_nan_before} NaN/valori non numerici trovati. Eseguito ffill/bfill.")
              df[current_feature_columns] = df[current_feature_columns].fillna(method='ffill').fillna(method='bfill')
              n_nan_after = df[current_feature_columns].isnull().sum().sum()
              if n_nan_after > 0:
                   raise ValueError(f"Ancora {n_nan_after} NaN dopo fill. Controlla inizio/fine dati.")

        # Successo
        st.session_state.df = df # Salva df in session_state
        st.sidebar.success(f"Dati caricati e validati ({data_source_info}): {len(df)} righe.")

    except Exception as e:
        df = None
        st.session_state.df = None
        df_load_error = f'Errore caricamento/processamento CSV ({data_source_info}): {type(e).__name__} - {e}'
        st.sidebar.error(f"-> ERRORE DATI: {df_load_error}")
        print(f"--- ERRORE CARICAMENTO DATI ({datetime.now()}) ---")
        traceback.print_exc()
        print("-----------------------------------------")

# Recupera df da session_state alla fine del blocco di caricamento
df = st.session_state.get('df', None)

# Mostra errore finale se necessario
if df is None and df_load_error:
    st.sidebar.error(f"Caricamento dati fallito: {df_load_error}")
elif df is None and not data_path_to_load: # Caso in cui né upload né default trovato
     pass

# --- Selezione Modello ---
st.sidebar.divider()
st.sidebar.header("Selezione Modello Predittivo")

available_models_dict = find_available_models(MODELS_DIR)
model_display_names = list(available_models_dict.keys()) # Nomi visualizzati

MODEL_CHOICE_UPLOAD = "Carica File Manualmente..."
MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"

selection_options = [MODEL_CHOICE_NONE] + model_display_names + [MODEL_CHOICE_UPLOAD]
current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
# Trova l'indice corretto basato sul nome visualizzato
try:
    current_index = selection_options.index(current_selection_name)
except ValueError:
    current_index = 0 # Default a Nessuno se il nome salvato non è più valido

selected_model_display_name = st.sidebar.selectbox(
    "Modello per Previsione/Simulazione:",
    selection_options,
    index=current_index,
    help="Scegli un modello pre-addestrato dalla cartella 'models' o carica i file manualmente."
)

# --- Caricamento Modello/Scaler in base alla selezione ---
config_to_load = None
model_to_load = None
device_to_load = None
scaler_f_to_load = None
scaler_t_to_load = None
load_error_sidebar = False
model_base_name_loaded = None # Nome file senza estensione

# Reset stato attivo prima del caricamento
st.session_state.active_model_name = None
st.session_state.active_config = None
st.session_state.active_model = None
st.session_state.active_device = None
st.session_state.active_scaler_features = None
st.session_state.active_scaler_targets = None

if selected_model_display_name == MODEL_CHOICE_NONE:
    st.sidebar.info("Nessun modello selezionato.")
    st.session_state.active_model_name = MODEL_CHOICE_NONE

elif selected_model_display_name == MODEL_CHOICE_UPLOAD:
    st.session_state.active_model_name = MODEL_CHOICE_UPLOAD # Segna la scelta
    with st.sidebar.expander("Carica File Modello", expanded=True):
        model_file_up = st.file_uploader('File Modello (.pth)', type=['pth'], key="up_pth")
        scaler_features_file_up = st.file_uploader('File Scaler Features (.joblib)', type=['joblib'], key="up_scf")
        scaler_targets_file_up = st.file_uploader('File Scaler Target (.joblib)', type=['joblib'], key="up_sct")

        st.subheader("Configurazione Modello Caricato")
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            input_window_up = st.number_input("Input Window (ore)", min_value=1, value=24, key="up_in")
            output_window_up = st.number_input("Output Window (ore)", min_value=1, value=12, key="up_out")
        with col_cfg2:
            hidden_size_up = st.number_input("Hidden Size", min_value=16, value=128, step=16, key="up_hidden")
            num_layers_up = st.number_input("Num Layers", min_value=1, value=2, key="up_layers")
            dropout_up = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="up_dropout")

        hydro_features_global = [col for col in st.session_state.feature_columns if 'Livello' in col]
        target_columns_up = st.multiselect(
             "Target Columns previste da questo modello", hydro_features_global, default=hydro_features_global, key="up_targets"
        )

        if model_file_up and scaler_features_file_up and scaler_targets_file_up and target_columns_up:
            st.info("Tentativo caricamento file manuali...")
            temp_config = {
                "input_window": input_window_up, "output_window": output_window_up,
                "hidden_size": hidden_size_up, "num_layers": num_layers_up, "dropout": dropout_up,
                "target_columns": target_columns_up,
                "feature_columns": st.session_state.feature_columns, # Usa feature globali
                "name": "uploaded_model"
            }
            model_to_load, device_to_load = load_specific_model(model_file_up, temp_config)
            scaler_f_to_load, scaler_t_to_load = load_specific_scalers(scaler_features_file_up, scaler_targets_file_up)

            if model_to_load and scaler_f_to_load and scaler_t_to_load:
                 config_to_load = temp_config
                 model_base_name_loaded = "uploaded_model" # Nome generico
            else: load_error_sidebar = True
        else:
             st.warning("Carica .pth, 2 .joblib e seleziona i target.")
             # Non è un errore fatale, l'utente sta ancora configurando

else: # Modello pre-addestrato selezionato
    model_info = available_models_dict[selected_model_display_name]
    model_base_name_loaded = model_info["config_name"] # Nome base file
    st.session_state.active_model_name = selected_model_display_name # Salva nome visualizzato
    st.sidebar.write(f"Caricamento modello: **{selected_model_display_name}**")

    config_to_load = load_model_config(model_info["config_path"])
    if config_to_load:
        # Aggiungi info percorso/nome alla config
        config_to_load["pth_path"] = model_info["pth_path"]
        config_to_load["scaler_features_path"] = model_info["scaler_features_path"]
        config_to_load["scaler_targets_path"] = model_info["scaler_targets_path"]
        config_to_load["name"] = model_base_name_loaded
         # Assicura che feature columns siano in config, altrimenti usa globali
        if "feature_columns" not in config_to_load:
             st.warning(f"'feature_columns' non trovate in {model_info['config_path']}, uso quelle globali.")
             config_to_load["feature_columns"] = st.session_state.feature_columns
        elif set(config_to_load["feature_columns"]) != set(st.session_state.feature_columns):
             st.warning(f"Le 'feature_columns' in {model_info['config_path']} differiscono da quelle globali!")
             # Potresti voler aggiornare st.session_state.feature_columns qui o gestire la discrepanza

        # Carica modello e scaler
        model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
        scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])

        if not (model_to_load and scaler_f_to_load and scaler_t_to_load):
            load_error_sidebar = True
            config_to_load = None
    else:
        load_error_sidebar = True

# Aggiorna session state alla fine
if config_to_load and model_to_load and device_to_load and scaler_f_to_load and scaler_t_to_load:
    st.session_state.active_config = config_to_load
    st.session_state.active_model = model_to_load
    st.session_state.active_device = device_to_load
    st.session_state.active_scaler_features = scaler_f_to_load
    st.session_state.active_scaler_targets = scaler_t_to_load
    # Aggiorna nome attivo solo se non era upload
    if selected_model_display_name != MODEL_CHOICE_UPLOAD:
         st.session_state.active_model_name = selected_model_display_name

# Messaggio di stato finale caricamento modello
if st.session_state.active_model and st.session_state.active_config:
    cfg = st.session_state.active_config
    st.sidebar.success(f"Modello '{st.session_state.active_model_name}' ATTIVO (In:{cfg['input_window']}h, Out:{cfg['output_window']}h)")
elif load_error_sidebar and selected_model_display_name not in [MODEL_CHOICE_NONE, MODEL_CHOICE_UPLOAD]:
     st.sidebar.error("Caricamento modello/config/scaler fallito.")
elif selected_model_display_name == MODEL_CHOICE_UPLOAD and not st.session_state.active_model:
     st.sidebar.info("Completa il caricamento manuale dei file.")


# --- Menu Navigazione ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
# Leggi lo stato attivo per abilitare/disabilitare opzioni
model_ready = st.session_state.active_model is not None and st.session_state.active_config is not None
data_ready = df is not None

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
radio_captions = [
    "Richiede Dati & Modello" if not (data_ready and model_ready) else "Visualizza dati e previsioni",
    "Richiede Modello" if not model_ready else "Esegui previsioni custom",
    "Richiede Dati" if not data_ready else "Esplora dati caricati",
    "Richiede Dati" if not data_ready else "Allena un nuovo modello"
]
# Disabilita opzioni se i requisiti non sono soddisfatti (Streamlit > 1.18)
# disabled_options = [
#     not (data_ready and model_ready), # Dashboard
#     not model_ready,                  # Simulazione
#     not data_ready,                   # Analisi
#     not data_ready                    # Allenamento
# ]

page = st.sidebar.radio(
    'Scegli una funzionalità',
    options=radio_options,
    captions=radio_captions
    # disabled=disabled_options # Attiva se usi Streamlit >= 1.18
)

# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================

# Leggi stato attivo all'inizio per le pagine
active_config = st.session_state.active_config
active_model = st.session_state.active_model
active_device = st.session_state.active_device
active_scaler_features = st.session_state.active_scaler_features
active_scaler_targets = st.session_state.active_scaler_targets
# Usa df da session_state per consistenza
df_current = st.session_state.get('df', None)
# Feature columns globali o specifiche del modello se presenti
feature_columns_current = active_config.get("feature_columns", st.session_state.feature_columns) if active_config else st.session_state.feature_columns
date_col_name_csv = st.session_state.date_col_name_csv

# --- PAGINA DASHBOARD ---
if page == 'Dashboard':
    st.header('Dashboard Idrologica')
    if not (model_ready and data_ready):
        st.warning("⚠️ Seleziona un Modello attivo e carica i Dati Storici (CSV) per usare il Dashboard.")
    else:
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns = active_config["target_columns"]

        st.info(f"Modello attivo: **{st.session_state.active_model_name}** (In: {input_window}h, Out: {output_window}h)")
        st.caption(f"Target previsti: {', '.join(target_columns)}")

        st.subheader('Ultimi dati disponibili')
        try:
            last_data = df_current.iloc[-1]
            last_date = last_data[date_col_name_csv]
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(label="Ultimo Rilevamento", value=last_date.strftime('%d/%m/%Y %H:%M'))
            with col2:
                st.write('**Livelli Idrometrici [m]**')
                st.dataframe(last_data[target_columns].round(3).to_frame(name="Valore"), use_container_width=True)
            with col3:
                rain_features_dash = [col for col in feature_columns_current if 'Cumulata' in col]
                st.write('**Precipitazioni [mm]**')
                if rain_features_dash: st.dataframe(last_data[rain_features_dash].round(2).to_frame(name="Valore"), use_container_width=True)
                humidity_feature_dash = [col for col in feature_columns_current if 'Umidita' in col]
                if humidity_feature_dash:
                     st.write('**Umidità [%]**')
                     st.metric(label=humidity_feature_dash[0].split('(')[0].strip(), value=f"{last_data[humidity_feature_dash[0]]:.1f}%")

            st.header('Previsione basata sugli ultimi dati')
            if st.button('Genera previsione', type="primary", key="dash_predict"):
                with st.spinner('Generazione previsione...'):
                    latest_data = df_current.iloc[-input_window:][feature_columns_current].values
                    if latest_data.shape[0] < input_window:
                        st.error(f"Dati insufficienti ({latest_data.shape[0]}) per finestra input ({input_window}).")
                    else:
                        predictions = predict(active_model, latest_data, active_scaler_features, active_scaler_targets, active_config, active_device)
                        if predictions is not None:
                            st.subheader(f'Previsione per le prossime {output_window} ore')
                            pred_times = [last_date + timedelta(hours=i+1) for i in range(output_window)]
                            results_df = pd.DataFrame(predictions, columns=target_columns)
                            results_df.insert(0, 'Ora previsione', [t.strftime('%d/%m %H:%M') for t in pred_times])
                            st.dataframe(results_df.round(3))
                            st.markdown(get_table_download_link(results_df, f"previsione_{last_date.strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                            st.subheader('Grafici Previsioni')
                            figs = plot_predictions(predictions, active_config, last_date)
                            for i, fig in enumerate(figs):
                                s_name = target_columns[i].split('(')[-1].replace(')','').replace('/','_').strip()
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown(get_plotly_download_link(fig, f"grafico_{s_name}_{last_date.strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
        except IndexError:
             st.error("Errore nel leggere l'ultimo dato dal DataFrame.")
        except KeyError as ke:
             st.error(f"Errore: Colonna '{ke}' non trovata nei dati caricati o nella configurazione.")
        except Exception as e:
             st.error(f"Errore imprevisto nel Dashboard: {e}")
             st.error(traceback.format_exc())


# --- PAGINA SIMULAZIONE ---
elif page == 'Simulazione':
    st.header('Simulazione Idrologica')
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo per usare la Simulazione.")
    else:
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns = active_config["target_columns"]

        st.info(f"Simulazione con: **{st.session_state.active_model_name}** (Input Richiesto: {input_window}h, Previsione: {output_window}h)")
        st.caption(f"Target previsti: {', '.join(target_columns)}")

        sim_data_input = None
        sim_method = st.radio(
            "Metodo preparazione dati simulazione",
            ['Manuale Costante', 'Importa da Google Sheet', 'Orario Dettagliato (Avanzato)']
        )

        # --- Simulazione: Manuale Costante ---
        if sim_method == 'Manuale Costante':
            st.subheader(f'Inserisci valori costanti per {input_window} ore')
            temp_sim_values = {}
            cols_manual = st.columns(3)
            with cols_manual[0]:
                 st.write("**Pioggia (mm)**")
                 for feature in [f for f in feature_columns_current if 'Cumulata' in f]:
                      default_val = df_current[feature].median() if df_current is not None and feature in df_current else 0.0
                      temp_sim_values[feature] = st.number_input(f'{feature.split("(")[0]}', min_value=0.0, value=round(default_val,1), step=0.5, format="%.1f")
            with cols_manual[1]:
                 st.write("**Umidità (%)**")
                 for feature in [f for f in feature_columns_current if 'Umidita' in f]:
                      default_val = df_current[feature].median() if df_current is not None and feature in df_current else 70.0
                      temp_sim_values[feature] = st.number_input(f'{feature.split("(")[0]}', min_value=0.0, max_value=100.0, value=round(default_val,1), step=1.0, format="%.1f")
            with cols_manual[2]:
                 st.write("**Livelli (m)**")
                 for feature in [f for f in feature_columns_current if 'Livello' in f]:
                      default_val = df_current[feature].median() if df_current is not None and feature in df_current else 0.5
                      temp_sim_values[feature] = st.number_input(f'{feature.split("[")[0]}', min_value=-2.0, max_value=15.0, value=round(default_val,2), step=0.05, format="%.2f")

            # Crea array numpy
            sim_data_list = []
            try:
                for feature in feature_columns_current: # Mantieni ordine corretto
                     sim_data_list.append(np.repeat(temp_sim_values[feature], input_window))
                sim_data_input = np.column_stack(sim_data_list)
                st.success("Dati costanti pronti.")
            except KeyError as ke:
                 st.error(f"Errore: Feature '{ke}' non trovata negli input manuali. Verifica la lista `feature_columns_current`.")
                 sim_data_input = None

        # --- Simulazione: Google Sheet ---
        elif sim_method == 'Importa da Google Sheet':
            st.subheader(f'Importa ultime {input_window} ore da Google Sheet')
            sheet_url = st.text_input("URL Foglio Google", "https://docs.google.com/spreadsheets/d/...")

            # Mappatura colonne GSheet -> Modello (DA PERSONALIZZARE!)
            column_mapping_gsheet_to_model = {
                'Data_Ora': date_col_name_csv, # Mappa sempre la data
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
                # Aggiungi la tua colonna umidità GSheet -> Modello qui se esiste
                # 'NomeColonnaUmiditaGSheet': 'Umidita\' Sensore 3452 (Montemurello)'
            }
            expected_google_sheet_cols = list(column_mapping_gsheet_to_model.keys())
            date_col_name_gsheet = 'Data_Ora' # Nome colonna data nel GSheet

            # Gestione Umidità manuale se non mappata
            humidity_col_in_model = next((f for f in feature_columns_current if 'Umidita' in f), None)
            humidity_mapped = False
            if humidity_col_in_model:
                 humidity_mapped = any(model_col == humidity_col_in_model for model_col in column_mapping_gsheet_to_model.values())

            selected_humidity_gsheet = None
            if humidity_col_in_model and not humidity_mapped:
                 st.warning(f"Colonna Umidità '{humidity_col_in_model}' non mappata da GSheet.")
                 selected_humidity_gsheet = st.number_input(
                     f"Inserisci umidità (%) costante per '{humidity_col_in_model}'",
                     min_value=0.0, max_value=100.0, value=75.0, step=1.0, format="%.1f"
                 )
            elif humidity_col_in_model and humidity_mapped:
                 st.info(f"Umidità '{humidity_col_in_model}' verrà letta da GSheet.")

            if st.button("Importa e Prepara da Google Sheet"):
                sheet_id = extract_sheet_id(sheet_url)
                if not sheet_id: st.error("URL GSheet non valido.")
                else:
                    st.info(f"Tentativo connessione GSheet ID: {sheet_id}")
                    with st.spinner("Importazione e pulizia dati GSheet..."):
                        sheet_data_cleaned = import_data_from_sheet(
                            sheet_id, expected_google_sheet_cols, input_window,
                            date_col_name=date_col_name_gsheet
                        )
                        if sheet_data_cleaned is not None:
                            mapped_data = pd.DataFrame()
                            successful_mapping = True
                            for gsheet_col, model_col in column_mapping_gsheet_to_model.items():
                                if gsheet_col in sheet_data_cleaned.columns:
                                    mapped_data[model_col] = sheet_data_cleaned[gsheet_col]
                                # Non serve else, import_data_from_sheet verifica le colonne

                            # Aggiungi umidità manuale se necessario
                            if humidity_col_in_model and not humidity_mapped and selected_humidity_gsheet is not None:
                                 mapped_data[humidity_col_in_model] = selected_humidity_gsheet
                            elif humidity_col_in_model and not humidity_mapped and selected_humidity_gsheet is None:
                                 st.error(f"Umidità '{humidity_col_in_model}' non mappata né inserita manualmente.")
                                 successful_mapping = False

                            if successful_mapping:
                                missing_model_features = [col for col in feature_columns_current if col not in mapped_data.columns]
                                if missing_model_features:
                                     st.error(f"Errore mappatura GSheet: mancano colonne modello: {', '.join(missing_model_features)}")
                                else:
                                     try:
                                          sim_data_input = mapped_data[feature_columns_current].values
                                          # Salva per evitare ricaricamenti
                                          st.session_state.imported_sim_data = sim_data_input
                                          st.session_state.imported_sim_df_preview = mapped_data
                                          st.success(f"Dati GSheet importati e mappati ({sim_data_input.shape[0]} righe).")
                                     except Exception as e_map:
                                          st.error(f"Errore finale creazione dati da GSheet: {e_map}")

            # Mostra anteprima se importato
            if 'imported_sim_data' in st.session_state:
                 st.subheader("Anteprima Dati da GSheet (ultime righe)")
                 preview_cols = [date_col_name_csv] + feature_columns_current
                 preview_cols = [col for col in preview_cols if col in st.session_state.imported_sim_df_preview.columns]
                 st.dataframe(st.session_state.imported_sim_df_preview[preview_cols].tail().round(3))
                 sim_data_input = st.session_state.imported_sim_data

        # --- Simulazione: Orario Dettagliato ---
        elif sim_method == 'Orario Dettagliato (Avanzato)':
            st.subheader(f'Inserisci dati per {input_window} ore precedenti')
            # Usa st.data_editor (come prima, ma assicurati gestisca input_window)
            # Inizializza/Recupera da session_state, gestendo cambio input_window
            session_key_hourly = f"sim_hourly_data_{input_window}" # Chiave dinamica
            if session_key_hourly not in st.session_state:
                 init_values = {}
                 for col in feature_columns_current:
                      init_values[col] = df_current[col].median() if df_current is not None and col in df_current else 0.0
                 st.session_state[session_key_hourly] = pd.DataFrame(
                     np.repeat([list(init_values.values())], input_window, axis=0),
                     columns=feature_columns_current,
                     index=[f"T-{input_window-i}" for i in range(input_window)]
                 )

            edited_df = st.data_editor(
                st.session_state[session_key_hourly],
                # num_rows="dynamic", # Attenzione: rende difficile mantenere input_window esatto
                height=(input_window + 1) * 35 + 3, # Altezza dinamica approx
                use_container_width=True,
                column_config={col: st.column_config.NumberColumn(format="%.3f") for col in feature_columns_current}
            )

            if len(edited_df) != input_window:
                 st.error(f"Tabella deve avere esattamente {input_window} righe (attuali: {len(edited_df)}). Modifica o resetta.")
                 sim_data_input = None
            else:
                 try:
                      sim_data_input = edited_df[feature_columns_current].values
                      st.session_state[session_key_hourly] = edited_df # Salva modifiche
                      st.success("Dati orari pronti.")
                 except KeyError as ke:
                      st.error(f"Errore colonne tabella: {ke}")
                      sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE ---
        st.divider()
        run_simulation = st.button('Esegui simulazione', type="primary", disabled=(sim_data_input is None), key="sim_run")

        if run_simulation and sim_data_input is not None:
             if sim_data_input.shape[0] != input_window or sim_data_input.shape[1] != len(feature_columns_current):
                  st.error(f"Errore shape input simulazione. Atteso:({input_window}, {len(feature_columns_current)}), Ottenuto:{sim_data_input.shape}")
             else:
                  with st.spinner('Simulazione in corso...'):
                       predictions_sim = predict(active_model, sim_data_input, active_scaler_features, active_scaler_targets, active_config, active_device)
                       if predictions_sim is not None:
                           st.subheader(f'Risultato Simulazione: Previsione per {output_window} ore')
                           current_time_sim = datetime.now()
                           pred_times_sim = [current_time_sim + timedelta(hours=i+1) for i in range(output_window)]
                           results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns)
                           results_df_sim.insert(0, 'Ora previsione', [t.strftime('%d/%m %H:%M') for t in pred_times_sim])
                           st.dataframe(results_df_sim.round(3))
                           st.markdown(get_table_download_link(results_df_sim, f"simulazione_{current_time_sim.strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                           st.subheader('Grafici Previsioni Simulate')
                           figs_sim = plot_predictions(predictions_sim, active_config, current_time_sim)
                           for i, fig_sim in enumerate(figs_sim):
                               s_name = target_columns[i].split('(')[-1].replace(')','').replace('/','_').strip()
                               st.plotly_chart(fig_sim, use_container_width=True)
                               st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_{s_name}_{current_time_sim.strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
        elif run_simulation and sim_data_input is None:
             st.error("Dati input simulazione non pronti o non validi.")


# --- PAGINA ANALISI DATI STORICI ---
elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')
    if not data_ready:
        st.warning("⚠️ Carica i Dati Storici (CSV) per usare l'Analisi.")
    else:
        st.info(f"Dataset caricato: {len(df_current)} righe, dal {df_current[date_col_name_csv].min().strftime('%d/%m/%Y')} al {df_current[date_col_name_csv].max().strftime('%d/%m/%Y')}")
        # ... (Logica selezione date e filtraggio come prima, usando df_current e date_col_name_csv) ...
        min_date = df_current[date_col_name_csv].min().date()
        max_date = df_current[date_col_name_csv].max().date()
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input('Data inizio', min_date, min_value=min_date, max_value=max_date)
        with col2: end_date = st.date_input('Data fine', max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date: st.error("Data inizio > Data fine.")
        else:
            mask = (df_current[date_col_name_csv].dt.date >= start_date) & (df_current[date_col_name_csv].dt.date <= end_date)
            filtered_df = df_current.loc[mask]
            if len(filtered_df) == 0: st.warning("Nessun dato nel periodo selezionato.")
            else:
                 st.success(f"Trovati {len(filtered_df)} record ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}).")
                 # ... (Logica Tabs con Plotly per Andamento, Statistiche, Correlazione come prima, usando filtered_df e feature_columns_current) ...
                 tab1, tab2, tab3 = st.tabs(["Andamento Temporale", "Statistiche/Distribuzione", "Correlazione"])
                 with tab1:
                      # ... (codice multiselect e plot plotly andamento) ...
                      pass # Mantieni codice precedente qui
                 with tab2:
                      # ... (codice selectbox, describe e plot plotly istogramma) ...
                      pass # Mantieni codice precedente qui
                 with tab3:
                      # ... (codice multiselect, heatmap e scatter plot plotly correlazione) ...
                      pass # Mantieni codice precedente qui
                 st.divider()
                 st.subheader('Download Dati Filtrati')
                 st.markdown(get_table_download_link(filtered_df, f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)

# --- PAGINA ALLENAMENTO MODELLO ---
elif page == 'Allenamento Modello':
    st.header('Allenamento Nuovo Modello LSTM')

    # DEBUG e Condizione iniziale
    df_training_page = st.session_state.get('df', None)
    st.write(f"<small>DEBUG (Training Page Entry): `df` is None: {df_training_page is None}</small>", unsafe_allow_html=True)
    if df_training_page is not None:
        st.write(f"<small>DEBUG (Training Page Entry): `df` shape: {df_training_page.shape}</small>", unsafe_allow_html=True)

    if df_training_page is None:
        st.warning("⚠️ Dati storici non caricati o caricamento fallito. Carica un file CSV valido.")
    else:
        st.success(f"Dati disponibili per l'addestramento: {len(df_training_page)} righe.")
        st.subheader('Configurazione Addestramento')

        save_model_name = st.text_input("Nome base per salvare il modello (es. 'modello_24_12_v1')", value=f"modello_{datetime.now().strftime('%Y%m%d_%H%M')}")

        st.write("**1. Seleziona Target:**")
        selected_targets_train = []
        hydro_features_train = [col for col in feature_columns_current if 'Livello' in col]
        cols_targets = st.columns(min(len(hydro_features_train), 5)) # Max 5 colonne
        for i, feature in enumerate(hydro_features_train):
            with cols_targets[i % len(cols_targets)]:
                 label = feature.split('(')[-1].replace(')','').strip()
                 if st.checkbox(label, value=(feature in active_config["target_columns"] if active_config else False), key=f"target_train_{i}"):
                     selected_targets_train.append(feature)

        st.write("**2. Imposta Parametri:**")
        with st.expander("Parametri Modello e Training", expanded=True):
             col1_train, col2_train, col3_train = st.columns(3)
             with col1_train:
                 input_window_train = st.number_input("Input Window (ore)", 6, 168, (active_config["input_window"] if active_config else 24), 6, key="t_in")
                 output_window_train = st.number_input("Output Window (ore)", 1, 72, (active_config["output_window"] if active_config else 12), 1, key="t_out")
                 val_split_train = st.slider("% Validazione", 5, 40, 20, 1, key="t_val")
             with col2_train:
                 hidden_size_train_cfg = st.number_input("Hidden Size", 16, 1024, (active_config["hidden_size"] if active_config else 128), 16, key="t_hidden")
                 num_layers_train_cfg = st.number_input("Num Layers", 1, 8, (active_config["num_layers"] if active_config else 2), 1, key="t_layers")
                 dropout_train_cfg = st.slider("Dropout", 0.0, 0.7, (active_config["dropout"] if active_config else 0.2), 0.05, key="t_dropout")
             with col3_train:
                 learning_rate_train = st.number_input("Learning Rate", 1e-5, 1e-2, 0.001, format="%.5f", step=1e-4, key="t_lr")
                 batch_size_train = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_batch")
                 epochs_train = st.number_input("Epoche", 5, 500, 50, 5, key="t_epochs")

        st.write("**3. Avvia Addestramento:**")
        # Validazione nome e targets prima di mostrare il bottone
        valid_name = bool(save_model_name and re.match(r'^[a-zA-Z0-9_-]+$', save_model_name))
        valid_targets = bool(selected_targets_train)
        ready_to_train = valid_name and valid_targets

        if not valid_targets: st.warning("Seleziona almeno un idrometro target.")
        if not valid_name: st.warning("Inserisci un nome valido per il modello (lettere, numeri, -, _).")

        train_button = st.button("Addestra Nuovo Modello", type="primary", disabled=not ready_to_train, key="train_run")

        if train_button and ready_to_train:
             st.info(f"Avvio addestramento per '{save_model_name}'...")
             # Preparazione Dati
             with st.spinner('Preparazione dati...'):
                  X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train = prepare_training_data(
                      df_training_page.copy(), feature_columns_current, selected_targets_train,
                      input_window_train, output_window_train, val_split_train
                  )
                  if X_train is None:
                       st.error("Preparazione dati fallita.")
                       st.stop()
                  st.success(f"Dati pronti: {len(X_train)} train, {len(X_val)} val.")

             # Addestramento
             st.subheader("Addestramento in corso...")
             input_size_train = len(feature_columns_current)
             output_size_train = len(selected_targets_train)
             try:
                 trained_model, train_losses, val_losses = train_model(
                     X_train, y_train, X_val, y_val,
                     input_size_train, output_size_train, output_window_train,
                     hidden_size_train_cfg, num_layers_train_cfg, epochs_train,
                     batch_size_train, learning_rate_train, dropout_train_cfg
                 )
             except Exception as e_train:
                  st.error(f"Errore durante addestramento: {e_train}")
                  st.error(traceback.format_exc())
                  st.stop()

             # Salvataggio Risultati
             if trained_model:
                 st.success("Addestramento completato!")
                 # Grafico Loss Finale (statico)
                 # ... (codice matplotlib per grafico loss finale) ...

                 st.subheader("Salvataggio Modello, Configurazione e Scaler")
                 os.makedirs(MODELS_DIR, exist_ok=True)
                 base_path = os.path.join(MODELS_DIR, save_model_name)
                 model_save_path = f"{base_path}.pth"
                 config_save_path = f"{base_path}.json"
                 scaler_f_save_path = f"{base_path}_features.joblib"
                 scaler_t_save_path = f"{base_path}_targets.joblib"

                 config_to_save = {
                     "input_window": input_window_train, "output_window": output_window_train,
                     "hidden_size": hidden_size_train_cfg, "num_layers": num_layers_train_cfg,
                     "dropout": dropout_train_cfg,
                     "target_columns": selected_targets_train,
                     "feature_columns": feature_columns_current, # Salva feature usate
                     "training_date": datetime.now().isoformat(),
                     "final_val_loss": min(val_losses) if val_losses else None,
                     # Aggiungi nome display se diverso dal nome file
                     "display_name": save_model_name # Default a nome file
                 }

                 try:
                     torch.save(trained_model.state_dict(), model_save_path)
                     with open(config_save_path, 'w') as f: json.dump(config_to_save, f, indent=4)
                     joblib.dump(scaler_features_train, scaler_f_save_path)
                     joblib.dump(scaler_targets_train, scaler_t_save_path)
                     st.success(f"Modello '{save_model_name}' salvato in '{MODELS_DIR}/'")
                     st.caption(f"Salvati: {model_save_path}, {config_save_path}, {scaler_f_save_path}, {scaler_t_save_path}")
                     # Opzionale: fornire link download se non in repo
                 except Exception as e_save:
                      st.error(f"Errore salvataggio file: {e_save}")

                 # Test rapido (opzionale, come prima)
                 # ...

# --- Footer ---
st.sidebar.divider()
st.sidebar.info('App Idrologica LSTM © 2024')
