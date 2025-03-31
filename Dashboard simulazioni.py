import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
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
import random
import mimetypes # Per indovinare i tipi MIME per i download
from streamlit_js_eval import streamlit_js_eval # Per forzare refresh periodico
import pytz # Per gestione timezone

# --- Configurazione Pagina Streamlit ---
st.set_page_config(page_title="Modello Predittivo Idrologico", layout="wide")

# --- Costanti Globali ---
MODELS_DIR = "models"
DEFAULT_DATA_PATH = "dati_idro.csv"
GSHEET_ID = "1pQI6cKrrT-gcVAfl-9ZhUx5b3J-edZRRj6nzDcCBRcA" # ID Foglio per Dashboard e Simulazione Base
GSHEET_DATE_COL = 'Data_Ora'
GSHEET_DATE_FORMAT = '%d/%m/%Y %H:%M'
GSHEET_RELEVANT_COLS = [
    GSHEET_DATE_COL,
    'Arcevia - Pioggia Ora (mm)', 'Barbara - Pioggia Ora (mm)', 'Corinaldo - Pioggia Ora (mm)',
    'Misa - Pioggia Ora (mm)', 'Umidita\' Sensore 3452 (Montemurello)', # Assicurati che sia nel foglio!
    'Serra dei Conti - Livello Misa (mt)', 'Pianello di Ostra - Livello Misa (m)',
    'Nevola - Livello Nevola (mt)', 'Misa - Livello Misa (mt)',
    'Ponte Garibaldi - Livello Misa 2 (mt)'
]
DASHBOARD_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_HISTORY_ROWS = 48 # 24 ore (48 step da 30 min)
DEFAULT_THRESHOLDS = {
    'Arcevia - Pioggia Ora (mm)': 10.0, 'Barbara - Pioggia Ora (mm)': 10.0, 'Corinaldo - Pioggia Ora (mm)': 10.0,
    'Misa - Pioggia Ora (mm)': 10.0, 'Umidita\' Sensore 3452 (Montemurello)': 95.0,
    'Serra dei Conti - Livello Misa (mt)': 2.5, 'Pianello di Ostra - Livello Misa (m)': 3.0,
    'Nevola - Livello Nevola (mt)': 2.0, 'Misa - Livello Misa (mt)': 2.8,
    'Ponte Garibaldi - Livello Misa 2 (mt)': 4.0
}
italy_tz = pytz.timezone('Europe/Rome')
HUMIDITY_COL_NAME = "Umidita' Sensore 3452 (Montemurello)" # Colonna specifica umidità

STATION_COORDS = {
    # Coordinate stazioni (invariate, usate per etichette)
    'Arcevia - Pioggia Ora (mm)': {'lat': 43.5228, 'lon': 12.9388, 'name': 'Arcevia (Pioggia)', 'type': 'Pioggia', 'location_id': 'Arcevia'},
    'Barbara - Pioggia Ora (mm)': {'lat': 43.5808, 'lon': 13.0277, 'name': 'Barbara (Pioggia)', 'type': 'Pioggia', 'location_id': 'Barbara'},
    'Corinaldo - Pioggia Ora (mm)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Pioggia)', 'type': 'Pioggia', 'location_id': 'Corinaldo'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo'},
    'Misa - Pioggia Ora (mm)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Pioggia)', 'type': 'Pioggia', 'location_id': 'Bettolelle'},
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle'},
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti'},
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'},
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'},
    HUMIDITY_COL_NAME: {'lat': 43.6, 'lon': 13.0, 'name': 'Montemurello (Umidità)', 'type': 'Umidità', 'location_id': 'Montemurello'}
}

# --- Definizioni Classi Modello (Dataset, LSTM, Encoder, Decoder, Seq2Seq) ---

# Dataset modificato per Seq2Seq (gestisce 1 o 3 tensor)
class TimeSeriesDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        # Converti in FloatTensor qui per evitare problemi di tipo
        self.tensors = tuple(torch.tensor(t, dtype=torch.float32) for t in tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

# Modello LSTM Standard (Invariato)
class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window # Numero di *passi* da 30 min
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window) # Prevede tutti i passi insieme

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Usa output dell'ultimo step
        out = self.fc(out)
        out = out.view(out.size(0), self.output_window, self.output_size) # Reshape
        return out

# Modello Encoder LSTM (Seq2Seq)
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x) # Ignora outputs, prendi solo stati finali
        return hidden, cell

# Modello Decoder LSTM (Seq2Seq)
class DecoderLSTM(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(forecast_input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size) # Prevede 1 step alla volta

    def forward(self, x_forecast_step, hidden, cell):
        # x_forecast_step shape: (batch, 1, forecast_input_size)
        output, (hidden, cell) = self.lstm(x_forecast_step, (hidden, cell))
        prediction = self.fc(output.squeeze(1)) # Shape: (batch, output_size)
        return prediction, hidden, cell

# Modello Seq2Seq Combinato
class Seq2SeqHydro(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps
        self.device = device

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0):
        batch_size = x_past.shape[0]
        forecast_window = x_future_forecast.shape[1]
        target_output_size = self.decoder.output_size

        # Gestione forecast input più corto dell'output richiesto (padding)
        if forecast_window < self.output_window:
             missing_steps = self.output_window - forecast_window
             last_forecast_step = x_future_forecast[:, -1:, :]
             padding = last_forecast_step.repeat(1, missing_steps, 1)
             x_future_forecast = torch.cat([x_future_forecast, padding], dim=1)
             print(f"Warning: forecast_window ({forecast_window}) < output_window ({self.output_window}). Padding forecast input.")
             forecast_window = self.output_window # Aggiorna per il loop

        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :] # Primo input forecast

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell
            )
            outputs[:, t, :] = decoder_output_step
            # Input per il prossimo step: sempre dal forecast fornito (durante predizione)
            if t < self.output_window - 1:
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]

        return outputs

# --- Funzioni Utilità Modello/Dati ---

# Funzione preparazione dati LSTM standard
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, val_split=20):
    """Prepara i dati per l'allenamento del modello LSTM standard."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM Standard...")
    # Controlli preliminari
    try:
        missing_features = [col for col in feature_columns if col not in df.columns]
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_features:
            st.error(f"Errore: Feature columns mancanti nel DataFrame: {missing_features}")
            return None, None, None, None, None, None
        if missing_targets:
            st.error(f"Errore: Target columns mancanti nel DataFrame: {missing_targets}")
            return None, None, None, None, None, None
        if df[feature_columns + target_columns].isnull().any().any():
             st.warning("NaN trovati PRIMA della creazione sequenze LSTM standard. Controllare pulizia dati.")
    except Exception as e:
        st.error(f"Errore controllo colonne in prepare_training_data: {e}")
        return None, None, None, None, None, None

    X, y = [], []
    total_len = len(df)
    input_steps = input_window * 2 # Converti ore in passi da 30 min
    output_steps = output_window * 2 # output_window nel config LSTM è già #steps
    required_len = input_steps + output_steps

    if total_len < required_len:
         st.error(f"Dati LSTM insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
         return None, None, None, None, None, None

    # Creazione sequenze
    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_steps][feature_columns].values)
        y.append(df.iloc[i + input_steps : i + required_len][target_columns].values)

    if not X or not y:
        st.error("Errore creazione sequenze X/y LSTM."); return None, None, None, None, None, None
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    # Scaling
    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima di scaling LSTM."); return None, None, None, None, None, None

    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    num_sequences, seq_len_in, num_features = X.shape
    num_sequences_y, seq_len_out, num_targets = y.shape

    # Verifica dimensioni sequenze
    if seq_len_in != input_steps or seq_len_out != output_steps:
        st.error(f"Errore shape sequenze LSTM: In={seq_len_in} (atteso {input_steps}), Out={seq_len_out} (atteso {output_steps})")
        return None, None, None, None, None, None

    X_flat = X.reshape(-1, num_features); y_flat = y.reshape(-1, num_targets)
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except Exception as e_scale:
        st.error(f"Errore scaling LSTM: {e_scale}"); return None, None, None, None, None, None
    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets)

    # Split Train/Validation
    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    if split_idx <= 0 or split_idx >= len(X_scaled):
         st.warning(f"Split indice LSTM ({split_idx}) non valido. Adattamento in corso.")
         if len(X_scaled) < 2: st.error("Dataset LSTM troppo piccolo per split."); return None, None, None, None, None, None
         split_idx = max(1, min(len(X_scaled) - 1, split_idx)) # Forza indice valido

    X_train = X_scaled[:split_idx]; y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]; y_val = y_scaled[split_idx:]

    if X_train.size == 0 or y_train.size == 0: st.error("Set Training LSTM vuoto."); return None, None, None, None, None, None
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set Validazione LSTM vuoto (split 0% o pochi dati).")
         X_val = np.empty((0, seq_len_in, num_features), dtype=np.float32) # Crea vuoto con shape corretta
         y_val = np.empty((0, seq_len_out, num_targets), dtype=np.float32)

    print(f"Dati LSTM pronti: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

# Funzione preparazione dati Seq2Seq
def prepare_training_data_seq2seq(df, past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps, val_split=20):
    """Prepara dati per modello Seq2Seq con 3 scaler separati."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq...")
    print(f"Input Steps: {input_window_steps}, Forecast Steps: {forecast_window_steps}, Output Steps: {output_window_steps}")

    all_needed_cols = list(set(past_feature_cols + forecast_feature_cols + target_cols))
    try:
        for col in all_needed_cols:
            if col not in df.columns: raise ValueError(f"Colonna '{col}' richiesta per Seq2Seq non trovata.")
        # Gestione NaN (FFill/BFill sull'intero df prima di creare sequenze)
        if df[all_needed_cols].isnull().any().any():
             st.warning("NaN trovati prima della creazione sequenze Seq2Seq. Applico ffill/bfill.")
             df_filled = df.copy()
             df_filled[all_needed_cols] = df_filled[all_needed_cols].fillna(method='ffill').fillna(method='bfill')
             if df_filled[all_needed_cols].isnull().any().any():
                  st.error("NaN RESIDUI dopo fillna. Controlla colonne completamente vuote.")
                  return None, None, None, None, None, None, None, None, None # 9 return values
             df_to_use = df_filled
        else:
            df_to_use = df # Usa l'originale se non ci sono NaN
    except ValueError as e:
        st.error(f"Errore colonne in prepare_seq2seq: {e}")
        return None, None, None, None, None, None, None, None, None

    X_encoder, X_decoder, y_target = [], [], []
    total_len = len(df_to_use)
    required_len = input_window_steps + max(forecast_window_steps, output_window_steps)

    if total_len < required_len:
        st.error(f"Dati Seq2Seq insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, None, None, None, None

    print(f"Creazione sequenze Seq2Seq: {total_len - required_len + 1} possibili...")
    for i in range(total_len - required_len + 1):
        enc_end = i + input_window_steps
        X_encoder.append(df_to_use.iloc[i : enc_end][past_feature_cols].values)
        dec_start = enc_end
        dec_end_forecast = dec_start + forecast_window_steps
        X_decoder.append(df_to_use.iloc[dec_start : dec_end_forecast][forecast_feature_cols].values)
        target_end = dec_start + output_window_steps
        y_target.append(df_to_use.iloc[dec_start : target_end][target_cols].values)

    if not X_encoder or not X_decoder or not y_target:
        st.error("Errore creazione sequenze X_enc/X_dec/Y_target Seq2Seq.")
        return None, None, None, None, None, None, None, None, None

    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    print(f"Shapes NumPy: Encoder={X_encoder.shape}, Decoder={X_decoder.shape}, Target={y_target.shape}")

    # Validazione Shapes
    if X_encoder.shape[1] != input_window_steps or X_decoder.shape[1] != forecast_window_steps or y_target.shape[1] != output_window_steps:
         st.error("Errore shape sequenze Seq2Seq dopo creazione.")
         return None, None, None, None, None, None, None, None, None
    if X_encoder.shape[2] != len(past_feature_cols) or X_decoder.shape[2] != len(forecast_feature_cols) or y_target.shape[2] != len(target_cols):
         st.error("Errore numero feature/target nelle sequenze Seq2Seq.")
         return None, None, None, None, None, None, None, None, None

    # Scaling con 3 scaler separati
    scaler_past_features = MinMaxScaler()
    scaler_forecast_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    num_sequences = X_encoder.shape[0]
    X_enc_flat = X_encoder.reshape(-1, len(past_feature_cols))
    X_dec_flat = X_decoder.reshape(-1, len(forecast_feature_cols))
    y_tar_flat = y_target.reshape(-1, len(target_cols))

    try:
        X_enc_scaled_flat = scaler_past_features.fit_transform(X_enc_flat)
        X_dec_scaled_flat = scaler_forecast_features.fit_transform(X_dec_flat)
        y_tar_scaled_flat = scaler_targets.fit_transform(y_tar_flat)
    except Exception as e_scale:
        st.error(f"Errore scaling Seq2Seq: {e_scale}")
        return None, None, None, None, None, None, None, None, None

    # Reshape back
    X_enc_scaled = X_enc_scaled_flat.reshape(num_sequences, input_window_steps, len(past_feature_cols))
    X_dec_scaled = X_dec_scaled_flat.reshape(num_sequences, forecast_window_steps, len(forecast_feature_cols))
    y_tar_scaled = y_tar_scaled_flat.reshape(num_sequences, output_window_steps, len(target_cols))

    # Split Train/Validation
    split_idx = int(num_sequences * (1 - val_split / 100))
    if split_idx <= 0 or split_idx >= num_sequences:
         st.warning(f"Split indice Seq2Seq ({split_idx}) non valido. Adattamento in corso.")
         if num_sequences < 2: st.error("Dataset Seq2Seq troppo piccolo per split."); return None, None, None, None, None, None, None, None, None
         split_idx = max(1, min(num_sequences - 1, split_idx))

    X_enc_train, X_dec_train, y_tar_train = X_enc_scaled[:split_idx], X_dec_scaled[:split_idx], y_tar_scaled[:split_idx]
    X_enc_val, X_dec_val, y_tar_val = X_enc_scaled[split_idx:], X_dec_scaled[split_idx:], y_tar_scaled[split_idx:]

    # Controlli finali su set vuoti
    if X_enc_train.size == 0 or X_dec_train.size == 0 or y_tar_train.size == 0:
        st.error("Set Training Seq2Seq vuoto dopo split.")
        return None, None, None, None, None, None, None, None, None
    if X_enc_val.size == 0 or X_dec_val.size == 0 or y_tar_val.size == 0:
        st.warning("Set Validazione Seq2Seq vuoto (split 0% o pochi dati).")
        # Crea array vuoti con shape corretta per evitare errori
        X_enc_val = np.empty((0, input_window_steps, len(past_feature_cols)), dtype=np.float32)
        X_dec_val = np.empty((0, forecast_window_steps, len(forecast_feature_cols)), dtype=np.float32)
        y_tar_val = np.empty((0, output_window_steps, len(target_cols)), dtype=np.float32)

    print(f"Dati Seq2Seq pronti: Train={len(X_enc_train)}, Val={len(X_enc_val)}")
    return (X_enc_train, X_dec_train, y_tar_train,
            X_enc_val, X_dec_val, y_tar_val,
            scaler_past_features, scaler_forecast_features, scaler_targets)


# --- Funzioni Caricamento Modello/Scaler (Aggiornate) ---

@st.cache_data(show_spinner="Ricerca modelli disponibili...")
def find_available_models(models_dir=MODELS_DIR):
    """ Trova modelli validi (pth, json, scalers) e determina il tipo (LSTM/Seq2Seq). """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE find_available_models")
    available = {}
    if not os.path.isdir(models_dir): return available

    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth_path in pth_files:
        base = os.path.splitext(os.path.basename(pth_path))[0]
        cfg_path = os.path.join(models_dir, f"{base}.json")

        if not os.path.exists(cfg_path): continue # Manca config

        model_info = {"config_name": base, "pth_path": pth_path, "config_path": cfg_path}
        model_type = "LSTM" # Default
        valid_model = False

        try:
            with open(cfg_path, 'r') as f: config_data = json.load(f)
            name = config_data.get("display_name", base) # Usa display_name se esiste
            model_type = config_data.get("model_type", "LSTM") # Legge tipo da config

            # Verifica file necessari in base al tipo
            if model_type == "Seq2Seq":
                required_keys = ["input_window_steps", "output_window_steps", "forecast_window_steps",
                                 "hidden_size", "num_layers", "dropout",
                                 "all_past_feature_columns", "forecast_input_columns", "target_columns"]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_fore_p = os.path.join(models_dir, f"{base}_forecast_features.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(s_past_p) and os.path.exists(s_fore_p) and os.path.exists(s_targ_p)):
                    model_info.update({
                        "scaler_past_features_path": s_past_p,
                        "scaler_forecast_features_path": s_fore_p,
                        "scaler_targets_path": s_targ_p,
                        "model_type": "Seq2Seq"
                    })
                    valid_model = True
            else: # Assume LSTM standard
                # feature_columns è opzionale in LSTM config (userà globali)
                required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(scf_p) and os.path.exists(sct_p)):
                     model_info.update({
                         "scaler_features_path": scf_p,
                         "scaler_targets_path": sct_p,
                         "model_type": "LSTM"
                     })
                     valid_model = True

        except Exception as e_cfg:
            st.warning(f"Modello '{base}' ignorato: errore lettura/config JSON ({cfg_path}) non valida: {e_cfg}")
            valid_model = False

        if valid_model:
            available[name] = model_info # Usa display_name come chiave
        else:
            print(f"Modello '{base}' ignorato: file associati mancanti o config JSON incompleta per tipo '{model_type}'.")


    return available

@st.cache_data
def load_model_config(_config_path):
    """Carica config e valida chiavi base."""
    try:
        with open(_config_path, 'r') as f: config = json.load(f)
        return config
    except Exception as e:
        st.error(f"Errore caricamento config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    """Carica modello Pytorch (LSTM o Seq2Seq) in base al config."""
    if not config: st.error("Config non valida per caricamento modello."); return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get("model_type", "LSTM")

    try:
        model = None
        if model_type == "Seq2Seq":
            # Istanzia Encoder, Decoder, Seq2Seq
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]
            layers = config["num_layers"]
            drop = config["dropout"]
            out_win = config["output_window_steps"]

            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_input_size, hidden, dec_output_size, layers, drop).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win, device).to(device)

        else: # Assume LSTM standard
            # Gestione feature_columns opzionali per LSTM
            f_cols_lstm = config.get("feature_columns")
            if not f_cols_lstm:
                 st.warning(f"Config LSTM '{config.get('display_name', 'N/A')}' non specifica feature_columns. Uso le globali.")
                 f_cols_lstm = st.session_state.get("feature_columns", []) # Usa globali da session state
                 if not f_cols_lstm: raise ValueError("Feature globali non definite per modello LSTM caricato.")
                 config["feature_columns"] = f_cols_lstm # Aggiorna config in memoria

            input_size_lstm = len(f_cols_lstm)
            target_size_lstm = len(config["target_columns"])
            out_win_lstm = config["output_window"] # Già numero di steps

            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm,
                              out_win_lstm, config["num_layers"], config["dropout"]).to(device)

        # Carica state_dict (da path o file object)
        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'): # File caricato
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")

        model.eval() # Imposta modello in modalità valutazione
        print(f"Modello '{config.get('display_name', 'N/A')}' (Tipo: {model_type}) caricato su {device}.")
        return model, device

    except Exception as e:
        st.error(f"Errore caricamento modello '{config.get('display_name', 'N/A')}' (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(config, model_info):
    """Carica 1, 2 o 3 scaler in base al tipo di modello nel config."""
    if not config or not model_info: st.error("Config o info modello mancanti per caricare scaler."); return None
    model_type = config.get("model_type", "LSTM")

    def _load_joblib(path):
         # Helper interno per caricare da path o file object
         if isinstance(path, str):
              if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
              return joblib.load(path)
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")

    try:
        if model_type == "Seq2Seq":
            sp_path = model_info["scaler_past_features_path"]
            sf_path = model_info["scaler_forecast_features_path"]
            st_path = model_info["scaler_targets_path"]
            scaler_past = _load_joblib(sp_path)
            scaler_forecast = _load_joblib(sf_path)
            scaler_targets = _load_joblib(st_path)
            print(f"Scaler Seq2Seq caricati.")
            # Restituisce un dizionario
            return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # Assume LSTM
            sf_path = model_info["scaler_features_path"]
            st_path = model_info["scaler_targets_path"]
            scaler_features = _load_joblib(sf_path)
            scaler_targets = _load_joblib(st_path)
            print(f"Scaler LSTM caricati.")
            # Restituisce una tupla per compatibilità
            return scaler_features, scaler_targets

    except Exception as e:
        st.error(f"Errore caricamento scaler (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc())
        if model_type == "Seq2Seq": return None # Singolo None per fallimento
        else: return None, None # Due None per LSTM

# --- Funzioni Predict (Standard e Seq2Seq) ---

# Predict LSTM Standard
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    """Esegue predizione per modello LSTM standard."""
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello, scaler o config mancanti."); return None

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"Funzione predict chiamata su modello non LSTM (tipo: {model_type})"); return None

    input_steps = config["input_window"] # Steps
    output_steps = config["output_window"] # Steps
    target_cols = config["target_columns"]
    f_cols_cfg = config.get("feature_columns", [])

    # Verifica shape input
    if input_data.shape[0] != input_steps:
        st.error(f"Predict LSTM: Input righe {input_data.shape[0]} != Steps richiesti {input_steps}."); return None
    expected_features = getattr(scaler_features, 'n_features_in_', len(f_cols_cfg) if f_cols_cfg else None)
    if expected_features is not None and input_data.shape[1] != expected_features:
        st.error(f"Predict LSTM: Input colonne {input_data.shape[1]} != Features attese {expected_features}."); return None

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device) # Aggiunge batch dim
        with torch.no_grad(): output = model(inp_tens) # Shape (1, output_steps, num_targets)

        if output.shape != (1, output_steps, len(target_cols)):
             st.error(f"Predict LSTM: Shape output modello {output.shape} inattesa.")
             return None

        out_np = output.cpu().numpy().squeeze(0) # (output_steps, num_targets)

        expected_targets = getattr(scaler_targets, 'n_features_in_', None)
        if expected_targets is None or expected_targets != len(target_cols):
             st.error(f"Predict LSTM: Problema scaler targets (attese {len(target_cols)}, trovate {expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds # Shape (output_steps, num_targets)

    except Exception as e:
        st.error(f"Errore durante predict LSTM: {e}")
        st.error(traceback.format_exc()); return None

# Predict Seq2Seq
def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    """Esegue predizione per modello Seq2Seq."""
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
         st.error("Predict Seq2Seq: Input mancanti."); return None

    model_type = config.get("model_type")
    if model_type != "Seq2Seq": st.error(f"Funzione predict_seq2seq chiamata su modello non Seq2Seq"); return None

    input_steps = config["input_window_steps"]
    forecast_steps = config["forecast_window_steps"] # Steps di forecast forniti
    output_steps = config["output_window_steps"] # Steps da prevedere
    past_cols = config["all_past_feature_columns"]
    forecast_cols = config["forecast_input_columns"]
    target_cols = config["target_columns"]

    # Recupera scaler dal dizionario
    scaler_past = scalers.get("past")
    scaler_forecast = scalers.get("forecast")
    scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]):
        st.error("Predict Seq2Seq: Scaler mancanti nel dizionario fornito."); return None

    # Verifica shape input
    if past_data.shape != (input_steps, len(past_cols)):
        st.error(f"Predict Seq2Seq: Shape dati passati {past_data.shape} errata."); return None
    if future_forecast_data.shape != (forecast_steps, len(forecast_cols)):
        st.error(f"Predict Seq2Seq: Shape dati forecast {future_forecast_data.shape} errata."); return None
    if forecast_steps < output_steps:
         st.warning(f"Predict Seq2Seq: Finestra forecast ({forecast_steps}) < finestra output ({output_steps}). Modello userà padding.")

    model.eval()
    try:
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(past_tens, future_tens) # Shape (1, output_steps, num_targets)

        if output.shape != (1, output_steps, len(target_cols)):
             st.error(f"Predict Seq2Seq: Shape output modello {output.shape} inattesa."); return None

        out_np = output.cpu().numpy().squeeze(0) # (output_steps, num_targets)

        expected_targets = getattr(scaler_targets, 'n_features_in_', None)
        if expected_targets is None or expected_targets != len(target_cols):
             st.error(f"Predict Seq2Seq: Problema scaler targets."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds # Shape (output_steps, num_targets)

    except Exception as e:
        st.error(f"Errore durante predict Seq2Seq: {e}")
        st.error(traceback.format_exc()); return None


# --- Funzione Grafici Previsioni ---
def plot_predictions(predictions, config, start_time=None):
    """Genera grafici Plotly per le previsioni (LSTM o Seq2Seq)."""
    if config is None or predictions is None: return []

    model_type = config.get("model_type", "LSTM")
    target_cols = config["target_columns"]
    output_steps = predictions.shape[0] # Steps previsti
    total_hours_predicted = output_steps * 0.5

    figs = []
    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        # Genera asse X (timestamp o passi)
        if start_time:
            time_steps = [start_time + timedelta(minutes=30 * (step + 1)) for step in range(output_steps)]
            x_axis, x_title = time_steps, "Data e Ora Previste"
        else:
            time_steps = np.arange(1, output_steps + 1) * 0.5 # Ore relative
            x_axis, x_title = time_steps, "Ore Future (passi da 30 min)"

        station_name_graph = get_station_label(sensor, short=False) # Etichetta completa

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsto'))

        # Estrai unità di misura per asse Y
        unit_match = re.search(r'\((.*?)\)', sensor)
        y_axis_unit = unit_match.group(1).strip() if unit_match else "Valore"

        fig.update_layout(
            title=f'Previsione {model_type} - {station_name_graph}',
            xaxis_title=x_title,
            yaxis_title=y_axis_unit,
            height=400,
            margin=dict(l=60, r=20, t=50, b=50), # Margini per leggibilità
            hovermode="x unified",
            template="plotly_white" # Tema pulito
        )
        if start_time:
            fig.update_xaxes(tickformat="%d/%m %H:%M") # Formato data/ora
        fig.update_yaxes(rangemode='tozero') # Asse Y parte da zero
        figs.append(fig)
    return figs

# --- Funzioni Fetch Dati Google Sheet (Dashboard e Simulazione) ---
# MODIFICATA: fetch_gsheet_dashboard_data per caricare TUTTI i dati e filtrare dopo
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format):
    """Recupera TUTTI i dati rilevanti, pulisce e li restituisce per il filtraggio successivo."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data (FULL FETCH)")
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
        # Carica TUTTI i valori (attenzione a fogli molto grandi!)
        all_values = worksheet.get_all_values()

        if not all_values or len(all_values) < 2:
             return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = all_values[0]
        data_rows = all_values[1:] # Prendi TUTTE le righe di dati

        # Verifica colonne mancanti
        headers_set = set(headers)
        missing_cols = [col for col in relevant_columns if col not in headers_set]
        if missing_cols:
            return None, f"Errore: Colonne GSheet richieste mancanti: {', '.join(missing_cols)}", actual_fetch_time

        df = pd.DataFrame(data_rows, columns=headers)
        # Seleziona solo colonne utili PRIMA della pulizia per efficienza
        try:
            # Filtra relevant_columns che esistono effettivamente in headers
            existing_relevant_cols = [col for col in relevant_columns if col in headers_set]
            df = df[existing_relevant_cols]
        except KeyError as e:
             return None, f"Errore selezione colonne rilevanti post-fetch: {e}", actual_fetch_time

        # Pulizia dati e conversione tipi (gestione data con timezone) - su tutto il DF
        error_parsing = []
        nan_stats = {}
        for col in existing_relevant_cols: # Itera sulle colonne effettivamente presenti
            nan_before = df[col].isnull().sum() # Conteggio NaN prima
            if col == date_col:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    if df[col].isnull().any(): error_parsing.append(f"Formato data non valido per '{col}'")
                    # Localizza o converti a timezone italiana
                    if not df[col].empty and df[col].notna().any(): # Prova solo se ci sono date valide
                         if df[col].dt.tz is None:
                             # Gestisci ambiguous='infer' e nonexistent='NaT' o 'shift_forward'
                             try:
                                 df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='NaT')
                             except pytz.exceptions.NonExistentTimeError:
                                 st.caption(f"Attenzione: Trovate ore non esistenti (cambio ora legale?) nella colonna '{col}'. Sostituite con NaT.")
                                 df[col] = pd.NaT # Oppure usa nonexistent='shift_forward'
                             except pytz.exceptions.AmbiguousTimeError:
                                 st.caption(f"Attenzione: Trovate ore ambigue (cambio ora solare?) nella colonna '{col}'. Inferenza tentata.")
                                 # infer è già usato, ma potresti aggiungere logica specifica se necessario
                                 pass # Continua con 'infer'
                         else:
                             df[col] = df[col].dt.tz_convert(italy_tz)
                except Exception as e_date: error_parsing.append(f"Errore data '{col}': {e_date}"); df[col] = pd.NaT
            else: # Colonne numeriche
                try:
                    # Conversione robusta: str -> pulisci -> numeric
                    if pd.api.types.is_object_dtype(df[col]):
                        df_col_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                        df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else: # Se non è object, prova comunque a convertire
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e_num: error_parsing.append(f"Errore numerico '{col}': {e_num}"); df[col] = np.nan
            nan_after = df[col].isnull().sum()
            if nan_after > nan_before: nan_stats[col] = nan_after - nan_before

        # Rimuovi righe dove la data è NaT dopo la conversione
        original_rows = len(df)
        df = df.dropna(subset=[date_col])
        rows_dropped = original_rows - len(df)
        if rows_dropped > 0:
            st.caption(f"Nota: {rows_dropped} righe rimosse per data non valida.")

        # Ordina per data
        df = df.sort_values(by=date_col).reset_index(drop=True)

        # Segnala NaN creati durante la pulizia
        if nan_stats:
             nan_msg_parts = [f"'{k}': {v} nuovi NaN" for k, v in nan_stats.items()]
             st.caption(f"Nota Dashboard: Valori non validi convertiti in NaN -> {', '.join(nan_msg_parts)}")

        error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_parsing) if error_parsing else None
        print(f"Fetch GSheet completato. Righe totali processate: {len(df)}")
        return df, error_message, actual_fetch_time

    except gspread.exceptions.APIError as api_e:
        try: error_details = api_e.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}";
        except: error_msg = str(api_e)
        return None, f"Errore API Google Sheets: {error_msg}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e: return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time

# ... (il resto delle funzioni rimane invariato) ...

# ==============================================================================
# --- LAYOUT STREAMLIT PRINCIPALE ---
# ==============================================================================
st.title('Modello Predittivo Idrologico')
st.caption('Applicazione per monitoraggio, simulazione e addestramento di modelli LSTM e Seq2Seq.')

# ... (Sidebar invariata fino alla fine) ...
# --- Menu Navigazione ---
st.header('Menu Navigazione')
model_ready = active_model_sess is not None and active_config_sess is not None

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
radio_captions = []
disabled_options = []
default_page_idx = 0

# Determina stato opzioni navigazione
for i, opt in enumerate(radio_options):
    caption = ""; disabled = False
    if opt == 'Dashboard': caption = "Monitoraggio GSheet"
    elif opt == 'Simulazione':
        if not model_ready: caption = "Richiede Modello attivo"; disabled = True
        else: caption = f"Esegui previsioni ({active_config_sess.get('model_type', 'LSTM')})"
    elif opt == 'Analisi Dati Storici':
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Esplora dati CSV"
    elif opt == 'Allenamento Modello':
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Allena un nuovo modello"
    radio_captions.append(caption); disabled_options.append(disabled)
    # Imposta Dashboard come default trovando il suo indice
    if opt == st.session_state.get('current_page', 'Dashboard'): default_page_idx = i


# Gestione pagina corrente e blocco accesso a pagine disabilitate
current_page = st.session_state.get('current_page', radio_options[default_page_idx])
try:
    current_page_index = radio_options.index(current_page)
    if disabled_options[current_page_index]:
        # Trova l'indice della dashboard per il reindirizzamento
        dashboard_index = radio_options.index('Dashboard')
        st.warning(f"Pagina '{current_page}' non disponibile ({radio_captions[current_page_index]}). Reindirizzato a Dashboard.")
        current_page = radio_options[dashboard_index]
        st.session_state['current_page'] = current_page
except ValueError: # Se la pagina in sessione non esiste più
    dashboard_index = radio_options.index('Dashboard')
    current_page = radio_options[dashboard_index]
    st.session_state['current_page'] = current_page

# Mostra il radio button (sempre abilitato)
chosen_page = st.radio(
    'Scegli una funzionalità:', options=radio_options, captions=radio_captions,
    index=radio_options.index(current_page), # Usa l'indice della pagina corrente (validata)
    key='page_selector_radio'
)

# Se l'utente sceglie una pagina DIVERSA e VALIDA, aggiorna lo stato e fai rerun
if chosen_page != current_page:
    chosen_page_index = radio_options.index(chosen_page)
    if not disabled_options[chosen_page_index]:
        st.session_state['current_page'] = chosen_page
        # Forzare il refresh della cache dei modelli potrebbe essere utile qui se l'utente cambia pagina
        # find_available_models.clear() # Opzionale: decommenta se vuoi forzare ricerca modelli al cambio pagina
        st.rerun() # Rerun per caricare la nuova pagina
    else:
        # Se sceglie una pagina disabilitata, mostra errore ma non cambiare pagina (rimane su 'current_page')
        st.error(f"La pagina '{chosen_page}' non è disponibile. {radio_captions[chosen_page_index]}.")
        # Non fare st.rerun() qui, altrimenti entrerebbe in loop di errore se la pagina corrente è disabilitata

# La pagina da visualizzare è sempre 'current_page' (che è stata validata)
page = current_page


# ==============================================================================
# --- Logica Pagine Principali (Contenuto nel Main Frame) ---
# ==============================================================================

# Recupera variabili attive da session state per usarle nelle pagine
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scalers = st.session_state.get('active_scalers') # Può essere tupla (LSTM) o dict (Seq2Seq)
active_model_type = active_config.get("model_type", "LSTM") if active_config else "LSTM"

df_current_csv = st.session_state.get('df', None) # Dati CSV caricati
date_col_name_csv = st.session_state.date_col_name_csv
data_ready_csv = df_current_csv is not None # Ricalcola qui per sicurezza


# --- PAGINA DASHBOARD (MODIFICATA) ---
if page == 'Dashboard':
    st.header('Dashboard Monitoraggio Idrologico')
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("Errore Configurazione: Credenziali Google mancanti nei secrets di Streamlit.")
        st.stop()

    # Fetch TUTTI i dati GSheet con cache basata su intervallo temporale
    now_ts = time.time()
    cache_time_key = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)
    # Chiamata modificata: non passa più DASHBOARD_HISTORY_ROWS
    df_full_dashboard_data, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
        cache_time_key, GSHEET_ID, GSHEET_RELEVANT_COLS,
        GSHEET_DATE_COL, GSHEET_DATE_FORMAT
    )
    # Salva in session state l'intero df caricato e l'errore/timestamp
    st.session_state.last_dashboard_full_data = df_full_dashboard_data
    st.session_state.last_dashboard_error = error_msg
    if df_full_dashboard_data is not None or error_msg is None:
        st.session_state.last_dashboard_fetch_time = actual_fetch_time

    # --- Visualizzazione Stato e Bottone Refresh ---
    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess
             fetch_secs_ago = int(fetch_time_ago.total_seconds())
             status_text = f"Dati GSheet aggiornati alle {last_fetch_dt_sess.strftime('%H:%M:%S')} ({fetch_secs_ago}s fa). Refresh automatico ogni {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
             st.caption(status_text)
        else: st.caption("Recupero dati GSheet in corso o fallito...")
    with col_refresh_btn:
        if st.button("Aggiorna Ora", key="dash_refresh_button"):
            fetch_gsheet_dashboard_data.clear() # Pulisce cache specifica dashboard
            fetch_sim_gsheet_data.clear() # Pulisce cache simulazione
            st.success("Cache GSheet pulita. Ricaricamento..."); time.sleep(0.5); st.rerun()

    # Mostra errori GSheet (se presenti)
    if st.session_state.last_dashboard_error:
        error_msg_display = st.session_state.last_dashboard_error
        if any(x in error_msg_display for x in ["API", "Foglio", "Credenziali", "Errore:"]): st.error(f"Errore GSheet: {error_msg_display}")
        else: st.warning(f"Attenzione Dati GSheet: {error_msg_display}")

    # --- Selettori Intervallo Date ---
    selected_start_date = None
    selected_end_date = None
    df_dashboard_filtered = None # DataFrame che conterrà i dati filtrati per i grafici
    latest_row_data = None # Riga con l'ultimo rilevamento assoluto

    df_full_sess = st.session_state.get('last_dashboard_full_data')

    if df_full_sess is not None and not df_full_sess.empty:
        date_col = GSHEET_DATE_COL
        if date_col not in df_full_sess.columns or df_full_sess[date_col].isnull().all():
            st.error(f"Colonna data '{date_col}' non trovata o vuota nei dati GSheet. Impossibile procedere con la dashboard.")
        else:
            # Determina date min/max dai dati CARICATI
            min_available_date = df_full_sess[date_col].min().date()
            max_available_date = df_full_sess[date_col].max().date()

            # Imposta default: ultimo giorno disponibile
            default_start = max(min_available_date, max_available_date - timedelta(days=1)) # Ultimo giorno o il minimo se c'è solo un giorno
            default_end = max_available_date

            st.subheader("Seleziona Periodo da Visualizzare")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                selected_start_date = st.date_input(
                    "Data Inizio",
                    value=st.session_state.get("dashboard_start_date", default_start), # Usa valore da sessione se esiste
                    min_value=min_available_date,
                    max_value=max_available_date,
                    key="dash_start_date_input"
                )
            with col_date2:
                selected_end_date = st.date_input(
                    "Data Fine",
                    value=st.session_state.get("dashboard_end_date", default_end), # Usa valore da sessione se esiste
                    min_value=min_available_date, # Minimo teorico
                    max_value=max_available_date,
                    key="dash_end_date_input"
                )

            # Salva le date selezionate in session_state per persistenza
            st.session_state.dashboard_start_date = selected_start_date
            st.session_state.dashboard_end_date = selected_end_date

            # Validazione date
            if selected_start_date > selected_end_date:
                st.error("Errore: La data di inizio non può essere successiva alla data di fine.")
                df_dashboard_filtered = pd.DataFrame(columns=df_full_sess.columns) # df vuoto per evitare errori dopo
                latest_row_data = df_full_sess.iloc[-1] # Prendi comunque l'ultimo dato
            else:
                # Filtra il DataFrame in base alle date selezionate (inclusive)
                try:
                    # Converti date selezionate in datetime con timezone per confronto corretto
                    filter_start_dt = italy_tz.localize(datetime.combine(selected_start_date, datetime.min.time()))
                    filter_end_dt = italy_tz.localize(datetime.combine(selected_end_date, datetime.max.time()))

                    df_dashboard_filtered = df_full_sess[
                        (df_full_sess[date_col] >= filter_start_dt) &
                        (df_full_sess[date_col] <= filter_end_dt)
                    ].copy() # Usa .copy() per evitare SettingWithCopyWarning
                    latest_row_data = df_full_sess.iloc[-1] # Ultimo dato assoluto
                    st.caption(f"Visualizzazione dati dal {selected_start_date.strftime('%d/%m/%Y')} al {selected_end_date.strftime('%d/%m/%Y')}. ({len(df_dashboard_filtered)} righe)")

                except Exception as e_filter:
                     st.error(f"Errore durante il filtraggio per data: {e_filter}")
                     df_dashboard_filtered = pd.DataFrame(columns=df_full_sess.columns) # df vuoto
                     latest_row_data = df_full_sess.iloc[-1] # Prendi comunque l'ultimo dato

    # --- Visualizzazione Dati e Grafici (basata su dati filtrati e ultimo dato) ---
    if latest_row_data is not None:
        # --- Stato Ultimo Rilevamento (USA L'ULTIMO DATO ASSOLUTO) ---
        last_update_time = latest_row_data.get(GSHEET_DATE_COL)
        if pd.notna(last_update_time) and isinstance(last_update_time, pd.Timestamp):
             time_now_italy = datetime.now(italy_tz)
             # Assicura timezone (dovrebbe già esserlo dal fetch)
             if last_update_time.tzinfo is None: last_update_time = italy_tz.localize(last_update_time)
             else: last_update_time = last_update_time.tz_convert(italy_tz)

             time_delta = time_now_italy - last_update_time; minutes_ago = int(time_delta.total_seconds() // 60)
             time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
             if minutes_ago < 2: time_ago_str = "pochi istanti fa"
             elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
             else: time_ago_str = f"circa {minutes_ago // 60}h e {minutes_ago % 60}min fa"

             st.markdown(f"**Ultimo rilevamento assoluto:** {time_str} ({time_ago_str})")
             if minutes_ago > 90: # Warning se > 1.5 ore
                 st.warning(f"Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa.")
        else: st.warning("Timestamp ultimo rilevamento non valido o mancante.")

        st.divider()
        # --- Tabella Valori Attuali e Stato Allerta (USA L'ULTIMO DATO ASSOLUTO) ---
        st.subheader("Valori Attuali e Stato Allerta (Basati sull'Ultimo Rilevamento)")
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL and col in latest_row_data.index]
        table_rows = []; current_alerts = []
        for col_name in cols_to_monitor:
            current_value = latest_row_data.get(col_name)
            threshold = st.session_state.dashboard_thresholds.get(col_name)
            alert_active = False; value_numeric = np.nan; value_display = "N/D"; unit = ""
            if 'Pioggia' in col_name: unit = 'mm'
            elif 'Livello' in col_name: unit = 'm'
            elif 'Umidit' in col_name: unit = '%'

            if pd.notna(current_value) and isinstance(current_value, (int, float, np.number)):
                 value_numeric = float(current_value)
                 fmt_spec = ".1f" if unit in ['mm', '%'] else ".2f" # Formato in base all'unità
                 try: value_display = f"{value_numeric:{fmt_spec}} {unit}".strip()
                 except ValueError: value_display = f"{value_numeric} {unit}".strip() # Fallback formattazione

                 if threshold is not None and isinstance(threshold, (int, float, np.number)) and value_numeric >= float(threshold):
                      alert_active = True; current_alerts.append((col_name, value_numeric, threshold))
            elif pd.notna(current_value): value_display = f"{current_value} (?)" # Valore non numerico

            # Assegna status testuale
            status = "ALLERTA" if alert_active else ("OK" if pd.notna(value_numeric) else "N/D")
            threshold_display = f"{float(threshold):.1f}" if threshold is not None else "-" # Mostra soglia con 1 decimale

            table_rows.append({"Stazione": get_station_label(col_name, short=True), "Valore": value_display, "Soglia": threshold_display, "Stato": status,
                               "Valore Numerico": value_numeric, "Soglia Numerica": float(threshold) if threshold else None}) # Colonne nascoste per styling

        df_display = pd.DataFrame(table_rows)

        def highlight_alert(row): # Funzione stile per evidenziare allerte
            style = [''] * len(row)
            if row['Stato'] == 'ALLERTA':
                style = ['background-color: rgba(255, 0, 0, 0.1);'] * len(row)
            return style

        # Mostra tabella formattata
        st.dataframe(
            df_display.style.apply(highlight_alert, axis=1),
            column_order=["Stazione", "Valore", "Soglia", "Stato"], # Ordine colonne visibili
            hide_index=True,
            use_container_width=True,
            column_config={ # Nascondi colonne usate solo per lo stile
                "Valore Numerico": None,
                "Soglia Numerica": None
            }
        )
        st.session_state.active_alerts = current_alerts # Salva allerte attive basate sull'ultimo dato

        # --- Grafici (USA DATI FILTRATI) ---
        if df_dashboard_filtered is not None and not df_dashboard_filtered.empty:
            st.divider()
            st.subheader(f"Grafici Storici Periodo Selezionato ({selected_start_date.strftime('%d/%m')} - {selected_end_date.strftime('%d/%m')})")

            # --- Grafico Comparativo Configurabile ---
            st.markdown("**Grafico Comparativo**")
            # Usa colonne presenti nei dati FILTRATI per la selezione
            cols_monitor_in_filtered = [c for c in cols_to_monitor if c in df_dashboard_filtered.columns]
            sensor_options_compare = {get_station_label(col, short=True): col for col in cols_monitor_in_filtered}
            # Filtra default per essere sicuro che siano presenti
            default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3]
            if not default_selection_labels: default_selection_labels = list(sensor_options_compare.keys())[:min(2, len(sensor_options_compare))]

            selected_labels_compare = st.multiselect("Seleziona sensori da confrontare:", options=list(sensor_options_compare.keys()), default=default_selection_labels, key="compare_select_multi")
            selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare if label in sensor_options_compare] # Sicurezza extra

            if selected_cols_compare:
                fig_compare = go.Figure()
                x_axis_data_filtered = df_dashboard_filtered[GSHEET_DATE_COL] # Usa date filtrate
                for col in selected_cols_compare:
                    if col in df_dashboard_filtered.columns:
                        fig_compare.add_trace(go.Scatter(x=x_axis_data_filtered, y=df_dashboard_filtered[col], mode='lines', name=get_station_label(col, short=True)))
                fig_compare.update_layout(title=f"Andamento Storico Comparato", xaxis_title='Data e Ora', yaxis_title='Valore',
                                          height=500, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_compare, use_container_width=True)
                # Link download grafico
                compare_filename_base = f"compare_{selected_start_date.strftime('%Y%m%d')}-{selected_end_date.strftime('%Y%m%d')}"
                st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)

            st.divider()
            # --- Grafici Individuali ---
            st.markdown("**Grafici Individuali**")
            num_cols_individual = 3 # Numero di colonne per i grafici
            graph_cols = st.columns(num_cols_individual)
            col_idx_graph = 0
            x_axis_data_indiv_filtered = df_dashboard_filtered[GSHEET_DATE_COL] # Usa date filtrate
            for col_name in cols_monitor_in_filtered: # Itera solo sulle colonne disponibili nei dati filtrati
                with graph_cols[col_idx_graph % num_cols_individual]:
                    threshold_individual = st.session_state.dashboard_thresholds.get(col_name)
                    label_individual = get_station_label(col_name, short=True)
                    unit = '(mm)' if 'Pioggia' in col_name else ('(m)' if 'Livello' in col_name else ('(%)' if 'Umidit' in col_name else ''))
                    yaxis_title_individual = f"Valore {unit}".strip()

                    fig_individual = go.Figure()
                    fig_individual.add_trace(go.Scatter(x=x_axis_data_indiv_filtered, y=df_dashboard_filtered[col_name], mode='lines', name=label_individual))
                    # Aggiungi linea soglia se definita
                    if threshold_individual is not None:
                        fig_individual.add_hline(y=threshold_individual, line_dash="dash", line_color="red",
                                                 annotation_text=f"Soglia ({float(threshold_individual):.1f})", annotation_position="bottom right")

                    fig_individual.update_layout(title=f"{label_individual}", xaxis_title=None, yaxis_title=yaxis_title_individual,
                                                 height=300, hovermode="x unified", showlegend=False, margin=dict(t=40, b=30, l=50, r=10), template="plotly_white")
                    fig_individual.update_yaxes(rangemode='tozero') # Asse Y da zero
                    st.plotly_chart(fig_individual, use_container_width=True)

                    # Link download grafico individuale
                    ind_filename_base = f"sensor_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{selected_start_date.strftime('%Y%m%d')}-{selected_end_date.strftime('%Y%m%d')}"
                    st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)
                col_idx_graph += 1
        elif df_dashboard_filtered is not None and df_dashboard_filtered.empty:
            st.warning("Nessun dato trovato nel periodo selezionato.")
        else: # df_dashboard_filtered è None (probabilmente errore filtro o date non valide)
            st.info("Seleziona un intervallo di date valido per visualizzare i grafici.")


        st.divider()
        # --- Riepilogo Alert Attivi (Basato sempre sull'ULTIMO dato) ---
        st.subheader("Riepilogo Allerte (Basato sull'Ultimo Rilevamento)")
        active_alerts_sess = st.session_state.get('active_alerts', [])
        if active_alerts_sess:
             st.warning("**Allerte Attive (Valori >= Soglia):**")
             alert_md = ""
             # Ordina per nome stazione per consistenza
             sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
             for col, val, thr in sorted_alerts:
                 label_alert = get_station_label(col, short=False); sensor_type_alert = STATION_COORDS.get(col, {}).get('type', '')
                 type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                 unit = '(mm)' if 'Pioggia' in col else ('(m)' if 'Livello' in col else ('(%)' if 'Umidit' in col else ''))
                 val_fmt = f"{val:.1f}" if unit in ['(mm)', '(%)'] else f"{val:.2f}"
                 thr_fmt = f"{float(thr):.1f}" if isinstance(thr, (int, float, np.number)) else str(thr)
                 alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit}** >= Soglia **{thr_fmt}{unit}**\n"
             st.markdown(alert_md)
        else: st.success("Nessuna soglia superata nell'ultimo rilevamento.")

    elif df_full_sess is not None and df_full_sess.empty:
        st.warning("Recupero dati GSheet riuscito, ma nessun dato trovato nel foglio.")
    elif not st.session_state.last_dashboard_error: # Se non ci sono errori e df è None, fetch iniziale
        st.info("Recupero dati dashboard in corso...")

    # --- Auto Refresh JS (Invariato) ---
    try:
        component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
        js_code = f"""
        (function() {{
            const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}';
            // Clear only if component is re-rendering, not on every script run
            if (window.hasOwnProperty(intervalIdKey)) {{
                // Optional: Add logic here if you need to reset based on specific conditions
            }} else {{
                 // Set interval only if it doesn't exist
                 window[intervalIdKey] = setInterval(function() {{
                    // Check if Streamlit hook is available before calling
                    if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') {{
                        console.log('Triggering Streamlit rerun via JS timer ({component_key})');
                        try {{
                             window.streamlitHook.rerunScript(null);
                        }} catch (e) {{
                             console.error('Error calling rerunScript:', e);
                             // Optionally clear interval if rerun fails consistently
                             // clearInterval(window[intervalIdKey]);
                             // delete window[intervalIdKey];
                        }}
                    }} else {{
                        console.warn('Streamlit hook not available for JS auto-refresh. Clearing interval.');
                        clearInterval(window[intervalIdKey]);
                        delete window[intervalIdKey]; // Prevent trying again
                    }}
                }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});
                console.log('Set new JS interval: ', window[intervalIdKey]);
            }}
        }})();
        """
        # Using a changing key might force re-evaluation, but let's try without first
        streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
    except Exception as e_js:
        st.caption(f"Nota: Impossibile impostare auto-refresh ({e_js})")


# --- PAGINA SIMULAZIONE (Invariata) ---
elif page == 'Simulazione':
    st.header(f'Simulazione Idrologica ({active_model_type})')
    if not model_ready:
        st.warning("Seleziona un Modello attivo dalla sidebar per eseguire la Simulazione.")
        st.stop()

    # Mostra dettagli modello attivo
    st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** ({active_model_type})")
    if active_model_type == "Seq2Seq":
        st.caption(f"Input Storico: {active_config['input_window_steps']} steps | Input Forecast: {active_config['forecast_window_steps']} steps | Output: {active_config['output_window_steps']} steps")
        with st.expander("Dettagli Colonne Modello Seq2Seq"):
             st.markdown("**Feature Storiche (Encoder Input):**")
             st.caption(f"`{', '.join(active_config['all_past_feature_columns'])}`")
             st.markdown("**Feature Forecast (Decoder Input):**")
             st.caption(f"`{', '.join(active_config['forecast_input_columns'])}`")
             st.markdown("**Target (Output):**")
             st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns'] # Definisci per uso successivo
    else: # LSTM Standard
        # Recupera feature columns (da config o globali)
        feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns)
        st.caption(f"Input: {active_config['input_window']} steps | Output: {active_config['output_window']} steps")
        with st.expander("Dettagli Colonne Modello LSTM"):
             st.markdown("**Feature Input:**")
             st.caption(f"`{', '.join(feature_columns_current_model)}`")
             st.markdown("**Target (Output):**")
             st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns'] # Definisci per uso successivo
        input_steps = active_config['input_window'] # Definisci per LSTM

    st.divider()

    # --- SEZIONE INPUT DATI SIMULAZIONE ---
    if active_model_type == "Seq2Seq":
         st.subheader(f"Preparazione Dati Input Simulazione Seq2Seq")
         # --- 1. Recupero dati storici passati da GSheet ---
         st.markdown("**Passo 1: Recupero Dati Storici (Input Encoder)**")
         # (Usa GSHEET_ID di default)
         st.caption(f"Verranno recuperati gli ultimi {active_config['input_window_steps']} steps dal Foglio Google (ID: `{GSHEET_ID}`).")

         # Mappatura colonne GSheet -> Feature Passate Modello Seq2Seq
         past_feature_cols_s2s = active_config["all_past_feature_columns"]
         column_mapping_gsheet_to_past_s2s = {
             # NOME_COLONNA_GSHEET : NOME_FEATURE_MODELLO_PASSATO (esempio, adattare!)
             'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
             'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
             'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
             'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
             'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME,
             'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
             'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
             'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
             'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
             'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
             GSHEET_DATE_COL: st.session_state.date_col_name_csv # Mappa colonna data
         }
         # Filtra solo quelle richieste dal modello
         relevant_model_cols_for_mapping = past_feature_cols_s2s + [st.session_state.date_col_name_csv]
         column_mapping_gsheet_to_past_s2s_filtered = {
             gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_past_s2s.items()
             if model_col in relevant_model_cols_for_mapping
         }

         # Gestione Imputazione Feature Passate non mappate
         past_features_set = set(past_feature_cols_s2s)
         mapped_past_features_target = set(column_mapping_gsheet_to_past_s2s_filtered.values()) - {st.session_state.date_col_name_csv}
         missing_past_features_in_map = list(past_features_set - mapped_past_features_target)
         imputed_values_sim_past_s2s = {}
         if missing_past_features_in_map:
             st.caption(f"Feature storiche non mappate da GSheet: `{', '.join(missing_past_features_in_map)}`. Saranno imputate (valore default 0 o mediana da CSV se disponibile).")
             for missing_f in missing_past_features_in_map:
                 default_val = 0.0
                 if data_ready_csv and missing_f in df_current_csv.columns and pd.notna(df_current_csv[missing_f].median()):
                     default_val = df_current_csv[missing_f].median()
                 imputed_values_sim_past_s2s[missing_f] = default_val

         # Bottone Fetch dati storici
         fetch_error_gsheet_s2s = None
         if st.button("Carica/Aggiorna Storico da GSheet", key="fetch_gsh_s2s_base"):
              fetch_sim_gsheet_data.clear() # Pulisce cache
              with st.spinner("Recupero dati storici (passato)..."):
                  required_cols_fetch = past_feature_cols_s2s + [st.session_state.date_col_name_csv]
                  imported_df_past, import_err_past, last_ts_past = fetch_sim_gsheet_data(
                       GSHEET_ID, active_config['input_window_steps'], GSHEET_DATE_COL, GSHEET_DATE_FORMAT,
                       column_mapping_gsheet_to_past_s2s_filtered,
                       required_cols_fetch,
                       imputed_values_sim_past_s2s
                  )
              if import_err_past:
                  st.error(f"Recupero storico Seq2Seq fallito: {import_err_past}")
                  st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = import_err_past
              elif imported_df_past is not None:
                  try:
                      # Seleziona e riordina SOLO le colonne necessarie per l'encoder
                      final_past_df = imported_df_past[past_feature_cols_s2s]
                      st.success(f"Recuperate e processate {len(final_past_df)} righe storiche.")
                      st.session_state.seq2seq_past_data_gsheet = final_past_df
                      st.session_state.seq2seq_last_ts_gsheet = last_ts_past if last_ts_past else datetime.now(italy_tz) # Salva timestamp
                      fetch_error_gsheet_s2s = None
                      st.rerun() # Rerun per mostrare subito stato e dati
                  except KeyError as e_cols:
                      st.error(f"Errore selezione colonne storiche dopo fetch: {e_cols}")
                      st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = f"Errore colonne: {e_cols}"
              else: st.error("Recupero storico Seq2Seq non riuscito (risultato vuoto)."); fetch_error_gsheet_s2s = "Errore recupero dati."

         # Mostra stato caricamento dati base e preview
         past_data_loaded_s2s = st.session_state.get('seq2seq_past_data_gsheet') is not None
         if past_data_loaded_s2s:
             st.caption("Dati storici base (input encoder) caricati.")
             with st.expander("Mostra dati storici caricati (Input Encoder)"):
                 st.dataframe(st.session_state.seq2seq_past_data_gsheet.round(3))
         elif fetch_error_gsheet_s2s: st.warning(f"Caricamento dati storici fallito ({fetch_error_gsheet_s2s}). Clicca il bottone per riprovare.")
         else: st.info("Clicca il bottone 'Carica/Aggiorna Storico da GSheet' per recuperare i dati necessari.")

         # --- 2. Input Previsioni Utente (Input Decoder) ---
         if past_data_loaded_s2s:
             st.markdown(f"**Passo 2: Inserisci Previsioni Future ({active_config['forecast_window_steps']} steps)**")
             forecast_input_cols_s2s = active_config['forecast_input_columns']
             st.caption(f"Inserisci i valori previsti per le seguenti feature: `{', '.join(forecast_input_cols_s2s)}`")

             # Data Editor per le previsioni future
             forecast_steps_s2s = active_config['forecast_window_steps']
             forecast_df_initial = pd.DataFrame(index=range(forecast_steps_s2s), columns=forecast_input_cols_s2s)
             # Tenta di usare l'ultimo valore noto dai dati passati caricati
             last_known_past_df_or_series = st.session_state.seq2seq_past_data_gsheet
             if last_known_past_df_or_series is not None and not last_known_past_df_or_series.empty:
                 last_known_past_data = last_known_past_df_or_series.iloc[-1]
                 for col in forecast_input_cols_s2s:
                      if 'pioggia' in col.lower() or 'cumulata' in col.lower(): forecast_df_initial[col] = 0.0 # Default 0 pioggia
                      elif col in last_known_past_data.index: forecast_df_initial[col] = last_known_past_data.get(col, 0.0) # Usa ultimo valore noto
                      else: forecast_df_initial[col] = 0.0 # Default 0 se non presente nel passato
             else: # Se non ci sono dati passati, inizializza a 0
                 forecast_df_initial = forecast_df_initial.fillna(0.0)


             forecast_editor_key = "seq2seq_forecast_editor"
             edited_forecast_df = st.data_editor(
                 forecast_df_initial.round(2),
                 key=forecast_editor_key,
                 num_rows="fixed",
                 use_container_width=True,
                 column_config={ # Configurazione dinamica colonne forecast
                     col: st.column_config.NumberColumn(
                         label=get_station_label(col, short=True), # Etichetta breve
                         help=f"Valore previsto per {col}",
                         min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else None,
                         max_value=100.0 if 'umidit' in col.lower() else None,
                         format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f",
                         step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1),
                         required=True # Rendi tutte le colonne richieste
                     ) for col in forecast_input_cols_s2s
                 }
             )

             # Validazione editor forecast
             forecast_data_valid_s2s = False
             if edited_forecast_df is not None and not edited_forecast_df.isnull().any().any():
                 # Ulteriore controllo che tutti i valori siano effettivamente numerici
                 is_all_numeric = True
                 try:
                     edited_forecast_df.astype(float)
                 except ValueError:
                     is_all_numeric = False

                 if is_all_numeric and edited_forecast_df.shape == (forecast_steps_s2s, len(forecast_input_cols_s2s)):
                     forecast_data_valid_s2s = True
                 elif not is_all_numeric:
                     st.warning("La tabella delle previsioni future contiene valori non numerici.")
                 else:
                      st.warning("Numero di righe o colonne errato nella tabella previsioni.")
             else:
                  st.warning("Completa o correggi la tabella delle previsioni future. Tutti i valori devono essere numerici.")

             # --- 3. Esecuzione Simulazione Seq2Seq ---
             st.divider()
             st.markdown("**Passo 3: Esegui Simulazione Seq2Seq**")
             can_run_s2s_sim = past_data_loaded_s2s and forecast_data_valid_s2s

             if st.button("Esegui Simulazione Seq2Seq", disabled=not can_run_s2s_sim, type="primary", key="run_s2s_sim_button"):
                  if not can_run_s2s_sim: st.error("Mancano dati storici validi o previsioni future valide.")
                  else:
                       with st.spinner("Simulazione Seq2Seq in corso..."):
                           # Assicura che i dati siano nel formato corretto (numpy float)
                           past_data_np = st.session_state.seq2seq_past_data_gsheet[past_feature_cols_s2s].astype(float).values
                           future_forecast_np = edited_forecast_df[forecast_input_cols_s2s].astype(float).values

                           predictions_s2s = predict_seq2seq(
                               active_model, past_data_np, future_forecast_np,
                               active_scalers, # Dizionario scaler
                               active_config, active_device
                           )

                       # Mostra risultati
                       if predictions_s2s is not None:
                           output_steps_actual = predictions_s2s.shape[0]
                           total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione Seq2Seq: Prossime {total_hours_output_actual:.1f} ore')
                           start_pred_time_s2s = st.session_state.get('seq2seq_last_ts_gsheet', datetime.now(italy_tz))
                           st.caption(f"Previsione calcolata a partire da: {start_pred_time_s2s.strftime('%d/%m/%Y %H:%M:%S %Z')}")

                           # Tabella Risultati
                           pred_times_s2s = [start_pred_time_s2s + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                           results_df_s2s = pd.DataFrame(predictions_s2s, columns=target_columns_model)
                           results_df_s2s.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_s2s])
                           # Rinomina colonne per leggibilità tabella display
                           rename_dict = {'Ora Prevista': 'Ora Prevista'}
                           for col in target_columns_model:
                               unit = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""
                               new_name = f"{get_station_label(col, short=True)} {unit_str}".strip()
                               count = 1; final_name = new_name
                               while final_name in rename_dict.values(): count += 1; final_name = f"{new_name}_{count}"
                               rename_dict[col] = final_name
                           results_df_s2s_display = results_df_s2s.copy().rename(columns=rename_dict)
                           st.dataframe(results_df_s2s_display.round(3), hide_index=True)
                           st.markdown(get_table_download_link(results_df_s2s, f"simulazione_seq2seq_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                           # Grafici Risultati
                           st.subheader('Grafici Previsioni Simulate (Seq2Seq)')
                           figs_sim_s2s = plot_predictions(predictions_s2s, active_config, start_pred_time_s2s)
                           num_graph_cols = min(len(figs_sim_s2s), 3)
                           sim_cols = st.columns(num_graph_cols)
                           for i, fig_sim in enumerate(figs_sim_s2s):
                              with sim_cols[i % num_graph_cols]:
                                   target_col_name = target_columns_model[i]
                                   s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                   st.plotly_chart(fig_sim, use_container_width=True)
                                   st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_s2s_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                       else:
                             st.error("Predizione simulazione Seq2Seq fallita.")

    else: # --- SEZIONE INPUT DATI SIMULAZIONE LSTM ---
         st.subheader("Metodo Input Dati Simulazione LSTM")
         sim_method_options = ['Manuale (Valori Costanti)', 'Importa da Google Sheet (Ultime Ore)', 'Orario Dettagliato (Tabella)']
         if data_ready_csv: sim_method_options.append('Usa Ultime Ore da CSV Caricato')
         sim_method = st.radio("Scegli come fornire i dati di input:", sim_method_options, key="sim_method_radio_select_lstm", horizontal=True, label_visibility="collapsed")

         # --- Logica per Metodi LSTM ---
         if sim_method == 'Manuale (Valori Costanti)':
             st.markdown(f'Inserisci valori costanti per le **{len(feature_columns_current_model)}** feature di input.')
             st.caption(f"Questi valori saranno ripetuti per i **{input_steps}** passi temporali ({input_steps*0.5:.1f} ore) richiesti dal modello.")
             temp_sim_values_lstm = {}
             cols_manual_lstm = st.columns(3) # Max 3 colonne per pulizia
             col_idx_manual = 0
             input_valid_manual = True
             for feature in feature_columns_current_model:
                 with cols_manual_lstm[col_idx_manual % 3]:
                     label = get_station_label(feature, short=True)
                     unit = re.search(r'\[(.*?)\]|\((.*?)\)', feature); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""
                     fmt = "%.1f" if 'pioggia' in feature.lower() or 'umidit' in feature.lower() else "%.2f"
                     step = 0.5 if 'pioggia' in feature.lower() else (1.0 if 'umidit' in feature.lower() else 0.1)
                     try:
                         val = st.number_input(f"{label} {unit_str}".strip(), key=f"man_{feature}", format=fmt, step=step)
                         temp_sim_values_lstm[feature] = float(val)
                     except Exception as e_num_input:
                         st.error(f"Valore non valido per {label}: {e_num_input}")
                         input_valid_manual = False
                 col_idx_manual += 1

             sim_data_input_manual = None
             if input_valid_manual:
                 try:
                     ordered_values = [temp_sim_values_lstm[feature] for feature in feature_columns_current_model]
                     sim_data_input_manual = np.tile(ordered_values, (input_steps, 1)).astype(float)
                 except KeyError as ke: st.error(f"Errore: Feature '{ke}' mancante."); input_valid_manual = False
                 except Exception as e: st.error(f"Errore creazione dati: {e}"); input_valid_manual = False

             input_ready_manual = sim_data_input_manual is not None and input_valid_manual
             st.divider()
             if st.button('Esegui Simulazione LSTM (Manuale)', type="primary", disabled=(not input_ready_manual), key="sim_run_exec_lstm_manual"):
                 if input_ready_manual:
                      with st.spinner('Simulazione LSTM (Manuale) in corso...'):
                           scaler_features_lstm, scaler_targets_lstm = active_scalers # Scaler da tupla
                           predictions_sim_lstm = predict(active_model, sim_data_input_manual, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                      # Mostra Risultati (simile a Seq2Seq)
                      if predictions_sim_lstm is not None:
                           output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione LSTM (Manuale): Prossime {total_hours_output_actual:.1f} ore')
                           start_pred_time_lstm = datetime.now(italy_tz) # Usa ora corrente
                           st.caption(f"Previsione calcolata a partire da: {start_pred_time_lstm.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                           # Tabella
                           pred_times_lstm = [start_pred_time_lstm + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                           results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                           rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'};
                           for col in target_columns_model:
                               unit = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""; new_name = f"{get_station_label(col, short=True)} {unit_str}".strip(); count = 1; final_name = new_name
                               while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                               rename_dict_lstm[col] = final_name
                           results_df_lstm_display = results_df_lstm.copy().rename(columns=rename_dict_lstm)
                           st.dataframe(results_df_lstm_display.round(3), hide_index=True)
                           st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_manuale_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                           # Grafici
                           st.subheader('Grafici Previsioni Simulate (LSTM Manuale)')
                           figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, start_pred_time_lstm); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                           for i, fig_sim in enumerate(figs_sim_lstm):
                               with sim_cols_lstm[i % num_graph_cols_lstm]:
                                    target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_manuale_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                      else: st.error("Predizione simulazione LSTM (Manuale) fallita.")
                 else: st.error("Dati input manuali non pronti o invalidi.")


         elif sim_method == 'Importa da Google Sheet (Ultime Ore)':
             st.markdown(f'Importa gli ultimi **{input_steps}** steps ({input_steps*0.5:.1f} ore) da Google Sheet per le **{len(feature_columns_current_model)}** feature di input.')
             # (Usa GSHEET_ID di default)
             st.caption(f"Verranno recuperati i dati dal Foglio Google (ID: `{GSHEET_ID}`).")

             # Mappatura GSheet -> Feature Modello LSTM
             column_mapping_gsheet_to_lstm = {
                 'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', 'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                 'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)', 'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
                 'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME,
                 'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                 'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', 'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                 'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
                 GSHEET_DATE_COL: st.session_state.date_col_name_csv # Mappa data
             }
             # Filtra mappatura solo per colonne richieste da feature_columns_current_model + data
             relevant_model_cols_lstm_map = feature_columns_current_model + [st.session_state.date_col_name_csv]
             column_mapping_gsheet_to_lstm_filtered = { gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_lstm.items() if model_col in relevant_model_cols_lstm_map }

             # Imputazione per feature LSTM non mappate
             lstm_features_set = set(feature_columns_current_model)
             mapped_lstm_features_target = set(column_mapping_gsheet_to_lstm_filtered.values()) - {st.session_state.date_col_name_csv}
             missing_lstm_features_in_map = list(lstm_features_set - mapped_lstm_features_target)
             imputed_values_sim_lstm_gs = {}
             if missing_lstm_features_in_map:
                 st.caption(f"Feature LSTM non mappate da GSheet: `{', '.join(missing_lstm_features_in_map)}`. Saranno imputate.")
                 for missing_f in missing_lstm_features_in_map:
                     default_val = 0.0
                     if data_ready_csv and missing_f in df_current_csv.columns and pd.notna(df_current_csv[missing_f].median()): default_val = df_current_csv[missing_f].median()
                     imputed_values_sim_lstm_gs[missing_f] = default_val

             # Bottone Importazione
             fetch_error_gsheet_lstm = None
             if st.button("Importa da Google Sheet (LSTM)", key="sim_run_gsheet_import_lstm"):
                 fetch_sim_gsheet_data.clear()
                 with st.spinner("Recupero dati LSTM da GSheet..."):
                     required_cols_lstm_fetch = feature_columns_current_model + [st.session_state.date_col_name_csv]
                     imported_df_lstm, import_err_lstm, last_ts_lstm = fetch_sim_gsheet_data(
                          GSHEET_ID, input_steps, GSHEET_DATE_COL, GSHEET_DATE_FORMAT,
                          column_mapping_gsheet_to_lstm_filtered,
                          required_cols_lstm_fetch, imputed_values_sim_lstm_gs
                     )
                 if import_err_lstm:
                     st.error(f"Recupero GSheet per LSTM fallito: {import_err_lstm}")
                     st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = import_err_lstm
                 elif imported_df_lstm is not None:
                     try:
                         final_lstm_df = imported_df_lstm[feature_columns_current_model] # Seleziona solo feature input
                         st.success(f"Recuperate e processate {len(final_lstm_df)} righe LSTM da GSheet.")
                         st.session_state.imported_sim_data_gs_df_lstm = final_lstm_df
                         st.session_state.imported_sim_start_time_gs_lstm = last_ts_lstm if last_ts_lstm else datetime.now(italy_tz)
                         fetch_error_gsheet_lstm = None
                         st.rerun() # Rerun per aggiornare stato
                     except KeyError as e_cols_lstm:
                         st.error(f"Errore selezione colonne LSTM dopo fetch: {e_cols_lstm}")
                         st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = f"Errore colonne: {e_cols_lstm}"
                 else: st.error("Recupero GSheet per LSTM non riuscito."); fetch_error_gsheet_lstm = "Errore sconosciuto."

             # Mostra stato caricamento e preview
             imported_df_lstm_gs = st.session_state.get("imported_sim_data_gs_df_lstm", None)
             if imported_df_lstm_gs is not None:
                 st.caption("Dati LSTM importati da Google Sheet pronti.")
                 with st.expander("Mostra dati importati (Input LSTM)"):
                     st.dataframe(imported_df_lstm_gs.round(3))

             # Bottone Esecuzione Simulazione LSTM GSheet
             sim_data_input_lstm_gs = None; sim_start_time_lstm_gs = None
             if isinstance(imported_df_lstm_gs, pd.DataFrame):
                 try:
                     sim_data_input_lstm_gs_ordered = imported_df_lstm_gs[feature_columns_current_model] # Assicura ordine
                     sim_data_input_lstm_gs = sim_data_input_lstm_gs_ordered.astype(float).values
                     sim_start_time_lstm_gs = st.session_state.get("imported_sim_start_time_gs_lstm", datetime.now(italy_tz))
                     if sim_data_input_lstm_gs.shape[0] != input_steps: # Verifica numero righe
                         st.warning(f"Numero righe dati GSheet ({sim_data_input_lstm_gs.shape[0]}) diverso da richiesto ({input_steps}). Verranno usati i dati disponibili, ma il modello potrebbe non funzionare correttamente.")
                     if np.isnan(sim_data_input_lstm_gs).any():
                          st.error("Trovati valori NaN nei dati GSheet importati. Impossibile procedere.")
                          sim_data_input_lstm_gs = None
                 except KeyError as e_key_gs: st.error(f"Colonna mancante nei dati LSTM GSheet: {e_key_gs}"); sim_data_input_lstm_gs = None
                 except Exception as e_prep_gs: st.error(f"Errore preparazione dati LSTM GSheet: {e_prep_gs}"); sim_data_input_lstm_gs = None

             input_ready_lstm_gs = sim_data_input_lstm_gs is not None and sim_data_input_lstm_gs.shape[0] >= input_steps # Richiede almeno N righe
             st.divider()
             if st.button('Esegui Simulazione LSTM (GSheet)', type="primary", disabled=(not input_ready_lstm_gs), key="sim_run_exec_lstm_gsheet"):
                  if input_ready_lstm_gs:
                      # Usa solo le ultime `input_steps` righe se ne abbiamo di più
                      sim_data_final_gs = sim_data_input_lstm_gs[-input_steps:]
                      if sim_data_final_gs.shape[0] < input_steps:
                           st.error(f"Dati GSheet insufficienti ({sim_data_final_gs.shape[0]}) per l'input richiesto ({input_steps}).")
                      else:
                           with st.spinner('Simulazione LSTM (GSheet) in corso...'):
                                scaler_features_lstm, scaler_targets_lstm = active_scalers
                                predictions_sim_lstm = predict(active_model, sim_data_final_gs, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                           # Mostra Risultati
                           if predictions_sim_lstm is not None:
                                output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                                st.subheader(f'Risultato Simulazione LSTM (GSheet): Prossime {total_hours_output_actual:.1f} ore')
                                st.caption(f"Previsione calcolata a partire da: {sim_start_time_lstm_gs.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                                # Tabella
                                pred_times_lstm = [sim_start_time_lstm_gs + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                                rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'};
                                for col in target_columns_model:
                                    unit = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""; new_name = f"{get_station_label(col, short=True)} {unit_str}".strip(); count = 1; final_name = new_name
                                    while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                                    rename_dict_lstm[col] = final_name
                                results_df_lstm_display = results_df_lstm.copy().rename(columns=rename_dict_lstm)
                                st.dataframe(results_df_lstm_display.round(3), hide_index=True)
                                st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_gsheet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                                # Grafici
                                st.subheader('Grafici Previsioni Simulate (LSTM GSheet)')
                                figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, sim_start_time_lstm_gs); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                                for i, fig_sim in enumerate(figs_sim_lstm):
                                    with sim_cols_lstm[i % num_graph_cols_lstm]:
                                         target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                         st.plotly_chart(fig_sim, use_container_width=True)
                                         st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_gsheet_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                           else: st.error("Predizione simulazione LSTM (GSheet) fallita.")
                  else: st.error("Dati input da GSheet non pronti o non importati.")


         elif sim_method == 'Orario Dettagliato (Tabella)':
             st.markdown(f'Inserisci i dati per i **{input_steps}** passi temporali ({input_steps*0.5:.1f} ore) precedenti.')
             st.caption(f"La tabella contiene le **{len(feature_columns_current_model)}** feature di input richieste dal modello.")

             # Crea DataFrame iniziale per l'editor, precompilato da CSV se possibile
             editor_df_initial = pd.DataFrame(index=range(input_steps), columns=feature_columns_current_model)
             if data_ready_csv and len(df_current_csv) >= input_steps:
                 try:
                    last_data_csv = df_current_csv[feature_columns_current_model].iloc[-input_steps:].reset_index(drop=True)
                    editor_df_initial = last_data_csv.astype(float).round(2) # Arrotonda per editor
                    st.caption("Tabella precompilata con gli ultimi dati CSV.")
                 except Exception as e_fill_csv:
                    st.caption(f"Impossibile precompilare con dati CSV ({e_fill_csv}). Inizializzata a 0.")
                    editor_df_initial = editor_df_initial.fillna(0.0)
             else: editor_df_initial = editor_df_initial.fillna(0.0) # Fallback a 0

             # Data Editor
             lstm_editor_key = "lstm_editor_sim"
             edited_lstm_df = st.data_editor(
                 editor_df_initial, key=lstm_editor_key, num_rows="fixed", use_container_width=True,
                 column_config={ # Configurazione dinamica colonne
                     col: st.column_config.NumberColumn(
                         label=get_station_label(col, short=True), help=f"Valore storico per {col}",
                         min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else None,
                         max_value=100.0 if 'umidit' in col.lower() else None,
                         format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f",
                         step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1),
                         required=True ) for col in feature_columns_current_model
                 }
             )

             # Validazione Editor
             sim_data_input_lstm_editor = None; validation_passed_editor = False
             if edited_lstm_df is not None and not edited_lstm_df.isnull().any().any():
                 # Controllo numerico
                 is_all_numeric_editor = True
                 try:
                     edited_lstm_df.astype(float)
                 except ValueError:
                      is_all_numeric_editor = False

                 if is_all_numeric_editor and edited_lstm_df.shape == (input_steps, len(feature_columns_current_model)):
                     try:
                         sim_data_input_lstm_editor = edited_lstm_df[feature_columns_current_model].astype(float).values
                         validation_passed_editor = True
                     except KeyError as e_key_edit: st.error(f"Errore colonna tabella: {e_key_edit}")
                     except Exception as e_conv_edit: st.error(f"Errore conversione dati tabella: {e_conv_edit}")
                 elif not is_all_numeric_editor:
                      st.warning("La tabella contiene valori non numerici.")
                 else: st.warning("Numero righe/colonne errato nella tabella.")
             else: st.warning("Completa o correggi la tabella. Tutti i valori devono essere numerici.")

             input_ready_lstm_editor = sim_data_input_lstm_editor is not None and validation_passed_editor
             st.divider()
             if st.button('Esegui Simulazione LSTM (Tabella)', type="primary", disabled=(not input_ready_lstm_editor), key="sim_run_exec_lstm_editor"):
                  if input_ready_lstm_editor:
                      with st.spinner('Simulazione LSTM (Tabella) in corso...'):
                           scaler_features_lstm, scaler_targets_lstm = active_scalers
                           predictions_sim_lstm = predict(active_model, sim_data_input_lstm_editor, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                      # Mostra Risultati
                      if predictions_sim_lstm is not None:
                           output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione LSTM (Tabella): Prossime {total_hours_output_actual:.1f} ore')
                           start_pred_time_lstm = datetime.now(italy_tz)
                           st.caption(f"Previsione calcolata a partire da: {start_pred_time_lstm.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                           # Tabella
                           pred_times_lstm = [start_pred_time_lstm + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                           rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'};
                           for col in target_columns_model:
                               unit = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""; new_name = f"{get_station_label(col, short=True)} {unit_str}".strip(); count = 1; final_name = new_name
                               while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                               rename_dict_lstm[col] = final_name
                           results_df_lstm_display = results_df_lstm.copy().rename(columns=rename_dict_lstm)
                           st.dataframe(results_df_lstm_display.round(3), hide_index=True)
                           st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_tabella_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                           # Grafici
                           st.subheader('Grafici Previsioni Simulate (LSTM Tabella)')
                           figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, start_pred_time_lstm); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                           for i, fig_sim in enumerate(figs_sim_lstm):
                               with sim_cols_lstm[i % num_graph_cols_lstm]:
                                    target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_tabella_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                      else: st.error("Predizione simulazione LSTM (Tabella) fallita.")
                  else: st.error("Dati input da tabella non pronti o invalidi.")


         elif sim_method == 'Usa Ultime Ore da CSV Caricato':
             st.markdown(f"Utilizza gli ultimi **{input_steps}** steps ({input_steps*0.5:.1f} ore) dai dati CSV caricati.")
             if not data_ready_csv: st.error("Dati CSV non caricati."); st.stop()
             if len(df_current_csv) < input_steps: st.error(f"Dati CSV insufficienti ({len(df_current_csv)} righe), richieste {input_steps}."); st.stop()

             # Estrazione e preparazione dati CSV
             sim_data_input_lstm_csv = None; sim_start_time_lstm_csv = None
             try:
                 latest_csv_data_df = df_current_csv.iloc[-input_steps:]
                 missing_cols_csv = [col for col in feature_columns_current_model if col not in latest_csv_data_df.columns]
                 if missing_cols_csv:
                     st.error(f"Colonne modello LSTM mancanti nel CSV: `{', '.join(missing_cols_csv)}`")
                 else:
                     sim_data_input_lstm_csv_ordered = latest_csv_data_df[feature_columns_current_model] # Ordine corretto
                     sim_data_input_lstm_csv = sim_data_input_lstm_csv_ordered.astype(float).values
                     last_csv_timestamp = df_current_csv[date_col_name_csv].iloc[-1]
                     if pd.notna(last_csv_timestamp): sim_start_time_lstm_csv = last_csv_timestamp
                     else: sim_start_time_lstm_csv = datetime.now(italy_tz) # Fallback
                     st.caption("Dati CSV pronti per la simulazione.")
                     with st.expander("Mostra dati CSV utilizzati (Input LSTM)"):
                         st.dataframe(latest_csv_data_df[[date_col_name_csv] + feature_columns_current_model].round(3))
                     if np.isnan(sim_data_input_lstm_csv).any():
                         st.error("Trovati valori NaN negli ultimi dati CSV. Impossibile procedere.")
                         sim_data_input_lstm_csv = None

             except Exception as e_prep_csv:
                 st.error(f"Errore preparazione dati da CSV: {e_prep_csv}")
                 sim_data_input_lstm_csv = None

             input_ready_lstm_csv = sim_data_input_lstm_csv is not None
             st.divider()
             if st.button('Esegui Simulazione LSTM (CSV)', type="primary", disabled=(not input_ready_lstm_csv), key="sim_run_exec_lstm_csv"):
                 if input_ready_lstm_csv:
                      with st.spinner('Simulazione LSTM (CSV) in corso...'):
                           scaler_features_lstm, scaler_targets_lstm = active_scalers
                           predictions_sim_lstm = predict(active_model, sim_data_input_lstm_csv, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                      # Mostra Risultati
                      if predictions_sim_lstm is not None:
                           output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione LSTM (CSV): Prossime {total_hours_output_actual:.1f} ore')
                           st.caption(f"Previsione calcolata a partire da: {sim_start_time_lstm_csv.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                           # Tabella
                           pred_times_lstm = [sim_start_time_lstm_csv + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                           rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'};
                           for col in target_columns_model:
                               unit = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str = f"({unit.group(1) or unit.group(2)})" if unit else ""; new_name = f"{get_station_label(col, short=True)} {unit_str}".strip(); count = 1; final_name = new_name
                               while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                               rename_dict_lstm[col] = final_name
                           results_df_lstm_display = results_df_lstm.copy().rename(columns=rename_dict_lstm)
                           st.dataframe(results_df_lstm_display.round(3), hide_index=True)
                           st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_csv_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                           # Grafici
                           st.subheader('Grafici Previsioni Simulate (LSTM CSV)')
                           figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, sim_start_time_lstm_csv); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                           for i, fig_sim in enumerate(figs_sim_lstm):
                               with sim_cols_lstm[i % num_graph_cols_lstm]:
                                    target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_csv_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                      else: st.error("Predizione simulazione LSTM (CSV) fallita.")
                 else: st.error("Dati input da CSV non pronti o invalidi.")


# --- PAGINA ANALISI DATI STORICI (Invariata) ---
elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici (da file CSV)')
    if not data_ready_csv:
        st.warning("Dati Storici CSV non disponibili. Caricane uno dalla sidebar per l'analisi.")
    else:
        st.info(f"Dataset CSV caricato: **{len(df_current_csv)}** righe.")
        if date_col_name_csv in df_current_csv and not df_current_csv[date_col_name_csv].isnull().all():
            try:
                min_dt_csv = df_current_csv[date_col_name_csv].min()
                max_dt_csv = df_current_csv[date_col_name_csv].max()
                st.caption(f"Periodo: da **{min_dt_csv.strftime('%d/%m/%Y %H:%M')}** a **{max_dt_csv.strftime('%d/%m/%Y %H:%M')}**")
            except AttributeError: # Se min/max non sono timestamp validi
                 st.warning(f"Date non valide nella colonna '{date_col_name_csv}'. Impossibile determinare il periodo.")
        else: st.warning(f"Colonna data '{date_col_name_csv}' non trovata o vuota per analisi periodo.")

        st.subheader("Esplorazione Dati")
        st.dataframe(df_current_csv.head())
        st.markdown(get_table_download_link(df_current_csv, f"dati_storici_completi_{datetime.now().strftime('%Y%m%d')}.csv", link_text="Scarica Dati Completi (CSV)"), unsafe_allow_html=True)

        st.divider()
        st.subheader("Statistiche Descrittive")
        numeric_cols_analysis = df_current_csv.select_dtypes(include=np.number).columns
        if not numeric_cols_analysis.empty:
            st.dataframe(df_current_csv[numeric_cols_analysis].describe().round(2))
        else: st.info("Nessuna colonna numerica trovata per le statistiche.")

        st.divider()
        st.subheader("Visualizzazione Temporale")
        if date_col_name_csv in df_current_csv and not numeric_cols_analysis.empty and not df_current_csv[date_col_name_csv].isnull().all():
            cols_to_plot = st.multiselect(
                "Seleziona colonne da visualizzare:",
                options=numeric_cols_analysis.tolist(),
                default=numeric_cols_analysis[:min(len(numeric_cols_analysis), 5)].tolist(),
                key="analysis_plot_select"
            )
            if cols_to_plot:
                fig_analysis = go.Figure()
                for col in cols_to_plot:
                    fig_analysis.add_trace(go.Scatter(x=df_current_csv[date_col_name_csv], y=df_current_csv[col], mode='lines', name=get_station_label(col, short=True)))
                fig_analysis.update_layout(
                    title="Andamento Temporale Selezionato", xaxis_title="Data e Ora", yaxis_title="Valore",
                    height=500, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_analysis, use_container_width=True)
                analysis_filename_base = f"analisi_temporale_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_analysis, analysis_filename_base), unsafe_allow_html=True)
            else: st.info("Seleziona almeno una colonna per visualizzare il grafico.")
        else: st.info("Colonna data valida o colonne numeriche mancanti per la visualizzazione temporale.")

        st.divider()
        st.subheader("Matrice di Correlazione")
        if len(numeric_cols_analysis) > 1:
            # Escludi colonne con varianza zero per evitare NaN nella correlazione
            cols_for_corr = df_current_csv[numeric_cols_analysis].loc[:, df_current_csv[numeric_cols_analysis].var() > 1e-9].columns
            if len(cols_for_corr) > 1:
                 corr_matrix = df_current_csv[cols_for_corr].corr()
                 fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                    colorscale='viridis', zmin=-1, zmax=1, colorbar=dict(title='Corr')
                 ))
                 fig_corr.update_layout(
                    title='Matrice di Correlazione tra Variabili Numeriche',
                    xaxis_tickangle=-45, height=600, template="plotly_white",
                    margin=dict(l=80, r=50, t=50, b=150) # Aumenta margini per etichette lunghe
                 )
                 fig_corr.update_xaxes(tickfont=dict(size=10)) # Riduci font etichette
                 fig_corr.update_yaxes(tickfont=dict(size=10))
                 st.plotly_chart(fig_corr, use_container_width=True)
            else:
                 st.info("Nessuna colonna numerica con varianza sufficiente per la matrice di correlazione.")
        else:
            st.info("Sono necessarie almeno due colonne numeriche per calcolare la matrice di correlazione.")


# --- PAGINA ALLENAMENTO MODELLO (Invariata)---
elif page == 'Allenamento Modello':
    st.header('Allenamento Nuovo Modello')
    if not data_ready_csv:
        st.warning("Dati Storici CSV non disponibili. Caricane uno dalla sidebar per avviare l'allenamento.")
        st.stop()

    st.success(f"Dati CSV disponibili per l'allenamento: {len(df_current_csv)} righe.")
    st.subheader('Configurazione Addestramento')

    # Selezione Tipo Modello da Allenare
    train_model_type = st.radio("Tipo di Modello da Allenare:", ["LSTM Standard", "Seq2Seq (Encoder-Decoder)"], key="train_select_type", horizontal=True)

    # Input nome modello (comune)
    default_save_name = f"modello_{train_model_type.split()[0].lower()}_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    save_name_input = st.text_input("Nome base per salvare il modello e i file associati:", default_save_name, key="train_save_filename")
    # Pulisci nome file
    save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_') or "modello_default"
    if save_name != save_name_input: st.caption(f"Nome file valido: `{save_name}`")

    # Crea directory modelli se non esiste
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Parametri Specifici per Tipo ---
    if train_model_type == "LSTM Standard":
        st.markdown("**1. Seleziona Feature e Target (LSTM)**")
        all_features_lstm = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
        # Default features: quelle globali presenti nel CSV
        default_features_lstm = [f for f in st.session_state.feature_columns if f in all_features_lstm]
        selected_features_train_lstm = st.multiselect("Feature Input LSTM:", options=all_features_lstm, default=default_features_lstm, key="train_lstm_feat")
        # Default target: primo livello trovato tra le feature
        level_options_lstm = [f for f in all_features_lstm if 'livello' in f.lower() or '[m]' in f.lower()]
        default_targets_lstm = level_options_lstm[:1]
        selected_targets_train_lstm = st.multiselect("Target Output LSTM (Livelli):", options=level_options_lstm, default=default_targets_lstm, key="train_lstm_target")

        st.markdown("**2. Parametri Modello e Training (LSTM)**")
        with st.expander("Impostazioni Allenamento LSTM", expanded=True):
            c1_lstm, c2_lstm, c3_lstm = st.columns(3)
            with c1_lstm:
                iw_t_lstm = st.number_input("Input Window (steps)", min_value=2, value=48, step=2, key="t_lstm_in", help="Numero di passi (30 min) passati usati come input.")
                ow_t_lstm = st.number_input("Output Window (steps)", min_value=1, value=24, step=1, key="t_lstm_out", help="Numero di passi (30 min) futuri da prevedere.")
                vs_t_lstm = st.slider("Split Validazione (%)", 0, 50, 20, 1, key="t_lstm_vs", help="Percentuale di dati usata per la validazione.")
            with c2_lstm:
                hs_t_lstm = st.number_input("Hidden Size", min_value=8, value=128, step=8, key="t_lstm_hs", help="Numero di unità nascoste nei layer LSTM.")
                nl_t_lstm = st.number_input("Numero Layers", min_value=1, value=2, step=1, key="t_lstm_nl", help="Numero di layer LSTM impilati.")
                dr_t_lstm = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_lstm_dr", help="Tasso di dropout tra i layer LSTM (se > 1 layer).")
            with c3_lstm:
                lr_t_lstm = st.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f", step=1e-4, key="t_lstm_lr", help="Tasso di apprendimento iniziale.")
                bs_t_lstm = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_lstm_bs", help="Dimensione dei batch per l'allenamento.")
                ep_t_lstm = st.number_input("Numero Epoche", min_value=1, value=50, step=5, key="t_lstm_ep", help="Numero di cicli di allenamento completi.")

            col_dev_lstm, col_save_lstm = st.columns(2)
            with col_dev_lstm: device_option_lstm = st.radio("Device Allenamento:", ['Auto (GPU se disponibile)', 'Forza CPU'], index=0, key='train_device_lstm', horizontal=True)
            with col_save_lstm: save_choice_lstm = st.radio("Strategia Salvataggio:", ['Migliore (su Validazione)', 'Modello Finale'], index=0, key='train_save_lstm', horizontal=True)

        st.divider()
        # --- Bottone Avvio Training LSTM ---
        ready_to_train_lstm = bool(save_name and selected_features_train_lstm and selected_targets_train_lstm)
        if st.button("Avvia Addestramento LSTM", type="primary", disabled=not ready_to_train_lstm, key="train_run_lstm"):
             st.info(f"Avvio addestramento LSTM Standard per '{save_name}'...")
             # 1. Prepara Dati
             with st.spinner("Preparazione dati LSTM..."):
                  # Usa direttamente gli STEPS nei parametri
                  X_tr, y_tr, X_v, y_v, sc_f, sc_t = prepare_training_data(
                       df_current_csv.copy(), selected_features_train_lstm, selected_targets_train_lstm,
                       iw_t_lstm, # Passa steps
                       ow_t_lstm, # Passa steps
                       val_split=vs_t_lstm
                  )
             if X_tr is None or y_tr is None or sc_f is None or sc_t is None:
                  st.error("Preparazione dati LSTM fallita. Controlla i parametri e i dati CSV.")
                  st.stop()
             if vs_t_lstm > 0 and (X_v is None or y_v is None):
                  st.warning("Split di validazione richiesto ma set di validazione vuoto. Allenamento continua solo su training set.")
             elif vs_t_lstm == 0 and X_v is not None and len(X_v)>0:
                 st.info("Split validazione 0%, nessun set di validazione sarà usato.")
                 X_v, y_v = None, None # Forza a None se split è 0

             # 2. Allena Modello
             trained_model_lstm, train_loss_history, val_loss_history = train_model(
                   X_tr, y_tr, X_v, y_v, len(selected_features_train_lstm), len(selected_targets_train_lstm), ow_t_lstm, # Passa output STEPS
                   hs_t_lstm, nl_t_lstm, ep_t_lstm, bs_t_lstm, lr_t_lstm, dr_t_lstm,
                   save_strategy=('migliore' if 'Migliore' in save_choice_lstm else 'finale'),
                   preferred_device=('auto' if 'Auto' in device_option_lstm else 'cpu')
             )

             # 3. Salva Risultati
             if trained_model_lstm:
                  st.success("Addestramento LSTM completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      m_path = f"{base_path}.pth"; torch.save(trained_model_lstm.state_dict(), m_path)
                      sf_path = f"{base_path}_features.joblib"; joblib.dump(sc_f, sf_path)
                      st_path = f"{base_path}_targets.joblib"; joblib.dump(sc_t, st_path)
                      c_path = f"{base_path}.json"
                      config_save_lstm = {
                          "model_type": "LSTM", "input_window": iw_t_lstm, "output_window": ow_t_lstm, # Salva STEPS
                          "hidden_size": hs_t_lstm, "num_layers": nl_t_lstm, "dropout": dr_t_lstm,
                          "feature_columns": selected_features_train_lstm, "target_columns": selected_targets_train_lstm,
                          "training_date": datetime.now(italy_tz).isoformat(), "val_split_percent": vs_t_lstm,
                          "learning_rate": lr_t_lstm, "batch_size": bs_t_lstm, "epochs": ep_t_lstm,
                          "display_name": save_name, # Usa nome file come display name
                      }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_lstm, f, indent=4)

                      st.success(f"Modello LSTM '{save_name}' salvato in '{MODELS_DIR}'.")
                      st.subheader("Download File Modello LSTM")
                      col_dl_lstm1, col_dl_lstm2 = st.columns(2)
                      with col_dl_lstm1:
                          st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True)
                          st.markdown(get_download_link_for_file(sf_path, "Scaler Features (.joblib)"), unsafe_allow_html=True)
                      with col_dl_lstm2:
                          st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True)
                          st.markdown(get_download_link_for_file(st_path, "Scaler Target (.joblib)"), unsafe_allow_html=True)
                      find_available_models.clear() # Aggiorna cache modelli

                  except Exception as e_save:
                      st.error(f"Errore salvataggio modello LSTM: {e_save}")
                      st.error(traceback.format_exc())
             else:
                 st.error("Addestramento LSTM fallito. Modello non salvato.")


    elif train_model_type == "Seq2Seq (Encoder-Decoder)":
         st.markdown("**1. Seleziona Feature (Seq2Seq)**")
         features_present_in_csv_s2s = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
         # Input Encoder: tutte le feature storiche da considerare
         default_past_features_s2s = [f for f in st.session_state.feature_columns if f in features_present_in_csv_s2s] # Default globali
         selected_past_features_s2s = st.multiselect("Feature Storiche (Input Encoder):", options=features_present_in_csv_s2s, default=default_past_features_s2s, key="train_s2s_past_feat", help="Tutte le feature considerate dal passato.")
         # Input Decoder: sottoinsieme delle feature storiche per cui si avranno previsioni
         options_forecast = selected_past_features_s2s # Può scegliere solo tra quelle passate
         default_forecast_cols = [f for f in options_forecast if 'pioggia' in f.lower() or 'cumulata' in f.lower() or f == HUMIDITY_COL_NAME] # Default piogge/umidità
         selected_forecast_features_s2s = st.multiselect("Feature Forecast (Input Decoder):", options=options_forecast, default=default_forecast_cols, key="train_s2s_forecast_feat", help="Feature per cui verranno fornite previsioni future al modello.")
         # Output: livelli da prevedere (devono essere tra le feature storiche)
         level_options_s2s = [f for f in selected_past_features_s2s if 'livello' in f.lower() or '[m]' in f.lower()]
         default_targets_s2s = level_options_s2s[:1] # Default primo livello
         selected_targets_s2s = st.multiselect("Target Output (Livelli):", options=level_options_s2s, default=default_targets_s2s, key="train_s2s_target_feat", help="Livelli idrometrici che il modello dovrà prevedere.")

         st.markdown("**2. Parametri Finestre e Training (Seq2Seq)**")
         with st.expander("Impostazioni Allenamento Seq2Seq", expanded=True):
             c1_s2s, c2_s2s, c3_s2s = st.columns(3)
             with c1_s2s:
                 iw_steps_s2s = st.number_input("Input Storico (steps)", min_value=2, value=48, step=2, key="t_s2s_in", help="Numero passi (30 min) storici per l'encoder.")
                 fw_steps_s2s = st.number_input("Input Forecast (steps)", min_value=1, value=24, step=1, key="t_s2s_fore", help="Numero passi (30 min) futuri forniti al decoder.")
                 ow_steps_s2s = st.number_input("Output Previsione (steps)", min_value=1, value=24, step=1, key="t_s2s_out", help="Numero passi (30 min) futuri da prevedere.")
                 if ow_steps_s2s > fw_steps_s2s: st.caption("Nota: Output Steps > Forecast Steps. Il modello userà padding durante la predizione.")
                 vs_t_s2s = st.slider("Split Validazione (%)", 0, 50, 20, 1, key="t_s2s_vs")
             with c2_s2s:
                 hs_t_s2s = st.number_input("Hidden Size", min_value=8, value=128, step=8, key="t_s2s_hs", help="Unità nascoste per Encoder/Decoder LSTM.")
                 nl_t_s2s = st.number_input("Numero Layers", min_value=1, value=2, step=1, key="t_s2s_nl", help="Numero layer LSTM per Encoder/Decoder.")
                 dr_t_s2s = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_s2s_dr", help="Dropout tra i layer.")
             with c3_s2s:
                 lr_t_s2s = st.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f", step=1e-4, key="t_s2s_lr")
                 bs_t_s2s = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_s2s_bs")
                 ep_t_s2s = st.number_input("Numero Epoche", min_value=1, value=50, step=5, key="t_s2s_ep")

             col_dev_s2s, col_save_s2s = st.columns(2)
             with col_dev_s2s: device_option_s2s = st.radio("Device Allenamento:", ['Auto (GPU se disponibile)', 'Forza CPU'], index=0, key='train_device_s2s', horizontal=True)
             with col_save_s2s: save_choice_s2s = st.radio("Strategia Salvataggio:", ['Migliore (su Validazione)', 'Modello Finale'], index=0, key='train_save_s2s', horizontal=True)

         st.divider()
         # --- Bottone Avvio Training Seq2Seq ---
         ready_to_train_s2s = bool(save_name and selected_past_features_s2s and selected_forecast_features_s2s and selected_targets_s2s)
         if st.button("Avvia Addestramento Seq2Seq", type="primary", disabled=not ready_to_train_s2s, key="train_run_s2s"):
             st.info(f"Avvio addestramento Seq2Seq per '{save_name}'...")
             # 1. Prepara Dati Seq2Seq
             with st.spinner("Preparazione dati Seq2Seq..."):
                  data_tuple = prepare_training_data_seq2seq(
                       df_current_csv.copy(), selected_past_features_s2s, selected_forecast_features_s2s, selected_targets_s2s,
                       iw_steps_s2s, fw_steps_s2s, ow_steps_s2s, vs_t_s2s
                  )
             if data_tuple is None or len(data_tuple) != 9 or data_tuple[0] is None:
                  st.error("Preparazione dati Seq2Seq fallita. Controlla parametri e dati."); st.stop()
             (X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v,
              sc_past, sc_fore, sc_tar) = data_tuple
             if not all([sc_past, sc_fore, sc_tar]): # Verifica scaler
                  st.error("Errore: Scaler Seq2Seq non generati."); st.stop()
             # Verifica set validazione
             if vs_t_s2s > 0 and (X_enc_v is None or X_enc_v.size == 0):
                 st.warning("Split validazione richiesto ma set validazione vuoto/non valido. Allenamento solo su training set.")
             elif vs_t_s2s == 0 and X_enc_v is not None and X_enc_v.size > 0:
                 st.info("Split validazione 0%, nessun set di validazione sarà usato.")
                 X_enc_v, X_dec_v, y_tar_v = None, None, None # Forza a None

             # 2. Istanzia e Allena Modello Seq2Seq
             try:
                 enc = EncoderLSTM(len(selected_past_features_s2s), hs_t_s2s, nl_t_s2s, dr_t_s2s)
                 dec = DecoderLSTM(len(selected_forecast_features_s2s), hs_t_s2s, len(selected_targets_s2s), nl_t_s2s, dr_t_s2s)
             except Exception as e_init:
                 st.error(f"Errore inizializzazione modello Seq2Seq: {e_init}"); st.stop()

             # Chiamata funzione training
             trained_model_s2s, train_loss_hist_s2s, val_loss_hist_s2s = train_model_seq2seq(
                  X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v, # Passa set validazione (possono essere None)
                  enc, dec, ow_steps_s2s, # Passa output STEPS
                  ep_t_s2s, bs_t_s2s, lr_t_s2s,
                  save_strategy=('migliore' if 'Migliore' in save_choice_s2s else 'finale'),
                  preferred_device=('auto' if 'Auto' in device_option_s2s else 'cpu'),
                  teacher_forcing_ratio_schedule=[0.6, 0.1] # Esempio: decadimento da 0.6 a 0.1
             )

             # 3. Salva Risultati Seq2Seq
             if trained_model_s2s:
                  st.success("Addestramento Seq2Seq completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      m_path = f"{base_path}.pth"; torch.save(trained_model_s2s.state_dict(), m_path)
                      sp_path = f"{base_path}_past_features.joblib"; joblib.dump(sc_past, sp_path)
                      sf_path = f"{base_path}_forecast_features.joblib"; joblib.dump(sc_fore, sf_path)
                      st_path = f"{base_path}_targets.joblib"; joblib.dump(sc_tar, st_path)
                      c_path = f"{base_path}.json"
                      config_save_s2s = {
                          "model_type": "Seq2Seq", "input_window_steps": iw_steps_s2s, "forecast_window_steps": fw_steps_s2s, "output_window_steps": ow_steps_s2s,
                          "hidden_size": hs_t_s2s, "num_layers": nl_t_s2s, "dropout": dr_t_s2s,
                          "all_past_feature_columns": selected_past_features_s2s, "forecast_input_columns": selected_forecast_features_s2s, "target_columns": selected_targets_s2s,
                          "training_date": datetime.now(italy_tz).isoformat(), "val_split_percent": vs_t_s2s,
                          "learning_rate": lr_t_s2s, "batch_size": bs_t_s2s, "epochs": ep_t_s2s,
                          "display_name": save_name,
                      }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_s2s, f, indent=4)
                      st.success(f"Modello Seq2Seq '{save_name}' salvato in '{MODELS_DIR}'.")
                      # Link download file
                      st.subheader("Download File Modello Seq2Seq")
                      col_dl_s2s_1, col_dl_s2s_2 = st.columns(2)
                      with col_dl_s2s_1:
                          st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True)
                          st.markdown(get_download_link_for_file(sp_path,"Scaler Passato (.joblib)"), unsafe_allow_html=True)
                          st.markdown(get_download_link_for_file(st_path,"Scaler Target (.joblib)"), unsafe_allow_html=True)
                      with col_dl_s2s_2:
                          st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True)
                          st.markdown(get_download_link_for_file(sf_path,"Scaler Forecast (.joblib)"), unsafe_allow_html=True)
                      find_available_models.clear() # Aggiorna cache modelli

                  except Exception as e_save_s2s:
                      st.error(f"Errore salvataggio modello Seq2Seq: {e_save_s2s}")
                      st.error(traceback.format_exc())
             else:
                 st.error("Addestramento Seq2Seq fallito. Modello non salvato.")

# --- Footer (Invariato) ---
st.sidebar.divider()
st.sidebar.caption(f'Modello Predittivo Idrologico © {datetime.now().year}')
